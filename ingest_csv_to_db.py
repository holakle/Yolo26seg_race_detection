"""Backfill existing run folders (CSV outputs) into the SQLite store.

Use this to load runs produced before the pipeline wrote to the DB directly, or
to re-index a whole test_matrix/ or full_video_chunks/ tree. Re-ingest is
idempotent and preserves any human review (verified_bib) already recorded for a
run, matching on (track_id, frame).
"""

import argparse
import csv
from pathlib import Path

import db
from common import CROSSINGS_HEADER, TRACK_CROSSINGS_HEADER
from summarize_tests import read_metrics


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_meta(run_dir, source_default=""):
    metrics = read_metrics(run_dir / "run.log")  # prefers metrics.json
    def num(key):
        val = metrics.get(key, "")
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    return {
        "source": metrics.get("source", source_default),
        "stride": num("stride"),
        "frame_start": num("frame_start"),
        "frame_end": num("frame_end"),
        "fps": num("fps"),
        "tracking_elapsed_sec": num("tracking_elapsed_sec"),
        "ocr_elapsed_sec": num("ocr_elapsed_sec"),
        "total_elapsed_sec": num("total_elapsed_sec"),
        "ocr_calls": num("ocr_calls"),
        "crossing_events": num("crossing_events"),
    }


def ingest_run(conn, run_dir, start_list_rows=None, result_list_rows=None):
    crossings = read_csv(run_dir / "crossings.csv")
    tracks = read_csv(run_dir / "track_crossings.csv")
    if not crossings and not tracks:
        return None

    run_id = db.upsert_run(conn, run_dir, run_meta(run_dir))
    # Preserve any human review before we clear+reinsert this run's rows.
    preserved = db.get_verified_map(conn, run_id)
    db.clear_run_rows(conn, run_id)

    for row in crossings:
        db.insert_crossing(conn, run_id, [row.get(c, "") for c in CROSSINGS_HEADER])
    for row in tracks:
        db.insert_track_crossing(conn, run_id, [row.get(c, "") for c in TRACK_CROSSINGS_HEADER])
    conn.commit()

    if preserved:
        db.apply_verified_map(conn, run_id, preserved)
    if start_list_rows:
        db.import_list(conn, run_id, start_list_rows, "start_list")
    if result_list_rows:
        db.import_list(conn, run_id, result_list_rows, "result_list")
    return run_id, len(crossings), len(tracks)


def find_run_dirs(root):
    root = Path(root)
    if (root / "crossings.csv").exists() or (root / "track_crossings.csv").exists():
        return [root]
    dirs = set()
    for name in ("crossings.csv", "track_crossings.csv"):
        for path in root.rglob(name):
            dirs.add(path.parent)
    return sorted(dirs)


def load_list(path):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Ingest run folders (crossings.csv / track_crossings.csv) into the SQLite store.")
    p.add_argument("--root", required=True, help="A run dir or a tree of run dirs (test_matrix/, full_video_chunks/).")
    p.add_argument("--db", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\race.db")
    p.add_argument("--start-list", help="Optional start-list CSV to attach to each ingested run.")
    p.add_argument("--result-list", help="Optional result-list CSV to attach to each ingested run.")
    args = p.parse_args()

    conn = db.connect(args.db)
    start_rows = load_list(args.start_list)
    result_rows = load_list(args.result_list)

    run_dirs = find_run_dirs(args.root)
    ingested = 0
    for run_dir in run_dirs:
        result = ingest_run(conn, run_dir, start_rows, result_rows)
        if result is None:
            continue
        run_id, n_cross, n_track = result
        ingested += 1
        print(f"ingested run_id={run_id} {run_dir} crossings={n_cross} tracks={n_track}")
    conn.close()
    print(f"db: {args.db}")
    print(f"runs_ingested: {ingested}")


if __name__ == "__main__":
    main()
