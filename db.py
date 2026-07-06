"""SQLite store for pipeline runs, crossings, and human review.

Chosen as the primary store because it is the same embedded, zero-config DB the
edge-device code can use: one file, no server. The desktop pipeline can still
export CSVs during the transition (`--export-csv`), but the DB is a complete
record on its own, including the human-review columns the Streamlit UI writes.

Schema is driven by ``common.CROSSINGS_HEADER`` so column names live in one place.
"""

import sqlite3
from datetime import datetime, timezone

from common import CROSSINGS_HEADER, START_LIST_FIELDS, TRACK_CROSSINGS_HEADER

# crossings columns beyond the shared CSV header: the human-review fields.
REVIEW_FIELDS = ["verified_bib", "verified_by", "verified_utc", "verify_note"]


def _cols(names):
    return ", ".join(names)


def connect(db_path):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL + a generous busy timeout let parallel chunk workers share one DB file
    # (concurrent readers, serialized writers) without "database is locked" errors.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    init_schema(conn)
    return conn


def init_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY,
            run_dir TEXT UNIQUE,
            source TEXT,
            created_utc TEXT,
            stride INTEGER,
            frame_start INTEGER,
            frame_end INTEGER,
            fps REAL,
            tracking_elapsed_sec REAL,
            ocr_elapsed_sec REAL,
            total_elapsed_sec REAL,
            ocr_calls INTEGER,
            crossing_events INTEGER
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS track_crossings (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
            {_cols(f'{c} TEXT' for c in TRACK_CROSSINGS_HEADER)}
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS crossings (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
            {_cols(f'{c} TEXT' for c in CROSSINGS_HEADER)},
            verified_bib TEXT DEFAULT NULL,
            verified_by TEXT,
            verified_utc TEXT,
            verify_note TEXT
        )
        """
    )
    for name in ("start_list", "result_list"):
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
                {_cols(f'{c} TEXT' for c in START_LIST_FIELDS)}
            )
            """
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crossings_run_status ON crossings(run_id, annotation_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crossings_matched_bib ON crossings(matched_bib)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crossings_verified_bib ON crossings(verified_bib)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_track_run_track ON track_crossings(run_id, track_id)")
    conn.commit()


def upsert_run(conn, run_dir, meta):
    """Insert or replace a run by its directory; returns run_id."""
    run_dir = str(run_dir)
    fields = ["run_dir", "source", "created_utc", "stride", "frame_start", "frame_end", "fps",
              "tracking_elapsed_sec", "ocr_elapsed_sec", "total_elapsed_sec", "ocr_calls", "crossing_events"]
    values = [run_dir] + [meta.get(f) for f in fields[1:]]
    if not meta.get("created_utc"):
        values[2] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = conn.execute("SELECT run_id FROM runs WHERE run_dir = ?", (run_dir,)).fetchone()
    if existing:
        run_id = existing["run_id"]
        conn.execute(
            f"UPDATE runs SET {', '.join(f'{f} = ?' for f in fields[1:])} WHERE run_id = ?",
            values[1:] + [run_id],
        )
    else:
        cur = conn.execute(
            f"INSERT INTO runs ({_cols(fields)}) VALUES ({_cols('?' for _ in fields)})",
            values,
        )
        run_id = cur.lastrowid
    conn.commit()
    return run_id


def update_run_metrics(conn, run_id, meta):
    fields = ["source", "stride", "frame_start", "frame_end", "fps",
              "tracking_elapsed_sec", "ocr_elapsed_sec", "total_elapsed_sec", "ocr_calls", "crossing_events"]
    present = [(f, meta[f]) for f in fields if f in meta]
    if not present:
        return
    conn.execute(
        f"UPDATE runs SET {', '.join(f'{f} = ?' for f, _ in present)} WHERE run_id = ?",
        [v for _, v in present] + [run_id],
    )
    conn.commit()


def clear_run_rows(conn, run_id):
    conn.execute("DELETE FROM crossings WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM track_crossings WHERE run_id = ?", (run_id,))


def insert_crossing(conn, run_id, row):
    """row is a CROSSINGS_HEADER-ordered sequence (as written to crossings.csv)."""
    cols = ["run_id"] + CROSSINGS_HEADER
    values = [run_id] + [str(v) for v in row]
    conn.execute(
        f"INSERT INTO crossings ({_cols(cols)}) VALUES ({_cols('?' for _ in cols)})",
        values,
    )


def insert_track_crossing(conn, run_id, row):
    cols = ["run_id"] + TRACK_CROSSINGS_HEADER
    values = [run_id] + [str(v) for v in row]
    conn.execute(
        f"INSERT INTO track_crossings ({_cols(cols)}) VALUES ({_cols('?' for _ in cols)})",
        values,
    )


def import_list(conn, run_id, rows, table):
    conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    cols = ["run_id"] + START_LIST_FIELDS
    conn.executemany(
        f"INSERT INTO {table} ({_cols(cols)}) VALUES ({_cols('?' for _ in cols)})",
        [[run_id] + [row.get(f, "") for f in START_LIST_FIELDS] for row in rows],
    )
    conn.commit()


def set_verified_bib(conn, crossing_id, bib, by="ui", note=""):
    conn.execute(
        "UPDATE crossings SET verified_bib = ?, verified_by = ?, verified_utc = ?, verify_note = ? WHERE id = ?",
        (bib, by, datetime.now(timezone.utc).isoformat(timespec="seconds"), note, crossing_id),
    )
    conn.commit()


def get_verified_map(conn, run_id):
    """(track_id, frame) -> (verified_bib, verified_by, verified_utc, verify_note) for a run."""
    out = {}
    for r in conn.execute(
        "SELECT track_id, frame, verified_bib, verified_by, verified_utc, verify_note "
        "FROM crossings WHERE run_id = ? AND verified_bib IS NOT NULL",
        (run_id,),
    ):
        out[(r["track_id"], r["frame"])] = (r["verified_bib"], r["verified_by"], r["verified_utc"], r["verify_note"])
    return out


def apply_verified_map(conn, run_id, verified):
    """Re-apply preserved human review after a re-ingest (match on track_id+frame)."""
    for (track_id, frame), (bib, by, utc, note) in verified.items():
        conn.execute(
            "UPDATE crossings SET verified_bib = ?, verified_by = ?, verified_utc = ?, verify_note = ? "
            "WHERE run_id = ? AND track_id = ? AND frame = ?",
            (bib, by, utc, note, run_id, str(track_id), str(frame)),
        )
    conn.commit()
