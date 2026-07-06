import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_REFERENCE = "3018,3933,4163,3987,16565,3883,4761,3646,3460,2621,3942,3456,4150,3713,4338,4122,4404,4902,3592,3893,158,4403,3958,4315,3638,3544"


def parse_args():
    p = argparse.ArgumentParser(description="Summarize completed test-matrix crossings.csv files.")
    p.add_argument("--root", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\test_matrix")
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--watch-bibs", default="3544,3460,4315,4761,16565,4163")
    return p.parse_args()


def read_metrics(log_path):
    metrics = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if re.match(r"^[a-z_]+$", key.strip()):
                metrics[key.strip()] = value.strip()
    return metrics


def bib_sequence(rows):
    return [row.get("matched_bib") or row.get("ocr_digits") for row in rows if row.get("matched_bib") or row.get("ocr_digits")]


def main():
    args = parse_args()
    root = Path(args.root)
    reference = [bib.strip() for bib in args.reference.split(",") if bib.strip()]
    watch_bibs = {bib.strip() for bib in args.watch_bibs.split(",") if bib.strip()}
    summary_rows = []
    missing_counts = Counter()
    extra_counts = Counter()

    for csv_path in sorted(root.glob("*/crossings.csv")):
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        preds = bib_sequence(rows)
        missing = [bib for bib in reference if bib not in preds]
        extra = [bib for bib in preds if bib not in reference]
        missing_counts.update(missing)
        extra_counts.update(extra)
        multi = [row for row in rows if int(row.get("ocr_scanned_count") or 0) > 1]
        multi_helped = [row for row in multi if (row.get("matched_bib") or row.get("ocr_digits")) in reference]
        metrics = read_metrics(csv_path.with_name("run.log"))
        summary_rows.append({
            "run": csv_path.parent.name,
            "events": len(rows),
            "predicted_count": len(preds),
            "ref_hits": len([bib for bib in reference if bib in preds]),
            "missing": ",".join(missing),
            "extra": ",".join(extra),
            "exact": sum(1 for row in rows if row.get("match_type") == "exact"),
            "fuzzy": sum(1 for row in rows if row.get("match_type") == "fuzzy"),
            "no_match": sum(1 for row in rows if not row.get("match_type")),
            "accepted": sum(1 for row in rows if row.get("annotation_status") == "accepted"),
            "review": sum(1 for row in rows if row.get("annotation_status") == "review"),
            "multi_scan_rows": len(multi),
            "multi_scan_helped": len(multi_helped),
            "tracking_elapsed_sec": metrics.get("tracking_elapsed_sec", ""),
            "ocr_elapsed_sec": metrics.get("ocr_elapsed_sec", ""),
            "total_elapsed_sec": metrics.get("total_elapsed_sec", ""),
            "tracking_real_time_factor": metrics.get("tracking_real_time_factor", ""),
            "total_real_time_factor": metrics.get("total_real_time_factor", ""),
            "ocr_calls": metrics.get("ocr_calls", ""),
        })

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "test_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["run"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    watch_path = root / "watchlist.csv"
    with watch_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bib", "type", "count", "watch_bib"])
        writer.writeheader()
        for bib, count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"bib": bib, "type": "missing", "count": count, "watch_bib": bib in watch_bibs})
        for bib, count in sorted(extra_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"bib": bib, "type": "extra", "count": count, "watch_bib": bib in watch_bibs})

    print(f"summary: {summary_path}")
    print(f"watchlist: {watch_path}")
    print(f"runs: {len(summary_rows)}")


if __name__ == "__main__":
    main()
