import argparse
import csv
import re
import statistics
from pathlib import Path


def digits_only(text):
    return "".join(re.findall(r"\d+", text or ""))


def finish_time_seconds(value):
    parts = str(value or "").strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = [int(part) for part in parts]
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def load_results(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {digits_only(row.get("bib_number")): row for row in rows if digits_only(row.get("bib_number"))}


def analyze(crossings_path, result_list_path, output_path, window_min):
    result_by_bib = load_results(result_list_path)
    with Path(crossings_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    checked = []
    for row in rows:
        bib = digits_only(row.get("matched_bib") or row.get("ocr_digits"))
        result = result_by_bib.get(bib, {})
        finish_time = result.get("finish_time", "")
        finish_sec = finish_time_seconds(finish_time)
        row["analysis_result_finish_time"] = finish_time
        row["analysis_result_time_delta_sec"] = ""
        row["analysis_result_time_likely"] = ""
        row["analysis_result_time_reason"] = ""
        if finish_sec is not None:
            checked.append((row, finish_sec))

    median_sec = statistics.median(finish_sec for _, finish_sec in checked) if checked else None
    window_sec = window_min * 60
    for row, finish_sec in checked:
        delta = finish_sec - median_sec
        row["analysis_result_time_delta_sec"] = f"{delta:.0f}"
        row["analysis_result_time_likely"] = "yes" if abs(delta) <= window_sec else "no"
        row["analysis_result_time_reason"] = (
            f"within {window_min:g} min of detected clip median"
            if abs(delta) <= window_sec
            else f"outside {window_min:g} min of detected clip median"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    for name in [
        "analysis_result_finish_time",
        "analysis_result_time_delta_sec",
        "analysis_result_time_likely",
        "analysis_result_time_reason",
    ]:
        if name not in fieldnames:
            fieldnames.append(name)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    likely_no = sum(1 for row in rows if row.get("analysis_result_time_likely") == "no")
    return len(rows), len(checked), likely_no


def parse_args():
    p = argparse.ArgumentParser(description="Post-pipeline analysis for crossings.csv. Does not run YOLO, tracking, or OCR.")
    p.add_argument("--crossings", required=True, help="Existing crossings.csv from yolo26_line_crossing.py.")
    p.add_argument("--result-list", required=True, help="Result CSV with bib_number and finish_time columns.")
    p.add_argument("--out", help="Analysis CSV path. Defaults to crossings_analysis.csv beside crossings.csv.")
    p.add_argument("--result-time-window-min", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    crossings = Path(args.crossings)
    out = Path(args.out) if args.out else crossings.with_name("crossings_analysis.csv")
    rows, checked, unlikely = analyze(crossings, args.result_list, out, args.result_time_window_min)
    print(f"analysis_csv: {out}")
    print(f"rows: {rows}")
    print(f"result_time_checked: {checked}")
    print(f"result_time_unlikely: {unlikely}")


if __name__ == "__main__":
    main()
