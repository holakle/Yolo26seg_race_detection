import argparse
import csv
from pathlib import Path


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def event_reason(a, b, time_window, x_window):
    dt = abs(as_float(a["time_sec"]) - as_float(b["time_sec"]))
    dx = abs(as_float(a["x"]) - as_float(b["x"]))
    if dt > time_window or dx > x_window:
        return ""

    bib_a = a.get("matched_bib") or a.get("ocr_digits")
    bib_b = b.get("matched_bib") or b.get("ocr_digits")
    if bib_a and bib_b and bib_a == bib_b:
        return f"same_bib dt={dt:.2f}s dx={dx:.0f}px"
    return ""


def representative_key(row):
    exact = row.get("match_type") == "exact"
    accepted = row.get("annotation_status") == "accepted"
    has_digits = bool(row.get("ocr_digits") or row.get("matched_bib"))
    is_line = row.get("event_source") == "line_crossing"
    return (accepted, exact, has_digits, is_line, -as_float(row["frame"]))


def duplicate_score(rows):
    if len(rows) < 2:
        return 0.0
    has_line = any(r.get("event_source") == "line_crossing" for r in rows)
    has_lost = any(r.get("event_source") == "lost_track_fallback" for r in rows)
    has_blank = any(not (r.get("ocr_digits") or r.get("matched_bib")) for r in rows)
    has_read = any(r.get("ocr_digits") or r.get("matched_bib") for r in rows)
    score = 0.55
    if has_line and has_lost:
        score += 0.20
    if has_blank and has_read:
        score += 0.15
    if len(rows) > 2:
        score += 0.10
    return min(score, 0.99)


def main():
    p = argparse.ArgumentParser(description="Post-process raw tracker crossings and flag likely duplicate IDs.")
    p.add_argument("--track-crossings", required=True, help="track_crossings.csv or merged_track_crossings.csv")
    p.add_argument("--crossings", required=True, help="crossings.csv or merged_crossings.csv")
    p.add_argument("--out", help="Output CSV. Defaults to suspected_duplicates.csv beside track crossings.")
    p.add_argument("--time-window-sec", type=float, default=2.0)
    p.add_argument("--x-window-px", type=float, default=300.0)
    args = p.parse_args()

    track_rows = load_csv(args.track_crossings)
    ocr_by_track = {r["track_id"]: r for r in load_csv(args.crossings)}
    rows = []
    for row in track_rows:
        merged = dict(row)
        merged.update({k: v for k, v in ocr_by_track.get(row["track_id"], {}).items() if k not in merged or v})
        rows.append(merged)

    rows.sort(key=lambda r: (as_float(r["frame"]), as_float(r["x"])))
    uf = UnionFind(len(rows))
    reasons = {}

    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if as_float(b["time_sec"]) - as_float(a["time_sec"]) > args.time_window_sec:
                break
            reason = event_reason(a, b, args.time_window_sec, args.x_window_px)
            if reason:
                uf.union(i, j)
                reasons.setdefault(tuple(sorted((i, j))), reason)

    blank_lost = [
        i for i, row in enumerate(rows)
        if row.get("event_source") == "lost_track_fallback" and not (row.get("ocr_digits") or row.get("matched_bib"))
    ]
    line_events = [
        i for i, row in enumerate(rows)
        if row.get("event_source") == "line_crossing"
    ]

    # Lost blank IDs often appear after the real line-crossing ID drops. Cluster
    # those blanks first, then attach each cluster to one prior line event.
    blank_groups = []
    used_blanks = set()
    for i in blank_lost:
        if i in used_blanks:
            continue
        group = [i]
        used_blanks.add(i)
        changed = True
        while changed:
            changed = False
            for j in blank_lost:
                if j in used_blanks:
                    continue
                if any(event_reason(rows[j], rows[k], args.time_window_sec, args.x_window_px) or (
                    abs(as_float(rows[j]["time_sec"]) - as_float(rows[k]["time_sec"])) <= args.time_window_sec
                    and abs(as_float(rows[j]["x"]) - as_float(rows[k]["x"])) <= args.x_window_px
                ) for k in group):
                    group.append(j)
                    used_blanks.add(j)
                    changed = True
        blank_groups.append(group)

    for group in blank_groups:
        first = min(group, key=lambda idx: as_float(rows[idx]["time_sec"]))
        first_row = rows[first]
        candidates = []
        for line_idx in line_events:
            line = rows[line_idx]
            dt = as_float(first_row["time_sec"]) - as_float(line["time_sec"])
            dx = abs(as_float(first_row["x"]) - as_float(line["x"]))
            if 0 <= dt <= args.time_window_sec and dx <= args.x_window_px:
                candidates.append((dt + dx / max(args.x_window_px, 1.0), line_idx, dt, dx))
        if not candidates:
            continue
        _, line_idx, dt, dx = min(candidates)
        for blank_idx in group:
            uf.union(line_idx, blank_idx)
            reasons.setdefault(
                tuple(sorted((line_idx, blank_idx))),
                f"blank_lost_after_line_crossing dt={dt:.2f}s dx={dx:.0f}px",
            )

    groups = {}
    for i in range(len(rows)):
        groups.setdefault(uf.find(i), []).append(i)
    duplicate_groups = [idxs for idxs in groups.values() if len(idxs) > 1]

    out_path = Path(args.out) if args.out else Path(args.track_crossings).with_name("suspected_duplicates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "duplicate_group_id", "is_group_representative", "duplicate_score", "duplicate_reason",
        "track_id", "frame", "time_sec", "direction", "x", "y", "event_source", "candidate_count",
        "ocr_scanned_count", "ocr_digits", "ocr_score", "matched_bib", "match_type",
        "match_probability", "annotation_status", "review_reason",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        group_no = 0
        for idxs in duplicate_groups:
            group_no += 1
            group_rows = [rows[i] for i in idxs]
            rep = max(group_rows, key=representative_key)
            reason_text = "; ".join(sorted(set(
                reasons.get(tuple(sorted((a, b))), "")
                for pos, a in enumerate(idxs)
                for b in idxs[pos + 1:]
                if reasons.get(tuple(sorted((a, b))), "")
            )))
            score = f"{duplicate_score(group_rows):.2f}"
            for row in group_rows:
                writer.writerow({
                    "duplicate_group_id": group_no,
                    "is_group_representative": "yes" if row is rep else "no",
                    "duplicate_score": score,
                    "duplicate_reason": reason_text,
                    **{name: row.get(name, "") for name in fieldnames[4:]},
                })

    print(f"events_read: {len(rows)}")
    print(f"duplicate_groups: {len(duplicate_groups)}")
    print(f"duplicate_rows: {sum(len(g) for g in duplicate_groups)}")
    print(f"suspected_duplicates: {out_path}")


if __name__ == "__main__":
    main()
