"""Offline A/B benchmark for bib-digit readers over already-saved candidate crops.

Runs each backend on the PNG crops the pipeline already exported (no YOLO/tracking
re-run) and compares assembled digits against a reference bib list plus per-crop
latency. This is how "is the SVHN digit detector actually faster/better than OCR?"
gets answered with data.

Example:
  python benchmark_readers.py --candidates-root yolo26_seg_test/<run>/ocr_candidates \
      --backends rapidocr,yolo_digits --digit-model digits_yolo.pt
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import cv2

from common import digits_only
from digit_readers import build_reader
from summarize_tests import DEFAULT_REFERENCE

FRAME_RE = re.compile(r"frame_(\d+)")


def event_dirs(root):
    root = Path(root)
    if any(root.glob("*.png")):
        return [root]
    return sorted(d for d in root.rglob("*") if d.is_dir() and any(d.glob("*.png")))


def event_frame(event_dir):
    m = FRAME_RE.search(event_dir.name)
    return int(m.group(1)) if m else None


def pick_crops(event_dir, nearest_only):
    pngs = sorted(event_dir.glob("*.png"))
    if not nearest_only or len(pngs) <= 1:
        return pngs
    target = event_frame(event_dir)
    if target is None:
        return pngs[:1]
    return [min(pngs, key=lambda p: abs((event_frame(p) or 0) - target))]


def rank(digits, score, reference):
    # Mirror the pipeline's preference: reference hit first, then length, then score.
    return (1 if digits and digits in reference else 0, len(digits), score)


def main():
    p = argparse.ArgumentParser(description="Benchmark digit readers over saved candidate crops.")
    p.add_argument("--candidates-root", required=True, help="A run's ocr_candidates/ dir or a tree of them.")
    p.add_argument("--backends", default="rapidocr", help="Comma list: rapidocr,yolo_digits")
    p.add_argument("--reference", default=DEFAULT_REFERENCE, help="Comma-separated reference bibs.")
    p.add_argument("--digit-model", help="YOLO digit-detector weight for the yolo_digits backend.")
    p.add_argument("--ocr-scale", type=float, default=2.0)
    p.add_argument("--nearest-only", action="store_true", help="Only read the crop nearest the crossing frame.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="Output benchmark.csv path (default beside the candidates root).")
    args = p.parse_args()

    reference = [b.strip() for b in args.reference.split(",") if b.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    readers = {name: build_reader(name, ocr_scale=args.ocr_scale, digit_model=args.digit_model, device=args.device)
               for name in backends}

    dirs = event_dirs(args.candidates_root)
    rows = []
    latencies = {name: [] for name in backends}
    for event_dir in dirs:
        crops = pick_crops(event_dir, args.nearest_only)
        for name, reader in readers.items():
            best_digits, best_score, best_rank, total_ms = "", 0.0, (-1, -1, -1.0), 0.0
            for png in crops:
                img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                _, digits, score, ms = reader.read(img)
                total_ms += ms
                latencies[name].append(ms)
                r = rank(digits, score, reference)
                if digits and r > best_rank:
                    best_digits, best_score, best_rank = digits, score, r
            rows.append({
                "event": event_dir.name,
                "backend": name,
                "digits": best_digits,
                "score": f"{best_score:.3f}",
                "ms": f"{total_ms:.1f}",
                "in_reference": "yes" if best_digits and best_digits in reference else "no",
            })

    out = Path(args.out) if args.out else Path(args.candidates_root).parent / "benchmark.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event", "backend", "digits", "score", "ms", "in_reference"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"benchmark_csv: {out}")
    print(f"events: {len(dirs)}")
    for name in backends:
        br = [r for r in rows if r["backend"] == name]
        preds = [r["digits"] for r in br if r["digits"]]
        ref_hits = len({r["digits"] for r in br if r["in_reference"] == "yes"})
        lat = latencies[name] or [0.0]
        lat_sorted = sorted(lat)
        p50 = statistics.median(lat_sorted)
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
        print(f"[{name}] events_with_digits={len(preds)} ref_hits={ref_hits}/{len(reference)} "
              f"mean_ms={statistics.mean(lat):.1f} p50_ms={p50:.1f} p95_ms={p95:.1f}")


if __name__ == "__main__":
    main()
