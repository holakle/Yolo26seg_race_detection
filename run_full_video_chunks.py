import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Run a long video in bounded frame chunks.")
    p.add_argument("--source", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\goldcoast2025_seg_020150-025150.mp4")
    p.add_argument("--out-root", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\full_video_chunks")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--start-list", default=r"C:\Users\holak\Documents\SAM detection\Startlist input\gold_coast_marathon_2025_results.csv")
    p.add_argument("--ignore-mask", default="")
    p.add_argument("--process-fps", default="3")
    p.add_argument("--chunk-seconds", type=float, default=300)
    p.add_argument("--chunk-overlap-seconds", type=float, default=2)
    p.add_argument("--max-chunks", type=int, help="Use for pilot/soak tests before the full file.")
    p.add_argument("--line", nargs=4, default=["0", "1060", "1920", "1060"])
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True, help="Write MP4 review videos for each chunk.")
    return p.parse_args()


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames


def run_chunk(args, out, start, end):
    cmd = [
        sys.executable, "yolo26_line_crossing.py",
        "--source", args.source,
        "--out", str(out),
        "--model", args.model,
        "--imgsz", "1080",
        "--process-fps", args.process_fps,
        "--line", *args.line,
        "--frame-start", str(start),
        "--frame-end", str(end),
        "--ocr",
        "--ocr-pre-frames", "3",
        "--ocr-post-frames", "5",
        "--ocr-backlog-fallback-only",
        "--ocr-fallback-min-digits", "3",
        "--start-list", args.start_list,
        "--device", "cpu",
        "--save-video" if args.save_video else "--no-video",
    ]
    if args.ignore_mask:
        cmd.extend(["--ignore-mask", args.ignore_mask])
    with (out / "run.log").open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True).returncode


def merge_csv(root, chunks, overlap_frames, source_name, merged_name):
    merged = root / merged_name
    fieldnames = None
    kept = 0
    with merged.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for index, start, end, out in chunks:
            csv_path = out / source_name
            if not csv_path.exists():
                continue
            reader = csv.DictReader(csv_path.open(newline="", encoding="utf-8"))
            rows = list(reader)
            if fieldnames is None:
                fieldnames = ["chunk_index", "chunk_start_frame", "chunk_end_frame", *(reader.fieldnames or [])]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            if not rows:
                continue
            min_frame = start if index == 0 else start + overlap_frames
            for row in rows:
                if int(row.get("frame") or 0) < min_frame:
                    continue
                row = {"chunk_index": index, "chunk_start_frame": start, "chunk_end_frame": end, **row}
                writer.writerow(row)
                kept += 1
    return merged, kept


def main():
    args = parse_args()
    fps, total_frames = video_meta(args.source)
    chunk_frames = max(1, round(args.chunk_seconds * fps))
    overlap_frames = max(0, round(args.chunk_overlap_seconds * fps))
    step = max(1, chunk_frames - overlap_frames)
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    chunks = []
    start = 0
    index = 0
    while start < total_frames and (args.max_chunks is None or index < args.max_chunks):
        end = min(total_frames, start + chunk_frames)
        out = root / f"chunk_{index:04d}_{start:06d}_{end:06d}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"running chunk {index}: {start}-{end}")
        code = run_chunk(args, out, start, end)
        if code:
            raise SystemExit(f"chunk {index} failed, see {out / 'run.log'}")
        chunks.append((index, start, end, out))
        start += step
        index += 1

    merged, kept = merge_csv(root, chunks, overlap_frames, "crossings.csv", "merged_crossings.csv")
    merged_tracks, kept_tracks = merge_csv(root, chunks, overlap_frames, "track_crossings.csv", "merged_track_crossings.csv")
    print(f"merged_crossings: {merged}")
    print(f"merged_rows: {kept}")
    print(f"merged_track_crossings: {merged_tracks}")
    print(f"merged_track_rows: {kept_tracks}")
    print(f"chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
