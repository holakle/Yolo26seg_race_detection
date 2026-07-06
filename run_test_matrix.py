import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run the short clip FPS/mask comparison matrix.")
    p.add_argument("--source", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\sample_middle.mp4")
    p.add_argument("--out-root", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\test_matrix")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--start-list", default=r"C:\Users\holak\Documents\SAM detection\Startlist input\gold_coast_marathon_2025_results.csv")
    p.add_argument("--ignore-mask", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test\ignore_mask.png")
    p.add_argument("--fps", default="2,3,4,6")
    p.add_argument("--conf", default="0.25", help="YOLO person detection confidence threshold.")
    p.add_argument("--line", nargs=4, default=["0", "1060", "1920", "1060"])
    p.add_argument("--save-video", action="store_true")
    return p.parse_args()


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def main():
    args = parse_args()
    root = Path(args.out_root)
    configs = []
    for fps in [v.strip() for v in args.fps.split(",") if v.strip()]:
        configs.append((fps, "mask", args.ignore_mask))
        configs.append((fps, "nomask", None))

    failed = []
    for fps, mask_name, mask_path in configs:
        out = root / f"fps{fps}_{mask_name}"
        cmd = [
            sys.executable, "yolo26_line_crossing.py",
            "--source", args.source,
            "--out", str(out),
            "--model", args.model,
            "--imgsz", "1080",
            "--conf", args.conf,
            "--process-fps", fps,
            "--line", *args.line,
            "--ocr",
            "--ocr-pre-frames", "3",
            "--ocr-post-frames", "5",
            "--ocr-backlog-fallback-only",
            "--ocr-fallback-min-digits", "3",
            "--start-list", args.start_list,
            "--device", "cpu",
            "--save-video" if args.save_video else "--no-video",
        ]
        if mask_path:
            cmd.extend(["--ignore-mask", mask_path])
        print(f"running {out.name}")
        code = run(cmd, out / "run.log")
        if code:
            failed.append(out.name)

    print(f"matrix_root: {root}")
    print(f"failed: {','.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
