import argparse
import shutil
import time
from pathlib import Path


PROTECTED = {"samples"}


def folder_size(path):
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def is_run_folder(path):
    if path.name in PROTECTED or not path.is_dir():
        return False
    markers = ["run.log", "crossings.csv", "track_crossings.csv", "merged_crossings.csv"]
    return any((path / marker).exists() for marker in markers) or any(path.glob("chunk_*"))


def main():
    p = argparse.ArgumentParser(description="List or delete generated YOLO/OCR run folders.")
    p.add_argument("--root", default=r"C:\Users\holak\Documents\SAM detection\yolo26_seg_test")
    p.add_argument("--contains", default="", help="Only include folder names containing this text.")
    p.add_argument("--older-than-days", type=float, default=0, help="Only include folders older than this many days.")
    p.add_argument("--keep-newest", type=int, default=0, help="Keep this many newest matching folders.")
    p.add_argument("--delete", action="store_true", help="Actually delete matching folders. Without this, only list them.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    now = time.time()
    candidates = []
    for path in root.iterdir():
        if not is_run_folder(path):
            continue
        if args.contains and args.contains.lower() not in path.name.lower():
            continue
        age_days = (now - path.stat().st_mtime) / 86400
        if args.older_than_days and age_days < args.older_than_days:
            continue
        candidates.append((path.stat().st_mtime, path, age_days))

    candidates.sort(reverse=True)
    if args.keep_newest:
        candidates = candidates[args.keep_newest:]

    total = 0
    for _, path, age_days in candidates:
        size = folder_size(path)
        total += size
        print(f"{size / 1024 / 1024:8.1f} MB  {age_days:5.1f} days  {path}")

    action = "deleted" if args.delete else "would_delete"
    if args.delete:
        for _, path, _ in candidates:
            resolved = path.resolve()
            if root not in resolved.parents:
                raise SystemExit(f"Refusing to delete outside root: {resolved}")
            shutil.rmtree(resolved)

    print(f"{action}_folders: {len(candidates)}")
    print(f"{action}_mb: {total / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
