# YOLO26 Race Line Crossing Test

Small local test harness for person segmentation, tracking, virtual line crossing, and OCR candidate export.

## What It Does

- Runs YOLO26 segmentation/tracking on person class only.
- Produces a black-background video showing only segmented people, track IDs, and the crossing line.
- Detects when a track crosses a configured line using the bottom-center of the person box.
- Saves a small backlog of segmented person crops before and after each crossing.
- Optionally runs RapidOCR on those crop candidates and keeps the best numeric result.

## Example

```powershell
python yolo26_line_crossing.py `
  --source "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\sample_middle.mp4" `
  --out "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\line_crossing_middle_1080" `
  --model yolo26n-seg.pt `
  --imgsz 1080 `
  --process-fps 4 `
  --line 0 1060 1920 1060 `
  --ocr `
  --ocr-pre-frames 3 `
  --ocr-post-frames 5 `
  --ocr-backlog-fallback-only `
  --ocr-fallback-min-digits 3 `
  --start-list "C:\Users\holak\Documents\SAM detection\Startlist input\gold_coast_marathon_2025_results.csv" `
  --device cpu
```

For a 60 fps source, `--process-fps 4` uses approximately `stride=15`.

## Notes

Comments in Python code do not add runtime lag. The script keeps comments short and focused around the tracking, crossing, and OCR backlog logic.

Generated outputs, model weights, and videos are ignored by Git. Track source files and docs only.

The final console output prints timing per layer: YOLO preprocess/inference/postprocess, tracking logic, video writing, deferred OCR, and start-list matching.

Every run writes `track_crossings.csv` immediately when a track crosses or triggers the lost-track fallback. That file is the all-detected-ID passing-time log and does not depend on OCR. `crossings.csv` is the OCR-enriched event file; unsure OCR rows are also copied to `review.csv`. Rows are `accepted` only for exact start-list hits with high OCR confidence. Fuzzy, low-confidence, lost-track, short-digit, mismatch, and watchlist bib rows are marked `review`.

Run the short FPS/mask matrix with:

```powershell
python run_test_matrix.py
python summarize_tests.py
```

Future runs save the MP4 review video by default. Add `--no-video` only for deliberate low-disk smoke tests.

Keep the YOLO person detection confidence at the default `--conf 0.25` unless there is a specific experiment. In the current `sample_middle.mp4` test, `--conf 0.15` did not reduce lost tracks or duplicate IDs and made CPU tracking much slower.

Run a bounded full-video pilot chunk with:

```powershell
python run_full_video_chunks.py --max-chunks 1
```

The full-video runner uses `--chunk-seconds 300`, `--chunk-overlap-seconds 2`, and saves chunk MP4 review videos by default. It writes chunk folders plus `merged_track_crossings.csv` and `merged_crossings.csv`.

Post-process likely duplicate tracker IDs without changing the raw crossing logs:

```powershell
python analyze_duplicate_crossings.py `
  --track-crossings "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\sample_middle_tracker_vs_ocr_3fps_nomask\track_crossings.csv" `
  --crossings "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\sample_middle_tracker_vs_ocr_3fps_nomask\crossings.csv"
```

This writes `suspected_duplicates.csv`. It is a review aid: the pipeline still keeps every raw tracker crossing, while this file groups nearby events that look like ID switches or lost-track fallbacks for the same runner.

List old generated runs and estimated disk use:

```powershell
python cleanup_runs.py
```

Delete matching old runs only after reviewing the list:

```powershell
python cleanup_runs.py --contains smoke --delete
```

Result lists are not part of the YOLO/OCR pipeline. Use them only after a run is complete:

```powershell
python analyze_crossings.py `
  --crossings "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\line_crossing_middle_1080_img1080_6fps_nomask_lostfallback\crossings.csv" `
  --result-list "C:\Users\holak\Documents\SAM detection\resultlist input\gold_coast_marathon_2025_results.csv"
```

This writes `crossings_analysis.csv` beside the original `crossings.csv`.

Future test clips can reuse the same command with:

```powershell
--source "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\sample_middle.mp4"
```

`--ocr-backlog-fallback-only` saves the full candidate backlog, but OCRs the crossing crop first. It only scans the remaining backlog if that first result has too few digits, or if `--start-list` is present and the digits are not an exact bib-number hit.

`--lost-track-fallback` is enabled by default. If an ID disappears for two processed frames close to or just below the crossing line before a strict crossing is registered, its recent crop backlog is still sent to OCR.

## Install

```powershell
pip install -r requirements.txt
```

Shared helpers and the canonical CSV column lists live in `common.py`; every script imports from it instead of re-defining `digits_only` / `as_float` or hardcoding headers.

## Performance flags

Each run writes a structured `metrics.json` beside the CSVs (timings, real-time factors, thread/worker settings); `summarize_tests.py` reads it and falls back to scraping `run.log` for older runs.

- `--threads N` pins torch/OMP CPU threads for the process. On a 6-physical-core box a single stream already uses ~6 threads, so the throughput lever is running independent work in parallel with `--threads` low enough that `workers * threads` stays near the physical core count.
- `--fast-mask` resizes only each person's bbox region of the segmentation mask instead of the whole frame (much cheaper for distant runners). It differs from the default by <1% of pixels at mask boundaries, so validate candidate crops before relying on it. Default off; the default path keeps the original crop bytes.
- `--ocr-workers N` fans the deferred OCR phase across worker processes (each builds its own RapidOCR). Keep it at 1 when chunks already run in parallel.

Run the matrix or full-video chunks in parallel (chunks/configs are independent frame ranges):

```powershell
python run_full_video_chunks.py --workers 3 --threads 2 --max-chunks 1
python run_test_matrix.py --workers 3 --threads 2
```

## SQLite store and review UI

SQLite is the primary, edge-friendly store (one embedded file, no server). Pass `--db path\race.db` to the pipeline to write runs, track crossings, and OCR crossings into it alongside the CSVs; `--no-export-csv` writes only the DB. WAL mode plus a busy timeout let parallel chunk workers share one DB file.

Backfill existing run folders (idempotent; preserves any human `verified_bib`):

```powershell
python ingest_csv_to_db.py --root yolo26_seg_test\test_matrix --db yolo26_seg_test\race.db --start-list "Startlist input\gold_coast_marathon_2025_results.csv"
```

Review and confirm bibs in a small Streamlit UI (filter to review rows / watchlist, see the segmented candidate crops inline, write `verified_bib`):

```powershell
streamlit run review_app.py -- --db yolo26_seg_test\race.db
```

## SVHN digit reader (experimental) and reader benchmark

`digit_readers.py` defines a pluggable `DigitReader.read(crop) -> (text, digits, score, ms)` interface with two backends:

- `rapidocr` — the current production OCR (the pipeline imports its extraction logic, so live behavior is unchanged).
- `yolo_digits` — the SVHN idea done right: a small YOLO **digit detector** (10 classes, 0-9) finds each digit's box on the masked person crop; boxes are ordered left-to-right and concatenated into the bib. Needs a trained weight — pass `--digit-model`. Source an SVHN-in-YOLO dataset or fine-tune `yolo26n`/`yolo11n` on SVHN full numbers.

Note: `tanganke/clip-vit-base-patch32_svhn` is a whole-image single-digit classifier (no localization), so it cannot read a multi-digit bib on its own. It is exposed only as an optional `ClipSvhnVerifier` to re-score individual digit boxes, off the critical path.

Compare backends on already-saved crops (no YOLO/tracking re-run) for accuracy and per-crop latency:

```powershell
python benchmark_readers.py --candidates-root yolo26_seg_test\<run>\ocr_candidates --backends rapidocr,yolo_digits --digit-model digits_yolo.pt --nearest-only
```

This writes `benchmark.csv` and prints a per-backend rollup (ref hits, mean/p50/p95 ms).
