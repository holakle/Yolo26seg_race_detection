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
  --no-video `
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

Use `python run_test_matrix.py --conf 0.15` to test a lower YOLO person detection threshold. On CPU this can be much slower, so compare `track_crossings.csv` and `suspected_duplicates.csv` before making it a default.

Run a bounded full-video pilot chunk with:

```powershell
python run_full_video_chunks.py --max-chunks 1
```

The full-video runner uses `--chunk-seconds 300`, `--chunk-overlap-seconds 2`, and `--no-video` by default. It writes chunk folders plus `merged_track_crossings.csv` and `merged_crossings.csv`.

Post-process likely duplicate tracker IDs without changing the raw crossing logs:

```powershell
python analyze_duplicate_crossings.py `
  --track-crossings "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\sample_middle_tracker_vs_ocr_3fps_nomask\track_crossings.csv" `
  --crossings "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\sample_middle_tracker_vs_ocr_3fps_nomask\crossings.csv"
```

This writes `suspected_duplicates.csv`. It is a review aid: the pipeline still keeps every raw tracker crossing, while this file groups nearby events that look like ID switches or lost-track fallbacks for the same runner.

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
