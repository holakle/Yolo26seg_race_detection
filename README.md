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
  --result-list "C:\Users\holak\Documents\SAM detection\resultlist input\gold_coast_marathon_2025_results.csv" `
  --device cpu
```

For a 60 fps source, `--process-fps 4` uses approximately `stride=15`.

## Notes

Comments in Python code do not add runtime lag. The script keeps comments short and focused around the tracking, crossing, and OCR backlog logic.

Generated outputs, model weights, and videos are ignored by Git. Track source files and docs only.

The final console output prints timing per layer: YOLO preprocess/inference/postprocess, tracking logic, video writing, deferred OCR, and start-list matching.

`--result-list` is used only after OCR is finished. It appends `result_*` columns to `crossings.csv` so obvious finish-time outliers can be reviewed without affecting detection or OCR.

Future test clips can reuse the same command with:

```powershell
--source "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\sample_middle.mp4"
```

`--ocr-backlog-fallback-only` saves the full candidate backlog, but OCRs the crossing crop first and only scans the remaining backlog if that first crop does not meet `--ocr-fallback-min-digits`.

`--lost-track-fallback` is enabled by default. If an ID disappears for two processed frames close to or just below the crossing line before a strict crossing is registered, its recent crop backlog is still sent to OCR.
