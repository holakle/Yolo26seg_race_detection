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
  --source "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\samples\sample_cropped_20s.mp4" `
  --out "C:\Users\holak\Documents\SAM detection\yolo26_seg_test\line_crossing_2fps_backlog" `
  --model yolo26n-seg.pt `
  --imgsz 640 `
  --process-fps 2 `
  --line 0 700 1280 700 `
  --ocr `
  --ocr-pre-frames 3 `
  --ocr-post-frames 3 `
  --device cpu
```

For a 60 fps source, `--process-fps 2` uses approximately `stride=30`.

## Notes

Comments in Python code do not add runtime lag. The script keeps comments short and focused around the tracking, crossing, and OCR backlog logic.

Generated outputs, model weights, and videos are ignored by Git. Track source files and docs only.
