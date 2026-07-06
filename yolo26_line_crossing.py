import argparse
import csv
import re
import time
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def side_of_line(point, a, b):
    # Signed area: the sign tells which side of the line the point is on.
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def distance_to_line(point, a, b):
    length = max(1.0, float(np.hypot(b[0] - a[0], b[1] - a[1])))
    return abs(side_of_line(point, a, b)) / length


def digits_only(text):
    return "".join(re.findall(r"\d+", text or ""))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clear_pngs(path):
    if path.exists():
        for png in path.glob("*.png"):
            png.unlink()


def mask_crop(frame, mask, box):
    # Export a person crop with alpha. Pixels outside the segment are black/transparent.
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[~crop_mask] = 0
    alpha = (crop_mask * 255).astype(np.uint8)
    return np.dstack([crop, alpha])


def read_ocr(ocr, crop_bgra, scale):
    # OCR sees the crop only; final CSV keeps a numeric-only field.
    image = crop_bgra[:, :, :3] if crop_bgra.shape[2] == 4 else crop_bgra
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = ocr(image)

    texts = [t for t in (result.txts or []) if t]
    scores = list(result.scores or [])
    if not texts and result.word_results:
        texts = [w[0] for w in result.word_results if w and w[0]]
        scores = [w[1] for w in result.word_results if w and w[0] and len(w) > 1 and w[1] is not None]

    return " ".join(texts), max(scores) if scores else ""


def load_start_list(path):
    if not path:
        return [], {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_bib = {}
    for row in rows:
        bib = digits_only(row.get("bib_number", ""))
        if bib:
            by_bib[bib] = row
    return rows, by_bib


def match_start_list(digits, ocr_score, start_rows, start_by_bib):
    if not digits or not start_by_bib:
        return "", "", "", "", "", "", ""

    conf = as_float(ocr_score)
    if digits in start_by_bib:
        row = start_by_bib[digits]
        probability = min(0.99, 0.75 + 0.24 * conf)
        return "exact", digits, row.get("name", ""), row.get("overall_position", ""), row.get("finish_time", ""), f"{probability:.3f}", ""

    candidates = []
    for row in start_rows:
        bib = digits_only(row.get("bib_number", ""))
        if not bib:
            continue
        similarity = SequenceMatcher(None, digits, bib).ratio()
        if digits in bib or bib in digits:
            similarity = max(similarity, min(len(digits), len(bib)) / max(len(digits), len(bib)))
        probability = min(0.85, 0.65 * similarity + 0.20 * conf)
        candidates.append((probability, bib, row))

    candidates.sort(reverse=True, key=lambda item: item[0])
    probability, bib, row = candidates[0]
    summary = "; ".join(f"{b}:{r.get('name', '')}:{p:.2f}" for p, b, r in candidates[:3])
    return "fuzzy", bib, row.get("name", ""), row.get("overall_position", ""), row.get("finish_time", ""), f"{probability:.3f}", summary


def fallback_result_ok(digits, min_digits, start_by_bib):
    if len(digits) < min_digits:
        return False
    return not start_by_bib or digits in start_by_bib


def ocr_rank(digits, score, start_by_bib):
    exact_start_hit = 1 if start_by_bib and digits in start_by_bib else 0
    return exact_start_hit, len(digits), score


def annotation_status(best, match, event, min_digits, accept_score, review_bibs):
    match_type, matched_bib = match[0], match[1]
    score = as_float(best["score"])
    reasons = []
    if match_type != "exact":
        reasons.append("not_exact_match")
    if best["digits"] != matched_bib:
        reasons.append("ocr_match_mismatch")
    if len(best["digits"]) < min_digits:
        reasons.append("short_digits")
    if score < accept_score:
        reasons.append("low_ocr_score")
    if event["direction"] == "lost_near_line":
        reasons.append("lost_track_fallback")
    if matched_bib in review_bibs:
        reasons.append("watchlist_bib")
    return ("review", ";".join(reasons)) if reasons else ("accepted", "")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="yolo26_seg_test/line_crossing")
    p.add_argument("--model", default="yolo26n-seg.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--process-fps", type=float, default=2.0, help="Target processed FPS. For 60 fps video, 2 fps becomes stride 30.")
    p.add_argument("--stride", type=int, help="Override process-fps with an explicit frame stride.")
    p.add_argument("--line", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    p.add_argument("--ignore-mask", help="White-on-black PNG mask. Detections with bottom-center inside white areas are ignored.")
    p.add_argument("--ocr", action="store_true", help="Run RapidOCR on crossing crop backlogs.")
    p.add_argument("--ocr-scale", type=float, default=2.0)
    p.add_argument("--ocr-pre-frames", type=int, default=3)
    p.add_argument("--ocr-post-frames", type=int, default=3)
    p.add_argument("--ocr-backlog-fallback-only", action=argparse.BooleanOptionalAction, default=False, help="OCR crossing crop first; scan backlog only if the first result is weak or not in start-list.")
    p.add_argument("--ocr-fallback-min-digits", type=int, default=1, help="With fallback-only OCR, require this many digits before skipping the backlog.")
    p.add_argument("--lost-track-fallback", action=argparse.BooleanOptionalAction, default=True, help="Create an OCR event when an uncrossed track disappears close to or beyond the line.")
    p.add_argument("--lost-track-line-window", type=float, default=40.0, help="Pixel distance from the line used by lost-track-fallback.")
    p.add_argument("--lost-track-miss-frames", type=int, default=2, help="Processed frames a track must be absent before lost-track-fallback fires.")
    p.add_argument("--start-list", help="CSV with a bib_number column for OCR result comparison.")
    p.add_argument("--accept-ocr-score", type=float, default=0.90)
    p.add_argument("--review-bibs", default="3544,3460,4315,4761,16565,4163")
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True, help="Write the mask-only review video.")
    p.add_argument("--no-video", dest="save_video", action="store_false", help="Skip writing the mask-only review video.")
    p.add_argument("--frame-start", type=int, default=0, help="Global source frame to start from.")
    p.add_argument("--frame-end", type=int, help="Exclusive global source frame to stop at.")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def save_and_ocr_event(event, candidate_dir, rows, review_rows, ocr, ocr_scale, fallback_only, fallback_min_digits, fps, start_rows, start_by_bib, accept_score, review_bibs):
    # One crossing may have many candidate crops. OCR all, then keep the best numeric read.
    event_dir = candidate_dir / f"id_{event['track_id']:04d}_frame_{event['frame']:06d}"
    event_dir.mkdir(parents=True, exist_ok=True)

    best = {"digits": "", "text": "", "score": "", "crop": "", "rank": (-1, -1, -1.0)}
    ocr_ms = 0.0
    saved = []
    candidate_reads = []

    for frame_i, crop in event["crops"]:
        name = f"id_{event['track_id']:04d}_frame_{frame_i:06d}.png"
        cv2.imwrite(str(event_dir / name), crop)
        saved.append((frame_i, name, crop))

    if ocr is not None and saved:
        crossing = min(saved, key=lambda item: abs(item[0] - event["frame"]))
        backlog = [item for item in saved if item != crossing]
        ocr_queue = [crossing] if fallback_only else saved
        if fallback_only:
            ocr_queue.extend(backlog)

    ocr_scanned = 0
    for frame_i, name, crop in ocr_queue if ocr is not None and saved else []:
        if fallback_only and ocr_scanned > 0 and fallback_result_ok(best["digits"], fallback_min_digits, start_by_bib):
            break
        if ocr is None:
            continue

        start = time.perf_counter()
        text, score = read_ocr(ocr, crop, ocr_scale)
        ocr_ms += (time.perf_counter() - start) * 1000
        ocr_scanned += 1
        digits = digits_only(text)
        numeric_score = float(score) if score != "" else 0.0
        if text or digits:
            candidate_reads.append(f"{name}:{digits}:{numeric_score:.3f}")
        rank = ocr_rank(digits, numeric_score, start_by_bib)
        if digits and rank > best["rank"]:
            best = {"digits": digits, "text": text, "score": score, "crop": name, "rank": rank}

    match_start = time.perf_counter()
    match = match_start_list(best["digits"], best["score"], start_rows, start_by_bib)
    match_sec = time.perf_counter() - match_start
    status, reason = annotation_status(best, match, event, fallback_min_digits, accept_score, review_bibs)

    row = [
        event["track_id"],
        event["frame"],
        f"{event['frame'] / fps:.3f}",
        event["direction"],
        event["x"],
        event["y"],
        event_dir.name,
        len(event["crops"]),
        ocr_scanned,
        best["crop"],
        best["text"],
        best["digits"],
        best["score"],
        f"{ocr_ms:.1f}" if ocr is not None else "",
        "|".join(candidate_reads),
        *match,
        status,
        reason,
    ]
    rows.writerow(row)
    if status == "review":
        review_rows.writerow(row)
    return ocr_ms / 1000, 1 if ocr is not None else 0, match_sec


def main():
    args = parse_args()
    total_start = time.perf_counter()
    out = Path(args.out)
    candidate_dir = out / "ocr_candidates"
    out.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for child in candidate_dir.glob("*"):
        if child.is_dir():
            clear_pngs(child)
            child.rmdir()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_end = min(args.frame_end or total, total)
    frame_start = max(0, min(args.frame_start, frame_end))
    stride = args.stride or max(1, round(fps / args.process_fps))
    out_fps = max(1, fps / stride)

    video_path = out / "mask_only_line_crossing.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)) if args.save_video else None

    csv_path = out / "crossings.csv"
    review_path = out / "review.csv"
    track_csv_path = out / "track_crossings.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    review_file = review_path.open("w", newline="", encoding="utf-8")
    track_csv_file = track_csv_path.open("w", newline="", encoding="utf-8")
    rows = csv.writer(csv_file)
    review_rows = csv.writer(review_file)
    track_rows = csv.writer(track_csv_file)
    csv_header = [
        "track_id", "frame", "time_sec", "direction", "x", "y", "candidate_dir", "candidate_count",
        "ocr_scanned_count", "best_crop", "ocr_text", "ocr_digits", "ocr_score", "ocr_ms", "ocr_candidate_reads",
        "match_type", "matched_bib", "matched_name", "matched_position", "matched_finish_time",
        "match_probability", "match_candidates", "annotation_status", "review_reason",
    ]
    rows.writerow(csv_header)
    review_rows.writerow(csv_header)
    track_rows.writerow(["track_id", "frame", "time_sec", "direction", "x", "y", "candidate_count", "event_source"])

    model = YOLO(args.model)
    ocr = None
    if args.ocr:
        from rapidocr import RapidOCR
        ocr = RapidOCR()
    start_rows, start_by_bib = load_start_list(args.start_list)
    review_bibs = {digits_only(bib) for bib in args.review_bibs.split(",") if digits_only(bib)}

    if args.warmup:
        model.predict(np.zeros((height, width, 3), dtype=np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)
        if ocr is not None:
            ocr(np.zeros((32, 96, 3), dtype=np.uint8))

    line_a = (args.line[0], args.line[1])
    line_b = (args.line[2], args.line[3])
    ignore_mask = None
    if args.ignore_mask:
        ignore_mask = cv2.imread(args.ignore_mask, cv2.IMREAD_GRAYSCALE)
        if ignore_mask is None:
            raise SystemExit(f"Could not read ignore mask: {args.ignore_mask}")
        if ignore_mask.shape[:2] != (height, width):
            ignore_mask = cv2.resize(ignore_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        ignore_mask = ignore_mask > 127

    # Per-track state is deliberately small: side, recent crops, and pending OCR events.
    last_side = {}
    last_point = {}
    last_frame = {}
    missed = defaultdict(int)
    crossed = set()
    recent = defaultdict(lambda: deque(maxlen=max(1, args.ocr_pre_frames + 1)))
    pending = {}
    completed_events = []
    processed = 0
    ocr_calls = 0
    ocr_total = 0.0
    match_total = 0.0
    layer = defaultdict(float)
    processing_start = time.perf_counter()

    def record_track_event(event, source):
        track_rows.writerow([
            event["track_id"],
            event["frame"],
            f"{event['frame'] / fps:.3f}",
            event["direction"],
            event["x"],
            event["y"],
            len(event["crops"]),
            source,
        ])
        track_csv_file.flush()

    def add_lost_event(track_id):
        side = last_side[track_id]
        point = last_point[track_id]
        if not recent[track_id] or (side < 0 and distance_to_line(point, line_a, line_b) > args.lost_track_line_window):
            return False
        event = {
            "track_id": track_id,
            "frame": last_frame[track_id],
            "direction": "lost_near_line",
            "x": point[0],
            "y": point[1],
            "crops": list(recent[track_id]),
            "post_left": 0,
        }
        completed_events.append(event)
        record_track_event(event, "lost_track_fallback")
        crossed.add(track_id)
        return True

    def iter_results():
        if frame_start == 0 and frame_end == total:
            results = model.track(
                source=args.source,
                stream=True,
                vid_stride=stride,
                persist=True,
                tracker=args.tracker,
                classes=[0],
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            for index, result in enumerate(results):
                yield result, index * stride
            return

        cap = cv2.VideoCapture(args.source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        frame_i = frame_start
        while frame_i < frame_end:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_i - frame_start) % stride == 0:
                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    classes=[0],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                yield result, frame_i
            frame_i += 1
        cap.release()

    for result, frame_i in iter_results():
        for name, ms in (result.speed or {}).items():
            layer[f"yolo_{name}"] += ms / 1000
        logic_start = time.perf_counter()
        frame = result.orig_img
        mask_only = np.zeros_like(frame) if writer is not None else None
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes is not None and result.boxes.id is not None else []
        seen = set()

        for mask_small, box, track_id in zip(masks, boxes, ids):
            mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
            x1, y1, x2, y2 = box
            point = (int((x1 + x2) / 2), int(y2))
            if ignore_mask is not None and ignore_mask[min(height - 1, max(0, point[1])), min(width - 1, max(0, point[0]))]:
                continue
            seen.add(track_id)
            missed[track_id] = 0

            if mask_only is not None:
                mask_only[mask] = frame[mask]
            crop = mask_crop(frame, mask, box)
            if crop is not None:
                recent[track_id].append((frame_i, crop))

            # If an ID already crossed, keep a few post-crossing crops before OCR.
            if track_id in pending and crop is not None and frame_i > pending[track_id]["frame"]:
                pending[track_id]["crops"].append((frame_i, crop))
                pending[track_id]["post_left"] -= 1
                if pending[track_id]["post_left"] <= 0:
                    completed_events.append(pending.pop(track_id))

            side = side_of_line(point, line_a, line_b)
            if track_id in last_side and track_id not in crossed and last_side[track_id] * side < 0:
                direction = "A_to_B" if last_side[track_id] < side else "B_to_A"
                event = {
                    "track_id": track_id,
                    "frame": frame_i,
                    "direction": direction,
                    "x": point[0],
                    "y": point[1],
                    "crops": list(recent[track_id]),
                    "post_left": args.ocr_post_frames,
                }
                pending[track_id] = event
                record_track_event(event, "line_crossing")
                crossed.add(track_id)

            last_side[track_id] = side
            last_point[track_id] = point
            last_frame[track_id] = frame_i
            if mask_only is not None:
                cv2.circle(mask_only, point, 4, (0, 255, 255), -1)
                cv2.putText(mask_only, str(track_id), (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If a track vanishes near or below the line, keep its backlog instead of losing it.
        if args.lost_track_fallback:
            for track_id in list(last_side):
                if track_id in seen or track_id in crossed or track_id in pending:
                    continue
                missed[track_id] += 1
                if missed[track_id] >= args.lost_track_miss_frames:
                    add_lost_event(track_id)

        if mask_only is not None:
            cv2.line(mask_only, line_a, line_b, (0, 0, 255), 2)
        layer["tracking_logic"] += time.perf_counter() - logic_start
        if writer is not None:
            write_start = time.perf_counter()
            writer.write(mask_only)
            layer["video_write"] += time.perf_counter() - write_start
        processed += 1
        if processed % 25 == 0:
            print(f"processed {processed} frames, source frame {frame_i}/{frame_end}")

    # End-of-file fallback: do not drop crossings just because post frames ran out.
    for event in list(pending.values()):
        completed_events.append(event)
    if args.lost_track_fallback and frame_end >= total:
        for track_id in list(last_side):
            if track_id not in crossed and track_id not in pending:
                add_lost_event(track_id)

    if writer is not None:
        writer.release()
    tracking_elapsed = time.perf_counter() - processing_start

    # OCR is deferred so the segmentation/tracking path can stay live-oriented.
    ocr_start = time.perf_counter()
    for event in completed_events:
        ocr_sec, calls, match_sec = save_and_ocr_event(event, candidate_dir, rows, review_rows, ocr, args.ocr_scale, args.ocr_backlog_fallback_only, args.ocr_fallback_min_digits, fps, start_rows, start_by_bib, args.accept_ocr_score, review_bibs)
        ocr_total += ocr_sec
        ocr_calls += calls
        match_total += match_sec
        csv_file.flush()
        review_file.flush()
    ocr_elapsed = time.perf_counter() - ocr_start

    csv_file.close()
    review_file.close()
    track_csv_file.close()
    total_elapsed = time.perf_counter() - total_start
    source_duration = (frame_end - frame_start) / fps if fps else 0

    print(f"video: {video_path if writer is not None else ''}")
    print(f"crossings: {csv_path}")
    print(f"review: {review_path}")
    print(f"track_crossings: {track_csv_path}")
    print(f"ocr_candidates: {candidate_dir}")
    print(f"frame_start: {frame_start}")
    print(f"frame_end: {frame_end}")
    print(f"stride: {stride}")
    print(f"processed_frames: {processed}")
    print(f"crossing_events: {len(completed_events)}")
    print(f"source_duration_sec: {source_duration:.3f}")
    print(f"init_sec: {total_elapsed - tracking_elapsed - ocr_elapsed:.3f}")
    print(f"tracking_elapsed_sec: {tracking_elapsed:.3f}")
    print(f"ocr_elapsed_sec: {ocr_elapsed:.3f}")
    print(f"match_elapsed_sec: {match_total:.3f}")
    print(f"total_elapsed_sec: {total_elapsed:.3f}")
    print(f"yolo_preprocess_sec: {layer['yolo_preprocess']:.3f}")
    print(f"yolo_inference_sec: {layer['yolo_inference']:.3f}")
    print(f"yolo_postprocess_sec: {layer['yolo_postprocess']:.3f}")
    print(f"tracking_logic_sec: {layer['tracking_logic']:.3f}")
    print(f"video_write_sec: {layer['video_write']:.3f}")
    print(f"tracking_fps: {processed / tracking_elapsed if tracking_elapsed else 0:.2f}")
    print(f"tracking_real_time_factor: {source_duration / tracking_elapsed if tracking_elapsed else 0:.2f}x")
    print(f"total_real_time_factor: {source_duration / total_elapsed if total_elapsed else 0:.2f}x")
    print(f"live_capable_tracking: {tracking_elapsed < source_duration}")
    print(f"live_capable_total: {total_elapsed < source_duration}")
    if ocr is not None:
        print(f"ocr_calls: {ocr_calls}")
        print(f"ocr_total_sec: {ocr_total:.3f}")


if __name__ == "__main__":
    main()
