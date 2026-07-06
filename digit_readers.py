"""Pluggable bib-digit readers so backends can be A/B benchmarked on the same crops.

Interface: ``DigitReader.read(crop) -> (text, digits, score, ms)`` where ``crop``
is a BGR or BGRA person crop (alpha is stripped). ``digits`` is the numeric-only
string used everywhere downstream.

- ``RapidOCRReader``  — the current production path (detection + recognition OCR).
- ``YoloDigitReader`` — the SVHN idea done right: a small YOLO digit *detector*
  finds each digit box on the masked crop; boxes are ordered left-to-right and
  concatenated into the bib. Needs a trained 10-class (0-9) weight.
- ``ClipSvhnVerifier`` — optional. tanganke/clip-vit-base-patch32_svhn is a
  whole-image single-digit classifier (no localization), so it can only re-score
  individual digit boxes, never read a multi-digit sequence. Off the critical path.

``rapidocr_read`` is the single source of the OCR text-extraction logic; the
pipeline imports it as ``read_ocr`` so live behavior is unchanged.
"""

import time
from pathlib import Path

import cv2

from common import digits_only


def to_bgr(crop):
    if crop is None:
        return None
    if crop.ndim == 3 and crop.shape[2] == 4:
        return crop[:, :, :3]
    return crop


def rapidocr_read(ocr, crop_bgra, scale):
    # OCR sees the crop only; caller keeps a numeric-only field. (Was yolo26_line_crossing.read_ocr.)
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


class DigitReader:
    name = "base"

    def read(self, crop):
        raise NotImplementedError


class RapidOCRReader(DigitReader):
    name = "rapidocr"

    def __init__(self, scale=2.0, ocr=None):
        self.scale = scale
        if ocr is None:
            from rapidocr import RapidOCR
            ocr = RapidOCR()
        self.ocr = ocr

    def read(self, crop):
        start = time.perf_counter()
        text, score = rapidocr_read(self.ocr, crop, self.scale)
        ms = (time.perf_counter() - start) * 1000
        return text, digits_only(text), (float(score) if score != "" else 0.0), ms


class YoloDigitReader(DigitReader):
    name = "yolo_digits"

    def __init__(self, model_path, conf=0.25, imgsz=160, min_rel_height=0.0, device="cpu"):
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"SVHN digit-detector weight not found: {model_path!r}. Train or download a "
                "YOLO digit detector with 10 classes (0-9) on SVHN-style data and pass it via "
                "--digit-model. See README 'SVHN digit reader'."
            )
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.min_rel_height = min_rel_height
        self.device = device

    def read(self, crop):
        start = time.perf_counter()
        bgr = to_bgr(crop)
        result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        dets = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            h = bgr.shape[0]
            for (x1, y1, x2, y2), cls, cf in zip(xyxy, classes, confs):
                if self.min_rel_height and (y2 - y1) < self.min_rel_height * h:
                    continue
                dets.append((float(x1), int(cls) % 10, float(cf)))
        dets.sort(key=lambda d: d[0])  # left to right -> assemble the number
        digits = "".join(str(cls) for _, cls, _ in dets)
        score = min((cf for _, _, cf in dets), default=0.0)
        ms = (time.perf_counter() - start) * 1000
        return digits, digits, score, ms


class ClipSvhnVerifier:
    """Experimental per-digit re-scorer using tanganke/clip-vit-base-patch32_svhn.

    NOT a sequence reader: the CLIP vision encoder classifies/embeds a whole crop
    as a single digit and cannot localize. Intended only to veto/re-score the
    individual digit boxes a detector produced. Kept off the critical path.
    """

    def __init__(self, model_name="tanganke/clip-vit-base-patch32_svhn", device="cpu"):
        from transformers import CLIPImageProcessor, CLIPVisionModel
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def embed(self, crop):
        import torch
        bgr = to_bgr(crop)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.pooler_output.squeeze(0).cpu().numpy()


def build_reader(name, ocr_scale=2.0, digit_model=None, device="cpu"):
    if name == "rapidocr":
        return RapidOCRReader(scale=ocr_scale)
    if name == "yolo_digits":
        return YoloDigitReader(digit_model, device=device)
    raise ValueError(f"unknown reader backend: {name}")
