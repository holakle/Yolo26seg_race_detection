"""Shared helpers and canonical CSV schemas.

Kept stdlib-only (no cv2/numpy) so every script can import it cheaply, including
the Streamlit UI and the offline benchmark. This is the single source of truth
for the small text helpers and the CSV column lists that used to be copy-pasted
across the pipeline and analysis scripts.
"""

import re


def digits_only(text):
    return "".join(re.findall(r"\d+", text or ""))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def finish_time_seconds(value):
    parts = str(value or "").strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = [int(part) for part in parts]
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


# Canonical CSV schemas. Edit these in one place; every writer/reader imports them.
CROSSINGS_HEADER = [
    "track_id", "frame", "time_sec", "direction", "x", "y", "candidate_dir", "candidate_count",
    "ocr_scanned_count", "best_crop", "ocr_text", "ocr_digits", "ocr_score", "ocr_ms", "ocr_candidate_reads",
    "match_type", "matched_bib", "matched_name", "matched_position", "matched_finish_time",
    "match_probability", "match_candidates", "annotation_status", "review_reason",
]

TRACK_CROSSINGS_HEADER = [
    "track_id", "frame", "time_sec", "direction", "x", "y", "candidate_count", "event_source",
]

DUPLICATES_HEADER = [
    "duplicate_group_id", "is_group_representative", "duplicate_score", "duplicate_reason",
    "track_id", "frame", "time_sec", "direction", "x", "y", "event_source", "candidate_count",
    "ocr_scanned_count", "ocr_digits", "ocr_score", "matched_bib", "match_type",
    "match_probability", "annotation_status", "review_reason",
]

START_LIST_FIELDS = ["overall_position", "name", "bib_number", "nation", "gender", "finish_time"]

ANALYSIS_EXTRA_FIELDS = [
    "analysis_result_finish_time",
    "analysis_result_time_delta_sec",
    "analysis_result_time_likely",
    "analysis_result_time_reason",
]


def bib_sequence(rows):
    """Bib reads from crossing rows, preferring the start-list match over raw OCR."""
    return [row.get("matched_bib") or row.get("ocr_digits") for row in rows if row.get("matched_bib") or row.get("ocr_digits")]


def reference_diff(rows, reference):
    """Compare detected bibs against a known reference list.

    Returns (predicted, ref_hits, missing, extra) where missing are reference
    bibs never detected and extra are detected bibs not in the reference.
    """
    predicted = bib_sequence(rows)
    ref_hits = [bib for bib in reference if bib in predicted]
    missing = [bib for bib in reference if bib not in predicted]
    extra = [bib for bib in predicted if bib not in reference]
    return predicted, ref_hits, missing, extra
