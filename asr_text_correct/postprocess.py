"""Transcript normalization and degenerate-output detection.

Vendored from ``reconstruction_production/1_stt/transcribe.py`` so this library
has no dependency on the API repository. Keep these copies in sync if the
serving-side normalization ever changes.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Optional

import numpy as np

REPETITION_COMPRESSION_RATIO = 2.4  # Whisper's degenerate-output heuristic


def ko_norm(text: Optional[str]) -> str:
    """Normalize Korean text similar to the training pipeline."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_degenerate(text: str) -> bool:
    """Flag repetitive/looping output (high zlib compression ratio)."""
    encoded = text.encode("utf-8")
    if not encoded:
        return False
    ratio = len(encoded) / len(zlib.compress(encoded))
    return ratio > REPETITION_COMPRESSION_RATIO


def per_sample_cer(prediction: str, reference: str) -> float:
    """Character error rate via Levenshtein distance (no extra deps)."""
    if not reference:
        return float(prediction != reference)
    len_ref = len(reference)
    len_pred = len(prediction)
    dp = np.zeros((len_ref + 1, len_pred + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len_ref + 1)
    dp[0, :] = np.arange(len_pred + 1)
    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[len_ref, len_pred] / max(1, len_ref))
