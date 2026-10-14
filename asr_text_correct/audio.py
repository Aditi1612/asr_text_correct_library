"""Audio loading and resampling.

Accepts the same inputs an app is likely to have on hand: a file path, raw
encoded bytes (wav/flac/ogg, plus mp3/m4a when ffmpeg is present), or an
in-memory float32 waveform. Unlike the CTC model, Whisper needs no mean/std
normalization or silence padding — its feature extractor handles that.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

AudioLike = Union[str, Path, bytes, bytearray, memoryview, np.ndarray, torch.Tensor]


def _decode_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(audio_bytes)
    try:
        speech, sr = sf.read(buf, dtype="float32", always_2d=False)
        return np.asarray(speech, dtype=np.float32), int(sr)
    except Exception:
        # soundfile can't handle MP3/M4A — fall back to torchaudio (needs ffmpeg).
        buf.seek(0)
        waveform, sr = torchaudio.load(buf)
        return waveform.mean(dim=0).numpy().astype(np.float32), int(sr)


def to_waveform(
    audio: AudioLike,
    input_sr: int | None = None,
    target_sr: int = 16000,
) -> np.ndarray:
    """Normalize any supported input into a mono float32 numpy array at target_sr.

    ``input_sr`` is only consulted for raw ndarray/tensor input; for files and
    encoded bytes the sample rate is read from the container.
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        speech, sr = _decode_bytes(path.read_bytes())
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        speech, sr = _decode_bytes(bytes(audio))
    else:
        if isinstance(audio, torch.Tensor):
            speech = audio.detach().cpu().numpy()
        else:
            speech = np.asarray(audio)
        speech = speech.astype(np.float32, copy=False)
        if input_sr is None:
            raise ValueError("input_sr is required when passing a raw waveform")
        sr = int(input_sr)

    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    if sr != target_sr:
        speech = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(speech)).float(), sr, target_sr
        ).numpy()

    return np.ascontiguousarray(speech, dtype=np.float32)
