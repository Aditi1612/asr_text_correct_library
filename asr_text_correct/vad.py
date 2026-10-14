"""Frame-level voice activity detection.

Uses ``webrtcvad`` when available (fast, robust), otherwise falls back to a
simple energy gate so the library still works with no extra dependency.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency
    import webrtcvad  # type: ignore

    _HAVE_WEBRTCVAD = True
except Exception:  # pragma: no cover
    _HAVE_WEBRTCVAD = False


def _float_to_pcm16(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class VAD:
    """Classifies fixed-length mono float32 frames as speech / not-speech."""

    SUPPORTED_RATES = (8000, 16000, 32000, 48000)
    SUPPORTED_FRAME_MS = (10, 20, 30)

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        energy_threshold: float = 1e-4,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.energy_threshold = energy_threshold
        self._backend = "energy"
        self._vad = None

        usable = (
            _HAVE_WEBRTCVAD
            and sample_rate in self.SUPPORTED_RATES
            and frame_ms in self.SUPPORTED_FRAME_MS
        )
        if usable:
            self._vad = webrtcvad.Vad(int(aggressiveness))
            self._backend = "webrtcvad"

    @property
    def backend(self) -> str:
        return self._backend

    def is_speech(self, frame: np.ndarray) -> bool:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape[0] != self.frame_len:
            # Pad/truncate a ragged trailing frame.
            if frame.shape[0] < self.frame_len:
                frame = np.pad(frame, (0, self.frame_len - frame.shape[0]))
            else:
                frame = frame[: self.frame_len]

        if self._vad is not None:
            return self._vad.is_speech(_float_to_pcm16(frame), self.sample_rate)

        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return rms >= self.energy_threshold
