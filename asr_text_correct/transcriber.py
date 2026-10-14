"""High-level, in-process API.

``Transcriber``          -- STT only: audio in, raw Korean transcript out.
``ASRCorrector``         -- STT followed by Gemini correction (the pipeline's
                            ``/stt-text-correction`` stage, without the server).
``StreamingTranscriber`` -- push mic frames continuously; get a finalized
                            (optionally corrected) transcript per utterance.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from . import audio as _audio
from .config import EngineConfig, resolve_language
from .corrector import GeminiCorrector
from .engine import WhisperEngine, get_engine
from .vad import VAD


class Transcriber:
    """Loads the Whisper model once and transcribes whole clips."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        *,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        force_cpu: bool = False,
        use_half: bool = True,
        language: str = "korean",
        cache: bool = True,
        **config_kwargs,
    ):
        self.config = EngineConfig.resolve(
            model_dir=model_dir,
            hf_repo=hf_repo,
            hf_revision=hf_revision,
            hf_token=hf_token,
            hf_cache_dir=hf_cache_dir,
            force_cpu=force_cpu,
            use_half=use_half,
            language=language,
            **config_kwargs,
        )
        self.engine: WhisperEngine = (
            get_engine(self.config) if cache else WhisperEngine(self.config)
        )

    @property
    def device(self) -> str:
        return self.engine.device.type

    @property
    def sample_rate(self) -> int:
        return self.config.target_sr

    def transcribe(
        self,
        audio: _audio.AudioLike,
        input_sr: Optional[int] = None,
        *,
        language: Optional[str] = None,
        reference_text: Optional[str] = None,
        detailed: bool = False,
        **decode_kwargs,
    ):
        """Return the normalized Hangul transcript for ``audio``.

        ``audio`` may be a file path, encoded bytes, or a float32 mono waveform
        (pass ``input_sr`` for a raw waveform). ``language`` accepts ``"KR"`` or
        ``"korean"``.

        ``detailed=False`` -> greedy decode, returns ``str``.
        ``detailed=True``  -> beam search, returns a dict with ``transcription``,
        ``confidence``, ``nbest``, ``is_uncertain`` (and ``cer`` if a reference
        is given). Extra kwargs (``num_beams``, ``nbest``, ``confidence_floor``)
        tune the beam search.
        """
        waveform = _audio.to_waveform(audio, input_sr=input_sr, target_sr=self.config.target_sr)
        lang = resolve_language(language) if language else None
        if detailed:
            return self.engine.transcribe_waveform_detailed(
                waveform, language=lang, reference_text=reference_text, **decode_kwargs
            )
        return self.engine.transcribe_waveform(
            waveform, language=lang, reference_text=reference_text
        )["transcription"]

    def stream(self, **kwargs) -> "StreamingTranscriber":
        """Create a StreamingTranscriber backed by this model (no correction)."""
        return StreamingTranscriber(self, **kwargs)


@dataclass
class Result:
    """Output of :meth:`ASRCorrector.process`."""

    transcript: str
    corrected_text: str
    correction_ok: bool
    confidence: Optional[float] = None
    nbest: List[Dict[str, Any]] = field(default_factory=list)
    is_uncertain: bool = False
    cer: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ASRCorrector:
    """STT + Gemini correction in one object.

    Builds its own :class:`Transcriber` and :class:`GeminiCorrector` unless you
    pass ready-made ones (``transcriber=`` / ``corrector=``).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        *,
        transcriber: Optional[Transcriber] = None,
        corrector: Optional[GeminiCorrector] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        gemini_model: Optional[str] = None,
        prompt: Optional[str] = None,
        **transcriber_kwargs,
    ):
        self.transcriber = transcriber or Transcriber(model_dir, **transcriber_kwargs)
        if corrector is None:
            corrector_kwargs: Dict[str, Any] = {}
            if gemini_model:
                corrector_kwargs["model"] = gemini_model
            if prompt:
                corrector_kwargs["prompt"] = prompt
            corrector = GeminiCorrector(api_key, api_keys=api_keys, **corrector_kwargs)
        self.corrector = corrector

    @property
    def device(self) -> str:
        return self.transcriber.device

    @property
    def sample_rate(self) -> int:
        return self.transcriber.sample_rate

    def correct(self, text: str, prompt: Optional[str] = None) -> str:
        """Correct an existing transcript (no audio)."""
        return self.corrector.correct(text, prompt=prompt)

    def process(
        self,
        audio: _audio.AudioLike,
        input_sr: Optional[int] = None,
        *,
        language: Optional[str] = None,
        reference_text: Optional[str] = None,
        detailed: bool = True,
        prompt: Optional[str] = None,
        **decode_kwargs,
    ) -> Result:
        """Transcribe ``audio`` then correct the transcript.

        ``detailed=True`` (default, as in the serving pipeline) uses beam search
        and fills ``confidence`` / ``nbest`` / ``is_uncertain``; ``False`` uses
        the faster greedy decode.
        """
        t0 = time.perf_counter()
        if detailed:
            stt = self.transcriber.transcribe(
                audio, input_sr, language=language, reference_text=reference_text,
                detailed=True, **decode_kwargs,
            )
        else:
            waveform = _audio.to_waveform(audio, input_sr=input_sr, target_sr=self.sample_rate)
            stt = self.transcriber.engine.transcribe_waveform(
                waveform,
                language=resolve_language(language) if language else None,
                reference_text=reference_text,
            )
        t1 = time.perf_counter()
        correction = self.corrector.correct_detailed(stt["transcription"], prompt=prompt)
        t2 = time.perf_counter()

        return Result(
            transcript=stt["transcription"],
            corrected_text=correction.text,
            correction_ok=correction.ok,
            confidence=stt.get("confidence"),
            nbest=stt.get("nbest", []),
            is_uncertain=stt.get("is_uncertain", not stt["transcription"]),
            cer=stt.get("cer"),
            timings={"stt": t1 - t0, "text_correction": t2 - t1, "total": t2 - t0},
        )

    def stream(self, *, correct: bool = True, **kwargs) -> "StreamingTranscriber":
        """Create a StreamingTranscriber; each segment is corrected when ``correct``."""
        return StreamingTranscriber(
            self.transcriber, corrector=self.corrector if correct else None, **kwargs
        )


@dataclass
class Segment:
    """One finalized utterance."""

    text: str
    start_time: float
    end_time: float
    raw_text: Optional[str] = None
    corrected_text: Optional[str] = None  # set when a corrector is attached

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class StreamingTranscriber:
    """VAD-segmented streaming.

    Feed arbitrary-length mono float32 chunks to :meth:`accept`. When the
    speaker pauses (``end_silence_ms`` of non-speech), the buffered utterance is
    transcribed (and corrected, if a corrector is attached) and returned as a
    :class:`Segment`. No partial hypotheses.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        corrector: Optional[GeminiCorrector] = None,
        frame_ms: int = 30,
        start_speech_frames: int = 3,
        end_silence_ms: int = 600,
        min_speech_ms: int = 250,
        max_speech_ms: int = 15000,
        pre_roll_ms: int = 200,
        vad_aggressiveness: int = 2,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ):
        self.transcriber = transcriber
        self.corrector = corrector
        self.sample_rate = transcriber.sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(self.sample_rate * frame_ms / 1000)
        self.start_speech_frames = start_speech_frames
        self.end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self.min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self.max_speech_frames = max(1, round(max_speech_ms / frame_ms))
        self.pre_roll_frames = max(0, round(pre_roll_ms / frame_ms))
        self.on_segment = on_segment

        self.vad = VAD(
            sample_rate=self.sample_rate,
            frame_ms=frame_ms,
            aggressiveness=vad_aggressiveness,
        )
        self.reset()

    @property
    def vad_backend(self) -> str:
        return self.vad.backend

    def reset(self) -> None:
        self._leftover = np.zeros(0, dtype=np.float32)
        self._preroll: Deque[np.ndarray] = deque(maxlen=self.pre_roll_frames or 1)
        self._triggered = False
        self._voiced: List[np.ndarray] = []
        self._ring: Deque[bool] = deque(maxlen=self.start_speech_frames)
        self._trailing_silence = 0
        self._speech_frames_in_seg = 0
        self._n_frames = 0          # total frames consumed
        self._seg_start_frame = 0

    # ------------------------------------------------------------------ #

    def accept(self, pcm, input_sr: Optional[int] = None) -> List[Segment]:
        """Consume a chunk; return any utterances finalized during this call."""
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if input_sr is not None and input_sr != self.sample_rate:
            samples = _audio.to_waveform(
                samples, input_sr=input_sr, target_sr=self.sample_rate
            )

        buf = np.concatenate([self._leftover, samples])
        n_full = len(buf) // self.frame_len
        segments: List[Segment] = []

        for i in range(n_full):
            frame = buf[i * self.frame_len : (i + 1) * self.frame_len]
            seg = self._push_frame(frame)
            if seg is not None:
                segments.append(seg)

        self._leftover = buf[n_full * self.frame_len :].copy()
        return segments

    def flush(self) -> Optional[Segment]:
        """Force-finalize whatever speech is currently buffered (call at end of stream)."""
        seg = None
        if self._triggered:
            seg = self._finalize()
        self.reset()
        return seg

    # ------------------------------------------------------------------ #

    def _push_frame(self, frame: np.ndarray) -> Optional[Segment]:
        self._n_frames += 1
        is_speech = self.vad.is_speech(frame)

        if not self._triggered:
            self._preroll.append(frame)
            self._ring.append(is_speech)
            if self._ring.count(True) >= self.start_speech_frames:
                self._triggered = True
                self._voiced = list(self._preroll)
                self._speech_frames_in_seg = self._ring.count(True)
                self._trailing_silence = 0
                self._seg_start_frame = self._n_frames - len(self._voiced)
                self._preroll.clear()
                self._ring.clear()
            return None

        # triggered
        self._voiced.append(frame)
        if is_speech:
            self._speech_frames_in_seg += 1
            self._trailing_silence = 0
        else:
            self._trailing_silence += 1

        if self._trailing_silence >= self.end_silence_frames:
            return self._finalize()
        if len(self._voiced) >= self.max_speech_frames:
            return self._finalize()
        return None

    def _finalize(self) -> Optional[Segment]:
        voiced = self._voiced
        speech_frames = self._speech_frames_in_seg
        start_frame = self._seg_start_frame
        end_frame = start_frame + len(voiced)

        # arm for the next utterance
        self._triggered = False
        self._voiced = []
        self._ring.clear()
        self._preroll.clear()
        self._trailing_silence = 0
        self._speech_frames_in_seg = 0

        if speech_frames < self.min_speech_frames or not voiced:
            return None

        wav = np.concatenate(voiced).astype(np.float32)
        result = self.transcriber.engine.transcribe_waveform(wav)
        text = result["transcription"].strip()
        if not text:
            return None

        seg = Segment(
            text=text,
            start_time=start_frame * self.frame_ms / 1000.0,
            end_time=end_frame * self.frame_ms / 1000.0,
            raw_text=result.get("raw_transcription"),
        )
        if self.corrector is not None:
            seg.corrected_text = self.corrector.correct(text)
        if self.on_segment is not None:
            self.on_segment(seg)
        return seg
