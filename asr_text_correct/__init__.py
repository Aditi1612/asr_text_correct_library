"""asr_text_correct -- in-process Korean ASR (fine-tuned Whisper) + Gemini correction.

No HTTP server. Load the model once, transcribe + correct clips or stream mic frames.

    from asr_text_correct import ASRCorrector

    pipe = ASRCorrector()                      # Whisper + Gemini (GEMINI_API_KEY)
    r = pipe.process("clip.wav")
    print(r.transcript, "->", r.corrected_text)

    stream = pipe.stream(on_segment=lambda s: print(s.corrected_text))
    for chunk in mic_frames():                 # float32 mono @ pipe.sample_rate
        stream.accept(chunk)
    stream.flush()
"""

from .config import CorrectorConfig, EngineConfig, default_model_dir, resolve_language
from .corrector import DEFAULT_PROMPT, Correction, GeminiCorrector
from .engine import WhisperEngine, get_engine
from .hub import resolve_from_hub, upload_checkpoint
from .postprocess import ko_norm, per_sample_cer
from .transcriber import ASRCorrector, Result, Segment, StreamingTranscriber, Transcriber

__all__ = [
    "ASRCorrector",
    "Result",
    "Transcriber",
    "StreamingTranscriber",
    "Segment",
    "GeminiCorrector",
    "Correction",
    "DEFAULT_PROMPT",
    "EngineConfig",
    "CorrectorConfig",
    "WhisperEngine",
    "get_engine",
    "ko_norm",
    "per_sample_cer",
    "default_model_dir",
    "resolve_language",
    "resolve_from_hub",
    "upload_checkpoint",
]

__version__ = "0.1.0"
