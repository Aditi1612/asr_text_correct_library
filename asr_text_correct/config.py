"""Engine / corrector configuration and default asset resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Best-effort default: the Whisper checkpoint that ships with the sibling
# reconstruction_production repo (only meaningful on the machine that repo lives
# on). Off-server, use a HF Hub repo (hf_repo=... / ASR_HF_REPO) or an explicit
# model_dir / ASR_MODEL_DIR.
_SIBLING_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "reconstruction_production" / "1_stt" / "checkpoint"
)

# Values used by the serving pipeline (production_pipeline/config.yaml + env defaults).
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 512


def default_model_dir() -> Optional[str]:
    env = os.environ.get("ASR_MODEL_DIR")
    if env:
        return env
    return str(_SIBLING_CHECKPOINT) if _SIBLING_CHECKPOINT.exists() else None


def gemini_api_keys_from_env() -> List[str]:
    """``GEMINI_API_KEYS`` (comma-separated) wins over a single ``GEMINI_API_KEY``."""
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi.strip():
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    return [single] if single else []


# Short language codes accepted by the API, mapped to Whisper language names.
LANGUAGE_MAP = {
    "KR": "korean",
    "EN": "english",
    "JP": "japanese",
    "ZH": "chinese",
    "ES": "spanish",
    "FR": "french",
}


def resolve_language(language: Optional[str]) -> str:
    """Accept either a code (``KR``) or a Whisper name (``korean``)."""
    if not language:
        return "korean"
    code = language.strip().upper()
    if code in LANGUAGE_MAP:
        return LANGUAGE_MAP[code]
    return language.strip().lower()


@dataclass(frozen=True)
class EngineConfig:
    """Immutable STT inference configuration.

    Frozen so it can be used as a cache key for a loaded engine.
    """

    model_dir: str
    target_sr: int = 16000
    language: str = "korean"
    task: str = "transcribe"
    force_cpu: bool = False
    # fp16 on CUDA matches the serving path; ignored on CPU.
    use_half: bool = True
    max_length: int = 448
    no_repeat_ngram_size: int = 3
    # Below this confidence a transcript is flagged is_uncertain.
    confidence_floor: float = 0.15

    @classmethod
    def resolve(
        cls,
        model_dir: Optional[str] = None,
        *,
        hf_repo: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "EngineConfig":
        """Resolve the checkpoint location.

        Priority: explicit ``model_dir`` > ``hf_repo`` (arg or ``ASR_HF_REPO``
        env) > ``ASR_MODEL_DIR`` env > sibling reconstruction_production checkpoint.
        """
        repo = hf_repo or os.environ.get("ASR_HF_REPO")

        if model_dir:
            md = model_dir
        elif repo:
            from .hub import resolve_from_hub

            md = resolve_from_hub(
                repo,
                revision=hf_revision or os.environ.get("ASR_HF_REVISION"),
                token=hf_token,
                cache_dir=hf_cache_dir or os.environ.get("ASR_HF_CACHE_DIR"),
            )
        else:
            md = default_model_dir()

        if not md:
            raise ValueError(
                "No Whisper checkpoint found. Pass model_dir=..., set ASR_MODEL_DIR, "
                "or pass hf_repo=... / set ASR_HF_REPO."
            )
        if "language" in kwargs:
            kwargs["language"] = resolve_language(kwargs["language"])
        return cls(model_dir=str(md), **kwargs)


@dataclass(frozen=True)
class CorrectorConfig:
    """Gemini transcript-correction settings."""

    api_keys: Tuple[str, ...]
    model: str = DEFAULT_GEMINI_MODEL
    temperature: float = DEFAULT_GEMINI_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    prompt: Optional[str] = None
    # On total failure, return whitespace-cleaned raw text instead of raising
    # (same behavior as the serving pipeline).
    fallback: bool = True

    @classmethod
    def resolve(
        cls,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> "CorrectorConfig":
        """Priority: ``api_keys`` > ``api_key`` > ``GEMINI_API_KEYS`` > ``GEMINI_API_KEY``."""
        keys = list(api_keys or ([api_key] if api_key else gemini_api_keys_from_env()))
        keys = [k for k in keys if k]
        if not keys:
            raise ValueError(
                "No Gemini API key found. Pass api_key=... / api_keys=[...], "
                "or set GEMINI_API_KEY / GEMINI_API_KEYS."
            )
        if "model" not in kwargs and os.environ.get("GEMINI_MODEL"):
            kwargs["model"] = os.environ["GEMINI_MODEL"]
        return cls(api_keys=tuple(keys), **kwargs)
