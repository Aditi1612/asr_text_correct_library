"""Gemini transcript correction.

Combines ``reconstruction_production/2_t2t/gemini_text_corrector.py`` (prompt +
request) with the key-rotation / fallback wrapper from
``production_pipeline/model_manager.py``, so output matches the API.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import CorrectorConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "You are an expert Korean language transcriber and linguistic corrector "
    "specializing in speech recognition and dysarthria rehabilitation. "
    "Your task is to correct raw, phonetically transcribed Korean text that may contain "
    "misheard syllables, incorrect spacing, grammatical errors, or misused words. "
    "Convert it into fluent, natural, and grammatically correct written Korean. "
    "Always preserve the speaker’s original intent and meaning, but fix all linguistic errors. "
    "Do not translate into English or explain your corrections—only output the corrected Korean sentence(s). "
    "If a word is unclear, infer the most likely intended word from context."
)


def _require_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Gemini correction needs the 'google-genai' package.\n    pip install google-genai"
        ) from exc
    return genai, types


def build_prompt(raw_text: str, custom_prompt: Optional[str] = None) -> str:
    instructions = (custom_prompt or DEFAULT_PROMPT).strip()
    return f"{instructions}\n\nRaw transcript:\n{raw_text}\n\nCorrected transcript:"


def basic_cleanup(text: str) -> str:
    """Fallback used when every Gemini key fails: collapse whitespace only."""
    return re.sub(r"\s+", " ", text.strip())


@dataclass
class Correction:
    """Result of one correction call."""

    text: str
    raw_text: str
    ok: bool  # False when the fallback cleanup was used
    model: str
    error: Optional[str] = None


class GeminiCorrector:
    """Corrects raw ASR transcripts with Gemini.

    Rotates across multiple API keys (round-robin for load balancing, and moves
    on to the next key on 429 / any error). If every key fails it returns the
    whitespace-cleaned raw text (``fallback=True``, the serving default) or
    raises (``fallback=False``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        api_keys: Optional[List[str]] = None,
        config: Optional[CorrectorConfig] = None,
        **config_kwargs,
    ):
        self.config = config or CorrectorConfig.resolve(
            api_key=api_key, api_keys=api_keys, **config_kwargs
        )
        self._genai, self._types = _require_genai()
        self._clients: Dict[str, object] = {}
        self._key_index = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.config.model

    def _client(self, api_key: str):
        client = self._clients.get(api_key)
        if client is None:
            client = self._genai.Client(api_key=api_key)
            self._clients[api_key] = client
        return client

    def _call(self, api_key: str, prompt: str) -> str:
        types = self._types
        response = self._client(api_key).models.generate_content(
            model=self.config.model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=0.9,
                top_k=40,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        if not response or not getattr(response, "text", None):
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()

    def correct_detailed(self, text: str, prompt: Optional[str] = None) -> Correction:
        """Correct ``text``; returns a :class:`Correction` with an ``ok`` flag."""
        if not text or not text.strip():
            return Correction(text=text, raw_text=text, ok=True, model=self.config.model)

        full_prompt = build_prompt(text, prompt or self.config.prompt)
        keys = self.config.api_keys
        with self._lock:
            start = self._key_index

        errors: List[str] = []
        for attempt in range(len(keys)):
            idx = (start + attempt) % len(keys)
            try:
                corrected = self._call(keys[idx], full_prompt)
            except Exception as exc:  # noqa: BLE001 - try the next key
                msg = str(exc)
                errors.append(f"key {idx + 1}: {msg[:100]}")
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    logger.warning("Gemini key %d/%d rate limited, trying next", idx + 1, len(keys))
                else:
                    logger.warning("Gemini error on key %d/%d: %s", idx + 1, len(keys), exc)
                continue
            with self._lock:
                self._key_index = (idx + 1) % len(keys)
            return Correction(text=corrected, raw_text=text, ok=True, model=self.config.model)

        error = "; ".join(errors)
        if not self.config.fallback:
            raise RuntimeError(f"All {len(keys)} Gemini API key(s) failed: {error}")
        logger.warning("All %d Gemini key(s) failed, using fallback cleanup: %s", len(keys), error)
        return Correction(
            text=basic_cleanup(text), raw_text=text, ok=False, model=self.config.model, error=error
        )

    def correct(self, text: str, prompt: Optional[str] = None) -> str:
        """Return the corrected text (or the fallback cleanup)."""
        return self.correct_detailed(text, prompt=prompt).text
