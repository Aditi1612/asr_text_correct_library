"""Low-level Whisper inference engine.

Greedy decoding, mirroring ``transcribe_audio`` in
``reconstruction_production/1_stt/transcribe.py`` so transcripts match the API,
with no FastAPI / web dependencies. Unlike the serving path's beam search
(~2.5x the GPU memory, 3-4x slower), confidence here comes from the greedy
decode's own token probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .config import EngineConfig
from .postprocess import is_degenerate, ko_norm, per_sample_cer

_ENGINE_CACHE: Dict[EngineConfig, "WhisperEngine"] = {}


class WhisperEngine:
    """Loads the checkpoint once; transcribes 16 kHz mono waveforms."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not config.force_cpu else "cpu"
        )
        self.compute_dtype = (
            torch.float16 if (self.device.type == "cuda" and config.use_half) else torch.float32
        )
        self.processor = WhisperProcessor.from_pretrained(config.model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.model_dir, torch_dtype=self.compute_dtype
        )
        self.model = self.model.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model.eval()

    # ------------------------------------------------------------------ #

    def _features(self, waveform: np.ndarray):
        out = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = out.input_features.to(device=self.device, dtype=self.model.dtype)
        mask = out.attention_mask.to(self.device) if "attention_mask" in out else None
        return features, mask

    def _base_generate_kwargs(self, language: Optional[str], task: Optional[str]) -> Dict[str, Any]:
        return {
            "max_length": self.config.max_length,
            "do_sample": False,
            "temperature": None,
            "language": language or self.config.language,
            "task": task or self.config.task,
            "suppress_tokens": [],
            "begin_suppress_tokens": [220, 50257],
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
        }

    @torch.inference_mode()
    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        *,
        language: Optional[str] = None,
        task: Optional[str] = None,
        confidence_floor: Optional[float] = None,
        reference_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Greedy decode with a 0..1 confidence and an uncertainty flag.

        Confidence is ``exp(mean token log-prob)`` of the decoded sequence — the
        same quantity the serving path's beam search reports, taken from the
        single greedy hypothesis instead.
        """
        floor = self.config.confidence_floor if confidence_floor is None else confidence_floor

        features, mask = self._features(waveform)
        kwargs = self._base_generate_kwargs(language, task)
        # Explicit: the checkpoint's generation_config defaults to num_beams=3.
        kwargs.update(num_beams=1, output_scores=True, return_dict_in_generate=True)
        if mask is not None:
            kwargs["attention_mask"] = mask

        out = self.model.generate(features, **kwargs)
        raw = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

        token_logprobs = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0].float()
        token_logprobs = token_logprobs[torch.isfinite(token_logprobs)]
        confidence = (
            float(torch.exp(token_logprobs.mean())) if token_logprobs.numel() else 0.0
        )

        if is_degenerate(raw):
            text = ""
            confidence = min(confidence, 0.05)
        else:
            text = ko_norm(raw)

        result = {
            "raw_transcription": raw,
            "transcription": text,
            "confidence": confidence,
            "is_uncertain": bool(confidence < floor or not text),
        }
        return self._with_reference(result, reference_text)

    @staticmethod
    def _with_reference(result: Dict[str, Any], reference_text: Optional[str]) -> Dict[str, Any]:
        reference_norm = ko_norm(reference_text) if reference_text else None
        result["reference_text"] = reference_text
        result["reference_norm"] = reference_norm
        result["cer"] = (
            per_sample_cer(result["transcription"], reference_norm) if reference_norm else None
        )
        return result


def get_engine(config: EngineConfig) -> WhisperEngine:
    """Return a process-cached engine for this exact config."""
    engine = _ENGINE_CACHE.get(config)
    if engine is None:
        engine = WhisperEngine(config)
        _ENGINE_CACHE[config] = engine
    return engine
