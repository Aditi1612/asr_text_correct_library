"""Low-level Whisper inference engine.

Mirrors ``reconstruction_production/1_stt/transcribe.py`` (``transcribe_audio``
and ``transcribe_audio_with_confidence``) so behavior matches the API, but with
no FastAPI / web dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        reference_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Greedy decode (fast path). Matches ``transcribe_audio`` in the serving code."""
        features, mask = self._features(waveform)
        kwargs = self._base_generate_kwargs(language, task)
        kwargs["num_beams"] = 1
        if mask is not None:
            kwargs["attention_mask"] = mask

        ids = self.model.generate(features, **kwargs)
        raw = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        text = "" if is_degenerate(raw) else ko_norm(raw)
        return self._with_reference(
            {"raw_transcription": raw, "transcription": text}, reference_text
        )

    @torch.inference_mode()
    def transcribe_waveform_detailed(
        self,
        waveform: np.ndarray,
        *,
        language: Optional[str] = None,
        task: Optional[str] = None,
        num_beams: Optional[int] = None,
        nbest: Optional[int] = None,
        confidence_floor: Optional[float] = None,
        reference_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Beam-search decode with a 0..1 confidence, N-best list and uncertainty flag.

        Matches ``transcribe_audio_with_confidence`` in the serving code.
        Confidence is ``exp(sequences_scores)`` — the beam's own length-normalized
        mean token log-prob.
        """
        num_beams = max(1, int(num_beams or self.config.num_beams))
        nbest = max(1, min(int(nbest or self.config.nbest), num_beams))
        floor = self.config.confidence_floor if confidence_floor is None else confidence_floor

        features, mask = self._features(waveform)
        kwargs = self._base_generate_kwargs(language, task)
        kwargs.update(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if mask is not None:
            kwargs["attention_mask"] = mask

        outputs = self.model.generate(features, **kwargs)
        # Deliberately not using compute_transition_scores(beam_indices): Whisper's
        # generate() fills beam_indices out of range and that crashes CUDA.
        scores = outputs.sequences_scores

        hypotheses: List[Dict[str, Any]] = []
        for i in range(outputs.sequences.shape[0]):
            confidence = float(np.exp(float(scores[i])))
            raw = self.processor.batch_decode(
                outputs.sequences[i].unsqueeze(0), skip_special_tokens=True
            )[0]
            text = ko_norm(raw)
            if is_degenerate(text):
                text = ""
                confidence = min(confidence, 0.05)
            hypotheses.append({"transcription": text, "confidence": confidence})

        # Beams often collapse to the same normalized text; keep the best of each.
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for hyp in sorted(hypotheses, key=lambda h: h["confidence"], reverse=True):
            if hyp["transcription"] in seen:
                continue
            seen.add(hyp["transcription"])
            deduped.append(hyp)

        top = deduped[:nbest] or [{"transcription": "", "confidence": 0.0}]
        nbest_list = [
            {"rank": r, "transcription": h["transcription"], "confidence": h["confidence"]}
            for r, h in enumerate(top, start=1)
        ]
        best = nbest_list[0]
        result = {
            "transcription": best["transcription"],
            "confidence": best["confidence"],
            "nbest": nbest_list,
            "is_uncertain": bool(best["confidence"] < floor or not best["transcription"]),
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
