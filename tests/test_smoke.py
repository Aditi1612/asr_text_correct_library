"""Smoke tests.

STT tests need the Whisper checkpoint (sibling reconstruction_production repo or
ASR_MODEL_DIR) and a sample wav. Gemini tests need GEMINI_API_KEY(S); the
fallback test needs neither network nor a valid key.

    pytest -q            # from the library root
"""

import glob
import os

import numpy as np
import pytest
import soundfile as sf

from asr_text_correct import ASRCorrector, GeminiCorrector, Transcriber
from asr_text_correct.config import default_model_dir, gemini_api_keys_from_env

_SAMPLE_GLOBS = [
    os.path.join(
        os.path.dirname(__file__), "..", "..", "reconstruction_production", "1_stt", "data", "*", "*.wav"
    ),
    os.path.join(os.path.dirname(__file__), "samples", "*.wav"),
]

needs_model = pytest.mark.skipif(default_model_dir() is None, reason="no checkpoint available")
needs_gemini = pytest.mark.skipif(not gemini_api_keys_from_env(), reason="no GEMINI_API_KEY")


def _find_sample():
    for pattern in _SAMPLE_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="session")
def asr():
    return Transcriber()


@pytest.fixture(scope="session")
def sample():
    path = _find_sample()
    if path is None:
        pytest.skip("no sample wav")
    return path


# --- correction ----------------------------------------------------------


def test_corrector_fallback_on_bad_key():
    corrector = GeminiCorrector(api_key="invalid-key")
    out = corrector.correct_detailed("  안녕   하세요  ")
    assert not out.ok
    assert out.text == "안녕 하세요"


def test_corrector_no_fallback_raises():
    corrector = GeminiCorrector(api_key="invalid-key", fallback=False)
    with pytest.raises(RuntimeError):
        corrector.correct("안녕 하세요")


def test_corrector_empty_passthrough():
    assert GeminiCorrector(api_key="unused").correct("   ") == "   "


@needs_gemini
def test_corrector_live():
    out = GeminiCorrector().correct_detailed("은 변원 앞에 있어요 약을 하우레 두 씩 드세요")
    assert out.ok and out.text


# --- STT -----------------------------------------------------------------


@needs_model
def test_transcribe_file(asr, sample):
    assert isinstance(asr.transcribe(sample), str)


@needs_model
def test_transcribe_waveform_matches_file(asr, sample):
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert asr.transcribe(sample) == asr.transcribe(wav, input_sr=sr)


@needs_model
def test_transcribe_detailed(asr, sample):
    out = asr.transcribe(sample, detailed=True)
    assert set(out) >= {"transcription", "confidence", "nbest", "is_uncertain"}
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["nbest"][0]["rank"] == 1


@needs_model
def test_streaming_segments(asr, sample):
    wav, sr = sf.read(sample, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # 300ms lead/tail silence so the VAD sees a clean utterance boundary.
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    wav = np.concatenate([pad, wav, pad])

    stream = asr.stream()
    got = []
    step = int(sr * 0.1)  # 100ms push chunks
    for i in range(0, len(wav), step):
        got.extend(stream.accept(wav[i : i + step], input_sr=sr))
    tail = stream.flush()
    if tail:
        got.append(tail)

    assert len(got) >= 1
    assert any(s.text for s in got)


# --- pipeline ------------------------------------------------------------


@needs_model
def test_pipeline_with_fallback(asr, sample):
    pipe = ASRCorrector(transcriber=asr, api_key="invalid-key")
    r = pipe.process(sample)
    assert r.correction_ok is False
    assert r.corrected_text == " ".join(r.transcript.split())


@needs_model
@needs_gemini
def test_pipeline_live(asr, sample):
    r = ASRCorrector(transcriber=asr).process(sample)
    assert r.correction_ok and r.corrected_text
