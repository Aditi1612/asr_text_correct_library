# asr_text_correct

In-process Korean **STT + transcript correction**: the fine-tuned Whisper model
from `reconstruction_production/1_stt` followed by Gemini correction from
`reconstruction_production/2_t2t`. There's **no HTTP server and no calls to the
pipeline API**. You load the model once inside your app, then transcribe and
correct clips or a live mic stream directly.

This is a standalone package. It does **not** modify or import
`APIs/reconstruction_production/`. By default it only reads the Whisper
checkpoint from there. The inference path is a faithful copy of the serving
code, so output matches the API's `/stt-text-correction` stage. That path
covers `ko_norm`, the repetition filter, beam-search confidence/N-best, the
correction prompt, and Gemini key rotation with fallback.

```
audio ──► Whisper (fine-tuned, ko) ──► transcript ──► Gemini ──► corrected_text
          + confidence / N-best                       (key rotation, fallback)
```

## Install

On this server (editable, from source):

```bash
conda activate lucy5            # already has torch / transformers / google-genai
pip install -e /data2/irfan/projects/APIs/asr_text_correct_library
```

On any other machine, install from GitHub:

```bash
pip install "asr-text-correct[hub] @ git+https://github.com/Aditi1612/asr_text_correct_library.git"
```

You can also build a wheel and hand over the file:

```bash
pip install build && python -m build      # -> dist/asr_text_correct-0.1.0-*.whl
pip install "asr_text_correct-0.1.0-py3-none-any.whl[hub]"
```

`torch` / `torchaudio` are heavy and hardware-specific. Install the right
CUDA/CPU build first, then install this.

Optional extras:
- `[hub]`: Hugging Face Hub download.
- `[vad]`: webrtcvad, which works better than the energy fallback.
- `[mic]`: sounddevice, needed for the mic demo.

## Gemini API key

Correction needs a Gemini key. You can supply it in any of these ways:

```bash
export GEMINI_API_KEY=...                  # single key
export GEMINI_API_KEYS=key1,key2,key3      # rotation (takes priority)
export GEMINI_MODEL=gemini-2.5-flash-lite  # optional, this is the default
```

You can also pass the key in code: `ASRCorrector(api_key=...)` or
`ASRCorrector(api_keys=[...])`.

How the corrector handles keys and failures:
- **Rotation:** with several keys it round-robins between them. On a 429 or any
  other error it tries the next key.
- **Fallback:** if every key fails, it returns the raw transcript with the
  whitespace collapsed and sets `correction_ok=False`. The serving pipeline
  does the same.
- **Strict mode:** pass `fallback=False` to raise an error instead.

## Model assets

The Whisper checkpoint (~3 GB of weights) is **not** in the package. Choose a
source:

### A. Hugging Face Hub (use this off this server)

The checkpoint is published to the **private** model repo
[`fluffy16/whisper-ko-asr`](https://huggingface.co/fluffy16/whisper-ko-asr).
It holds only the inference files (~5.8 GB); optimizer and scheduler states are
skipped.

```python
ASRCorrector(hf_repo="fluffy16/whisper-ko-asr")   # downloads + caches on first use
```

The env-var form is `export ASR_HF_REPO=fluffy16/whisper-ko-asr`, with no code
change.

The repo is private, so each user needs read access and a token. The repo owner
grants access in the repo settings on Hugging Face (or by adding the user to an
org that owns it). The user then authenticates in any one of these ways:
- `huggingface-cli login`
- `export HF_TOKEN=hf_xxx`
- `ASRCorrector(hf_repo=..., hf_token="hf_xxx")`

To pin an exact model version, pass `hf_revision="<commit-sha>"`.

To publish a new checkpoint (run once, from a machine that has it):

```bash
huggingface-cli login
python examples/upload_to_hf.py fluffy16/whisper-ko-asr /path/to/checkpoint
```

### B. Local files

```python
ASRCorrector("/abs/path/checkpoint")
# or
export ASR_MODEL_DIR=/abs/path/checkpoint
```

### C. This server only

With no arguments, the library auto-finds the sibling
`../reconstruction_production/1_stt/checkpoint`.

**Resolution priority:** explicit `model_dir` → `hf_repo` / `ASR_HF_REPO` →
`ASR_MODEL_DIR` → sibling checkpoint.

## One-shot: STT + correction

```python
from asr_text_correct import ASRCorrector

pipe = ASRCorrector()                 # loads Whisper once (~3 GB VRAM fp16)
r = pipe.process("clip.wav")
r.transcript        # '나는 바지를 입고 단추를 채웁니다'
r.corrected_text    # '나는 바지를 입고 단추를 채웠다.'
r.confidence        # 0.899   (beam-search, 0..1)
r.nbest             # [{'rank': 1, 'transcription': ..., 'confidence': ...}, ...]
r.is_uncertain      # True if confidence < 0.15 or empty transcript
r.correction_ok     # False if the Gemini fallback was used
r.timings           # {'stt': 0.76, 'text_correction': 0.77, 'total': 1.53}
r.to_dict()
```

Options for `process()`:
- `language="KR"` (or `"korean"`).
- `reference_text=...` fills `r.cer`.
- `detailed=False` switches to the faster greedy decode, which leaves
  confidence and N-best empty.
- `prompt=...` overrides the correction prompt for that call.

`process()` accepts a file path, encoded bytes (`wav/flac/ogg`, plus `mp3/m4a`
if ffmpeg is present), or a numpy/torch waveform passed with `input_sr=`.

### Only one half

```python
from asr_text_correct import Transcriber, GeminiCorrector

asr = Transcriber()                                  # STT only, no API key needed
asr.transcribe("clip.wav")                           # greedy -> str
asr.transcribe("clip.wav", detailed=True)            # beam -> dict with confidence / nbest

fix = GeminiCorrector()                              # correction only, no model load
fix.correct("은 변원 앞에 있어요 약을 하우레 두 씩 드세요")
# -> '병원 앞에 있어요. 약을 하루에 두 시 씩 드세요.'
fix.correct_detailed(text)                           # Correction(text, raw_text, ok, model, error)

pipe = ASRCorrector(transcriber=asr, corrector=fix)  # combine existing objects
```

## Streaming (VAD-segmented)

Your app captures audio from any source and pushes mono float32 frames. Each
time the speaker pauses, the buffered utterance is transcribed, then corrected:

```python
pipe = ASRCorrector()
stream = pipe.stream(on_segment=lambda s: print(s.text, "->", s.corrected_text))

for chunk in your_mic_source():                    # any length, float32 mono
    for seg in stream.accept(chunk, input_sr=48000):
        ...                                        # seg.text, seg.corrected_text, seg.start_time

stream.flush()                                     # finalize trailing speech
```

- `pipe.stream(correct=False)` and `Transcriber().stream()` return raw STT
  only, and `corrected_text` stays `None`.
- Streaming uses the greedy decode to keep latency low.
- `accept()` blocks while a finished utterance is transcribed and corrected.
  That takes about 0.3–0.8 s for STT plus one Gemini round-trip. If you can't
  block your capture thread, run the corrector in a worker.

Runnable mic demo: `python examples/stream_mic.py`

### Tuning `stream(...)`

| arg | default | meaning |
|---|---|---|
| `end_silence_ms` | 600 | pause length that ends an utterance |
| `min_speech_ms` | 250 | shorter blips are dropped |
| `max_speech_ms` | 15000 | force-cut a segment that never pauses (Whisper window is 30 s) |
| `start_speech_frames` | 3 | consecutive speech frames to trigger |
| `pre_roll_ms` | 200 | audio kept before the trigger point |
| `vad_aggressiveness` | 2 | webrtcvad 0–3 (higher = stricter) |

## Notes / limits

- Output text is `ko_norm`-normalized: NFC, lowercase, and only Hangul, digits
  and spaces. Degenerate repetitive output (zlib ratio > 2.4) is returned as
  `""`.
- Correction is an LLM rewrite. It may change verb endings and add punctuation,
  so keep `transcript` if you need the literal STT output.
- One `ASRCorrector` is fine for sequential calls but runs one inference at a
  time. If you need parallelism, use one per worker. `Transcriber` caches the
  loaded model per config, so creating it again is cheap.

## Layout

```
asr_text_correct/
  config.py        EngineConfig / CorrectorConfig + asset & API-key resolution
  hub.py           HF Hub download + one-time checkpoint upload
  audio.py         decode / resample to 16 kHz mono
  postprocess.py   ko_norm, repetition filter, CER   (vendored from 1_stt/transcribe.py)
  engine.py        WhisperEngine: greedy + beam/confidence/N-best (mirrors 1_stt/transcribe.py)
  corrector.py     GeminiCorrector: prompt, key rotation, fallback (mirrors 2_t2t + model_manager)
  transcriber.py   Transcriber, ASRCorrector, StreamingTranscriber, Segment, Result
  vad.py           webrtcvad wrapper + energy fallback
examples/          transcribe_file.py, correct_text.py, stream_mic.py, upload_to_hf.py
tests/             test_smoke.py
```
