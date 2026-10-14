"""One-shot STT + correction of an audio file.

    export GEMINI_API_KEY=...
    python examples/transcribe_file.py /abs/path/clip.wav [model_dir]
"""

import sys

from asr_text_correct import ASRCorrector


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: transcribe_file.py <audio> [model_dir]")
    audio = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else None

    pipe = ASRCorrector(model_dir)
    print(f"[device={pipe.device}]")
    r = pipe.process(audio)
    print("stt       :", r.transcript)
    print("corrected :", r.corrected_text, "" if r.correction_ok else "(fallback)")
    print(f"confidence: {r.confidence:.3f}  uncertain={r.is_uncertain}")
    print("timings   :", {k: round(v, 3) for k, v in r.timings.items()})


if __name__ == "__main__":
    main()
