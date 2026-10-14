"""Correct a raw transcript with Gemini only (no audio, no model load).

    export GEMINI_API_KEY=...
    python examples/correct_text.py "은 변원 앞에 있어요 약을 하우레 두 씩 드세요"
"""

import sys

from asr_text_correct import GeminiCorrector


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: correct_text.py <raw text>")
    corrector = GeminiCorrector()
    result = corrector.correct_detailed(" ".join(sys.argv[1:]))
    print(result.text)
    if not result.ok:
        print(f"(fallback used: {result.error})", file=sys.stderr)


if __name__ == "__main__":
    main()
