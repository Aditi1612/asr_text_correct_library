"""Publish the local Whisper checkpoint to a private Hugging Face Hub repo (run once).

    huggingface-cli login          # or export HF_TOKEN=hf_xxx
    python examples/upload_to_hf.py fluffy16/whisper-ko-asr \
        /data2/irfan/projects/APIs/reconstruction_production/1_stt/checkpoint
"""

import sys

from asr_text_correct import upload_checkpoint


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: upload_to_hf.py <repo_id> <checkpoint_dir>")
    url = upload_checkpoint(repo_id=sys.argv[1], checkpoint_dir=sys.argv[2], private=True)
    print("uploaded ->", url)


if __name__ == "__main__":
    main()
