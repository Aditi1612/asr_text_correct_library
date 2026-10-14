"""Hugging Face Hub integration for distributing the Whisper checkpoint.

The weights (several GB) are too large to ship inside the pip package, so they
live in a (private) HF Hub model repo. On first use the library downloads and
caches them; afterwards it loads from the local cache offline.

Expected repo layout (either works)::

    repo-root/                      repo-root/
      config.json                     checkpoint/
      model*.safetensors                config.json
      tokenizer.json ...                model*.safetensors ...

Upload once with :func:`upload_checkpoint` (or ``huggingface-cli``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Files the Whisper model + processor need. Skips optimizer.pt, scheduler.pt,
# rng_state.pth, training_args.bin etc. that sit in a training checkpoint dir.
_CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "model.safetensors.index.json",
]
_ALLOW_PATTERNS = (
    _CHECKPOINT_FILES
    + ["*.safetensors", "pytorch_model*.bin"]
    + [f"checkpoint/{n}" for n in _CHECKPOINT_FILES]
    + ["checkpoint/*.safetensors", "checkpoint/pytorch_model*.bin"]
)


def _require_hf():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Hugging Face Hub support needs the 'huggingface_hub' package.\n"
            "    pip install 'asr-text-correct[hub]'"
        ) from exc
    return __import__("huggingface_hub")


def resolve_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Download (or reuse cached) checkpoint; return the local model directory.

    ``token`` falls back to ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the
    environment, or a prior ``huggingface-cli login``.
    """
    hf = _require_hf()
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    local = hf.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        allow_patterns=_ALLOW_PATTERNS,
    )
    root = Path(local)
    return str(root / "checkpoint" if (root / "checkpoint").is_dir() else root)


def upload_checkpoint(
    repo_id: str,
    checkpoint_dir: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Publish a local Whisper checkpoint to an HF Hub model repo (creating it if needed).

    Uploads only the files inference needs. Returns the repo URL.
    """
    hf = _require_hf()
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = hf.HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    ckpt = Path(checkpoint_dir)
    wanted = list(_CHECKPOINT_FILES)
    wanted += [p.name for p in ckpt.glob("*.safetensors")]
    if not any(n.endswith(".safetensors") for n in wanted):
        wanted += [p.name for p in ckpt.glob("pytorch_model*.bin")]

    # One commit for all files; large files go through LFS/xet with resumable chunks.
    api.upload_folder(
        folder_path=str(ckpt),
        path_in_repo="checkpoint",
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        allow_patterns=[n for n in wanted if (ckpt / n).is_file()],
        commit_message="Upload Whisper checkpoint (inference files only)",
    )

    return f"https://huggingface.co/{repo_id}"
