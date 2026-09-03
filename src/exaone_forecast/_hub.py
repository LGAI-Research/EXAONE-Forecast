"""Shared helper for fetching a released checkpoint from the Hugging Face Hub.

A model publishes its weights in its own Hub repository. This helper fetches
``config.json`` plus one weight file and stages them as a checkpoint directory,
for models whose loader expects that layout.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["download_checkpoint"]


def download_checkpoint(
    repo_id: str,
    weight_filename: str,
    staging_name: str,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> str:
    """Download ``config.json`` + ``weight_filename`` and return a checkpoint dir.

    Released weight files carry a descriptive name on the Hub, while the loader
    expects a directory that looks like a standard checkpoint. The file is
    therefore materialised as ``model.safetensors`` inside a local staging
    directory, which is what this function returns.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download the weights. Install it with "
            "`pip install huggingface_hub`, or pass an explicit local checkpoint "
            "directory to exaone_forecast.forecast.EXAONEFinance(ckpt_dir=...)."
        ) from exc

    kwargs = {"repo_id": repo_id, "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    config_path = Path(hf_hub_download(filename="config.json", **kwargs))
    weight_path = Path(hf_hub_download(filename=weight_filename, **kwargs))

    staging = config_path.parent / staging_name
    staging.mkdir(parents=True, exist_ok=True)
    _link_or_copy(config_path, staging / "config.json")
    _link_or_copy(weight_path, staging / "model.safetensors")
    return str(staging)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Point ``dst`` at ``src`` without duplicating a multi-hundred-MB file."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src.resolve():
            return
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:  # filesystems without symlink support
        shutil.copyfile(src, dst)
