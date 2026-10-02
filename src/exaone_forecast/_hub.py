"""Shared helper for fetching a released checkpoint from the Hugging Face Hub.

A model publishes its weights in its own Hub repository. This helper fetches
``config.json`` plus one weight file and stages them as a checkpoint directory,
for models whose loader expects that layout.

``huggingface_hub`` is used when it is installed. A model that does not depend
on it (EXAONE Demand) falls back to a plain HTTPS download of the same files
from the same repository, using the standard library only.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
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
    except ImportError:
        hf_hub_download = None

    if hf_hub_download is not None:
        kwargs = {"repo_id": repo_id, "revision": revision}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        config_path = Path(hf_hub_download(filename="config.json", **kwargs))
        weight_path = Path(hf_hub_download(filename=weight_filename, **kwargs))
    else:
        config_path = _download(repo_id, "config.json", revision, cache_dir)
        weight_path = _download(repo_id, weight_filename, revision, cache_dir)

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


def _download(
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | os.PathLike | None,
) -> Path:
    """Fetch one file of a public Hub repository without ``huggingface_hub``.

    The file is cached under ``cache_dir`` (default ``$HF_HOME/exaone_forecast``,
    i.e. ``~/.cache/huggingface/exaone_forecast``) and reused on later calls.
    ``HF_ENDPOINT`` and ``HF_TOKEN`` are honored as in ``huggingface_hub``.
    """
    revision = revision or "main"
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME") or os.path.join(
            os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
            "huggingface",
        )
        cache_dir = os.path.join(hf_home, "exaone_forecast")
    target = Path(cache_dir) / repo_id.replace("/", "--") / revision / filename
    if target.exists():
        return target

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    request = urllib.request.Request(f"{endpoint}/{repo_id}/resolve/{revision}/{filename}")
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    # Write to a temporary file first so an interrupted download never leaves a
    # truncated file where a later call would take it for a finished one.
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(dir=target.parent, prefix=f".{filename}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(request) as response:
            shutil.copyfileobj(response, out, length=16 * 1024 * 1024)
        os.replace(partial, target)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return target
