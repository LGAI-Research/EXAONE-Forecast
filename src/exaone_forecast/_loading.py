"""Shared helper for loading a released checkpoint into its model class.

Every model in the family publishes a checkpoint directory holding ``config.json``
and ``model.safetensors``. The weights are loaded by building the model from its
config and then calling ``load_state_dict`` on the raw tensors — that is,
**bypassing** ``PreTrainedModel.from_pretrained``'s weight-conversion path.

This is deliberate. A model whose config declares a ``model_type`` that
transformers recognises, and whose layers reuse that architecture's parameter
names, will have its weights rewritten by the conversion transformers >= 5
applies on ``from_pretrained`` — silently, and to something the checkpoint never
meant. A plain ``load_state_dict`` is verbatim and transformers-version
independent.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_checkpoint"]


def load_checkpoint(ckpt_dir: str | os.PathLike, model_cls, config_cls, device=None):
    """Build ``model_cls`` from the checkpoint's config and load its weights verbatim.

    Args:
        ckpt_dir: directory holding ``config.json`` and ``model.safetensors``.
        model_cls: the model class to instantiate.
        config_cls: the ``PretrainedConfig`` subclass that reads ``config.json``.
        device: optional torch device to move the model to.

    Returns:
        The model in ``.eval()`` mode.
    """
    ckpt = Path(ckpt_dir)
    if not (ckpt / "config.json").exists() and (ckpt / "hf_final_model" / "config.json").exists():
        ckpt = ckpt / "hf_final_model"

    weights = ckpt / "model.safetensors"
    if weights.exists():
        config = config_cls.from_pretrained(str(ckpt))   # config only, no weight conversion
        model = model_cls(config)
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(weights)), strict=True)
    else:
        # non-safetensors or sharded checkpoints
        model = model_cls.from_pretrained(str(ckpt))

    if device is not None:
        model = model.to(device)
    return model.eval()
