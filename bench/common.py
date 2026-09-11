"""Shared CLI helpers for the pilot experiments."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from verifier_anchored_sd.experiment_data import (  # noqa: F401
    token_windows,
    token_windows_with_sources,
)
from verifier_anchored_sd.model_contracts import validate_tokenizer_pair


def resolve_dtype(dtype: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


def load_hf_tokenizer(model_id: str, *, revision: str = "main"):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )


def load_hf_model(
    model_id: str,
    device: str,
    dtype: str = "bfloat16",
    *,
    revision: str = "main",
    gpu_memory_gib: int | None = None,
    offload_folder: str | Path | None = None,
):
    """Load one exact-weight model, optionally with a bounded CUDA allocation."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
    load_kwargs = {
        "torch_dtype": resolve_dtype(dtype),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "revision": revision,
    }
    if gpu_memory_gib is not None and torch.cuda.is_available() and device != "cpu":
        if gpu_memory_gib <= 0:
            raise ValueError("gpu_memory_gib must be positive")
        if offload_folder is None:
            raise ValueError("offload_folder is required with a GPU memory cap")
        load_kwargs.update(
            device_map="auto",
            max_memory={0: f"{gpu_memory_gib}GiB", "cpu": "80GiB"},
            offload_state_dict=True,
            offload_folder=str(offload_folder),
        )
    else:
        load_kwargs["device_map"] = (
            device if device in {"auto", "balanced", "balanced_low_0"} else {"": device}
        )
    return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()


def load_hf_pair(
    target_id: str,
    draft_id: str,
    device: str,
    dtype: str = "bfloat16",
    *,
    low_vram: bool = False,
    target_gpu_memory_gib: int = 6,
    draft_gpu_memory_gib: int = 4,
    target_device: str | None = None,
    draft_device: str | None = None,
    target_revision: str = "main",
    draft_revision: str = "main",
):
    target_device = device if target_device is None else target_device
    draft_device = device if draft_device is None else draft_device
    tokenizer = load_hf_tokenizer(target_id, revision=target_revision)
    draft_tokenizer = load_hf_tokenizer(draft_id, revision=draft_revision)
    validate_tokenizer_pair(tokenizer, draft_tokenizer)
    if low_vram and torch.cuda.is_available() and target_device == draft_device and device != "cpu":
        # Keep exact BF16 weights; offload only reduces residency. This profile is
        # for smoke/feasibility runs, not for paper wall-clock gates.
        target = load_hf_model(
            target_id,
            target_device,
            dtype,
            revision=target_revision,
            gpu_memory_gib=target_gpu_memory_gib,
            offload_folder=".cache/vakv_offload_target",
        )
        draft = load_hf_model(
            draft_id,
            draft_device,
            dtype,
            revision=draft_revision,
            gpu_memory_gib=draft_gpu_memory_gib,
            offload_folder=".cache/vakv_offload_draft",
        )
    else:
        target = load_hf_model(target_id, target_device, dtype, revision=target_revision)
        draft = load_hf_model(draft_id, draft_device, dtype, revision=draft_revision)
    if target.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError("target embedding table does not cover the shared tokenizer")
    if draft.get_input_embeddings().num_embeddings < len(tokenizer):
        raise ValueError("draft embedding table does not cover the shared tokenizer")
    return tokenizer, target, draft


def iter_texts(text_file: str | None, *, limit: int) -> Iterator[str]:
    if text_file:
        path = Path(text_file)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    yield str(row.get("text", row)) if isinstance(row, dict) else str(row)
                except json.JSONDecodeError:
                    yield line
                limit -= 1
                if limit <= 0:
                    return
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("provide --text-file or install the HF extra for FineWeb-Edu") from exc
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        yield row["text"]
        limit -= 1
        if limit <= 0:
            return


def model_dims(model):
    config = getattr(model.config, "text_config", model.config)
    layers = int(config.num_hidden_layers)
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    attn_heads = int(config.num_attention_heads)
    dim = int(getattr(config, "head_dim", config.hidden_size // attn_heads))
    return layers, heads, dim
