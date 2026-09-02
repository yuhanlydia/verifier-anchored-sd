"""Shared CLI helpers for the pilot experiments."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def _tokenizer_contract(tokenizer) -> dict:
    return {
        "vocab": tokenizer.get_vocab(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def load_hf_pair(
    target_id: str,
    draft_id: str,
    device: str,
    dtype: str = "bfloat16",
    *,
    low_vram: bool = False,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_id, trust_remote_code=True)
    if _tokenizer_contract(tokenizer) != _tokenizer_contract(draft_tokenizer):
        raise ValueError(
            "target and draft tokenizer contracts differ; cross-model cache positions/token IDs are unsafe"
        )
    device_map = {"": device} if device not in {"auto", "balanced", "balanced_low_0"} else device
    load_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True, "low_cpu_mem_usage": True}
    if low_vram and torch.cuda.is_available() and device != "cpu":
        # Keep exact BF16 weights; offload only reduces residency. This profile is
        # for smoke/feasibility runs, not for paper wall-clock gates.
        device_map = "auto"
        load_kwargs.update(
            max_memory={0: "13GiB", "cpu": "48GiB"},
            offload_state_dict=True,
            offload_folder=".cache/vakv_offload_target",
        )
    target = AutoModelForCausalLM.from_pretrained(
        target_id, device_map=device_map, **load_kwargs
    ).eval()
    if low_vram and torch.cuda.is_available() and device != "cpu":
        load_kwargs["offload_folder"] = ".cache/vakv_offload_draft"
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, device_map=device_map, **load_kwargs
    ).eval()
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


def token_windows(tokenizer, texts, *, seq_len: int, count: int):
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if ids.numel() < seq_len:
            continue
        yield ids[:seq_len]
        count -= 1
        if count <= 0:
            return


def model_dims(model):
    config = getattr(model.config, "text_config", model.config)
    layers = int(config.num_hidden_layers)
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    attn_heads = int(config.num_attention_heads)
    dim = int(getattr(config, "head_dim", config.hidden_size // attn_heads))
    return layers, heads, dim
