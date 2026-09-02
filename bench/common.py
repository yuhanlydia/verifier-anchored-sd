"""Shared CLI helpers for the three pilot experiments."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def load_hf_pair(target_id: str, draft_id: str, device: str, dtype: str = "bfloat16"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)
    device_map = {"": device} if device not in {"auto", "balanced", "balanced_low_0"} else device
    target = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True
    ).eval()
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
    config = model.config
    layers = int(config.num_hidden_layers)
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    attn_heads = int(config.num_attention_heads)
    dim = int(getattr(config, "head_dim", config.hidden_size // attn_heads))
    return layers, heads, dim
