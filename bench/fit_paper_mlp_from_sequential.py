#!/usr/bin/env python3
"""Train the paper's independent full-head nonlinear KV mapper.

For each receiver layer and K/V kind, all receiver heads are trained together only
for implementation efficiency. Parameters remain independent across receiver heads.
The input is the same cross-head concatenation and the same selected source layers
as a preregistered paper-faithful full-head ridge mapper.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from verifier_anchored_sd.cache_artifacts import (
    exact_shard_paths,
    load_cache_shard,
    load_manifest,
    validate_capture_pair,
)
from verifier_anchored_sd.experiment_artifacts import atomic_write_json, sha256_file
from verifier_anchored_sd.paper_mapper_baselines import (
    GroupedHeadMLPMapper,
    MLPMapperMetadata,
)
from verifier_anchored_sd.paper_mapper_fit import (
    paper_mlp_profile,
    validate_mlp_ridge_metadata,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


class _LayerKindMLP(nn.Module):
    """Independent MLPs for every receiver head of one receiver layer/KV kind."""

    def __init__(self, heads: int, in_dim: int, hidden: int, out_dim: int, *, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.w1 = nn.Parameter(torch.empty(heads, in_dim, hidden))
        self.b1 = nn.Parameter(torch.empty(heads, hidden))
        self.w2 = nn.Parameter(torch.empty(heads, hidden, hidden))
        self.b2 = nn.Parameter(torch.empty(heads, hidden))
        self.w3 = nn.Parameter(torch.empty(heads, hidden, out_dim))
        self.b3 = nn.Parameter(torch.empty(heads, out_dim))
        # PyTorch Linear-like Kaiming-uniform initialization, independently sampled.
        for weight, fan_in in ((self.w1, in_dim), (self.w2, hidden), (self.w3, hidden)):
            bound = math.sqrt(3.0 / fan_in)
            with torch.no_grad():
                weight.uniform_(-bound, bound, generator=generator)
        for bias, fan_in in ((self.b1, in_dim), (self.b2, hidden), (self.b3, hidden)):
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                bias.uniform_(-bound, bound, generator=generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(torch.einsum("np,hpf->nhf", x, self.w1) + self.b1)
        hidden = torch.relu(torch.einsum("nhf,hfg->nhg", hidden, self.w2) + self.b2)
        return torch.einsum("nhf,hfd->nhd", hidden, self.w3) + self.b3


def _features(cache, selected: list[int], kind: int) -> torch.Tensor:
    rows = []
    for layer in selected:
        tensor = cache.layers[layer].key if kind == 0 else cache.layers[layer].value
        batch, heads, tokens, dim = tensor.shape
        rows.append(tensor.permute(0, 2, 1, 3).reshape(batch * tokens, heads * dim))
    return torch.cat(rows, dim=-1)


def _targets(cache, layer: int, kind: int) -> torch.Tensor:
    tensor = cache.layers[layer].key if kind == 0 else cache.layers[layer].value
    batch, heads, tokens, dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch * tokens, heads, dim)


def _iter_batches(
    source_paths: list[Path],
    draft_paths: list[Path],
    *,
    selected: list[int],
    target_layer: int,
    kind: int,
    batch_size: int,
    device: torch.device,
):
    pending_x: list[torch.Tensor] = []
    pending_y: list[torch.Tensor] = []
    pending = 0
    for source_path, draft_path in zip(source_paths, draft_paths, strict=True):
        source, source_meta = load_cache_shard(source_path)
        draft, draft_meta = load_cache_shard(draft_path)
        if source_meta.get("sequence_id") != draft_meta.get("sequence_id") or source_meta.get(
            "token_digest"
        ) != draft_meta.get("token_digest"):
            raise RuntimeError("MLP source/draft shard binding mismatch")
        if kind == 0:
            source = source.to_content_space()
            draft = draft.to_content_space()
        x = _features(source, selected, kind).float()
        y = _targets(draft, target_layer, kind).float()
        pending_x.append(x)
        pending_y.append(y)
        pending += x.shape[0]
        while pending >= batch_size:
            merged_x = torch.cat(pending_x, dim=0)
            merged_y = torch.cat(pending_y, dim=0)
            yield (
                merged_x[:batch_size].to(device, non_blocking=True),
                merged_y[:batch_size].to(device, non_blocking=True),
            )
            merged_x, merged_y = merged_x[batch_size:], merged_y[batch_size:]
            pending_x = [merged_x] if merged_x.numel() else []
            pending_y = [merged_y] if merged_y.numel() else []
            pending = merged_x.shape[0]
    if pending:
        yield (
            torch.cat(pending_x, dim=0).to(device, non_blocking=True),
            torch.cat(pending_y, dim=0).to(device, non_blocking=True),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-dir", required=True)
    ap.add_argument("--ridge-metadata", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile", choices=["pilot", "paper"], default="pilot")
    ap.add_argument("--sequences", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--storage-dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    args = ap.parse_args()

    pair_root = Path(args.pair_dir)
    source_root, draft_root = pair_root / "source", pair_root / "draft"
    source_manifest = load_manifest(source_root)
    draft_manifest = load_manifest(draft_root)
    geometry = validate_capture_pair(source_manifest, draft_manifest)
    count = int(source_manifest["capture"]["count"])
    profile = paper_mlp_profile(args.profile)
    if args.sequences is None:
        sequences = 500 if profile.paper_faithful else min(32, count)
    else:
        sequences = args.sequences
    if sequences <= 0 or sequences > count:
        raise ValueError("MLP sequences must lie inside the captured calibration count")
    if profile.paper_faithful:
        capture = source_manifest["capture"]
        if sequences != 500 or capture.get("seq_len") != 1024 or capture.get("stride") != 4:
            raise RuntimeError(
                "paper MLP profile requires exactly 500 x 1024 calibration windows with stride 4"
            )

    ridge_metadata_path = Path(args.ridge_metadata)
    ridge_metadata = json.loads(ridge_metadata_path.read_text(encoding="utf-8"))
    selected_layers = validate_mlp_ridge_metadata(
        ridge_metadata,
        pair=source_manifest["pair"],
        target_revision=source_manifest["model"]["revision"],
        draft_revision=draft_manifest["model"]["revision"],
        draft_layers=geometry["draft_layers"],
    )
    source_paths = exact_shard_paths(source_root / "shards", count)[:sequences]
    draft_paths = exact_shard_paths(draft_root / "shards", count)[:sequences]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA MLP training requested but CUDA is unavailable")

    m = MLPMapperMetadata(
        target_layers=geometry["target_layers"],
        draft_layers=geometry["draft_layers"],
        target_kv_heads=geometry["kv_heads"],
        draft_kv_heads=geometry["kv_heads"],
        head_dim=geometry["head_dim"],
        layer_selection=selected_layers,
        content_space=True,
        hidden_dim=profile.hidden_dim,
    )
    max_features = max(m.feature_width(layer) for layer in range(m.draft_layers))
    store_dtype = torch.bfloat16 if args.storage_dtype == "bfloat16" else torch.float32
    prefix = (m.draft_layers, 2, m.draft_kv_heads)
    # Host RAM is the artifact store; only one receiver layer/KV-kind MLP is live on GPU.
    w1 = torch.zeros(*prefix, max_features, m.hidden_dim, dtype=store_dtype)
    b1 = torch.zeros(*prefix, m.hidden_dim, dtype=store_dtype)
    w2 = torch.zeros(*prefix, m.hidden_dim, m.hidden_dim, dtype=store_dtype)
    b2 = torch.zeros(*prefix, m.hidden_dim, dtype=store_dtype)
    w3 = torch.zeros(*prefix, m.hidden_dim, m.head_dim, dtype=store_dtype)
    b3 = torch.zeros(*prefix, m.head_dim, dtype=store_dtype)
    diagnostics: list[dict] = []
    started = time.perf_counter()

    try:
        for layer in range(m.draft_layers):
            in_dim = m.feature_width(layer)
            for kind in range(2):
                block = _LayerKindMLP(
                    m.draft_kv_heads,
                    in_dim,
                    m.hidden_dim,
                    m.head_dim,
                    seed=args.seed + 2 * layer + kind,
                ).to(device=device, dtype=torch.float32)
                optimizer = torch.optim.Adam(block.parameters(), lr=profile.learning_rate)
                final_loss = float("nan")
                updates = 0
                for epoch in range(profile.epochs):
                    weighted_loss = 0.0
                    seen = 0
                    for x, y in _iter_batches(
                        source_paths,
                        draft_paths,
                        selected=selected_layers[layer],
                        target_layer=layer,
                        kind=kind,
                        batch_size=profile.batch_size,
                        device=device,
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        prediction = block(x)
                        loss = torch.nn.functional.mse_loss(prediction, y)
                        if not torch.isfinite(loss):
                            raise RuntimeError("paper MLP training produced a non-finite loss")
                        loss.backward()
                        optimizer.step()
                        batch = x.shape[0]
                        weighted_loss += float(loss.detach()) * batch
                        seen += batch
                        updates += 1
                    if seen == 0:
                        raise RuntimeError("paper MLP training data iterator is empty")
                    final_loss = weighted_loss / seen
                with torch.no_grad():
                    w1[layer, kind, :, :in_dim] = block.w1.detach().to(
                        device="cpu", dtype=store_dtype
                    )
                    b1[layer, kind] = block.b1.detach().to(device="cpu", dtype=store_dtype)
                    w2[layer, kind] = block.w2.detach().to(device="cpu", dtype=store_dtype)
                    b2[layer, kind] = block.b2.detach().to(device="cpu", dtype=store_dtype)
                    w3[layer, kind] = block.w3.detach().to(device="cpu", dtype=store_dtype)
                    b3[layer, kind] = block.b3.detach().to(device="cpu", dtype=store_dtype)
                diagnostics.append(
                    {
                        "draft_layer": layer,
                        "kind": "key" if kind == 0 else "value",
                        "final_train_mse": final_loss,
                        "optimizer_updates": updates,
                    }
                )
                del block, optimizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"paper-MLP layer={layer} kind={'K' if kind == 0 else 'V'} "
                    f"mse={final_loss:.6g}",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError as exc:
        output = Path(args.output)
        atomic_write_json(
            f"{output}.failure.json",
            {
                "schema_version": 1,
                "status": "incomplete",
                "phase": "paper_mlp_training",
                "completed_layer_kinds": len(diagnostics),
                "requested_layer_kinds": 2 * m.draft_layers,
                "profile": args.profile,
                "sequences": sequences,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "rule": "do_not_auto_shrink_hidden_dim_batch_model_dtype_or_k",
            },
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    mapper = GroupedHeadMLPMapper(m, w1, b1, w2, b2, w3, b3)
    output = Path(args.output)
    mapper.save(output)
    metadata = {
        "schema_version": 3,
        "git_commit": _git_commit(),
        "pair": source_manifest["pair"],
        "source_model": source_manifest["model"],
        "draft_model": draft_manifest["model"],
        "capture": source_manifest["capture"],
        "dtype": source_manifest["dtype"],
        "calibration_input_sha256": source_manifest["input_sha256"],
        "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "draft_manifest_sha256": sha256_file(draft_root / "manifest.json"),
        "token_rows_digest": source_manifest["token_rows_digest"],
        "token_row_digests": source_manifest["token_row_digests"][:sequences],
        "mapper": {
            "kind": "mlp_full_head",
            "head_mode": "full",
            "content_space": True,
            "k": max(len(row) for row in selected_layers),
            "selected_layers": selected_layers,
            "hidden_dim": profile.hidden_dim,
            "optimizer": profile.optimizer,
            "learning_rate": profile.learning_rate,
            "epochs": profile.epochs,
            "batch_size": profile.batch_size,
            "loss": profile.loss,
            "profile": args.profile,
            "paper_faithful": profile.paper_faithful,
            "sequences": sequences,
            "ridge_selection_metadata_sha256": sha256_file(ridge_metadata_path),
            "feature_width_max": max_features,
            "parameter_sharing_across_receiver_heads": False,
        },
        "training_diagnostics": diagnostics,
        "elapsed_s": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(output),
    }
    atomic_write_json(f"{output}.json", metadata)
    print(
        json.dumps(
            {
                "output": str(output),
                "profile": args.profile,
                "paper_faithful": profile.paper_faithful,
                "sequences": sequences,
                "mean_final_train_mse": sum(x["final_train_mse"] for x in diagnostics)
                / len(diagnostics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
