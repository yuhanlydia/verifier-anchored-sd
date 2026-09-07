import pytest
import torch

from verifier_anchored_sd.transfer_metrics import (
    attention_output_cosines,
    finalize_screen,
    validate_screen_inputs,
)


def mapper_metadata():
    return {
        "pair": {"target": "Qwen/T", "draft": "Qwen/D"},
        "source_model": {"revision": "target-rev", "tokenizer_hash": "tok"},
        "draft_model": {"revision": "draft-rev", "tokenizer_hash": "tok"},
        "token_row_digests": ["cal-a", "cal-b"],
    }


def screen_manifest():
    return {
        "pair": {"target": "Qwen/T", "draft": "Qwen/D"},
        "model": {"revision": "target-rev", "tokenizer_hash": "tok"},
        "token_row_digests": ["eval-a", "eval-b"],
    }


def test_screen_inputs_validate_pair_revision_tokenizer_and_disjointness():
    validate_screen_inputs(mapper_metadata(), screen_manifest())

    overlap = screen_manifest()
    overlap["token_row_digests"] = ["eval-a", "cal-b"]
    with pytest.raises(ValueError, match="overlap"):
        validate_screen_inputs(mapper_metadata(), overlap)


def test_screen_cannot_pass_with_missing_rows():
    result = finalize_screen(
        [{"a_transfer": 0.99, "kl_native_mapped": 0.01, "top1_agreement": 1}],
        requested=2,
        samples=100,
        seed=0,
    )

    assert result["gate"]["status"] == "incomplete"


def test_attention_cosine_compares_matching_layer_outputs():
    native = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 2.0]])]
    mapped = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, -3.0]])]

    values = attention_output_cosines(native, mapped)

    assert values == pytest.approx([1.0, -1.0])
