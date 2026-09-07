import pytest
import torch

from verifier_anchored_sd.experiment_data import (
    collect_tokenized_prompts,
    token_windows,
    token_windows_with_sources,
)


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert isinstance(add_special_tokens, bool)
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[int(token) for token in text.split()]])}


def test_token_windows_uses_multiple_disjoint_chunks_from_long_documents():
    rows = list(
        token_windows(
            _Tokenizer(),
            ["0 1 2 3 4 5 6 7", "8 9 10 11"],
            seq_len=4,
            count=3,
        )
    )

    assert [row.tolist() for row in rows] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def test_token_windows_preserve_document_and_chunk_provenance():
    rows = list(
        token_windows_with_sources(
            _Tokenizer(),
            ["0 1 2 3 4 5 6 7", "8 9 10 11"],
            seq_len=4,
            count=3,
        )
    )

    assert [(item.document_id, item.chunk_id) for item in rows] == [(0, 0), (0, 1), (1, 0)]


def test_collect_prompts_skips_short_rows_but_requires_requested_count():
    rows = collect_tokenized_prompts(
        _Tokenizer(), ["1", "2 3", "4 5 6"], max_tokens=2, count=2
    )

    assert rows == [[2, 3], [4, 5]]
    with pytest.raises(RuntimeError, match="1/2"):
        collect_tokenized_prompts(
            _Tokenizer(), ["1", "2 3"], max_tokens=2, count=2
        )
