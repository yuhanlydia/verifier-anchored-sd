import torch

from verifier_anchored_sd.experiment_data import token_windows


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert not add_special_tokens
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
