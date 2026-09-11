from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import common  # noqa: E402


class _Tokenizer:
    def __len__(self):
        return 4


class _Model:
    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=4)


def test_low_vram_pair_uses_requested_gpu_residency_budgets(monkeypatch):
    calls = []

    monkeypatch.setattr(common, "load_hf_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(common, "validate_tokenizer_pair", lambda *args: None)
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)

    def load_model(model_id, device, dtype, **kwargs):
        calls.append((model_id, kwargs["gpu_memory_gib"]))
        return _Model()

    monkeypatch.setattr(common, "load_hf_model", load_model)

    common.load_hf_pair(
        "target",
        "draft",
        "cuda",
        "bfloat16",
        low_vram=True,
        target_gpu_memory_gib=10,
        draft_gpu_memory_gib=7,
    )

    assert calls == [("target", 10), ("draft", 7)]
