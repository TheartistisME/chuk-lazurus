"""Tests for ``tools/_window_router/encoder.py``.

The BowCharacterEncoder tests are exhaustive (pure python, no deps).
The GemmaEmbedEncoder and HFSentenceEncoder tests stay offline by
replacing ``transformers`` with small stubs, so the suite never touches
network or a real HF cache.
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools._window_router.encoder import BowCharacterEncoder


def test_fit_then_encode_is_deterministic() -> None:
    e1 = BowCharacterEncoder(n=3).fit(["alpha beta", "gamma", "delta epsilon"])
    e2 = BowCharacterEncoder(n=3).fit(["alpha beta", "gamma", "delta epsilon"])

    v1 = e1.encode("alpha omega")
    v2 = e2.encode("alpha omega")
    assert v1 == v2


def test_vocab_size_stable_across_encode_calls() -> None:
    encoder = BowCharacterEncoder(n=3).fit(["hello world"])
    initial_size = encoder.vocab_size

    encoder.encode("completely unseen text with new trigrams")
    encoder.encode("another batch of unseen content")

    assert encoder.vocab_size == initial_size


def test_dim_equals_vocab_size() -> None:
    encoder = BowCharacterEncoder(n=3).fit(["a quick brown fox"])
    assert encoder.dim == encoder.vocab_size
    assert encoder.dim > 0


def test_encode_returns_list_of_floats_with_correct_length() -> None:
    encoder = BowCharacterEncoder(n=3).fit(["hello world"])
    vector = encoder.encode("hello")
    assert isinstance(vector, list)
    assert len(vector) == encoder.dim
    assert all(isinstance(v, float) for v in vector)


def test_encode_counts_ngrams_from_training_vocab() -> None:
    encoder = BowCharacterEncoder(n=3).fit(["abc"])
    # Only trigram "abc" is in vocab.
    vector = encoder.encode("abcabc")
    assert vector.count(2.0) == 1
    # "abcabc" has two "abc" trigrams and two "bca" / "cab" which are OOV.
    total_in_vocab = sum(vector)
    assert total_in_vocab == 2.0


def test_encode_without_fit_raises_runtime_error() -> None:
    encoder = BowCharacterEncoder()
    with pytest.raises(RuntimeError):
        encoder.encode("anything")


def test_set_vocab_restores_fitted_state() -> None:
    encoder = BowCharacterEncoder(n=2)
    encoder.set_vocab({"ab": 0, "bc": 1, "cd": 2})
    assert encoder.dim == 3
    vector = encoder.encode("abcd")
    assert vector == [1.0, 1.0, 1.0]


def test_clause_hierarchy_features_append_level_onehots() -> None:
    encoder = BowCharacterEncoder(n=2).fit(
        ["Clause 1.4.72 Scope", "Clause 2.6.3.2.2 Protection"]
    )

    assert encoder.clause_level_vocabs[0] == {"1": 0, "2": 1}
    assert encoder.clause_level_vocabs[1] == {"1.4": 0, "2.6": 1}
    assert encoder.clause_level_vocabs[2] == {"1.4.72": 0, "2.6.3.2.2": 1}

    text_dim = len(encoder.vocab)
    vector = encoder.encode("Explain clause 1.4.72 and clause 2.6.3.2.2")
    assert len(vector) == encoder.dim
    assert vector[text_dim:] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_set_clause_level_vocabs_restores_hierarchy_state() -> None:
    encoder = BowCharacterEncoder(n=2)
    encoder.set_vocab({"ab": 0, "bc": 1})
    encoder.set_clause_level_vocabs([["1"], ["1.4"], ["1.4.72"]])

    vector = encoder.encode("1.4.72 abcd")
    assert vector == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        BowCharacterEncoder(n=0)


def test_gemma_encoder_import_error_points_to_bow_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``transformers`` cannot be imported, the error must mention the bow fallback."""

    for mod_name in list(sys.modules):
        if mod_name == "transformers" or mod_name.startswith("transformers."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("mock: transformers unavailable in this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from tools._window_router.encoder import GemmaEmbedEncoder

    with pytest.raises(ImportError) as excinfo:
        GemmaEmbedEncoder(model_id="google/gemma-3-4b-it")

    message = str(excinfo.value).lower()
    assert "transformers" in message
    assert "bow" in message


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loads: dict[str, int],
    hidden_values: list[list[list[float]]],
    attention_mask_row: list[int],
    hidden_dtype: object | None = None,
) -> None:
    """Install a fake ``transformers`` module into ``sys.modules``.

    The fake model's ``__call__`` returns a predictable ``last_hidden_state``
    so tests can verify the mean-pool math exactly.
    """
    import torch  # noqa: PLC0415

    resolved_dtype = hidden_dtype if hidden_dtype is not None else torch.float32

    class FakeTokenizer:
        def __call__(
            self,
            text: str | list[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, object]:
            texts = [text] if isinstance(text, str) else list(text)
            assert texts
            assert return_tensors == "pt"
            assert padding is True
            assert truncation is True
            assert max_length == 512
            rows = len(texts)
            return {
                "input_ids": torch.tensor(
                    [[1, 2, 3, 4]] * rows, dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [list(attention_mask_row)] * rows, dtype=torch.long
                ),
            }

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeTokenizer:
            assert model_id == "google/gemma-test"
            loads["tokenizer"] += 1
            return FakeTokenizer()

    hidden_dim = len(hidden_values[0][0]) if hidden_values and hidden_values[0] else 2
    embedding = torch.nn.Embedding(8, hidden_dim)

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(hidden_size=hidden_dim, text_config=None)
            self.model = SimpleNamespace(embedding_scale=1.0)

        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

        def eval(self) -> FakeModel:
            return self

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return embedding

        def __call__(
            self,
            *,
            input_ids: object,
            attention_mask: object,
            output_hidden_states: bool,
        ) -> object:
            loads["forward"] += 1
            assert output_hidden_states is False
            assert attention_mask is not None
            batch = int(getattr(input_ids, "shape", [len(hidden_values)])[0])
            # Broadcast the canonical row to every batch entry.
            last_hidden_state = torch.tensor(
                [hidden_values[0] for _ in range(batch)],
                dtype=resolved_dtype,
            )
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeModel:
            assert model_id == "google/gemma-test"
            loads["model"] += 1
            return FakeModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_gemma_encoder_version_uses_last_hidden_state_prefix() -> None:
    """Bump to ``last-hidden-state-v1::`` invalidates stale embedding-layer caches."""
    from tools._window_router.cache import encoder_version

    version = encoder_version(encoder_type="gemma-embed", model_id="google/gemma-x")
    assert version.startswith("last-hidden-state-v1::")
    assert version == "last-hidden-state-v1::google/gemma-x"
    assert "embedding-layer-v1" not in version


def test_gemma_encoder_version_preserves_model_key_exactly() -> None:
    from tools._window_router.cache import encoder_version

    model_id = "/home/u/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/abc"
    version = encoder_version(encoder_type="gemma-embed", model_id=model_id)
    assert version == f"last-hidden-state-v1::{model_id}"


def test_gemma_encoder_lazy_loads_and_pools_last_hidden_state_with_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full forward pass: mean-pool ``last_hidden_state`` with attention_mask."""
    torch = pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}

    # 4-token sequence with the last position padded out via attention_mask.
    # Hidden states are deliberately distinct so the pool is easy to verify.
    hidden_values = [
        [
            [1.0, 10.0],  # token 0
            [2.0, 20.0],  # token 1
            [4.0, 40.0],  # token 2
            [999.0, 999.0],  # padded - must be excluded by mask
        ]
    ]
    attention_mask_row = [1, 1, 1, 0]

    _install_fake_transformers(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import GemmaEmbedEncoder

    encoder = GemmaEmbedEncoder(model_id="google/gemma-test", device="cpu").fit(
        ["Clause 1.4.72 Scope", "Clause 2.6.3.2.2 Protection"]
    )

    # fit() must not trigger any HF loading.
    assert loads == {"tokenizer": 0, "model": 0, "forward": 0}

    # Clause-hierarchy channel remains intact.
    assert encoder.clause_level_vocabs[0] == {"1": 0, "2": 1}
    assert encoder.clause_level_vocabs[1] == {"1.4": 0, "2.6": 1}
    assert encoder.clause_level_vocabs[2] == {"1.4.72": 0, "2.6.3.2.2": 1}
    # 2 text dims + 6 clause one-hots.
    assert encoder.dim == 8
    assert encoder.vocab_size == 8
    assert loads == {"tokenizer": 0, "model": 1, "forward": 0}

    vector = encoder.encode("gemma prompt clause 1.4.72")

    # Expected pooled text embedding is the mean over the 3 unmasked tokens:
    #   mean([1, 2, 4]) = 7/3, mean([10, 20, 40]) = 70/3.
    # Followed by clause-hierarchy one-hots (levels: 1, 1.4, 1.4.72).
    assert vector == pytest.approx(
        [7.0 / 3.0, 70.0 / 3.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )
    assert loads == {"tokenizer": 1, "model": 1, "forward": 1}

    batched = encoder.encode_many_tensor(
        ["gemma prompt clause 1.4.72", "gemma prompt clause 1.4.72"],
        batch_size=2,
    )
    assert tuple(batched.shape) == (2, 8)
    # fp32 must be preserved in the batched output.
    assert batched.dtype == torch.float32


def test_gemma_encoder_mean_pool_exact_with_all_ones_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an all-ones ``last_hidden_state`` and partial mask, pooled text = 1.0."""
    pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    hidden_values = [
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    ]
    attention_mask_row = [1, 1, 0, 0]

    _install_fake_transformers(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import GemmaEmbedEncoder

    # Fit with no clause IDs so clause feature dim is zero.
    encoder = GemmaEmbedEncoder(model_id="google/gemma-test", device="cpu").fit(
        ["no clause ids here", "plain text"]
    )
    assert encoder.clause_feature_dim == 0
    # 3-dim all-ones vector: mean over any unmasked subset is (1, 1, 1).
    vector = encoder.encode("plain text")
    assert vector == pytest.approx([1.0, 1.0, 1.0])


def test_gemma_encoder_preserves_fp32_even_when_hidden_is_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when the model emits fp16, the pooled output must be fp32."""
    torch = pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    hidden_values = [
        [
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 4.0],
        ]
    ]
    attention_mask_row = [1, 1, 1, 1]

    _install_fake_transformers(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
        hidden_dtype=torch.float16,
    )

    from tools._window_router.encoder import GemmaEmbedEncoder

    encoder = GemmaEmbedEncoder(model_id="google/gemma-test", device="cpu").fit(
        ["plain text"]
    )
    batched = encoder.encode_many_tensor(["plain text"], batch_size=1)
    assert batched.dtype == torch.float32
    assert batched.shape == (1, 2)
    assert batched.reshape(-1).tolist() == pytest.approx([2.0, 4.0])


# ── HFSentenceEncoder tests ──────────────────────────────────────────


def _install_fake_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loads: dict[str, int],
    hidden_values: list[list[list[float]]],
    attention_mask_row: list[int],
    model_id_expected: str = "sentence-transformers/all-MiniLM-L6-v2",
    hidden_dtype: object | None = None,
) -> None:
    """Install a fake ``transformers`` module shaped like a sentence-transformer.

    The model reports ``config.hidden_size`` matching ``hidden_values`` and its
    ``__call__`` returns a deterministic ``last_hidden_state`` so tests can
    verify masked-mean-pool + L2 normalisation precisely.
    """
    import torch  # noqa: PLC0415

    resolved_dtype = hidden_dtype if hidden_dtype is not None else torch.float32
    hidden_dim = len(hidden_values[0][0]) if hidden_values and hidden_values[0] else 2

    class FakeTokenizer:
        def __call__(
            self,
            text: str | list[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, object]:
            texts = [text] if isinstance(text, str) else list(text)
            assert texts
            assert return_tensors == "pt"
            assert padding is True
            assert truncation is True
            assert max_length == 512
            rows = len(texts)
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]] * rows, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [list(attention_mask_row)] * rows, dtype=torch.long
                ),
            }

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeTokenizer:
            assert model_id == model_id_expected
            loads["tokenizer"] += 1
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(hidden_size=hidden_dim, text_config=None)

        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

        def eval(self) -> FakeModel:
            return self

        def __call__(
            self,
            *,
            input_ids: object,
            attention_mask: object,
            output_hidden_states: bool,
        ) -> object:
            loads["forward"] += 1
            assert output_hidden_states is False
            assert attention_mask is not None
            batch = int(getattr(input_ids, "shape", [len(hidden_values)])[0])
            last_hidden_state = torch.tensor(
                [hidden_values[0] for _ in range(batch)],
                dtype=resolved_dtype,
            )
            return SimpleNamespace(last_hidden_state=last_hidden_state)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeModel:
            assert model_id == model_id_expected
            loads["model"] += 1
            return FakeModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_hf_sentence_encoder_import_error_points_to_bow_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``transformers`` is unavailable the error must mention the bow fallback."""
    for mod_name in list(sys.modules):
        if mod_name == "transformers" or mod_name.startswith("transformers."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("mock: transformers unavailable in this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from tools._window_router.encoder import HFSentenceEncoder

    with pytest.raises(ImportError) as excinfo:
        HFSentenceEncoder(model_id="sentence-transformers/all-MiniLM-L6-v2")

    message = str(excinfo.value).lower()
    assert "transformers" in message
    assert "bow" in message


def test_hf_sentence_encoder_lazy_loads_and_pools_with_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full forward pass: masked mean-pool over ``last_hidden_state`` with L2 norm."""
    pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    # 2-dim hidden; last token padded out by attention_mask.
    hidden_values = [
        [
            [3.0, 4.0],  # token 0
            [3.0, 4.0],  # token 1
            [3.0, 4.0],  # token 2
            [999.0, 999.0],  # padded - must be excluded
        ]
    ]
    attention_mask_row = [1, 1, 1, 0]

    _install_fake_sentence_transformer(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import HFSentenceEncoder

    encoder = HFSentenceEncoder(
        model_id="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ).fit(["Clause 1.4.72 Scope", "Clause 2.6.3.2.2 Protection"])

    # fit() must not trigger any HF loading.
    assert loads == {"tokenizer": 0, "model": 0, "forward": 0}

    # Clause-hierarchy channel remains intact.
    assert encoder.clause_level_vocabs[0] == {"1": 0, "2": 1}
    assert encoder.clause_level_vocabs[1] == {"1.4": 0, "2.6": 1}
    assert encoder.clause_level_vocabs[2] == {"1.4.72": 0, "2.6.3.2.2": 1}
    # 2 text dims + 6 clause one-hots.
    assert encoder.dim == 8
    assert encoder.vocab_size == 8
    assert encoder.hidden_size == 2
    assert loads == {"tokenizer": 0, "model": 1, "forward": 0}

    vector = encoder.encode("sentence prompt clause 1.4.72")

    # Mean over unmasked tokens = (3.0, 4.0); L2-normalised = (0.6, 0.8).
    # Clause one-hots for levels 1, 1.4, 1.4.72 appended AFTER the sentence vector.
    assert vector == pytest.approx(
        [0.6, 0.8, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )
    assert loads == {"tokenizer": 1, "model": 1, "forward": 1}


def test_hf_sentence_encoder_encode_many_tensor_shape_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``encode_many_tensor`` must be float32 CPU with [N, dim] shape."""
    torch = pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    hidden_values = [
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    ]
    attention_mask_row = [1, 1, 1, 1]

    _install_fake_sentence_transformer(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import HFSentenceEncoder

    encoder = HFSentenceEncoder(
        model_id="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ).fit(["plain text one", "plain text two"])

    # No clause IDs means clause_feature_dim == 0.
    assert encoder.clause_feature_dim == 0
    assert encoder.dim == 3

    batched = encoder.encode_many_tensor(["a", "b", "c"], batch_size=2)
    assert isinstance(batched, torch.Tensor)
    assert batched.dtype == torch.float32
    assert tuple(batched.shape) == (3, 3)
    assert batched.device.type == "cpu"


def test_hf_sentence_encoder_normalize_true_gives_unit_norm_sentence_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``normalize=True`` the first hidden_size columns have L2 norm == 1."""
    pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    hidden_values = [
        [
            [5.0, 12.0],  # 13 -> normalises to (5/13, 12/13)
            [5.0, 12.0],
            [5.0, 12.0],
            [0.0, 0.0],
        ]
    ]
    attention_mask_row = [1, 1, 1, 0]

    _install_fake_sentence_transformer(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import HFSentenceEncoder

    # Fit with a clause id so clause_feature_dim > 0 — we need to prove the
    # sentence portion is normalised independently of the clause-feature tail.
    encoder = HFSentenceEncoder(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize=True,
    ).fit(["Clause 1.4.72 Protection"])

    assert encoder.hidden_size == 2
    assert encoder.clause_feature_dim == 3  # one token at each of 3 levels

    batched = encoder.encode_many_tensor(["Clause 1.4.72 again"], batch_size=1)
    row = batched[0].tolist()

    sentence_part = row[: encoder.hidden_size]
    clause_part = row[encoder.hidden_size :]

    # Sentence part is the L2-normalised pooled vector.
    norm = (sentence_part[0] ** 2 + sentence_part[1] ** 2) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)
    assert sentence_part == pytest.approx([5.0 / 13.0, 12.0 / 13.0], abs=1e-5)

    # Clause portion must match _encode_clause_features output and sit AFTER
    # the sentence vector (columns [hidden_size:] of the row).
    expected_clause = encoder._encode_clause_features("Clause 1.4.72 again")
    assert clause_part == pytest.approx(expected_clause)


def test_hf_sentence_encoder_normalize_false_keeps_raw_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``normalize=False`` the sentence vector is the unnormalised mean."""
    pytest.importorskip("torch")

    loads = {"tokenizer": 0, "model": 0, "forward": 0}
    hidden_values = [
        [
            [3.0, 4.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [0.0, 0.0],
        ]
    ]
    attention_mask_row = [1, 1, 1, 0]

    _install_fake_sentence_transformer(
        monkeypatch,
        loads=loads,
        hidden_values=hidden_values,
        attention_mask_row=attention_mask_row,
    )

    from tools._window_router.encoder import HFSentenceEncoder

    encoder = HFSentenceEncoder(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize=False,
    ).fit(["plain text"])

    vector = encoder.encode("plain text")
    # Mean over 3 unmasked tokens of (3, 4) is (3, 4); no normalisation applied.
    assert vector == pytest.approx([3.0, 4.0])


def test_hf_sentence_encoder_version_prefix() -> None:
    """encoder_version must use the sentence-transformer prefix for cache invalidation."""
    from tools._window_router.cache import encoder_version

    version = encoder_version(
        encoder_type="sentence-transformer",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert version == "sentence-transformer-v1::sentence-transformers/all-MiniLM-L6-v2"
    assert version.startswith("sentence-transformer-v1::")
    # Must be distinct from the gemma-embed version for the same model_id.
    assert (
        version
        != encoder_version(
            encoder_type="gemma-embed",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
        )
    )
