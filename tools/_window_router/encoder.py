"""Encoders for the window-router classifier.

Three encoders share a common surface so the training loop is indifferent
to which one is active:

    .encode(text: str) -> list[float]
    .vocab_size: int
    .dim: int

``BowCharacterEncoder`` is pure Python and has no external dependencies;
``GemmaEmbedEncoder`` and ``HFSentenceEncoder`` lazy-import ``transformers``
inside ``__init__`` and defer ``from_pretrained`` calls until the HF path
is actually used, so hosts without the library (including CI) never hit
the heavy dependency until someone explicitly asks for ``--encoder
gemma-embed`` or ``--encoder sentence-transformer``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_CLAUSE_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)(?![A-Za-z0-9])"
)
_NUM_CLAUSE_LEVELS = 3


def _extract_clause_ids(text: str) -> list[str]:
    seen: set[str] = set()
    clause_ids: list[str] = []
    for match in _CLAUSE_ID_PATTERN.finditer(str(text)):
        clause_id = str(match.group(1)).strip()
        if not clause_id or clause_id in seen:
            continue
        seen.add(clause_id)
        clause_ids.append(clause_id)
    return clause_ids


def _clause_hierarchy(clause_id: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in str(clause_id).split(".") if part.strip()]
    if not parts:
        return ("", "", "")
    level1 = parts[0]
    level2 = ".".join(parts[:2]) if len(parts) >= 2 else level1
    level3 = ".".join(parts)
    return (level1, level2, level3)


def _coerce_clause_level_vocabs(
    raw_vocabs: Iterable[dict[str, int] | Iterable[str]] | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    if raw_vocabs is None:
        raw_items: list[dict[str, int] | Iterable[str]] = []
    else:
        raw_items = list(raw_vocabs)

    levels: list[dict[str, int]] = []
    for level_idx in range(_NUM_CLAUSE_LEVELS):
        raw_level = raw_items[level_idx] if level_idx < len(raw_items) else ()
        vocab: dict[str, int] = {}
        if isinstance(raw_level, dict):
            ordered_items = sorted(raw_level.items(), key=lambda item: int(item[1]))
            for token, _raw_idx in ordered_items:
                token_key = str(token)
                if token_key not in vocab:
                    vocab[token_key] = len(vocab)
        else:
            for token in raw_level:
                token_key = str(token)
                if token_key not in vocab:
                    vocab[token_key] = len(vocab)
        levels.append(vocab)
    return tuple(levels)  # type: ignore[return-value]


def _build_clause_level_vocabs(
    texts: Iterable[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    levels: list[dict[str, int]] = [{} for _ in range(_NUM_CLAUSE_LEVELS)]
    for text in texts:
        for clause_id in _extract_clause_ids(text):
            for level_idx, value in enumerate(_clause_hierarchy(clause_id)):
                if not value or value in levels[level_idx]:
                    continue
                levels[level_idx][value] = len(levels[level_idx])
    return tuple(levels)  # type: ignore[return-value]


class _ClauseHierarchyFeatureMixin:
    def _init_clause_level_vocabs(
        self,
        clause_level_vocabs: Iterable[dict[str, int] | Iterable[str]] | None = None,
    ) -> None:
        self._clause_level_vocabs = _coerce_clause_level_vocabs(clause_level_vocabs)

    def _fit_clause_level_vocabs(self, texts: Iterable[str]) -> None:
        self._clause_level_vocabs = _build_clause_level_vocabs(texts)

    def set_clause_level_vocabs(
        self,
        clause_level_vocabs: Iterable[dict[str, int] | Iterable[str]] | None,
    ) -> None:
        self._clause_level_vocabs = _coerce_clause_level_vocabs(clause_level_vocabs)

    @property
    def clause_level_vocabs(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        return tuple(dict(vocab) for vocab in self._clause_level_vocabs)  # type: ignore[return-value]

    @property
    def clause_feature_dim(self) -> int:
        return sum(len(vocab) for vocab in self._clause_level_vocabs)

    def _encode_clause_features(self, text: str) -> list[float]:
        level_vectors = [[0.0] * len(vocab) for vocab in self._clause_level_vocabs]
        for clause_id in _extract_clause_ids(text):
            for level_idx, value in enumerate(_clause_hierarchy(clause_id)):
                if not value:
                    continue
                vocab = self._clause_level_vocabs[level_idx]
                column = vocab.get(value)
                if column is not None:
                    level_vectors[level_idx][column] = 1.0
        features: list[float] = []
        for level_vector in level_vectors:
            features.extend(level_vector)
        return features


def _resolve_input_embeddings(model: Any) -> Any:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None:
            return embeddings

    nested = getattr(getattr(model, "model", None), "embed_tokens", None)
    if nested is not None:
        return nested

    embeddings = getattr(model, "embed_tokens", None)
    if embeddings is not None:
        return embeddings

    raise ValueError("Could not resolve an input embedding layer for GemmaEmbedEncoder")


def _infer_embedding_dim(model: Any, embeddings: Any) -> int:
    dim = getattr(embeddings, "embedding_dim", None)
    if isinstance(dim, int) and dim > 0:
        return int(dim)

    weight = getattr(embeddings, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 2:
        try:
            dim = int(shape[-1])
        except (TypeError, ValueError):
            dim = None
        if isinstance(dim, int) and dim > 0:
            return dim

    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    for source in (config, text_config):
        if source is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim"):
            value = getattr(source, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    raise ValueError("Could not infer embedding dimension for GemmaEmbedEncoder")


def _resolve_embedding_scale(model: Any) -> float:
    for source in (
        getattr(model, "model", None),
        model,
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if source is None:
            continue
        for attr in ("embedding_scale", "embed_scale"):
            value = getattr(source, attr, None)
            if value is None:
                continue
            item = getattr(value, "item", None)
            if callable(item):
                value = item()
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 1.0


class BowCharacterEncoder(_ClauseHierarchyFeatureMixin):
    """Deterministic character-n-gram bag-of-words encoder.

    Vocab is populated by :meth:`fit` and never grows during
    :meth:`encode`. Dimensionality equals vocab size. Results are stable
    given stable training data (``fit`` preserves insertion order of
    distinct n-grams).

    Args:
        n: n-gram width (default 3).
        lowercase: folded to lowercase before n-gram extraction.
    """

    def __init__(self, n: int = 3, lowercase: bool = True) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        self.lowercase = bool(lowercase)
        self._vocab: dict[str, int] = {}
        self._fitted: bool = False
        self._init_clause_level_vocabs()

    def _normalise(self, text: str) -> str:
        return str(text).lower() if self.lowercase else str(text)

    def _ngrams(self, text: str) -> list[str]:
        text = self._normalise(text)
        if not text:
            return []
        if len(text) < self.n:
            return [text]
        return [text[i : i + self.n] for i in range(len(text) - self.n + 1)]

    def fit(self, texts: Iterable[str]) -> BowCharacterEncoder:
        """Populate the vocab from ``texts``. Returns ``self`` for chaining."""
        vocab: dict[str, int] = {}
        clause_levels: list[dict[str, int]] = [{} for _ in range(_NUM_CLAUSE_LEVELS)]
        for text in texts:
            for ngram in self._ngrams(text):
                if ngram not in vocab:
                    vocab[ngram] = len(vocab)
            for clause_id in _extract_clause_ids(text):
                for level_idx, value in enumerate(_clause_hierarchy(clause_id)):
                    if not value or value in clause_levels[level_idx]:
                        continue
                    clause_levels[level_idx][value] = len(clause_levels[level_idx])
        self._vocab = vocab
        self._clause_level_vocabs = tuple(clause_levels)  # type: ignore[assignment]
        self._fitted = True
        return self

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def set_vocab(self, vocab: dict[str, int]) -> None:
        """Restore a fitted vocab (used when loading a checkpoint)."""
        self._vocab = {str(g): int(i) for g, i in vocab.items()}
        self._fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + self.clause_feature_dim

    @property
    def dim(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError(
                "BowCharacterEncoder has not been fitted; call .fit(texts) first"
            )
        features = [0.0] * len(self._vocab)
        for ngram in self._ngrams(text):
            idx = self._vocab.get(ngram)
            if idx is not None:
                features[idx] += 1.0
        return features + self._encode_clause_features(text)


class GemmaEmbedEncoder(_ClauseHierarchyFeatureMixin):
    """Mean-pooled embedding-layer encoder backed by ``transformers.AutoModel``.

    ``transformers`` is imported lazily inside ``__init__`` and the HF
    objects are loaded on first use, so merely constructing this class on a
    host without a complete HF cache does not trigger downloads. When the
    import fails the error message points the caller at the ``--encoder bow``
    fallback.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        clause_level_vocabs: Iterable[dict[str, int] | Iterable[str]] | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self._init_clause_level_vocabs(clause_level_vocabs)

        try:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
            raise ImportError(
                "GemmaEmbedEncoder requires the `transformers` package "
                f"({exc!s}). Install it with `uv pip install transformers`, "
                "or re-run with `--encoder bow` which uses no external weights."
            ) from exc

        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GemmaEmbedEncoder requires `torch`. Install it or use `--encoder bow`."
            ) from exc

        self._torch = torch
        self._auto_tokenizer_cls = AutoTokenizer
        self._auto_model_cls = AutoModel
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._embeddings: Any | None = None
        self._embedding_scale = 1.0
        self._dim: int | None = None

    def fit(self, texts: Iterable[str]) -> GemmaEmbedEncoder:
        self._fit_clause_level_vocabs(texts)
        return self

    def _ensure_embeddings(self) -> None:
        if self._model is not None:
            return

        model = self._auto_model_cls.from_pretrained(self.model_id).to(self.device)
        model.eval()
        embeddings = _resolve_input_embeddings(model)

        self._embedding_scale = _resolve_embedding_scale(model)
        self._dim = _infer_embedding_dim(model, embeddings)
        self._embeddings = embeddings
        self._model = model

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        self._tokenizer = self._auto_tokenizer_cls.from_pretrained(self.model_id)

    @property
    def vocab_size(self) -> int:
        return self.dim

    @property
    def dim(self) -> int:
        self._ensure_embeddings()
        if self._dim is None:  # pragma: no cover - guarded by _ensure_embeddings
            raise RuntimeError("GemmaEmbedEncoder did not resolve an embedding dimension")
        return self._dim + self.clause_feature_dim

    def _pooled_batch_embeddings(self, texts: list[str]) -> Any:
        if not texts:
            return []

        self._ensure_embeddings()
        self._ensure_tokenizer()

        torch = self._torch
        encoded = self._tokenizer(
            [str(text) for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Gemma tokenizer output did not include input_ids")
        input_ids = input_ids.to(self.device)

        attention_mask: Any = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state  # [B, T, D]
            hidden = hidden.to(torch.float32)

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            else:
                pooled = hidden.mean(dim=1)
        return pooled.detach().cpu().to(torch.float32)

    def encode_many_tensor(self, texts: Iterable[str], batch_size: int = 32) -> Any:
        torch = self._torch
        batch = [str(text) for text in texts]
        if not batch:
            return torch.empty((0, self.dim), dtype=torch.float32)

        rows: list[Any] = []
        for start in range(0, len(batch), max(1, int(batch_size))):
            chunk = batch[start : start + max(1, int(batch_size))]
            text_embeddings = self._pooled_batch_embeddings(chunk)
            clause_rows = torch.tensor(
                [self._encode_clause_features(text) for text in chunk],
                dtype=torch.float32,
            )
            rows.append(torch.cat((text_embeddings.to(dtype=torch.float32), clause_rows), dim=1))
        return torch.cat(rows, dim=0)

    def encode_many(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        encoded = self.encode_many_tensor(texts, batch_size=batch_size)
        return [[float(v) for v in row] for row in encoded.tolist()]

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text], batch_size=1)[0]


def _infer_hidden_size(model: Any) -> int:
    """Resolve the transformer ``hidden_size`` for sentence-transformer models.

    Preference order:
        1. ``model.config.hidden_size``
        2. ``model.config.text_config.hidden_size`` (multi-modal configs)
        3. Fall back to the embedding-dim heuristics used for Gemma.

    The last fallback is kept as a defensive last resort; for all supported
    sentence-transformer checkpoints the transformer hidden size is the
    correct sentence-vector dimension.
    """
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    for source in (config, text_config):
        if source is None:
            continue
        value = getattr(source, "hidden_size", None)
        if isinstance(value, int) and value > 0:
            return int(value)

    # Defensive fallback: derive the dim from the input embedding layer.
    embeddings = _resolve_input_embeddings(model)
    return _infer_embedding_dim(model, embeddings)


class HFSentenceEncoder(_ClauseHierarchyFeatureMixin):
    """Mean-pooled ``last_hidden_state`` encoder for sentence-transformer models.

    Unlike :class:`GemmaEmbedEncoder` this loads via plain
    ``transformers.AutoModel`` and relies on ``model.config.hidden_size`` for
    the sentence-vector dimension, which is the correct surface for
    sentence-transformers/all-MiniLM-L6-v2 and similar encoder-only
    checkpoints (hidden_size=384, max_length=512).

    ``transformers`` and ``torch`` are lazy-imported inside ``__init__`` and
    HF ``from_pretrained`` calls are deferred until first use, matching the
    Gemma encoder's lazy-loading discipline.

    Args:
        model_id: HuggingFace model id or local path.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        clause_level_vocabs: Optional pre-built clause-hierarchy vocabs.
        normalize: When ``True`` (default) the pooled sentence vector is
            L2-normalised before the clause-feature one-hots are appended.
            This matches sentence-transformers behaviour and keeps
            cosine-similarity ranking well-scaled for Gate 0.2 probes.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        clause_level_vocabs: Iterable[dict[str, int] | Iterable[str]] | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.normalize = bool(normalize)
        self._init_clause_level_vocabs(clause_level_vocabs)

        try:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
            raise ImportError(
                "HFSentenceEncoder requires the `transformers` package "
                f"({exc!s}). Install it with `uv pip install transformers`, "
                "or re-run with `--encoder bow` which uses no external weights."
            ) from exc

        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HFSentenceEncoder requires `torch`. Install it or use `--encoder bow`."
            ) from exc

        self._torch = torch
        self._auto_tokenizer_cls = AutoTokenizer
        self._auto_model_cls = AutoModel
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._hidden_size: int | None = None

    def fit(self, texts: Iterable[str]) -> HFSentenceEncoder:
        self._fit_clause_level_vocabs(texts)
        return self

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model = self._auto_model_cls.from_pretrained(self.model_id).to(self.device)
        model.eval()
        self._hidden_size = int(_infer_hidden_size(model))
        if self._hidden_size <= 0:  # pragma: no cover - defensive
            raise RuntimeError(
                "HFSentenceEncoder could not resolve a positive hidden_size"
            )
        self._model = model

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        self._tokenizer = self._auto_tokenizer_cls.from_pretrained(self.model_id)

    @property
    def hidden_size(self) -> int:
        self._ensure_model()
        assert self._hidden_size is not None  # for type-checkers
        return self._hidden_size

    @property
    def dim(self) -> int:
        return self.hidden_size + self.clause_feature_dim

    @property
    def vocab_size(self) -> int:
        return self.dim

    def _pooled_batch_embeddings(self, texts: list[str]) -> Any:
        if not texts:
            return []

        self._ensure_model()
        self._ensure_tokenizer()

        torch = self._torch
        encoded = self._tokenizer(
            [str(text) for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Sentence-transformer tokenizer output did not include input_ids")
        input_ids = input_ids.to(self.device)

        attention_mask: Any = encoded.get("attention_mask")
        if attention_mask is None:
            raise ValueError(
                "Sentence-transformer tokenizer output did not include attention_mask"
            )
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            hidden = outputs.last_hidden_state.to(torch.float32)  # [B, T, H]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().to(torch.float32)

    def encode_many_tensor(self, texts: Iterable[str], batch_size: int = 32) -> Any:
        torch = self._torch
        batch = [str(text) for text in texts]
        if not batch:
            return torch.empty((0, self.dim), dtype=torch.float32)

        rows: list[Any] = []
        for start in range(0, len(batch), max(1, int(batch_size))):
            chunk = batch[start : start + max(1, int(batch_size))]
            text_embeddings = self._pooled_batch_embeddings(chunk)
            clause_rows = torch.tensor(
                [self._encode_clause_features(text) for text in chunk],
                dtype=torch.float32,
            )
            rows.append(torch.cat((text_embeddings.to(dtype=torch.float32), clause_rows), dim=1))
        return torch.cat(rows, dim=0)

    def encode_many(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        encoded = self.encode_many_tensor(texts, batch_size=batch_size)
        return [[float(v) for v in row] for row in encoded.tolist()]

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text], batch_size=1)[0]


__all__ = ["BowCharacterEncoder", "GemmaEmbedEncoder", "HFSentenceEncoder"]
