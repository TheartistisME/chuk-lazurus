#!/usr/bin/env python3
"""Generic window-router training CLI.

Three subcommands operate against any ``TorchKnowledgeStore``-compatible
store and any benchmark fixture shaped like the AUS3000 epic1 fixture:

    build-dataset   — JSONL of (text, window_id) training pairs
    train           — TorchMLPClassifier via TorchClassificationTrainer
    eval            — top-1 / top-3 / MRR plus TF-IDF baseline report

No MLX imports. ``transformers`` is lazy-imported from the encoder module
so hosts without it can still run ``--encoder bow``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_torch_mlp_class() -> Any:
    """Load :class:`TorchMLPClassifier` without triggering the MLX-heavy
    ``classifiers`` package ``__init__`` (which imports ``mlx.core``).
    """
    import importlib.util  # noqa: PLC0415

    module_path = (
        SRC_ROOT
        / "chuk_lazarus"
        / "models_v2"
        / "models"
        / "classifiers"
        / "torch_mlp.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_window_router_torch_mlp", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TorchMLPClassifier from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TorchMLPClassifier


def _load_store(store_path: Path) -> Any:
    from chuk_lazarus.inference.context.knowledge.torch_store import (  # noqa: PLC0415
        TorchKnowledgeStore,
    )

    return TorchKnowledgeStore.load(store_path)


def _maybe_load_tokenizer(model_id: str | None) -> Any | None:
    """Lazy-load ``transformers.AutoTokenizer`` only when a model id is given.

    When ``model_id`` is falsy, returns ``None`` so excerpt sampling /
    the TF-IDF baseline are skipped cleanly. A missing ``transformers``
    dependency raises ``SystemExit`` with a message pointing to the
    ``--tokenizer-model-id`` flag as the opt-in surface.
    """
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for --tokenizer-model-id; "
            "install transformers or omit the flag."
        ) from exc
    return AutoTokenizer.from_pretrained(model_id)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_sidecar(path: Path, payload: dict[str, Any]) -> Path:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def _resolve_runtime_device(
    requested_device: str, *, checkpoint_device: str | None = None
) -> str:
    from tools._window_router.cache import resolve_device  # noqa: PLC0415

    return resolve_device(requested_device, preferred_device=checkpoint_device)


def _load_benchmark_cases(benchmark_fixture: str | Path | None) -> list[dict[str, Any]]:
    if benchmark_fixture is None:
        return []
    payload = json.loads(Path(benchmark_fixture).read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def _build_encoder_for_training(args: argparse.Namespace, texts: list[str]) -> Any:
    from tools._window_router.encoder import (  # noqa: PLC0415
        BowCharacterEncoder,
        GemmaEmbedEncoder,
        HFSentenceEncoder,
    )

    if args.encoder == "bow":
        return BowCharacterEncoder().fit(texts)
    if args.encoder == "gemma-embed":
        if not args.model_id:
            raise SystemExit("--model-id is required with --encoder gemma-embed")
        resolved_device = _resolve_runtime_device(str(args.device))
        return GemmaEmbedEncoder(model_id=args.model_id, device=resolved_device).fit(texts)
    if args.encoder == "sentence-transformer":
        if not args.model_id:
            raise SystemExit("--model-id is required with --encoder sentence-transformer")
        resolved_device = _resolve_runtime_device(str(args.device))
        return HFSentenceEncoder(model_id=args.model_id, device=resolved_device).fit(texts)
    raise SystemExit(f"unknown encoder {args.encoder!r}")


def _restore_clause_level_vocabs(payload: dict[str, Any]) -> tuple[dict[str, int], ...]:
    raw_levels = payload.get("clause_level_vocabs")
    if not isinstance(raw_levels, list):
        return ({}, {}, {})

    restored: list[dict[str, int]] = []
    for level_idx in range(3):
        raw_level = raw_levels[level_idx] if level_idx < len(raw_levels) else []
        vocab: dict[str, int] = {}
        if isinstance(raw_level, list):
            for token in raw_level:
                token_key = str(token)
                if token_key not in vocab:
                    vocab[token_key] = len(vocab)
        restored.append(vocab)
    return tuple(restored)


def _ordered_vocab_tokens(vocab: dict[str, int]) -> list[str]:
    return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]


def _extract_texts_and_labels(
    records: Iterable[dict[str, Any]]
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for record in records:
        text = record.get("text")
        window_id = record.get("window_id")
        if text is None or window_id is None:
            continue
        texts.append(str(text))
        labels.append(int(window_id))
    return texts, labels


def _build_feature_cache_for_dataset(
    *,
    dataset_path: Path,
    samples: list[dict[str, Any]],
    dataset_hash: str,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]] | None:
    from tools._window_router.cache import (  # noqa: PLC0415
        default_feature_cache_path,
        encoder_version,
        feature_cache_key,
        write_feature_cache,
    )
    from tools._window_router.encoder import (  # noqa: PLC0415
        GemmaEmbedEncoder,
        HFSentenceEncoder,
    )

    feature_encoder = str(args.feature_encoder or "").strip()
    if not feature_encoder:
        return None
    if feature_encoder not in ("gemma-embed", "sentence-transformer"):
        raise SystemExit(f"unsupported feature encoder {feature_encoder!r}")
    if not args.model_id:
        raise SystemExit(
            f"--model-id is required with --feature-encoder {feature_encoder}"
        )

    texts, labels = _extract_texts_and_labels(samples)
    if not texts:
        raise SystemExit("dataset has no usable (text, window_id) pairs for feature caching")

    resolved_device = _resolve_runtime_device(str(args.device))
    resolved_encoder_version = encoder_version(
        encoder_type=feature_encoder,
        model_id=str(args.model_id),
    )
    cache_path = (
        Path(args.feature_cache_out)
        if args.feature_cache_out is not None
        else default_feature_cache_path(
            dataset_path,
            encoder_type=feature_encoder,
            encoder_version=resolved_encoder_version,
            dataset_hash=dataset_hash,
        )
    )
    if feature_encoder == "gemma-embed":
        encoder = GemmaEmbedEncoder(
            model_id=str(args.model_id), device=resolved_device
        ).fit(texts)
    else:
        encoder = HFSentenceEncoder(
            model_id=str(args.model_id), device=resolved_device
        ).fit(texts)
    features = encoder.encode_many_tensor(texts, batch_size=int(args.feature_batch_size))
    cache_meta = {
        "encoder_type": feature_encoder,
        "encoder_version": resolved_encoder_version,
        "dataset_hash": dataset_hash,
        "cache_key": feature_cache_key(
            encoder_type=feature_encoder,
            encoder_version=resolved_encoder_version,
            dataset_hash=dataset_hash,
        ),
        "model_id": str(args.model_id),
        "device": resolved_device,
        "sample_count": len(texts),
        "num_labels": max(labels) + 1,
        "input_size": int(encoder.dim),
        "clause_level_vocabs": [
            _ordered_vocab_tokens(vocab)
            for vocab in getattr(encoder, "clause_level_vocabs", ())
        ],
    }
    write_feature_cache(cache_path, features=features, labels=labels, meta=cache_meta)
    return cache_path, cache_meta


def _resolve_training_feature_cache(
    *,
    dataset_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    from tools._window_router.cache import (  # noqa: PLC0415
        default_feature_cache_path,
        encoder_version,
        load_feature_cache,
        read_dataset_meta,
    )

    encoder_type = str(args.encoder)
    if encoder_type not in ("gemma-embed", "sentence-transformer"):
        raise SystemExit(
            f"_resolve_training_feature_cache does not support encoder {encoder_type!r}"
        )
    if not args.model_id:
        raise SystemExit(f"--model-id is required with --encoder {encoder_type}")

    dataset_meta = read_dataset_meta(dataset_path)
    dataset_hash = str(dataset_meta.get("dataset_hash") or "")
    if not dataset_hash:
        raise SystemExit(
            f"dataset sidecar for {dataset_path} is missing dataset_hash; rebuild with build-dataset"
        )

    resolved_encoder_version = encoder_version(
        encoder_type=encoder_type,
        model_id=str(args.model_id),
    )
    cache_path = (
        Path(args.feature_cache)
        if args.feature_cache is not None
        else default_feature_cache_path(
            dataset_path,
            encoder_type=encoder_type,
            encoder_version=resolved_encoder_version,
            dataset_hash=dataset_hash,
        )
    )
    if not cache_path.exists():
        raise SystemExit(
            f"feature cache {cache_path} not found; build it via build-dataset "
            f"with --feature-encoder {encoder_type}"
        )
    cache_payload = load_feature_cache(cache_path)
    cache_meta = cache_payload["meta"]
    if str(cache_meta.get("dataset_hash") or "") != dataset_hash:
        raise SystemExit(
            f"feature cache {cache_path} does not match dataset hash {dataset_hash}"
        )
    if str(cache_meta.get("encoder_type") or "") != encoder_type:
        raise SystemExit(
            f"feature cache {cache_path} is not a {encoder_type} cache "
            f"(found {cache_meta.get('encoder_type')!r})"
        )
    if str(cache_meta.get("encoder_version") or "") != resolved_encoder_version:
        raise SystemExit(
            f"feature cache {cache_path} encoder_version mismatch: "
            f"{cache_meta.get('encoder_version')!r} != {resolved_encoder_version!r}"
        )
    sample_count = int(cache_meta.get("sample_count") or 0)
    features = cache_payload["features"]
    labels = cache_payload["labels"]
    if int(getattr(features, "shape", [0])[0]) != sample_count:
        raise SystemExit(
            f"feature cache {cache_path} sample_count does not match features shape"
        )
    if int(getattr(labels, "shape", [0])[0]) != sample_count:
        raise SystemExit(
            f"feature cache {cache_path} sample_count does not match labels shape"
        )
    return cache_path, cache_payload


def _build_encoder_for_eval(
    args: argparse.Namespace, meta: dict[str, Any], payload: dict[str, Any]
) -> Any:
    from tools._window_router.encoder import (  # noqa: PLC0415
        BowCharacterEncoder,
        GemmaEmbedEncoder,
        HFSentenceEncoder,
    )

    kind = str(meta.get("encoder") or "")
    clause_level_vocabs = _restore_clause_level_vocabs(payload)
    if kind == "bow":
        vocab_list = payload.get("encoder_vocab") or []
        ngram = int(payload.get("encoder_n") or 3)
        encoder = BowCharacterEncoder(n=ngram)
        encoder.set_vocab({str(g): i for i, g in enumerate(vocab_list)})
        encoder.set_clause_level_vocabs(clause_level_vocabs)
        return encoder
    if kind == "gemma-embed":
        model_id = args.model_id or meta.get("model_id")
        if not model_id:
            raise SystemExit("gemma-embed ckpt requires --model-id or meta['model_id']")
        resolved_device = _resolve_runtime_device(
            str(args.device),
            checkpoint_device=str(meta.get("device") or "").strip() or None,
        )
        return GemmaEmbedEncoder(
            model_id=model_id,
            device=resolved_device,
            clause_level_vocabs=clause_level_vocabs,
        )
    if kind == "sentence-transformer":
        model_id = args.model_id or meta.get("model_id")
        if not model_id:
            raise SystemExit(
                "sentence-transformer ckpt requires --model-id or meta['model_id']"
            )
        resolved_device = _resolve_runtime_device(
            str(args.device),
            checkpoint_device=str(meta.get("device") or "").strip() or None,
        )
        return HFSentenceEncoder(
            model_id=model_id,
            device=resolved_device,
            clause_level_vocabs=clause_level_vocabs,
        )
    raise SystemExit(f"unknown encoder in ckpt: {kind!r}")


# ── build-dataset ────────────────────────────────────────────────────


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from tools._window_router.augmentation import (  # noqa: PLC0415
        DEFAULT_TITLE_AUGMENTATION_TEMPLATES,
    )
    from tools._window_router.cache import compute_file_sha256  # noqa: PLC0415
    from tools._window_router.dataset import (  # noqa: PLC0415
        DEFAULT_MULTI_CLAUSE_TEMPLATES,
        DEFAULT_PARAPHRASE_TEMPLATES,
        build_router_dataset,
    )

    store = _load_store(args.store_path)
    templates: tuple[str, ...] = DEFAULT_PARAPHRASE_TEMPLATES
    if args.paraphrases is not None:
        count = max(0, int(args.paraphrases))
        templates = DEFAULT_PARAPHRASE_TEMPLATES[:count]

    tokenizer = _maybe_load_tokenizer(args.tokenizer_model_id)
    benchmark_cases = _load_benchmark_cases(args.benchmark_fixture)
    samples = build_router_dataset(
        store,
        paraphrase_templates=templates,
        augment=bool(args.augment),
        tokenizer=tokenizer,
        benchmark_cases=benchmark_cases,
    )
    out_path = Path(args.out_jsonl)
    _write_jsonl(out_path, samples)
    dataset_hash = compute_file_sha256(out_path)

    num_windows = int(getattr(store, "num_windows", 0) or len(store.window_metadata))
    sidecar_payload = {
        "num_windows": num_windows,
        "num_samples": len(samples),
        "num_templates": len(templates),
        "augment": bool(args.augment),
        "num_augmentation_templates": (
            len(DEFAULT_TITLE_AUGMENTATION_TEMPLATES) if args.augment else 0
        ),
        "num_multi_clause_templates": len(DEFAULT_MULTI_CLAUSE_TEMPLATES),
        "num_multi_clause_cases": len(
            [
                case
                for case in benchmark_cases
                if len(case.get("primary_clause_ids") or ()) >= 2
            ]
        ),
        "store_path": str(args.store_path),
        "dataset_hash": dataset_hash,
    }
    feature_cache_info = _build_feature_cache_for_dataset(
        dataset_path=out_path,
        samples=samples,
        dataset_hash=dataset_hash,
        args=args,
    )
    if feature_cache_info is not None:
        feature_cache_path, feature_cache_meta = feature_cache_info
        sidecar_payload["feature_cache"] = {
            "path": str(feature_cache_path),
            "encoder_type": str(feature_cache_meta.get("encoder_type") or ""),
            "encoder_version": str(feature_cache_meta.get("encoder_version") or ""),
            "cache_key": str(feature_cache_meta.get("cache_key") or ""),
            "input_size": int(feature_cache_meta.get("input_size") or 0),
        }

    sidecar = _write_sidecar(out_path, sidecar_payload)
    print(f"wrote {len(samples)} samples to {out_path}")
    print(f"wrote sidecar meta to {sidecar}")
    if feature_cache_info is not None:
        print(f"wrote feature cache to {feature_cache_info[0]}")
    return 0


# ── train ────────────────────────────────────────────────────────────


def cmd_train(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    from chuk_lazarus.training.torch import (  # noqa: PLC0415
        TorchClassificationTrainer,
        TorchTrainerConfig,
    )
    from tools._window_router.cache import CachedFeatureDataset  # noqa: PLC0415

    TorchMLPClassifier = _load_torch_mlp_class()
    resolved_device = _resolve_runtime_device(str(args.device))

    samples = _read_jsonl(Path(args.dataset))
    if not samples:
        raise SystemExit(f"dataset {args.dataset} is empty")
    encoder = None
    dataset: Any

    if args.encoder in ("gemma-embed", "sentence-transformer"):
        _cache_path, cache_payload = _resolve_training_feature_cache(
            dataset_path=Path(args.dataset),
            args=args,
        )
        cache_meta = cache_payload["meta"]
        input_size = int(cache_meta.get("input_size") or 0)
        if input_size <= 0:
            raise SystemExit(
                f"feature cache {args.feature_cache or _cache_path} has invalid input_size {input_size}"
            )
        dataset = CachedFeatureDataset(
            cache_payload["features"],
            cache_payload["labels"],
        )
        num_labels = int(cache_meta.get("num_labels") or 0)
        if num_labels <= 0:
            raise SystemExit(
                f"feature cache {args.feature_cache or _cache_path} has invalid num_labels {num_labels}"
            )
        clause_level_vocabs = [
            list(level)
            for level in (cache_meta.get("clause_level_vocabs") or [[], [], []])
        ]
        encoder_version_value = str(cache_meta.get("encoder_version") or "")
    else:
        texts, labels = _extract_texts_and_labels(samples)
        if not texts:
            raise SystemExit("dataset has no usable (text, window_id) pairs")
        num_labels = max(labels) + 1
        encoder = _build_encoder_for_training(args, texts)
        input_size = int(encoder.dim)
        if input_size <= 0:
            raise SystemExit(f"encoder produced zero-length features (dim={input_size})")
        dataset = [{"label": label, "text": text} for text, label in zip(texts, labels)]
        clause_level_vocabs = [
            _ordered_vocab_tokens(vocab)
            for vocab in getattr(encoder, "clause_level_vocabs", ())
        ]
        from tools._window_router.cache import encoder_version as _encoder_version  # noqa: PLC0415

        encoder_version_value = _encoder_version(
            encoder_type=str(args.encoder),
            model_id=str(args.model_id or ""),
        )

    model = TorchMLPClassifier(
        input_size=input_size,
        hidden_size=int(args.hidden),
        num_labels=num_labels,
    )

    ckpt_path = Path(args.out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = TorchTrainerConfig(
        learning_rate=float(args.lr),
        batch_size=int(args.batch_size),
        num_epochs=int(args.epochs),
        log_interval=100,
        checkpoint_interval=10_000_000,
        checkpoint_dir=str(ckpt_path.parent),
        device=resolved_device,
    )
    trainer = TorchClassificationTrainer(model, encoder=encoder, config=cfg)
    trainer.train(dataset)

    model_cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    payload: dict[str, Any] = {
        "state_dict": model_cpu_state,
        "meta": {
            "encoder": args.encoder,
            "num_labels": int(num_labels),
            "input_size": int(input_size),
            "hidden_size": int(args.hidden),
            "model_id": args.model_id,
            "device": resolved_device,
            "encoder_version": encoder_version_value,
        },
        "clause_level_vocabs": clause_level_vocabs,
    }
    if args.encoder == "bow":
        payload["encoder_vocab"] = _ordered_vocab_tokens(encoder.vocab)
        payload["encoder_n"] = int(getattr(encoder, "n", 3))
    torch.save(payload, ckpt_path)
    print(f"saved checkpoint {ckpt_path} (num_labels={num_labels}, dim={input_size})")
    return 0


# ── eval ─────────────────────────────────────────────────────────────


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("ckpt payload is not a dict")
    for key in ("state_dict", "meta"):
        if key not in payload:
            raise ValueError(f"ckpt payload missing required key {key!r}")
    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("ckpt payload 'meta' is not a dict")
    for key in ("encoder", "num_labels", "input_size"):
        if key not in meta:
            raise ValueError(f"ckpt meta missing required key {key!r}")
    return payload


def cmd_eval(args: argparse.Namespace) -> int:
    import torch  # noqa: PLC0415

    from tools._window_router.eval import evaluate, write_report  # noqa: PLC0415

    TorchMLPClassifier = _load_torch_mlp_class()

    ckpt_path = Path(args.ckpt)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    payload = _validate_payload(payload)
    meta = payload["meta"]

    num_labels = int(meta["num_labels"])
    input_size = int(meta["input_size"])
    hidden_size = int(meta.get("hidden_size", 256))

    encoder = _build_encoder_for_eval(args, meta, payload)
    if int(encoder.dim) != input_size:
        raise ValueError(
            f"encoder dim {encoder.dim} does not match ckpt input_size {input_size}"
        )

    model = TorchMLPClassifier(
        input_size=input_size, hidden_size=hidden_size, num_labels=num_labels
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    store = _load_store(args.store_path)
    fixture = json.loads(Path(args.benchmark_fixture).read_text(encoding="utf-8"))
    cases = [case for case in fixture.get("cases", []) if case.get("primary_clause_ids")]
    if not cases:
        raise SystemExit(f"benchmark fixture {args.benchmark_fixture} has no usable cases")

    tokenizer = _maybe_load_tokenizer(args.tokenizer_model_id)
    report = evaluate(
        model=model,
        encoder=encoder,
        store=store,
        cases=cases,
        tokenizer=tokenizer,
    )

    out_dir = (
        Path(args.out_report)
        if args.out_report is not None
        else ckpt_path.with_name(f"{ckpt_path.stem}_eval")
    )
    write_report(out_dir, report)
    print(
        "model: "
        f"top1={report['model']['top_1_accuracy']:.3f} "
        f"top3={report['model']['top_3_accuracy']:.3f} "
        f"mrr={report['model']['mrr']:.3f} "
        f"n={report['model']['n']}"
    )
    if tokenizer is not None:
        print(
            "tfidf: "
            f"top1={report['tfidf_baseline']['top_1_accuracy']:.3f} "
            f"top3={report['tfidf_baseline']['top_3_accuracy']:.3f} "
            f"mrr={report['tfidf_baseline']['mrr']:.3f} "
            f"n={report['tfidf_baseline']['n']}"
        )
    print(f"wrote report to {out_dir}")
    return 0


# ── argparse plumbing ────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic window-router training and evaluation CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bd = sub.add_parser("build-dataset", help="Build a (text, window_id) JSONL training set.")
    bd.add_argument("--store-path", required=True, type=Path)
    bd.add_argument("--out-jsonl", required=True, type=Path)
    bd.add_argument("--paraphrases", type=int, default=None, help="Cap on default template count")
    bd.add_argument(
        "--benchmark-fixture",
        default=None,
        type=Path,
        help="Optional benchmark fixture used to inject explicit multi-clause prompts.",
    )
    augment_group = bd.add_mutually_exclusive_group()
    augment_group.add_argument(
        "--augment",
        dest="augment",
        action="store_true",
        help="Enable deterministic clause-title augmentation (default).",
    )
    augment_group.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable deterministic clause-title augmentation.",
    )
    bd.add_argument(
        "--tokenizer-model-id",
        default=None,
        help=(
            "HuggingFace tokenizer id for excerpt samples. "
            "When omitted, excerpt samples are skipped."
        ),
    )
    bd.add_argument(
        "--feature-encoder",
        default=None,
        choices=("gemma-embed", "sentence-transformer"),
        help="Optional encoder used to precompute feature vectors during dataset build.",
    )
    bd.add_argument(
        "--feature-cache-out",
        default=None,
        type=Path,
        help="Optional explicit output path for precomputed feature vectors.",
    )
    bd.add_argument(
        "--feature-batch-size",
        type=int,
        default=32,
        help="Batch size for precomputing encoder features.",
    )
    bd.add_argument(
        "--model-id",
        default=None,
        help="Required with --feature-encoder gemma-embed/sentence-transformer.",
    )
    bd.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Defaults to auto-detecting cuda when available.",
    )
    bd.set_defaults(augment=True)
    bd.set_defaults(func=cmd_build_dataset)

    tr = sub.add_parser("train", help="Train a TorchMLPClassifier window router.")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument(
        "--encoder",
        required=True,
        choices=("bow", "gemma-embed", "sentence-transformer"),
    )
    tr.add_argument("--out-ckpt", required=True, type=Path)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    tr.add_argument(
        "--model-id",
        default=None,
        help="Required with --encoder gemma-embed/sentence-transformer",
    )
    tr.add_argument(
        "--feature-cache",
        default=None,
        type=Path,
        help=(
            "Optional explicit feature-cache path for "
            "gemma-embed/sentence-transformer training."
        ),
    )
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Score a trained router + TF-IDF baseline.")
    ev.add_argument("--ckpt", required=True, type=Path)
    ev.add_argument("--benchmark-fixture", required=True, type=Path)
    ev.add_argument("--store-path", required=True, type=Path)
    ev.add_argument(
        "--out-report",
        default=None,
        type=Path,
        help=(
            "Output directory for report.json/report.md. "
            "Defaults to <ckpt_stem>_eval beside the checkpoint."
        ),
    )
    ev.add_argument("--encoder-cache", default=None, help="Reserved for future encoder caching")
    ev.add_argument("--model-id", default=None)
    ev.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Defaults to auto-detecting cuda when available.",
    )
    ev.add_argument(
        "--tokenizer-model-id",
        default=None,
        help=(
            "HuggingFace tokenizer id for the TF-IDF baseline. "
            "When omitted, the baseline is skipped."
        ),
    )
    ev.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
