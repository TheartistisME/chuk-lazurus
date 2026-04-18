"""End-to-end pipeline smoke: build-dataset → train → eval.

Uses an in-``tmp_path`` synthetic 4-window knowledge store so the entire
pipeline runs fully offline (no network, no HF download, no MLX).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.train_window_router import main as cli_main


def _synthetic_store(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 12,
        "num_windows": 4,
        "num_tokens": 40,
        "entries_per_window": 8,
        "crystal_layer": 2,
        "window_size": 16,
        "arch_config": {
            "retrieval_layer": 1,
            "query_head": 0,
            "injection_layer": 2,
        },
        "has_residuals": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    window_tokens: dict[str, np.ndarray] = {
        "0": np.array([10, 20, 30, 40], dtype=np.uint32),
        "1": np.array([20, 50, 60, 70], dtype=np.uint32),
        "2": np.array([80, 90, 100, 110], dtype=np.uint32),
        "3": np.array([40, 120, 130, 140], dtype=np.uint32),
    }
    np.savez(str(root / "window_tokens.npz"), **window_tokens)
    np.savez(str(root / "window_token_lists.npz"), **window_tokens)

    metadata: dict[str, dict[str, Any]] = {
        "0": {"clause_id": "1.1", "clause_title": "Alpha Prime", "part_index": 1},
        "1": {"clause_id": "2.2", "clause_title": "Bravo Foxtrot", "part_index": 1},
        "2": {"clause_id": "3.3", "clause_title": "Charlie Delta", "part_index": 1},
        "3": {"clause_id": "4.4", "clause_title": "Echo November", "part_index": 1},
    }
    (root / "window_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return root


def _tiny_benchmark_fixture(path: Path) -> Path:
    payload = {
        "benchmark_version": "window_router_smoke_v1",
        "cases": [
            {
                "name": "alpha_case",
                "category": "lookup",
                "prompt": "Define Alpha Prime",
                "primary_clause_ids": ["1.1"],
            },
            {
                "name": "bravo_case",
                "category": "lookup",
                "prompt": "What is Bravo Foxtrot?",
                "primary_clause_ids": ["2.2"],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def test_pipeline_build_train_eval_smoke(tmp_path: Path) -> None:
    # Seed both torch and stdlib RNGs — trainer shuffles via ``random``.
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    store_path = _synthetic_store(tmp_path / "store")
    dataset_jsonl = tmp_path / "dataset" / "pairs.jsonl"
    ckpt_path = tmp_path / "ckpt" / "router.pt"
    report_dir = tmp_path / "report"
    fixture_path = _tiny_benchmark_fixture(tmp_path / "bench.json")

    # build-dataset ----------------------------------------------------
    exit_code = cli_main(
        [
            "build-dataset",
            "--store-path",
            str(store_path),
            "--out-jsonl",
            str(dataset_jsonl),
        ]
    )
    assert exit_code == 0
    assert dataset_jsonl.exists()
    meta_sidecar = dataset_jsonl.with_suffix(dataset_jsonl.suffix + ".meta.json")
    assert meta_sidecar.exists()
    sidecar_payload = json.loads(meta_sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["num_windows"] == 4
    assert sidecar_payload["num_samples"] > 0

    # train ------------------------------------------------------------
    exit_code = cli_main(
        [
            "train",
            "--dataset",
            str(dataset_jsonl),
            "--encoder",
            "bow",
            "--out-ckpt",
            str(ckpt_path),
            "--epochs",
            "2",
            "--hidden",
            "16",
            "--batch-size",
            "4",
            "--lr",
            "5e-2",
        ]
    )
    assert exit_code == 0
    assert ckpt_path.exists()

    # Verify the checkpoint payload is safe-tensor-loadable.
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assert {"state_dict", "meta"}.issubset(payload)
    assert payload["meta"]["encoder"] == "bow"
    assert payload["meta"]["num_labels"] == 4
    assert payload["clause_level_vocabs"] == [["1", "2", "3", "4"], ["1.1", "2.2", "3.3", "4.4"], ["1.1", "2.2", "3.3", "4.4"]]

    # eval -------------------------------------------------------------
    exit_code = cli_main(
        [
            "eval",
            "--ckpt",
            str(ckpt_path),
            "--benchmark-fixture",
            str(fixture_path),
            "--store-path",
            str(store_path),
            "--out-report",
            str(report_dir),
        ]
    )
    assert exit_code == 0

    report_json = report_dir / "report.json"
    report_md = report_dir / "report.md"
    assert report_json.exists()
    assert report_md.exists()

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["model"]["n"] == 2
    assert report["k"] == 3

    # eval without explicit out-report should derive a sibling directory
    exit_code = cli_main(
        [
            "eval",
            "--ckpt",
            str(ckpt_path),
            "--benchmark-fixture",
            str(fixture_path),
            "--store-path",
            str(store_path),
        ]
    )
    assert exit_code == 0

    default_report_dir = ckpt_path.with_name(f"{ckpt_path.stem}_eval")
    assert (default_report_dir / "report.json").exists()
    assert (default_report_dir / "report.md").exists()


def test_build_dataset_no_augment_opt_out(tmp_path: Path) -> None:
    store_path = _synthetic_store(tmp_path / "store")
    dataset_default = tmp_path / "dataset_default.jsonl"
    dataset_noaug = tmp_path / "dataset_noaug.jsonl"

    assert (
        cli_main(
            [
                "build-dataset",
                "--store-path",
                str(store_path),
                "--out-jsonl",
                str(dataset_default),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "build-dataset",
                "--store-path",
                str(store_path),
                "--out-jsonl",
                str(dataset_noaug),
                "--no-augment",
            ]
        )
        == 0
    )

    default_meta = json.loads(
        dataset_default.with_suffix(dataset_default.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    noaug_meta = json.loads(
        dataset_noaug.with_suffix(dataset_noaug.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert default_meta["augment"] is True
    assert default_meta["num_augmentation_templates"] > 0
    assert noaug_meta["augment"] is False
    assert noaug_meta["num_augmentation_templates"] == 0
    assert int(default_meta["num_samples"]) > int(noaug_meta["num_samples"])


class _StubTokenizer:
    """Minimal tokenizer for offline smoke testing.

    Implements the two surfaces the pipeline consumes:
      * ``.decode(ids, skip_special_tokens=bool)`` — used by
        ``build_router_dataset`` when producing first-N-token excerpts.
      * ``.encode(text, add_special_tokens=bool)`` — used by
        ``TorchKnowledgeStore.route_top_k`` for the TF-IDF baseline.

    ``encode`` deliberately emits token ids that overlap with the
    4-window synthetic store (values 10, 20, 30, 40, …, 140) so the
    TF-IDF baseline actually has non-zero overlap to score.
    """

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{int(i)}" for i in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Map each character to a store-present token id; keeps overlap
        # deterministic without depending on any HF tokenizer.
        store_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
        return [store_tokens[ord(ch) % len(store_tokens)] for ch in str(text)[:32]]


def test_pipeline_with_tokenizer_wires_baseline_and_excerpts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``--tokenizer-model-id`` must flow through build-dataset and eval.

    Asserts:
      1. Excerpt samples fire: tokenized dataset has strictly more samples
         than the no-tokenizer build.
      2. TF-IDF baseline actually runs: baseline predictions are non-empty
         and the baseline metric block reports ``n == model.n``.
      3. Baseline metrics are numeric floats — not ``None`` and not empty.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    store_path = _synthetic_store(tmp_path / "store")
    fixture_path = _tiny_benchmark_fixture(tmp_path / "bench.json")

    import tools.train_window_router as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "_maybe_load_tokenizer",
        lambda model_id: _StubTokenizer() if model_id else None,
    )

    # baseline build-dataset WITHOUT tokenizer (counts only templates + titles)
    dataset_plain = tmp_path / "dataset_plain.jsonl"
    assert (
        cli_main(
            [
                "build-dataset",
                "--store-path",
                str(store_path),
                "--out-jsonl",
                str(dataset_plain),
            ]
        )
        == 0
    )
    plain_sidecar = json.loads(
        dataset_plain.with_suffix(dataset_plain.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    plain_num_samples = int(plain_sidecar["num_samples"])
    assert plain_num_samples > 0

    # build-dataset WITH tokenizer — excerpts must add samples
    dataset_tok = tmp_path / "dataset_tok.jsonl"
    assert (
        cli_main(
            [
                "build-dataset",
                "--store-path",
                str(store_path),
                "--out-jsonl",
                str(dataset_tok),
                "--tokenizer-model-id",
                "stub",
            ]
        )
        == 0
    )
    tok_sidecar = json.loads(
        dataset_tok.with_suffix(dataset_tok.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    tok_num_samples = int(tok_sidecar["num_samples"])
    assert tok_num_samples > plain_num_samples, (
        f"tokenizer path did not produce excerpt samples: "
        f"plain={plain_num_samples} tok={tok_num_samples}"
    )

    # train on the excerpt-augmented dataset
    ckpt_path = tmp_path / "ckpt_tok" / "router.pt"
    assert (
        cli_main(
            [
                "train",
                "--dataset",
                str(dataset_tok),
                "--encoder",
                "bow",
                "--out-ckpt",
                str(ckpt_path),
                "--epochs",
                "2",
                "--hidden",
                "16",
                "--batch-size",
                "4",
                "--lr",
                "5e-2",
            ]
        )
        == 0
    )
    assert ckpt_path.exists()

    # eval WITH tokenizer — TF-IDF baseline must populate
    report_dir = tmp_path / "report_tok"
    assert (
        cli_main(
            [
                "eval",
                "--ckpt",
                str(ckpt_path),
                "--benchmark-fixture",
                str(fixture_path),
                "--store-path",
                str(store_path),
                "--out-report",
                str(report_dir),
                "--tokenizer-model-id",
                "stub",
            ]
        )
        == 0
    )
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))

    assert report["model"]["n"] == 2
    baseline = report["tfidf_baseline"]
    assert baseline["n"] == 2, f"expected baseline n==2, got {baseline!r}"
    assert isinstance(baseline["top_1_accuracy"], float)
    assert baseline["top_1_accuracy"] >= 0.0
    assert isinstance(baseline["top_3_accuracy"], float)
    assert isinstance(baseline["mrr"], float)
    # Strongest signal that the baseline ran rather than being skipped:
    # at least one per-case baseline_top3 list is non-empty.
    assert any(case.get("baseline_top3") for case in report["cases"]), (
        "TF-IDF baseline produced only empty prediction lists — "
        "tokenizer was not wired through to store.route_top_k"
    )


def test_build_dataset_writes_feature_cache_and_train_uses_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store_path = _synthetic_store(tmp_path / "store")
    dataset_jsonl = tmp_path / "dataset" / "pairs.jsonl"
    ckpt_path = tmp_path / "ckpt" / "router.pt"

    import tools._window_router.encoder as encoder_mod

    class _FakeGemmaEncoder:
        def __init__(
            self,
            model_id: str,
            device: str = "cpu",
            clause_level_vocabs: Any | None = None,
        ) -> None:
            self.model_id = model_id
            self.device = device
            self._clause_level_vocabs = clause_level_vocabs or (
                {"1": 0},
                {"1.1": 0},
                {"1.1": 0},
            )

        def fit(self, texts: Any) -> _FakeGemmaEncoder:
            return self

        @property
        def clause_level_vocabs(self) -> Any:
            return self._clause_level_vocabs

        @property
        def dim(self) -> int:
            return 5

        def encode_many_tensor(self, texts: Any, batch_size: int = 32) -> torch.Tensor:
            rows = []
            for idx, text in enumerate(texts):
                rows.append(
                    torch.tensor(
                        [float(len(str(text))), float(idx), 1.0, 0.0, 0.0],
                        dtype=torch.float32,
                    )
                )
            return torch.stack(rows)

    monkeypatch.setattr(encoder_mod, "GemmaEmbedEncoder", _FakeGemmaEncoder)

    assert (
        cli_main(
            [
                "build-dataset",
                "--store-path",
                str(store_path),
                "--out-jsonl",
                str(dataset_jsonl),
                "--feature-encoder",
                "gemma-embed",
                "--model-id",
                "stub-model",
            ]
        )
        == 0
    )
    sidecar = json.loads(
        dataset_jsonl.with_suffix(dataset_jsonl.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert "feature_cache" in sidecar
    cache_path = Path(sidecar["feature_cache"]["path"])
    assert cache_path.exists()

    class _RaisingGemmaEncoder:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("cached gemma training should not instantiate the encoder")

    monkeypatch.setattr(encoder_mod, "GemmaEmbedEncoder", _RaisingGemmaEncoder)

    assert (
        cli_main(
            [
                "train",
                "--dataset",
                str(dataset_jsonl),
                "--encoder",
                "gemma-embed",
                "--model-id",
                "stub-model",
                "--out-ckpt",
                str(ckpt_path),
                "--epochs",
                "1",
                "--hidden",
                "8",
                "--batch-size",
                "4",
            ]
        )
        == 0
    )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert payload["meta"]["encoder"] == "gemma-embed"
    assert payload["meta"]["model_id"] == "stub-model"
    assert payload["meta"]["device"] in {"cpu", "cuda"}
    assert "encoder_vocab" not in payload


def test_eval_gemma_embed_uses_device_from_checkpoint_meta(
    tmp_path: Path, monkeypatch: Any
) -> None:
    store_path = _synthetic_store(tmp_path / "store")
    fixture_path = _tiny_benchmark_fixture(tmp_path / "bench.json")
    report_dir = tmp_path / "report"
    ckpt_path = tmp_path / "gemma.pt"

    import tools._window_router.encoder as encoder_mod
    import tools.train_window_router as cli_mod

    seen_devices: list[str] = []

    class _FakeEvalGemmaEncoder:
        def __init__(
            self,
            model_id: str,
            device: str = "cpu",
            clause_level_vocabs: Any | None = None,
        ) -> None:
            assert model_id == "stub-model"
            seen_devices.append(device)
            self._dim = 4

        @property
        def dim(self) -> int:
            return self._dim

        def encode(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(encoder_mod, "GemmaEmbedEncoder", _FakeEvalGemmaEncoder)

    TorchMLPClassifier = cli_mod._load_torch_mlp_class()
    model = TorchMLPClassifier(input_size=4, hidden_size=8, num_labels=4)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": {
            "encoder": "gemma-embed",
            "num_labels": 4,
            "input_size": 4,
            "hidden_size": 8,
            "model_id": "stub-model",
            "device": "cpu",
        },
        "clause_level_vocabs": [[], [], []],
    }
    torch.save(payload, ckpt_path)

    assert (
        cli_main(
            [
                "eval",
                "--ckpt",
                str(ckpt_path),
                "--benchmark-fixture",
                str(fixture_path),
                "--store-path",
                str(store_path),
                "--out-report",
                str(report_dir),
            ]
        )
        == 0
    )
    assert seen_devices == ["cpu"]
