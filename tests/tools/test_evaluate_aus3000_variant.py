from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.evaluate_aus3000_variant as evaluator


def _make_record(clause_id: str) -> evaluator.CorpusRecord:
    return evaluator.CorpusRecord(
        clause_id=clause_id,
        clause_title=f"Clause {clause_id}",
        clause_content=f"Clause {clause_id} content",
        standard_id="AS/NZS 3000-2018",
        standard_title="Electrical installations",
    )


def _answer_for_case(case: evaluator.EvalCase) -> str:
    if case.expect_insufficient:
        return "I can't answer from the retrieved store context."

    parts = list(case.required_answer_anchors)
    for group in case.required_literal_groups:
        parts.append(group[0])
    return " ".join(parts)


def _build_window_map(cases: list[evaluator.EvalCase]) -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    ordered_clause_ids: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for clause_id in case.expected_clause_ids:
            if clause_id not in seen:
                seen.add(clause_id)
                ordered_clause_ids.append(clause_id)

    window_metadata: dict[int, dict[str, object]] = {}
    window_ids_by_clause_id: dict[str, int] = {}
    for index, clause_id in enumerate(ordered_clause_ids, start=100):
        window_metadata[index] = {
            "clause_id": clause_id,
            "clause_title": f"Clause {clause_id}",
            "part_index": 1,
            "part_count": 1,
            "source_file": f"{clause_id}.json",
        }
        window_ids_by_clause_id[clause_id] = index
    return window_metadata, window_ids_by_clause_id


def _make_response(case: evaluator.EvalCase, window_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        text=_answer_for_case(case),
        window_ids=window_ids,
        mode="strict",
        routing_mode="metadata",
        expansion_terms=[],
        residual_reason="none",
        boundary_shape="[]",
        stats_summary="{}",
    )


def _build_records(cases: list[evaluator.EvalCase]) -> list[evaluator.CorpusRecord]:
    records: list[evaluator.CorpusRecord] = []
    seen: set[str] = set()
    for case in cases:
        for clause_id in case.expected_clause_ids:
            if clause_id in seen:
                continue
            seen.add(clause_id)
            records.append(_make_record(clause_id))
    return records


def _write_window_metadata(store_path: Path, window_metadata: dict[int, dict[str, object]]) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / "window_metadata.json").write_text(json.dumps(window_metadata, indent=2) + "\n")


class _DummyRuntime:
    def clear_cache(self) -> None:
        return None


class _DummyStore:
    def __init__(self, window_metadata: dict[int, dict[str, object]]) -> None:
        self.window_metadata = window_metadata

    def get_window_text(self, window_id: int, tokenizer) -> str:  # noqa: ANN001
        metadata = self.window_metadata[window_id]
        return f"{metadata['clause_id']} {metadata['clause_title']}"


def test_fixture_is_machine_readable_and_docs_aligned() -> None:
    fixture = evaluator.load_benchmark_fixture()
    cases = evaluator.build_case_suite()

    assert fixture["gate_order"] == [
        "store_evidence_gate",
        "single_pass_gate",
        "soak_gate",
    ]
    assert len(cases) == 23
    assert len(evaluator.unique_primary_clause_ids(cases)) == 16
    assert len(fixture["no_regression_case_ids"]) == 17


def test_parser_defaults_use_canonical_clause_aligned_paths() -> None:
    parser = evaluator.build_parser()
    args = parser.parse_args([])

    assert args.mode == "report"
    assert args.checkpoint == evaluator.DEFAULT_CHECKPOINT
    assert args.output == evaluator.DEFAULT_OUTPUT
    assert args.benchmark_fixture == evaluator.DEFAULT_BENCHMARK_FIXTURE


def test_exact_route_scoring_rejects_wrong_window_even_when_text_mentions_clause() -> None:
    case = evaluator.build_case_suite()[0]
    expected_records = [_make_record("1.4.2")]
    routed_ids = ("7.9.3.3",)

    evaluation = evaluator.evaluate_report_case(
        case,
        "Clause 1.4.2 Accessible means inspection maintenance repairs destructive dismantling.",
        expected_records,
        routed_ids,
    )

    assert evaluation.verdict == "FAIL"
    assert evaluation.route_pass is False
    assert any("route exact match: no" in note.lower() for note in evaluation.notes)


def test_comparison_case_uses_effective_top_k_and_exact_route_order() -> None:
    cases = evaluator.build_case_suite()
    case = next(item for item in cases if item.name == "accessible_vs_readily")
    expected_records = [_make_record("1.4.2"), _make_record("1.4.3")]

    pass_eval = evaluator.evaluate_strict_case(
        case,
        "inspection maintenance quickly without climbing movable ladder 2.0 m",
        expected_records,
        ("1.4.2", "1.4.3"),
    )
    fail_eval = evaluator.evaluate_strict_case(
        case,
        "inspection maintenance quickly without climbing movable ladder 2.0 m",
        expected_records,
        ("1.4.3", "1.4.2"),
    )

    assert evaluator.case_effective_top_k(case) == 2
    assert pass_eval.verdict == "PASS"
    assert fail_eval.route_pass is False
    assert fail_eval.verdict == "FAIL"


def test_ood_refusal_with_electrical_bleed_fails() -> None:
    case = next(item for item in evaluator.build_case_suite() if item.name == "capital_of_france")
    evaluation = evaluator.evaluate_strict_case(
        case,
        "I can't answer from the retrieved store context, but the electrical installation clause applies.",
        [],
        (),
    )

    assert evaluation.verdict == "FAIL"
    assert evaluation.ood_pass is False
    assert any("electrical bleed" in note.lower() for note in evaluation.notes)


def test_insulation_resistance_accepts_unicode_ohm_literal_variants() -> None:
    case = next(item for item in evaluator.build_case_suite() if item.name == "insulation_resistance_results")
    expected_records = [_make_record("8.3.6.3")]

    evaluation = evaluator.evaluate_strict_case(
        case,
        "not less than 1 MΩ consumer mains submains",
        expected_records,
        ("8.3.6.3",),
    )

    assert evaluation.verdict == "PASS"
    assert evaluation.grounding_pass is True
    assert evaluator._literal_groups_present(
        "not less than 1 MΩ consumer mains submains",
        case.required_literal_groups,
    )


def test_store_evidence_gate_verifies_all_16_unique_primary_clause_ids() -> None:
    fixture = evaluator.load_benchmark_fixture()
    cases = evaluator.build_case_suite()
    window_metadata, _ = _build_window_map(cases)

    outcome = evaluator.run_store_evidence_gate(fixture, window_metadata)

    assert outcome.overall_pass is True
    assert outcome.pass_count == 1
    assert outcome.fail_count == 0
    assert len(fixture["unique_primary_clause_ids"]) == 16


def test_single_pass_gate_counts_strict_pass_fail_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = evaluator.build_case_suite()
    fixture = evaluator.load_benchmark_fixture()
    window_metadata, window_ids_by_clause_id = _build_window_map(cases)
    store_path = tmp_path / "checkpoint" / "torch_store"
    _write_window_metadata(store_path, window_metadata)

    records = _build_records(cases)
    records_by_clause_id = {record.clause_id: record for record in records}

    response_map: dict[str, SimpleNamespace] = {}
    for case in cases:
        if case.expect_insufficient:
            response_map[case.prompt] = _make_response(
                case,
                [window_ids_by_clause_id[next(iter(window_ids_by_clause_id))]],
            )
            continue

        if case.category == "comparison":
            window_ids = [window_ids_by_clause_id[clause_id] for clause_id in case.primary_clause_ids]
        else:
            window_ids = [window_ids_by_clause_id[case.primary_clause_ids[0]]]
        response_map[case.prompt] = _make_response(case, window_ids)

    def _fake_prepare_store_response(**kwargs):  # noqa: ANN001
        return response_map[kwargs["question"]]

    monkeypatch.setattr(evaluator, "_prepare_store_response", _fake_prepare_store_response)
    monkeypatch.setattr(evaluator, "_load_torch_runtime", lambda *args, **kwargs: (_DummyRuntime(), object()))
    monkeypatch.setattr(evaluator, "TorchKnowledgeStore", SimpleNamespace(load=lambda path: _DummyStore(window_metadata)))

    args = SimpleNamespace(
        mode="single_pass_gate",
        model="test-model",
        checkpoint=tmp_path / "checkpoint",
        corpus=evaluator.DEFAULT_CORPUS,
        output=tmp_path / "report.txt",
        benchmark_fixture=evaluator.DEFAULT_BENCHMARK_FIXTURE,
        duration_minutes=1.0,
        max_cases=None,
        max_new_tokens=120,
        temperature=0.0,
        top_k=3,
        clear_cache_every=0,
        no_chat_template=False,
        device="cpu",
    )

    outcome = evaluator._run_strict_mode(
        args=args,
        metadata={"windowing": {"num_windows": len(window_metadata), "num_tokens": 0}},
        store_path=store_path,
        records=records,
        cases=cases,
    )

    assert outcome.executed == 23
    assert outcome.pass_count == 23
    assert outcome.fail_count == 0
    assert outcome.overall_pass is True
    assert outcome.route_pass_count == 17
    assert outcome.grounding_pass_count == 17
    assert outcome.ood_pass_count == 6
    assert outcome.regression_pass_count == 17


def test_strict_mode_creates_output_before_runtime_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = evaluator.build_case_suite()
    fixture = evaluator.load_benchmark_fixture()
    window_metadata, window_ids_by_clause_id = _build_window_map(cases)
    store_path = tmp_path / "checkpoint" / "torch_store"
    _write_window_metadata(store_path, window_metadata)

    records = _build_records(cases)
    response_map: dict[str, SimpleNamespace] = {}
    for case in cases:
        if case.expect_insufficient:
            response_map[case.prompt] = _make_response(
                case,
                [window_ids_by_clause_id[next(iter(window_ids_by_clause_id))]],
            )
            continue

        if case.category == "comparison":
            window_ids = [window_ids_by_clause_id[clause_id] for clause_id in case.primary_clause_ids]
        else:
            window_ids = [window_ids_by_clause_id[case.primary_clause_ids[0]]]
        response_map[case.prompt] = _make_response(case, window_ids)

    def _fake_prepare_store_response(**kwargs):  # noqa: ANN001
        return response_map[kwargs["question"]]

    def _fake_load_torch_runtime(*args, **kwargs):  # noqa: ANN001
        report_path = output_path
        assert report_path.exists()
        report_text = report_path.read_text(encoding="utf-8")
        assert "Startup: loading torch runtime and knowledge store" in report_text
        return _DummyRuntime(), object()

    output_path = tmp_path / "strict_report.txt"

    monkeypatch.setattr(evaluator, "_prepare_store_response", _fake_prepare_store_response)
    monkeypatch.setattr(evaluator, "_load_torch_runtime", _fake_load_torch_runtime)
    monkeypatch.setattr(evaluator, "TorchKnowledgeStore", SimpleNamespace(load=lambda path: _DummyStore(window_metadata)))

    args = SimpleNamespace(
        mode="single_pass_gate",
        model="test-model",
        checkpoint=tmp_path / "checkpoint",
        corpus=evaluator.DEFAULT_CORPUS,
        output=output_path,
        benchmark_fixture=evaluator.DEFAULT_BENCHMARK_FIXTURE,
        duration_minutes=1.0,
        max_cases=None,
        max_new_tokens=120,
        temperature=0.0,
        top_k=3,
        clear_cache_every=0,
        no_chat_template=False,
        device="cpu",
    )

    outcome = evaluator._run_strict_mode(
        args=args,
        metadata={"windowing": {"num_windows": len(window_metadata), "num_tokens": 0}},
        store_path=store_path,
        records=records,
        cases=cases,
    )

    assert outcome.overall_pass is True
