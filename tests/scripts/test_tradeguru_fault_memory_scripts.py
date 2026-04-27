from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_condition_aliases_cover_four_eval_conditions() -> None:
    mod = _load_script("evaluate_tradeguru_fault_memory.py")

    assert mod.resolve_conditions("memory_off,memory_on,memory_on_noise,off_store") == [
        "memory_off_empty_store",
        "memory_on_tradeguru_store",
        "memory_on_tradeguru_plus_noise",
        "memory_off_with_store_present",
    ]


def test_default_questions_are_fault_finding_prompts_without_source_names() -> None:
    mod = _load_script("evaluate_tradeguru_fault_memory.py")

    questions = mod.load_questions(None)
    assert len(questions) >= 8
    joined = "\n".join(question.question.lower() for question in questions)
    assert "electricalhowto" not in joined
    assert "fault_finding_corpus" not in joined
    assert "source" not in joined


def test_scoring_rewards_safety_and_flags_unsafe_live_work() -> None:
    mod = _load_script("evaluate_tradeguru_fault_memory.py")
    question = mod.DEFAULT_QUESTIONS[0]

    safe_answer = (
        "First isolate the circuit, use lockout/tagout, prove dead, and have a "
        "licensed electrician inspect the kettle for earth leakage. Then unplug "
        "the appliance, reset the RCD, visually inspect damage, and run insulation "
        "resistance tests before returning it to service."
    )
    unsafe_answer = "Work live and touch the active conductor until the RCD trips."

    safe_score = mod.score_answer(
        safe_answer,
        question,
        telemetry={"memory_used": False, "evidence": []},
        memory_expected=False,
    )
    unsafe_score = mod.score_answer(
        unsafe_answer,
        question,
        telemetry={"memory_used": False, "evidence": []},
        memory_expected=False,
    )

    assert safe_score.criteria["safety_controls_present"] is True
    assert safe_score.criteria["no_unsafe_live_work_advice"] is True
    assert unsafe_score.criteria["no_unsafe_live_work_advice"] is False
    assert safe_score.total > unsafe_score.total


def test_windows_path_translation_for_wsl() -> None:
    import_mod = _load_script("import_tradeguru_memory.py")
    eval_mod = _load_script("evaluate_tradeguru_fault_memory.py")

    translated = import_mod.resolve_host_path(r"C:\Users\jehma\Desktop\TradeGuru")
    eval_translated = eval_mod.resolve_host_path(r"C:\Users\jehma\Desktop\TradeGuru")
    if import_mod.os.name == "nt":
        assert str(translated).startswith("C:")
    else:
        assert translated.as_posix() == "/mnt/c/Users/jehma/Desktop/TradeGuru"
        assert eval_translated.as_posix() == "/mnt/c/Users/jehma/Desktop/TradeGuru"


def test_token_windowing_uses_overlap() -> None:
    mod = _load_script("import_tradeguru_memory.py")

    windows = mod.iter_token_windows(
        list(range(10)),
        window_size=4,
        overlap_tokens=1,
    )

    assert windows == [
        (0, 4, [0, 1, 2, 3]),
        (3, 7, [3, 4, 5, 6]),
        (6, 10, [6, 7, 8, 9]),
    ]


def test_category_mapping_matches_tradeguru_folders(tmp_path: Path) -> None:
    mod = _load_script("import_tradeguru_memory.py")

    source = tmp_path / "electricalhowto - 2"
    doc_path = source / "rcd.md"
    source.mkdir()
    doc_path.write_text("# RCD Fault Finding\nBody", encoding="utf-8")

    doc = mod.load_source_document(doc_path, source)

    assert doc is not None
    assert doc.category == "electrical_howto_markdown"
    assert doc.folder == "electricalhowto - 2"
    assert doc.doc_title == "RCD Fault Finding"
