from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david.quality_gates import QualityGateCandidate, discover_quality_gates


def test_selected_tests_generate_targeted_pytest_and_reject_unsafe_paths(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "test_feature.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    outside = tmp_path.parent / "outside_test.py"
    outside.write_text("def test_outside():\n    assert True\n", encoding="utf-8")

    candidates = discover_quality_gates(
        tmp_path,
        selected_tests=[
            "tests/test_feature.py::test_feature",
            "../outside_test.py",
            str(outside),
            "scripts/run_swebench_pro_parity.py",
        ],
    )

    targeted = candidates[0]
    assert targeted.name == "pytest_selected"
    assert targeted.command == ["python", "-m", "pytest", "tests/test_feature.py::test_feature"]
    assert targeted.confidence == 0.95
    rejected = targeted.provenance["rejected_selected_paths"]
    assert "../outside_test.py" in rejected
    assert str(outside) in rejected
    assert "scripts/run_swebench_pro_parity.py" in rejected
    for candidate in candidates:
        assert isinstance(candidate.command, list)
        assert "../outside_test.py" not in candidate.command
        assert "scripts/run_swebench_pro_parity.py" not in candidate.command


def test_discovers_pytest_from_pyproject_and_tests_dir(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        encoding="utf-8",
    )

    candidates = discover_quality_gates(tmp_path)

    workspace = _by_name(candidates, "pytest_workspace")
    assert workspace.command == ["python", "-m", "pytest", "tests"]
    assert "pytest configuration" in workspace.reason
    assert workspace.provenance["signal"] == "tool.pytest"


def test_discovers_package_json_scripts_as_argv_without_script_bodies(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        """
        {
          "scripts": {
            "test": "vitest run",
            "lint": "eslint .",
            "build": "vite build"
          }
        }
        """,
        encoding="utf-8",
    )

    candidates = discover_quality_gates(tmp_path)

    assert _by_name(candidates, "package_test").command == ["npm", "run", "test"]
    assert _by_name(candidates, "package_lint").command == ["npm", "run", "lint"]
    assert all("vitest run" not in candidate.command for candidate in candidates)
    assert _by_name(candidates, "package_test").provenance == {
        "source": "package.json",
        "script": "test",
        "package_manager": "npm",
    }


def test_discovers_py_compile_for_selected_python_sources_only(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.py"
    test = tmp_path / "tests" / "test_app.py"
    source.parent.mkdir()
    test.parent.mkdir()
    source.write_text("def app():\n    return 1\n", encoding="utf-8")
    test.write_text("def test_app():\n    assert True\n", encoding="utf-8")

    candidates = discover_quality_gates(
        tmp_path,
        selected_paths=["src/app.py", "tests/test_app.py", "../escaped.py"],
    )

    py_compile = _by_name(candidates, "py_compile_selected")
    assert py_compile.command == ["python", "-m", "py_compile", "src/app.py"]
    rejected = py_compile.provenance["rejected_selected_paths"]
    assert rejected["tests/test_app.py"] == ["path is not a selected Python source file"]
    assert "../escaped.py" in rejected


def test_candidate_count_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        """
        {
          "scripts": {
            "test": "vitest",
            "lint": "eslint .",
            "typecheck": "tsc --noEmit",
            "check": "npm run lint",
            "build": "vite build"
          }
        }
        """,
        encoding="utf-8",
    )

    candidates = discover_quality_gates(tmp_path, max_candidates=2)

    assert len(candidates) == 2
    assert [candidate.name for candidate in candidates] == ["pytest_workspace", "package_test"]


def test_candidate_to_dict_preserves_argv_list() -> None:
    candidate = QualityGateCandidate(
        name="example",
        command=["python", "-m", "pytest"],
        reason="unit test",
        confidence=0.5,
        provenance={"source": "test"},
    )

    data = candidate.to_dict()

    assert data["command"] == ["python", "-m", "pytest"]
    assert data["provenance"] == {"source": "test"}


def _by_name(candidates: list[QualityGateCandidate], name: str) -> QualityGateCandidate:
    return next(candidate for candidate in candidates if candidate.name == name)
