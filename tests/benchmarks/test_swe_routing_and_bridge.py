from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_locobench_benchmark import (  # noqa: E402
    DIFF_PROTOCOL_CONSTRAINT,
    LoCoBenchCase,
    magnetized_search_bridge_decision,
    route_domain_score,
    route_domain_weight,
    visible_prompt,
)
from run_swebench_pro_parity import (  # noqa: E402
    RULER_JIT_INDEXING_COMPAT_SIGNAL,
    RULER_JIT_INDEXING_IMPLEMENTATION,
    SWEBenchProCase,
    ruler_jit_case_metadata,
    swe_ruler_jit_benchmark_slug,
)


def test_domain_score_prefers_backend_source_over_frontend_asset() -> None:
    backend = "src/database/redis/main.js"
    frontend = "public/styles/admin/theme.css"

    assert route_domain_weight(backend) > route_domain_weight(frontend)
    assert route_domain_score(10.0, backend) > route_domain_score(10.0, frontend)


def test_visible_prompt_keeps_diff_protocol_out_of_user_prompt() -> None:
    case = LoCoBenchCase(
        row_index=0,
        instruction="Update the counter.",
        files={"src/app.py": "def counter():\n    return 1\n"},
        expected_patch="",
        raw={},
    )

    prompt = visible_prompt(case)

    assert DIFF_PROTOCOL_CONSTRAINT not in prompt
    assert "MANDATORY_FORMAT" not in prompt
    assert "examples/alpha.py" not in prompt
    assert "<<<< SEARCH" not in prompt
    assert "==== REPLACE" not in prompt


def test_magnetized_bridge_forces_replace_transition_for_matching_search() -> None:
    target_span = "def counter(value):\n    return value + 1\n"
    generated = "src/app.py\n<<<< SEARCH\n" + target_span.rstrip()

    decision = magnetized_search_bridge_decision(
        generated,
        target_path="src/app.py",
        target_span=target_span,
    )

    assert decision["active"] is True
    assert decision["similarity"] > 0.90
    assert decision["emitted_prefix"] == "\n==== REPLACE\n"
    assert decision["state_transition"] == "SEARCH_BLOCK->REPLACE_BLOCK"


def test_swe_ruler_jit_metadata_uses_selected_row_cache_scope() -> None:
    args = argparse.Namespace(
        dataset_id="ScaleAI/SWE-bench_Pro",
        split="test",
        start_index=25,
        limit=1,
        window_tokens=512,
        activation_overlap_tokens=128,
        exact_context_packing=False,
    )
    case = SWEBenchProCase(
        row_index=25,
        instruction="Fix the route handler.",
        files={"src/router.py": "def route():\n    return None\n"},
        expected_patch="",
        raw={},
        repo="example/project",
        instance_id="example__project-25",
        base_commit="abc123",
        repo_language="python",
        requirements="",
        interface="",
        fail_to_pass=[],
        pass_to_pass=[],
        before_repo_set_cmd="",
        selected_test_files_to_run=[],
        dockerhub_tag="",
        checkout_path="",
    )

    slug = swe_ruler_jit_benchmark_slug([case], args)
    args.jit_indexing = {
        "mode": "real_gemma4_fast_batch_layer12",
        "layer": 12,
        "benchmark_slug": slug,
        "metadata_path": ".benchmarks/jit/example_jit_index_metadata.json",
        "activation_route_dir": ".benchmarks/jit/example_activation_routes",
        "ruler_compat": {
            "active": True,
            "signal": RULER_JIT_INDEXING_COMPAT_SIGNAL,
            "implementation": RULER_JIT_INDEXING_IMPLEMENTATION,
            "cache_scope": "selected_swebench_rows",
            "case_ids": ["example__project-25"],
        },
    }

    metadata = ruler_jit_case_metadata(args, case)

    assert "rows_25_25" in slug
    assert metadata["active"] is True
    assert metadata["signal"] == RULER_JIT_INDEXING_COMPAT_SIGNAL
    assert metadata["implementation"] == RULER_JIT_INDEXING_IMPLEMENTATION
    assert metadata["case_index_in_jit_corpus"] == 0
