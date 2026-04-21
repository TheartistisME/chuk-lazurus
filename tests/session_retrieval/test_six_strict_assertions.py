"""CUDA-gated integration test for the six strict-mode assertions.

Mirrors ``examples/inference/demo_clause_aligned_strict.py`` (lines 97-298):
a successful :meth:`SessionRetriever.query_exact_id` on CUDA must populate
``QueryResult.strict_assertions`` with all six keys == True and a non-empty
``generated_answer`` that contains a planted verbatim phrase from the fixture.

Runnability:
- Skipped when ``torch`` is unimportable (``pytest.importorskip``).
- Skipped when ``torch.cuda.is_available()`` is False.
- Currently also hard-skipped because a fully-built clause-aligned checkpoint
  + calibrated Gemma-4 model on CUDA is required, and constructing that in a
  unit-test harness is out of scope for this test dispatch. The CI harness
  should provide a ``CHUK_LAZARUS_TEST_CHECKPOINT_ROOT`` env var pointing to
  a real checkpoint directory with at least one planted entry; when present
  the test body below becomes runnable.

TODO(ci): wire up a minimal real-checkpoint fixture and drop the outer skip.
The fixture artifact must contain:
  - exactly one session_id directory under checkpoint_root/
  - a torch_store/ produced by tools/build_clause_aligned_store.py
  - a matching original_input_root/<session_id>/NNN_ti_ci.json carrying the
    planted verbatim phrase ``TEST_PHRASE`` below in ``clause_content``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Import torch via importorskip so this module silently passes on CPU-only hosts.
torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for strict-mode demo"
    ),
]

# Verbatim phrase the real fixture must plant in the routed window's source
# record so we can assert window-routing correctness via substring match.
TEST_PHRASE: str = "Lazarus strict-mode demo phrase"


def _fixture_root_from_env() -> Path | None:
    """Return a fully-built checkpoint root from env, or None if unavailable."""
    raw = os.environ.get("CHUK_LAZARUS_TEST_CHECKPOINT_ROOT")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_dir():
        return None
    return path


@pytest.fixture(scope="session")
def real_checkpoint_root() -> Path:
    """Session-scoped: a real clause-aligned checkpoint root for integration."""
    root = _fixture_root_from_env()
    if root is None:
        pytest.skip(
            "Real checkpoint fixture not supplied via "
            "CHUK_LAZARUS_TEST_CHECKPOINT_ROOT; CI integration test skipped."
        )
    return root


@pytest.fixture(scope="session")
def real_original_input_root() -> Path | None:
    """Session-scoped: optional original_input_root."""
    raw = os.environ.get("CHUK_LAZARUS_TEST_ORIGINAL_INPUT_ROOT")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_dir():
        return None
    return path


@pytest.fixture(scope="session")
def real_dotted_handle() -> str:
    """Session-scoped: the dotted handle CI knows points at the planted window."""
    handle = os.environ.get("CHUK_LAZARUS_TEST_DOTTED_HANDLE")
    if not handle:
        pytest.skip(
            "Real dotted handle not supplied via CHUK_LAZARUS_TEST_DOTTED_HANDLE."
        )
    return handle


@pytest.fixture(scope="session")
def retriever(
    real_checkpoint_root: Path, real_original_input_root: Path | None
):
    """Session-scoped: construct a real :class:`SessionRetriever` on CUDA."""
    from chuk_lazarus.session_retrieval import SessionRetriever  # local import

    return SessionRetriever.from_checkpoint_root(
        real_checkpoint_root,
        device="cuda",
        original_input_root=real_original_input_root,
    )


def test_query_exact_id_passes_all_six_strict_assertions(
    retriever,
    real_dotted_handle: str,
) -> None:
    """``query_exact_id`` yields all six strict assertions True + planted text."""
    result = retriever.query_exact_id(real_dotted_handle)

    expected_strict_assertions = {
        "cuda_available": True,
        "model_on_cuda": True,
        "residual_compatible": True,
        "hook_fired": True,
        "gpu_memory_grew": True,
        "store_window_nonempty": True,
    }
    assert result.strict_assertions == expected_strict_assertions

    # Non-empty generated answer proves the full pipeline ran.
    assert isinstance(result.generated_answer, str)
    assert result.generated_answer.strip() != ""

    # Routing correctness: the planted phrase from CI must appear in the
    # matched window text verbatim.
    assert TEST_PHRASE in result.matched_window_text, (
        "Planted phrase missing from matched_window_text; window routing is "
        "incorrect or the fixture is stale. Expected substring: "
        f"{TEST_PHRASE!r}. Actual window text head: "
        f"{result.matched_window_text[:200]!r}"
    )
