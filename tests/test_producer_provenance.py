"""Producer scripts emit canonical 5-field provenance.

Walks the AST of every paper-output producer, finds dict literals matching
the canonical provenance shape, and asserts the key set + ordering match
(`model_revision, script, timestamp, value_source, device`).

Complements `test_canonical_provenance_all_files` in `test_results_schema.py`,
which checks committed JSONs. This test pins the contract at the producer
source level: a future edit that drops a field, reorders, adds an extra,
or stops emitting fails here even before any JSON is regenerated.

Run: uv run pytest tests/test_producer_provenance.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ("model_revision", "script", "timestamp", "value_source", "device")
CANONICAL_SET = set(CANONICAL)

PRODUCERS = [
    "scripts/run_model.py",
    "scripts/run_stream_model.py",
    "scripts/pythia_checkpoint_dynamics.py",
    "scripts/gpt2_shuffle_test.py",
    "scripts/nonlinear_probe.py",
    "scripts/medqa_selective.py",
    "scripts/truthfulqa_hallucination.py",
    "scripts/rag_hallucination.py",
    "scripts/pythia_1.4b_shuffle.py",
    "scripts/pythia_12b_backfill.py",
    "scripts/mechanistic_llama.py",
    "scripts/mechanistic_mistral.py",
    "scripts/roc_width_sweep.py",
    "scripts/split_bootstrap_gpu.py",
    "src/transformer_observe.py",
]


def _provenance_dict_literals(tree: ast.AST) -> list[tuple[int, tuple[str, ...]]]:
    """Return (lineno, key-tuple) for every dict literal whose key set is canonical."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = tuple(k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
        if set(keys) == CANONICAL_SET:
            found.append((node.lineno, keys))
    return found


@pytest.mark.parametrize("producer", PRODUCERS)
def test_producer_emits_canonical_provenance(producer: str) -> None:
    """Producer source contains a dict literal with the canonical 5-field shape, in order."""
    path = REPO_ROOT / producer
    assert path.is_file(), f"{producer}: not found"

    tree = ast.parse(path.read_text(), filename=str(path))
    candidates = _provenance_dict_literals(tree)

    assert candidates, (
        f"{producer}: no dict literal with canonical provenance keys ({', '.join(CANONICAL)}) found in source"
    )

    for lineno, keys in candidates:
        assert keys == CANONICAL, (
            f"{producer}:{lineno}: provenance dict key order != canonical\n"
            f"  expected: {CANONICAL}\n"
            f"  got:      {keys}"
        )


def test_producer_list_locked() -> None:
    """PRODUCERS lists every paper-output script."""
    expected_count = 15
    assert len(PRODUCERS) == expected_count, (
        f"PRODUCERS has {len(PRODUCERS)} entries; expected {expected_count}. "
        "Update PRODUCERS when the paper-output script set changes."
    )
    for p in PRODUCERS:
        assert (REPO_ROOT / p).is_file(), f"PRODUCERS references missing file: {p}"
