"""Verify that inlined probe functions in producer scripts match src/probe.py.

The standalone-runnable producer design inlines core probe functions
instead of importing them. This test catches drift between the
src/probe.py reference and every script that inlines partial_spearman or
compute_loss_residuals, by compiling both copies and running them on the
same synthetic data.

Auto-discovery: any script under scripts/ that defines `def partial_spearman(`
or `def compute_loss_residuals(` is parametrized in.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIFT_FUNCS = ("partial_spearman", "compute_loss_residuals")

DRIFT_EXEMPT = {
    # MPS-only dev scripts return rho directly, not (rho, p).
    ("mistral7b_instruct_full_mps.py", "partial_spearman"),
    ("phi3_layer_sweep_mps.py", "partial_spearman"),
}


def _extract_function(filepath: Path, func_name: str) -> str | None:
    """Extract a function's source by name; return None if absent."""
    lines = filepath.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
            break
    if start is None:
        return None
    func_lines = [lines[start]]
    for line in lines[start + 1 :]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def _compile_function(source: str, name: str):
    """Compile a function source string and return the callable."""
    namespace = {"np": np, "rankdata": rankdata, "pearsonr": pearsonr}
    exec(compile(source, f"<{name}>", "exec"), namespace)
    return namespace[name]


def _discover_inlined_scripts() -> list[tuple[Path, str]]:
    """Return (script_path, func_name) pairs for every script that inlines
    a drift-checked function, excluding entries in DRIFT_EXEMPT."""
    pairs: list[tuple[Path, str]] = []
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        for func in DRIFT_FUNCS:
            if (script.name, func) in DRIFT_EXEMPT:
                continue
            if _extract_function(script, func) is not None:
                pairs.append((script, func))
    return pairs


@pytest.fixture(scope="module")
def probe_funcs():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import probe

    return {name: getattr(probe, name) for name in DRIFT_FUNCS}


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 500
    losses = rng.exponential(2.0, n)
    max_softmax = rng.beta(5, 2, n)
    activation_norm = rng.lognormal(0, 1, n)
    probe_scores = 0.3 * losses + 0.5 * rng.randn(n)
    return losses, max_softmax, activation_norm, probe_scores


_DRIFT_PAIRS = _discover_inlined_scripts()


@pytest.mark.parametrize(
    "script_path,func_name",
    _DRIFT_PAIRS,
    ids=[f"{p.name}::{f}" for p, f in _DRIFT_PAIRS],
)
def test_inlined_function_matches_probe(script_path, func_name, probe_funcs, synthetic_data):
    """Every inlined copy of a drift-checked probe function must produce
    identical output to src/probe.py on the same synthetic input.
    """
    src_text = _extract_function(script_path, func_name)
    inlined = _compile_function(src_text, func_name)
    reference = probe_funcs[func_name]
    losses, max_softmax, activation_norm, probe_scores = synthetic_data

    if func_name == "partial_spearman":
        covariates = [max_softmax, activation_norm]
        r_ref, p_ref = reference(probe_scores, losses, covariates)
        r_inl, p_inl = inlined(probe_scores, losses, covariates)
        assert r_ref == pytest.approx(r_inl, abs=1e-12), (
            f"partial_spearman drift in {script_path.name}: ref={r_ref}, inlined={r_inl}"
        )
        assert p_ref == pytest.approx(p_inl, abs=1e-12)
    elif func_name == "compute_loss_residuals":
        resid_ref = reference(losses, max_softmax, activation_norm)
        resid_inl = inlined(losses, max_softmax, activation_norm)
        np.testing.assert_array_almost_equal(
            resid_ref,
            resid_inl,
            decimal=12,
            err_msg=f"compute_loss_residuals drift in {script_path.name}",
        )
    else:
        pytest.fail(f"unhandled drift function: {func_name}")
