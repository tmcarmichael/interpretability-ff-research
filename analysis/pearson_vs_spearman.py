"""Pearson vs Spearman comparison for models with control sensitivity data.

Reports the delta between standard (rank-based) and nonlinear (raw-value)
controls across models that have both measurements.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from analysis.load_results import SCOPES, load_all_models, load_control_sensitivity


def partial_pearson(
    x: np.ndarray,
    y: np.ndarray,
    covariates: list[np.ndarray],
) -> tuple[float, float]:
    X = np.column_stack(covariates + [np.ones(len(x))])
    coef_x = np.linalg.lstsq(X, x, rcond=None)[0]
    coef_y = np.linalg.lstsq(X, y, rcond=None)[0]
    r, p = pearsonr(x - X @ coef_x, y - X @ coef_y)
    return float(r), float(p)


def report(scope: str | None = "control_sensitivity_14") -> None:
    """Print the Pearson-vs-Spearman delta table for all available models."""
    load_all_models(verbose=True, scope=scope)
    print()

    models = load_control_sensitivity(scope=scope)
    if not models:
        print("No control sensitivity data.")
        return

    print(f"  {'Model':<15} {'Standard':>10} {'Nonlinear':>11} {'Delta':>8}")
    print(f"  {'-' * 45}")
    deltas = []
    for m in models:
        s = m["standard"]
        n = m["nonlinear"]
        delta = n - s
        deltas.append(delta)
        print(f"  {m['name']:<15} {s:+.4f}    {n:+.4f}  {delta:+.4f}")

    mean_delta = np.mean(deltas)
    print(f"\n  Mean delta: {mean_delta:+.4f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scope", default="control_sensitivity_14", choices=sorted(SCOPES), help="Named model scope."
    )
    args = p.parse_args()
    report(scope=args.scope)
