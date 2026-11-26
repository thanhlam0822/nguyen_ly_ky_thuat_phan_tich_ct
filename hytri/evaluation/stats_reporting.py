"""Statistical summaries for per-build evaluation scores."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np


def mean_and_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean. Returns (mean, low, high)."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    rng = random.Random(seed)
    n = len(arr)
    means = []
    for _ in range(n_bootstrap):
        sample = [arr[rng.randrange(n)] for _ in range(n)]
        means.append(float(np.mean(sample)))
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1.0 - alpha) * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return float(arr.mean()), means[lo_idx], means[hi_idx]


def wilcoxon_signed_rank_pvalue(method_a: list[float], method_b: list[float]) -> float:
    """Paired two-sided Wilcoxon signed-rank test p-value (method_a vs method_b)."""
    if len(method_a) != len(method_b) or len(method_a) < 2:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        stat = wilcoxon(method_a, method_b, alternative="two-sided")
        return float(stat.pvalue)
    except Exception:
        return 1.0


def compare_methods_paired(
    scores_by_method: dict[str, list[float]],
    metric_name: str,
    baseline: str | None = None,
) -> list[dict]:
    """Pairwise Wilcoxon comparisons vs baseline (or all pairs if baseline None)."""
    methods = list(scores_by_method.keys())
    rows: list[dict] = []
    if baseline and baseline in methods:
        others = [m for m in methods if m != baseline]
        for m in others:
            a = scores_by_method[baseline]
            b = scores_by_method[m]
            if len(a) == len(b) and len(a) >= 2:
                rows.append({
                    "metric": metric_name,
                    "method_a": baseline,
                    "method_b": m,
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "p_value": wilcoxon_signed_rank_pvalue(a, b),
                })
    return rows


def format_ci_line(name: str, values: list[float], seed: int = 42) -> str:
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{name}: mean={mean:.4f} std={std:.4f} 95%CI=[{lo:.4f}, {hi:.4f}] n={len(values)}"
