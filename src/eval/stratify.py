# -*- coding: utf-8 -*-
"""
src/eval/stratify.py

Stratify run summaries by coupling index buckets.

Why not hard-code thresholds?
- coupling_index scale depends on graph size and dataset
- fixed (5,10) will be meaningless across benchmarks

This module supports:
- explicit thresholds (low_th, high_th)
- or quantile-based thresholds computed from available coupling scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from src.io.schemas import RunSummary


@dataclass(frozen=True)
class StratifyResult:
    thresholds: Tuple[float, float]
    groups: Dict[str, List[RunSummary]]  # {"low":..., "mid":..., "high":...}
    counts: Dict[str, int]


def _quantile(values: List[float], q: float) -> float:
    """
    Simple deterministic quantile without numpy dependency.
    q in [0,1].
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    # linear interpolation
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def stratify_by_bucket(
    results: List[RunSummary],
    coupling_scores: Dict[str, float],  # prompt_id -> coupling_index
    thresholds: Optional[Tuple[float, float]] = None,
    quantiles: Tuple[float, float] = (0.33, 0.66),
) -> StratifyResult:
    """
    Stratify results into low/mid/high coupling buckets.

    Args:
        results: list of RunSummary
        coupling_scores: mapping prompt_id -> coupling_index
        thresholds: explicit (low_th, high_th). If provided, quantiles are ignored.
        quantiles: used when thresholds is None, default (0.33, 0.66)

    Returns:
        StratifyResult with thresholds used and grouped results.
    """
    # gather coupling values for prompts that appear in results
    vals: List[float] = []
    for r in results:
        if r.prompt_id in coupling_scores:
            vals.append(float(coupling_scores[r.prompt_id]))

    if thresholds is None:
        q1, q2 = quantiles
        low_th = _quantile(vals, q1) if vals else 0.0
        high_th = _quantile(vals, q2) if vals else 0.0
        thresholds = (low_th, high_th)

    low_th, high_th = thresholds

    low: List[RunSummary] = []
    mid: List[RunSummary] = []
    high: List[RunSummary] = []

    for r in results:
        ci = float(coupling_scores.get(r.prompt_id, 0.0))
        if ci < low_th:
            low.append(r)
        elif ci < high_th:
            mid.append(r)
        else:
            high.append(r)

    groups = {"low": low, "mid": mid, "high": high}
    counts = {k: len(v) for k, v in groups.items()}

    return StratifyResult(thresholds=(float(low_th), float(high_th)), groups=groups, counts=counts)
