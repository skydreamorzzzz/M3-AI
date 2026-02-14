# -*- coding: utf-8 -*-
"""
src/scheduler/conflict_aware.py

Conflict-aware scoring helpers.

Core idea:
  score = w_out_degree * out_degree + w_coupling_degree * coupling_degree - alpha_conflict * conflict_risk
"""

from __future__ import annotations

from typing import Dict


def score_one(
    cid: str,
    out_degree: Dict[str, int],
    coupling_degree: Dict[str, int],
    conflict_risk: Dict[str, float],
    w_out_degree: float,
    w_coupling_degree: float,
    alpha_conflict: float,
) -> float:
    structural = (
        w_out_degree * float(out_degree.get(cid, 0))
        + w_coupling_degree * float(coupling_degree.get(cid, 0))
    )
    risk = float(conflict_risk.get(cid, 0.0))
    return structural - alpha_conflict * risk


def compute_conflict_scores(
    out_degree: Dict[str, int],
    coupling_degree: Dict[str, int],
    conflict_risk: Dict[str, float],
    w_out_degree: float = 1.0,
    w_coupling_degree: float = 0.3,
    alpha_conflict: float = 0.5,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    # union keys to be robust
    keys = set(out_degree.keys()) | set(coupling_degree.keys()) | set(conflict_risk.keys())
    for cid in keys:
        scores[cid] = score_one(
            cid,
            out_degree=out_degree,
            coupling_degree=coupling_degree,
            conflict_risk=conflict_risk,
            w_out_degree=w_out_degree,
            w_coupling_degree=w_coupling_degree,
            alpha_conflict=alpha_conflict,
        )
    return scores
