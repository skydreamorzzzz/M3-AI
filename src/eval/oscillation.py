# -*- coding: utf-8 -*-
"""
src/eval/oscillation.py

Oscillation detection for refinement traces.

We detect two kinds of oscillation:
1) Self-oscillation (single constraint flips): A True->False->True or False->True->False.
2) Cross-oscillation (pairwise ping-pong): fixing A tends to degrade B, and fixing B tends to degrade A.

TraceStep fields used:
- selected_constraint: which constraint we attempted to fix in this step
- status_after: status dict after edit (for self-oscillation)
- degraded_constraints: list of constraints that got worse/broken due to this edit (for cross-oscillation)

This module is deterministic and does not depend on LLM/editor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from src.io.schemas import TraceStep


@dataclass(frozen=True)
class OscillationResult:
    has_oscillation: bool
    self_oscillation: bool
    cross_oscillation: bool
    # Most suspicious cross pairs (A,B) where A->B and B->A both appear
    top_pairs: List[Tuple[str, str]]
    # Evidence counts: (A,B) -> times we observed "fix A degraded B"
    pair_counts: Dict[Tuple[str, str], int]


def _detect_self_oscillation(trace: List[TraceStep]) -> bool:
    """
    Detect A status flip pattern within the history of selections.

    We record the status_after[selected_constraint] along the times it was selected.
    If we see x != y and x == z in consecutive triple, we call it self-oscillation.
    """
    history: Dict[str, List[bool]] = defaultdict(list)

    for step in trace:
        cid = step.selected_constraint
        # If missing, treat as False (failed) conservatively
        st = bool(step.status_after.get(cid, False))
        history[cid].append(st)

    for states in history.values():
        if len(states) < 3:
            continue
        for i in range(len(states) - 2):
            a, b, c = states[i], states[i + 1], states[i + 2]
            if a != b and a == c:
                return True
    return False


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical undirected key for pair ranking (A,B) == (B,A) in top_pairs."""
    return (a, b) if a <= b else (b, a)


def _detect_cross_oscillation(
    trace: List[TraceStep],
    min_pair_support: int = 1,
) -> Tuple[bool, Dict[Tuple[str, str], int], List[Tuple[str, str]]]:
    """
    Detect cross-oscillation via degraded_constraints.

    We count directed events:
        (A -> B) means: when we selected/fixed A, B appeared in degraded_constraints.

    Cross-oscillation exists if there is at least one pair (A,B) such that:
        count(A->B) >= min_pair_support AND count(B->A) >= min_pair_support

    Return:
      has_cross, directed_counts, top_pairs (undirected) sorted by mutual support.
    """
    directed = Counter()

    for step in trace:
        a = step.selected_constraint
        for b in (step.degraded_constraints or []):
            if not b or b == a:
                continue
            directed[(a, b)] += 1

    # Find mutual pairs
    mutual_undirected_score: Dict[Tuple[str, str], int] = {}
    has_cross = False

    for (a, b), cab in directed.items():
        cba = directed.get((b, a), 0)
        if cab >= min_pair_support and cba >= min_pair_support:
            has_cross = True
            k = _pair_key(a, b)
            # mutual score = cab + cba
            mutual_undirected_score[k] = max(mutual_undirected_score.get(k, 0), cab + cba)

    # Rank top pairs by mutual score desc, then lexicographically
    top_pairs = sorted(mutual_undirected_score.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    top_pairs = [p for p, _ in top_pairs[:10]]

    return has_cross, dict(directed), top_pairs


def detect_oscillation(
    trace: List[TraceStep],
    min_pair_support: int = 1,
) -> OscillationResult:
    """
    Unified oscillation detector.

    Args:
        trace: list of TraceStep
        min_pair_support: threshold for cross-oscillation mutual pair support

    Returns:
        OscillationResult with evidence.
    """
    self_osc = _detect_self_oscillation(trace)
    cross_osc, pair_counts, top_pairs = _detect_cross_oscillation(trace, min_pair_support=min_pair_support)

    return OscillationResult(
        has_oscillation=bool(self_osc or cross_osc),
        self_oscillation=self_osc,
        cross_oscillation=cross_osc,
        top_pairs=top_pairs,
        pair_counts=pair_counts,
    )
