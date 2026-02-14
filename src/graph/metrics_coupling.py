# -*- coding: utf-8 -*-
"""
src/graph/metrics_coupling.py

Coupling / dependency metrics for DRG.

IMPORTANT:
- This module should be the single source of truth for coupling-related metrics,
  to avoid duplicated logic drifting across files.
"""

from __future__ import annotations

from typing import Tuple

from src.io.schemas import ConstraintGraph, GraphEdge


def split_edges(graph: ConstraintGraph) -> Tuple[int, int]:
    """
    Return (num_dependency_edges, num_coupling_edges).
    """
    dep = 0
    coup = 0
    for e in graph.edges:
        if e.edge_type == "dependency":
            dep += 1
        elif e.edge_type == "coupling":
            coup += 1
    return dep, coup


def coupling_index(graph: ConstraintGraph, lam: float = 0.5) -> float:
    """
    CI = |E_dep| + lam * |E_coupling|
    """
    dep, coup = split_edges(graph)
    return float(dep + lam * coup)


def dependency_density(graph: ConstraintGraph) -> float:
    """
    Density over directed possible edges: dep / (n*(n-1))
    """
    n = len(graph.nodes)
    if n <= 1:
        return 0.0
    dep, _ = split_edges(graph)
    return float(dep) / float(n * (n - 1))


def coupling_density(graph: ConstraintGraph) -> float:
    n = len(graph.nodes)
    if n <= 1:
        return 0.0
    _, coup = split_edges(graph)
    return float(coup) / float(n * (n - 1))
