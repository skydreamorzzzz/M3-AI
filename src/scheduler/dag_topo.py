# -*- coding: utf-8 -*-
"""
src/scheduler/dag_topo.py

Deterministic DAG-based scheduler for M3-style refinement.

Features:
- Uses dependency edges only for topo sorting
- Optional coupling-degree tie-break
- Optional conflict-risk penalty
- Robust fallback if DAG incomplete
- Fully deterministic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.io.schemas import ConstraintGraph, GraphEdge
from src.graph.validate_graph import ensure_dag


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class DagTopoParams:
    tie_break: str = "hybrid"  # none | degree | coupling | hybrid

    w_out_degree: float = 1.0
    w_coupling_degree: float = 0.3

    skip_passed: bool = True
    validate_and_fix_dag: bool = True
    failed_only: bool = True

    use_conflict_risk: bool = False
    alpha_conflict: float = 0.5


# ============================================================
# Scheduler
# ============================================================

class DagTopoScheduler:

    def __init__(self, params: Optional[DagTopoParams] = None) -> None:
        self.params = params or DagTopoParams()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def schedule(
        self,
        graph: ConstraintGraph,
        status: Dict[str, bool],
        conflict_risk: Optional[Dict[str, float]] = None,
    ) -> List[str]:

        g = graph

        if self.params.validate_and_fix_dag:
            g, _ = ensure_dag(g)

        dep_edges = [e for e in g.edges if e.edge_type == "dependency"]
        coup_edges = [e for e in g.edges if e.edge_type == "coupling"]

        nodes = [c.id for c in g.nodes]

        candidates = self._select_candidates(nodes, status)

        if not candidates:
            return []

        adj, indeg = self._build_dep_graph(dep_edges, candidates)

        out_deg = self._out_degree(adj)
        coup_deg = self._coupling_degree(coup_edges, candidates)

        order = self._kahn_topo(
            candidates=candidates,
            adj=adj,
            indeg=indeg,
            out_degree=out_deg,
            coupling_degree=coup_deg,
            conflict_risk=conflict_risk or {},
        )

        return order

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    def _select_candidates(self, nodes: List[str], status: Dict[str, bool]) -> List[str]:

        cand = []

        for cid in nodes:
            passed = bool(status.get(cid, False))

            if self.params.failed_only:
                if passed:
                    continue

            cand.append(cid)

        if self.params.skip_passed:
            cand = [cid for cid in cand if not bool(status.get(cid, False))]

        return cand

    # --------------------------------------------------------
    # Build dependency subgraph
    # --------------------------------------------------------

    def _build_dep_graph(
        self,
        dep_edges: List[GraphEdge],
        candidates: List[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, int]]:

        cand_set = set(candidates)

        adj: Dict[str, List[str]] = {cid: [] for cid in candidates}
        indeg: Dict[str, int] = {cid: 0 for cid in candidates}

        for e in dep_edges:
            if e.source in cand_set and e.target in cand_set:
                adj[e.source].append(e.target)
                indeg[e.target] += 1

        return adj, indeg

    def _out_degree(self, adj: Dict[str, List[str]]) -> Dict[str, int]:
        return {u: len(v) for u, v in adj.items()}

    def _coupling_degree(
        self,
        coup_edges: List[GraphEdge],
        candidates: List[str],
    ) -> Dict[str, int]:

        cand_set = set(candidates)
        deg = {cid: 0 for cid in candidates}

        for e in coup_edges:
            if e.source in cand_set and e.target in cand_set:
                deg[e.source] += 1

        return deg

    # --------------------------------------------------------
    # Topo sort
    # --------------------------------------------------------

    def _kahn_topo(
        self,
        candidates: List[str],
        adj: Dict[str, List[str]],
        indeg: Dict[str, int],
        out_degree: Dict[str, int],
        coupling_degree: Dict[str, int],
        conflict_risk: Dict[str, float],
    ) -> List[str]:

        Q = sorted([cid for cid in candidates if indeg[cid] == 0])
        order = []

        while Q:
            u = self._pop_best(
                Q,
                out_degree,
                coupling_degree,
                conflict_risk,
            )
            order.append(u)

            for v in adj.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    Q.append(v)

            Q.sort()  # keep deterministic

        if len(order) < len(candidates):
            remaining = [cid for cid in candidates if cid not in order]
            remaining.sort(
                key=lambda cid: (
                    -out_degree.get(cid, 0),
                    -coupling_degree.get(cid, 0),
                    conflict_risk.get(cid, 0.0),
                    cid,
                )
            )
            order.extend(remaining)

        return order

    # --------------------------------------------------------
    # Tie-break logic
    # --------------------------------------------------------

    def _pop_best(
        self,
        Q: List[str],
        out_degree: Dict[str, int],
        coupling_degree: Dict[str, int],
        conflict_risk: Dict[str, float],
    ) -> str:

        if self.params.tie_break == "none":
            return Q.pop(0)

        def score(cid: str) -> float:

            if self.params.tie_break == "degree":
                s = float(out_degree.get(cid, 0))

            elif self.params.tie_break == "coupling":
                s = float(coupling_degree.get(cid, 0))

            else:  # hybrid or fallback
                s = (
                    self.params.w_out_degree * float(out_degree.get(cid, 0))
                    + self.params.w_coupling_degree * float(coupling_degree.get(cid, 0))
                )

            if self.params.use_conflict_risk:
                s -= self.params.alpha_conflict * float(conflict_risk.get(cid, 0.0))

            return s

        best_idx = 0
        best_score = score(Q[0])

        for i in range(1, len(Q)):
            s_i = score(Q[i])
            if s_i > best_score:
                best_score = s_i
                best_idx = i
            elif s_i == best_score:
                if Q[i] < Q[best_idx]:  # lexical deterministic tie-break
                    best_idx = i

        return Q.pop(best_idx)
