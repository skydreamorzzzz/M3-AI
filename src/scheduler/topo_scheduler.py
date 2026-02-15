# -*- coding: utf-8 -*-
"""
src/scheduler/topo_scheduler.py

TopoScheduler: schedule constraints by
- using ONLY "dependency" edges for topo order
- breaking cycles (safety)
- returning an ordered list of constraint ids

Contract (matches loop_core.py):
    schedule(graph, status, conflict_risk=None) -> List[str]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.graph.dag import normalize_edges, break_cycles, topo_sort


@dataclass(frozen=True)
class TopoSchedulerParams:
    prefer_unpassed: bool = True
    risk_weight_mode: str = "dst"  # "dst" or "avg"
    dependency_edge_type: str = "dependency"


class TopoScheduler:
    def __init__(self, params: Optional[TopoSchedulerParams] = None) -> None:
        self.params = params or TopoSchedulerParams()

    def schedule(
        self,
        graph: Any,
        status: Dict[str, bool],
        conflict_risk: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        node_ids = _get_node_ids(graph)
        if not node_ids:
            return []

        # ✅ IMPORTANT: only dependency edges participate in topo order
        dep_edges_raw = _get_dependency_edges(graph, self.params.dependency_edge_type)
        dep_edges = normalize_edges(dep_edges_raw)

        risk = conflict_risk or {}

        def edge_weight(u: str, v: str) -> float:
            # Higher weight => less likely removed
            ru = float(risk.get(u, 0.5))
            rv = float(risk.get(v, 0.5))
            if self.params.risk_weight_mode == "avg":
                return 0.5 * (ru + rv)
            return rv  # default: dst risk

        dag_edges, _removed = break_cycles(node_ids=node_ids, edges=dep_edges, edge_weight=edge_weight)

        order = topo_sort(node_ids=node_ids, edges=dag_edges)
        if order is None:
            # Fallback: just use node order
            order = list(node_ids)

        if self.params.prefer_unpassed:
            unpassed = [cid for cid in order if not bool(status.get(cid, False))]
            passed = [cid for cid in order if bool(status.get(cid, False))]
            return unpassed + passed

        return order


def _get_node_ids(graph: Any) -> List[str]:
    nodes = getattr(graph, "nodes", None)
    if not nodes:
        return []
    out: List[str] = []
    for n in nodes:
        if n is None:
            continue
        if isinstance(n, str):
            out.append(n)
        elif hasattr(n, "id"):
            out.append(str(getattr(n, "id")))
        else:
            out.append(str(n))
    return out


def _get_dependency_edges(graph: Any, dep_type: str) -> List[Any]:
    """
    Return a list of edges that represent hard dependencies.
    Supports:
    - graph.edges is already list[(u,v)] -> treat as dependency edges
    - graph.edges list[GraphEdge] -> filter by edge_type == dep_type
    """
    edges = getattr(graph, "edges", None)
    if not edges:
        return []

    # if edges are tuples already, assume they are dependency edges
    if isinstance(edges, list) and edges and isinstance(edges[0], (tuple, list)) and len(edges[0]) == 2:
        return edges

    dep: List[Any] = []
    if isinstance(edges, list):
        for e in edges:
            et = getattr(e, "edge_type", None)
            # dict style
            if isinstance(e, dict):
                et = e.get("edge_type", et)
            if et is None:
                # Unknown typed edge -> be conservative: DO NOT include
                # (avoids pulling coupling into topo by mistake)
                continue
            if str(et) == dep_type:
                dep.append(e)
    return dep
