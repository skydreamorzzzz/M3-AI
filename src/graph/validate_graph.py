# -*- coding: utf-8 -*-
"""
src/graph/validate_graph.py

Validate and fix cycles in dependency edges so that topo sort won't crash.

Key points:
- Only dependency edges must be DAG.
- Coupling edges are ignored for DAG validation (they may form cycles).
- If cycle exists, break it by removing "weak" dependency edges first, then lowest-weight edges.

Outputs:
- fixed ConstraintGraph
- ValidationReport (removed edges + reasons)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.io.schemas import ConstraintGraph, GraphEdge, ConstraintType


@dataclass
class RemovedEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    reason: str


@dataclass
class ValidationReport:
    had_cycle: bool
    removed: List[RemovedEdge] = field(default_factory=list)
    # number of iterations of cycle breaking
    break_rounds: int = 0

    def to_dict(self) -> dict:
        return {
            "had_cycle": self.had_cycle,
            "break_rounds": self.break_rounds,
            "removed": [r.__dict__ for r in self.removed],
        }


def _build_dep_adj(dep_edges: List[GraphEdge]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for e in dep_edges:
        adj.setdefault(e.source, []).append(e.target)
    return adj


def _all_nodes(graph: ConstraintGraph) -> Set[str]:
    return {c.id for c in graph.nodes}


def _dep_edges(graph: ConstraintGraph) -> List[GraphEdge]:
    return [e for e in graph.edges if e.edge_type == "dependency"]


def _coup_edges(graph: ConstraintGraph) -> List[GraphEdge]:
    return [e for e in graph.edges if e.edge_type != "dependency"]


def _find_one_cycle(dep_edges: List[GraphEdge]) -> Optional[List[str]]:
    """
    Find one directed cycle in dependency edges.
    Return list of node ids in the cycle path (e.g., [a,b,c,a]) if found else None.
    DFS-based.
    """
    adj = _build_dep_adj(dep_edges)
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    parent: Dict[str, str] = {}

    def dfs(u: str) -> Optional[List[str]]:
        visited.add(u)
        in_stack.add(u)
        for v in adj.get(u, []):
            if v not in visited:
                parent[v] = u
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif v in in_stack:
                # found back-edge u -> v, reconstruct cycle
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent.get(cur)
                    if cur is None:
                        break
                cycle.append(v)
                cycle.reverse()
                return cycle
        in_stack.remove(u)
        return None

    # ensure we traverse all nodes present in edges
    nodes = set()
    for e in dep_edges:
        nodes.add(e.source)
        nodes.add(e.target)

    for n in list(nodes):
        if n not in visited:
            parent[n] = n
            cyc = dfs(n)
            if cyc is not None:
                return cyc
    return None


def _edge_lookup(dep_edges: List[GraphEdge]) -> Dict[Tuple[str, str], GraphEdge]:
    mp: Dict[Tuple[str, str], GraphEdge] = {}
    for e in dep_edges:
        mp[(e.source, e.target)] = e
    return mp


def _node_type_map(graph: ConstraintGraph) -> Dict[str, str]:
    """
    constraint_id -> ConstraintType.value (string)
    """
    mp = {}
    for c in graph.nodes:
        mp[c.id] = c.type.value
    return mp


def _weak_dependency_score(edge: GraphEdge, node_type: Dict[str, str]) -> Tuple[int, float]:
    """
    Heuristic: return (weak_rank, weight) where higher weak_rank = more removable.
    We prefer removing edges that are likely "soft" dependencies:
      - edges into SPATIAL/RELATION/ATTRIBUTE are considered weaker than into COUNT/TEXT/OBJECT
      - lower weight is weaker
    """
    tgt_t = node_type.get(edge.target, "")
    # rank: larger => remove earlier
    if tgt_t in ("spatial", "relation", "attribute"):
        rank = 3
    elif tgt_t in ("count", "text"):
        rank = 2
    else:
        # object or unknown: should be quite strong; remove last
        rank = 1
    return (rank, edge.weight)


def _choose_edge_to_remove(cycle_nodes: List[str], dep_edges: List[GraphEdge], node_type: Dict[str, str]) -> GraphEdge:
    """
    Given a cycle node path like [a,b,c,a], select one edge on the cycle to remove.
    Strategy:
      1) maximize weak_rank (remove weakest target-type edges first)
      2) then minimize weight
    """
    lookup = _edge_lookup(dep_edges)
    cycle_edges: List[GraphEdge] = []
    for i in range(len(cycle_nodes) - 1):
        u, v = cycle_nodes[i], cycle_nodes[i + 1]
        e = lookup.get((u, v))
        if e is not None:
            cycle_edges.append(e)

    # fallback: if lookup fails (shouldn't), just remove the last dep edge
    if not cycle_edges:
        return dep_edges[-1]

    # sort by: weak_rank desc, weight asc
    cycle_edges.sort(
        key=lambda e: (-_weak_dependency_score(e, node_type)[0], _weak_dependency_score(e, node_type)[1])
    )
    return cycle_edges[0]


def ensure_dag(
    graph: ConstraintGraph,
    max_break_rounds: int = 20,
) -> Tuple[ConstraintGraph, ValidationReport]:
    """
    Ensure dependency edges form a DAG by iteratively breaking cycles.

    Returns:
      fixed_graph, report
    """
    dep = _dep_edges(graph)
    coup = _coup_edges(graph)
    node_type = _node_type_map(graph)

    report = ValidationReport(had_cycle=False)

    for k in range(max_break_rounds):
        cycle = _find_one_cycle(dep)
        if cycle is None:
            # already a DAG
            return ConstraintGraph(nodes=graph.nodes, edges=dep + coup), report

        report.had_cycle = True
        report.break_rounds += 1

        # choose one edge on that cycle to remove
        e_rm = _choose_edge_to_remove(cycle, dep, node_type)

        dep = [e for e in dep if not (e.source == e_rm.source and e.target == e_rm.target and e.edge_type == e_rm.edge_type)]
        report.removed.append(
            RemovedEdge(
                source=e_rm.source,
                target=e_rm.target,
                edge_type=e_rm.edge_type,
                weight=float(getattr(e_rm, "weight", 1.0)),
                reason=f"break_cycle: cycle={'->'.join(cycle)}; removed_edge={e_rm.source}->{e_rm.target}",
            )
        )

    # If still cyclic after max rounds, last resort: drop all dependency edges and rely on sequential
    # (do NOT crash batch runs)
    report.had_cycle = True
    report.removed.append(
        RemovedEdge(
            source="*",
            target="*",
            edge_type="dependency",
            weight=0.0,
            reason=f"max_break_rounds_exceeded={max_break_rounds}; dropped_all_dependency_edges_as_fallback",
        )
    )
    fixed = ConstraintGraph(nodes=graph.nodes, edges=coup)  # no dependency edges
    return fixed, report
