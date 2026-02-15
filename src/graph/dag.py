# -*- coding: utf-8 -*-
"""
src/graph/dag.py

Local graph utilities:
- normalize edges from various schema representations
- break cycles (turn a directed graph into a DAG)
- topological sort (Kahn)

Compatibility:
- supports edges as (u,v) tuples
- supports dict edges: {"src","dst"} / {"u","v"} / {"from","to"} / {"source","target"}
- supports dataclass/object edges with attributes:
  (src,dst) / (u,v) / (from,to) / (source,target) / (parent,child)
"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, deque


Edge = Tuple[str, str]
EdgeWeightFn = Callable[[str, str], float]


# ============================================================
# Edge normalization
# ============================================================

def _edge_from_obj(e: Any) -> Optional[Edge]:
    """Best-effort parse of an edge into (src, dst)."""
    if e is None:
        return None

    # Common: tuple/list of length 2
    if isinstance(e, (tuple, list)) and len(e) == 2:
        a, b = e[0], e[1]
        if a is None or b is None:
            return None
        return str(a), str(b)

    # Dict styles
    if isinstance(e, dict):
        for k1, k2 in [
            ("src", "dst"),
            ("u", "v"),
            ("from", "to"),
            ("parent", "child"),
            ("source", "target"),  # ✅ support GraphEdge-like dicts
        ]:
            if k1 in e and k2 in e:
                return str(e[k1]), str(e[k2])
        return None

    # Dataclass / object with attributes
    if is_dataclass(e) or hasattr(e, "__dict__"):
        for a1, a2 in [
            ("src", "dst"),
            ("u", "v"),
            ("from_", "to"),
            ("from", "to"),
            ("parent", "child"),
            ("source", "target"),  # ✅ support your GraphEdge(source,target)
        ]:
            if hasattr(e, a1) and hasattr(e, a2):
                return str(getattr(e, a1)), str(getattr(e, a2))
        return None

    return None


def normalize_edges(edges: Any) -> List[Edge]:
    """
    Normalize edges to List[(src, dst)].

    - Filters self-loops and invalid edges
    - Deduplicates, stable order
    """
    if edges is None:
        return []

    out: List[Edge] = []
    if isinstance(edges, (list, tuple)):
        for e in edges:
            p = _edge_from_obj(e)
            if p is None:
                continue
            u, v = p
            if not u or not v:
                continue
            if u == v:
                continue
            out.append((u, v))
    else:
        p = _edge_from_obj(edges)
        if p is not None:
            u, v = p
            if u and v and u != v:
                out.append((u, v))

    seen = set()
    dedup: List[Edge] = []
    for u, v in out:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        dedup.append((u, v))
    return dedup


# ============================================================
# Cycle breaking
# ============================================================

def break_cycles(
    node_ids: Iterable[str],
    edges: List[Edge],
    edge_weight: Optional[EdgeWeightFn] = None,
) -> Tuple[List[Edge], List[Edge]]:
    """
    Turn a directed graph into a DAG by removing edges.

    Strategy:
    - Try topo sort; if cycle exists, find a cycle (DFS back-edge reconstruction)
    - Remove the lowest-weight edge on that cycle (or arbitrary if no weights)
    """
    nodes = [str(x) for x in node_ids]
    cur_edges = list(edges)
    removed: List[Edge] = []

    def w(u: str, v: str) -> float:
        if edge_weight is None:
            return 1.0
        try:
            return float(edge_weight(u, v))
        except Exception:
            return 1.0

    def build_adj(es: List[Edge]) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = defaultdict(list)
        for u, v in es:
            adj[u].append(v)
        return adj

    def find_cycle_edges(es: List[Edge]) -> Optional[List[Edge]]:
        adj = build_adj(es)
        color: Dict[str, int] = {n: 0 for n in nodes}  # 0=unseen,1=visiting,2=done
        parent: Dict[str, str] = {}

        def dfs(u: str) -> Optional[List[Edge]]:
            color[u] = 1
            for v in adj.get(u, []):
                if v not in color:
                    continue
                if color[v] == 0:
                    parent[v] = u
                    cyc = dfs(v)
                    if cyc is not None:
                        return cyc
                elif color[v] == 1:
                    # back-edge u->v
                    path_nodes = [u]
                    cur = u
                    while cur != v and cur in parent:
                        cur = parent[cur]
                        path_nodes.append(cur)
                    if cur != v:
                        return [(u, v)]
                    path_nodes.reverse()  # v ... u
                    cyc_edges: List[Edge] = []
                    for i in range(len(path_nodes) - 1):
                        cyc_edges.append((path_nodes[i], path_nodes[i + 1]))
                    cyc_edges.append((u, v))
                    return cyc_edges
            color[u] = 2
            return None

        for n in nodes:
            if color.get(n, 0) == 0:
                cyc = dfs(n)
                if cyc is not None:
                    return cyc
        return None

    while True:
        order = topo_sort(nodes, cur_edges, return_partial=True)
        if order is not None and len(order) == len(nodes):
            break

        cyc = find_cycle_edges(cur_edges)
        if not cyc:
            if cur_edges:
                removed.append(cur_edges.pop())
                continue
            break

        e_min = min(cyc, key=lambda e: w(e[0], e[1]))
        if e_min in cur_edges:
            cur_edges.remove(e_min)
            removed.append(e_min)
        else:
            cyc_set = set(cyc)
            removed_any = False
            for e in list(cur_edges):
                if e in cyc_set:
                    cur_edges.remove(e)
                    removed.append(e)
                    removed_any = True
                    break
            if not removed_any and cur_edges:
                removed.append(cur_edges.pop())

    return cur_edges, removed


# ============================================================
# Topological sort
# ============================================================

def topo_sort(
    node_ids: Iterable[str],
    edges: List[Edge],
    *,
    return_partial: bool = False,
) -> Optional[List[str]]:
    """
    Kahn topo sort.
    Returns None if cycle (unless return_partial=True).
    """
    nodes = [str(x) for x in node_ids]
    node_set = set(nodes)

    indeg: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, List[str]] = defaultdict(list)

    for u, v in edges:
        if u not in node_set or v not in node_set:
            continue
        adj[u].append(v)
        indeg[v] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    order: List[str] = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        return order if return_partial else None
    return order
