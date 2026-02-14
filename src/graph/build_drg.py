# -*- coding: utf-8 -*-
"""
src/graph/build_drg.py

Build a Dependency-Relation Graph (DRG) from extracted constraints.

Design goals:
- Deterministic (no LLM here).
- Robust to minor extraction noise via normalization hooks.
- Edges are typed:
    - "dependency": hard prerequisite (should be respected by topo sort)
    - "coupling": soft coupling (optional; can be used for tie-break / coupling index)
- Graph is intended to be a DAG for dependency edges; coupling edges may create cycles
  but should NOT be used for topo sorting (unless you explicitly choose to).

Typical dependency rules:
- Any non-object constraint depends on its subject object existence.
- Any constraint with `reference` depends on reference object existence.
- Some hierarchical dependencies (optional):
    - TEXT content depends on TEXT region (if you model region separately)
    - RELATION depends on SPATIAL / OBJECT (configurable)

Input: List[Constraint]
Output: ConstraintGraph(nodes, edges)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.io.schemas import Constraint, ConstraintGraph, GraphEdge, ConstraintType


@dataclass(frozen=True)
class DRGParams:
    # Whether to add soft coupling edges between constraints sharing the same object
    add_object_coupling: bool = True
    # Whether to add soft coupling edges between constraints referencing the same reference object
    add_reference_coupling: bool = False

    # If True, add dependency edges between non-object constraints and their object-existence constraint
    add_object_dependencies: bool = True
    # If True, add dependency edges for reference object existence
    add_reference_dependencies: bool = True

    # Weighting (purely for analysis / tie-break; not required for topo)
    w_dependency: float = 1.0
    w_coupling: float = 0.3

    # If True, add an extra dependency: RELATION depends on COUNT for same object
    # (useful when prompts bind "two cats ... between ..." etc.)
    relation_depends_on_count: bool = True

    # If True, add an extra dependency: SPATIAL depends on COUNT for same object
    spatial_depends_on_count: bool = True


def _norm_obj(s: Optional[str]) -> Optional[str]:
    """Lightweight normalization for object names. Keep it conservative."""
    if s is None:
        return None
    s2 = s.strip().lower()
    # optional: collapse plurals very lightly (safe-ish for many prompts)
    if s2.endswith("s") and len(s2) > 3:
        # avoid turning "glass" -> "glas"
        if not s2.endswith("ss"):
            s2 = s2[:-1]
    return s2


def _index_object_constraints(constraints: List[Constraint]) -> Dict[str, str]:
    """
    Return mapping: object_name -> constraint_id for OBJECT existence constraint.
    If multiple OBJECT constraints exist for same object, prefer the first.
    """
    obj2cid: Dict[str, str] = {}
    for c in constraints:
        if c.type == ConstraintType.OBJECT:
            obj = _norm_obj(c.object)
            if obj and obj not in obj2cid:
                obj2cid[obj] = c.id
    return obj2cid


def _index_by_type_and_object(constraints: List[Constraint]) -> Dict[Tuple[str, str], List[str]]:
    """(type, object) -> [constraint_id, ...]"""
    mp: Dict[Tuple[str, str], List[str]] = {}
    for c in constraints:
        obj = _norm_obj(c.object) or ""
        key = (str(c.type.value), obj)
        mp.setdefault(key, []).append(c.id)
    return mp


def _add_edge(
    edges: List[GraphEdge],
    seen: Set[Tuple[str, str, str]],
    src: str,
    dst: str,
    edge_type: str,
    weight: float,
) -> None:
    if src == dst:
        return
    key = (src, dst, edge_type)
    if key in seen:
        return
    seen.add(key)
    edges.append(GraphEdge(source=src, target=dst, edge_type=edge_type, weight=weight))


def build_drg(constraints: List[Constraint], params: Optional[DRGParams] = None) -> ConstraintGraph:
    """
    Build DRG graph from constraints.

    Notes:
    - If the LLM didn't output OBJECT constraints, we do NOT invent them here.
      In that case, object-dependency edges will be skipped automatically.
    """
    params = params or DRGParams()

    # normalize objects in a local view (do not mutate input objects)
    obj2cid = _index_object_constraints(constraints)
    type_obj_index = _index_by_type_and_object(constraints)

    edges: List[GraphEdge] = []
    seen: Set[Tuple[str, str, str]] = set()

    # ---------- 1) Hard dependency edges ----------
    if params.add_object_dependencies:
        for c in constraints:
            if c.type == ConstraintType.OBJECT:
                continue
            subj = _norm_obj(c.object)
            if subj and subj in obj2cid:
                _add_edge(
                    edges, seen,
                    src=obj2cid[subj],
                    dst=c.id,
                    edge_type="dependency",
                    weight=params.w_dependency,
                )

    if params.add_reference_dependencies:
        for c in constraints:
            ref = _norm_obj(c.reference)
            if ref and ref in obj2cid:
                _add_edge(
                    edges, seen,
                    src=obj2cid[ref],
                    dst=c.id,
                    edge_type="dependency",
                    weight=params.w_dependency,
                )

    # Optional: relation/spatial depend on count for same object (helps avoid early layout moves before count settles)
    if params.relation_depends_on_count or params.spatial_depends_on_count:
        # find COUNT constraints by object
        count_by_obj: Dict[str, List[str]] = {}
        for c in constraints:
            if c.type == ConstraintType.COUNT:
                obj = _norm_obj(c.object)
                if obj:
                    count_by_obj.setdefault(obj, []).append(c.id)

        for c in constraints:
            obj = _norm_obj(c.object)
            if not obj:
                continue
            if obj not in count_by_obj:
                continue

            if params.relation_depends_on_count and c.type == ConstraintType.RELATION:
                for count_id in count_by_obj[obj]:
                    _add_edge(
                        edges, seen,
                        src=count_id,
                        dst=c.id,
                        edge_type="dependency",
                        weight=params.w_dependency,
                    )

            if params.spatial_depends_on_count and c.type == ConstraintType.SPATIAL:
                for count_id in count_by_obj[obj]:
                    _add_edge(
                        edges, seen,
                        src=count_id,
                        dst=c.id,
                        edge_type="dependency",
                        weight=params.w_dependency,
                    )

    # ---------- 2) Soft coupling edges (optional, not for topo sorting) ----------
    if params.add_object_coupling:
        # group constraints by subject object (excluding OBJECT itself)
        group: Dict[str, List[str]] = {}
        for c in constraints:
            if c.type == ConstraintType.OBJECT:
                continue
            obj = _norm_obj(c.object)
            if obj:
                group.setdefault(obj, []).append(c.id)

        # add undirected coupling as two directed edges
        for obj, cids in group.items():
            if len(cids) <= 1:
                continue
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = cids[i], cids[j]
                    _add_edge(edges, seen, a, b, "coupling", params.w_coupling)
                    _add_edge(edges, seen, b, a, "coupling", params.w_coupling)

    if params.add_reference_coupling:
        # group constraints by reference object
        ref_group: Dict[str, List[str]] = {}
        for c in constraints:
            ref = _norm_obj(c.reference)
            if ref:
                ref_group.setdefault(ref, []).append(c.id)

        for ref, cids in ref_group.items():
            if len(cids) <= 1:
                continue
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = cids[i], cids[j]
                    _add_edge(edges, seen, a, b, "coupling", params.w_coupling)
                    _add_edge(edges, seen, b, a, "coupling", params.w_coupling)

    # Return graph
    return ConstraintGraph(nodes=constraints, edges=edges)


# ---------------------------
# Optional helpers for analysis
# ---------------------------

def split_edges(graph: ConstraintGraph) -> Tuple[List[GraphEdge], List[GraphEdge]]:
    """Return (dependency_edges, coupling_edges)."""
    dep, coup = [], []
    for e in graph.edges:
        if e.edge_type == "dependency":
            dep.append(e)
        elif e.edge_type == "coupling":
            coup.append(e)
    return dep, coup


def coupling_index(graph: ConstraintGraph, lam: float = 0.5) -> float:
    """
    A simple coupling index for stratification:
        CI = |E_dep| + lam * |E_coupling|
    """
    dep, coup = split_edges(graph)
    return float(len(dep) + lam * len(coup))
