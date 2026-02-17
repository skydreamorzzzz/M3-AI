# -*- coding: utf-8 -*-
"""
src/refine/loop_core.py

Refinement loop (global-check_all version).

Design:
- check_all(): authoritative global evaluation
- check_one(): only generates edit instruction for selected constraint
- conflict statistics updated from global status delta
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import copy

from src.io.schemas import PromptItem, Constraint, TraceStep, RunSummary, ConstraintGraph
from src.refine.checker import Checker
from src.refine.editor import Editor, ArtifactHandle
from src.refine.verifier import Verifier, Decision
from src.scheduler.conflict_matrix import ConflictMatrix


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class LoopParams:
    max_rounds: int = 8
    early_stop: bool = True
    accept_on_same: bool = False
    max_conflict_count: int = 5


# ============================================================
# Utilities
# ============================================================

def _constraint_map(constraints: List[Constraint]) -> Dict[str, Constraint]:
    return {c.id: c for c in constraints}


def _all_pass(status: Dict[str, bool]) -> bool:
    return all(bool(v) for v in status.values()) if status else True


def _diff_status(before: Dict[str, bool], after: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    degraded: List[str] = []
    improved: List[str] = []
    for cid, ok_before in before.items():
        ok_after = bool(after.get(cid, False))
        if ok_before and not ok_after:
            degraded.append(cid)
        if not ok_before and ok_after:
            improved.append(cid)
    return degraded, improved


# ============================================================
# Main loop
# ============================================================

def run_refine_loop(
    item: PromptItem,
    scheduler: Any,
    checker: Checker,
    editor: Editor,
    verifier: Verifier,
    params: Optional[LoopParams] = None,
    initial_artifact: Optional[ArtifactHandle] = None,
    conflict_matrix: Optional[ConflictMatrix] = None,
) -> Tuple[ArtifactHandle, List[TraceStep], RunSummary]:

    params = params or LoopParams()

    constraints = item.constraints or []
    cmap = _constraint_map(constraints)

    graph: ConstraintGraph
    if item.graph is not None:
        graph = item.graph
    else:
        graph = ConstraintGraph(nodes=constraints, edges=[])

    best = initial_artifact or ArtifactHandle(
        payload=f"artifact://{item.prompt_id}/init",
        meta={"prompt_id": item.prompt_id},
    )

    cm = conflict_matrix or ConflictMatrix()
    traces: List[TraceStep] = []
    conflict_count = 0

    # ============================================================
    # Initial global evaluation
    # ============================================================

    status_best = checker.check_all(
        prompt_text=item.text,
        artifact=best,
        constraints=constraints,
    )

    # ============================================================
    # Iterative refinement
    # ============================================================

    for t in range(params.max_rounds):

        if params.early_stop and _all_pass(status_best):
            break

        if conflict_count >= params.max_conflict_count:
            break

        # conflict risk from matrix
        conflict_risk = cm.export_dict()

        order = scheduler.schedule(
            graph=graph,
            status=status_best,
            conflict_risk=conflict_risk,
        )

        if not order:
            break

        selected = order[0]
        if selected not in cmap:
            continue

        constraint = cmap[selected]
        status_before = copy.deepcopy(status_best)

        # ============================================================
        # Generate edit instruction (local)
        # ============================================================

        local_res = checker.check_one(
            prompt_text=item.text,
            artifact=best,
            constraint=constraint,
        )

        if local_res.passed:
            # already satisfied — skip edit
            continue

        candidate = editor.edit(best, local_res.edit_instruction)

        # ============================================================
        # Global evaluation of candidate
        # ============================================================

        status_candidate = checker.check_all(
            prompt_text=item.text,
            artifact=candidate,
            constraints=constraints,
        )

        # ============================================================
        # Verifier decision
        # ============================================================

        decision: Decision = verifier.verify(
            prompt_text=item.text,
            best_artifact=best,
            candidate_artifact=candidate,
            status_best=status_best,
            status_candidate=status_candidate,
            extra={"selected": selected, "round": t},
        )

        accepted = False

        if decision == "better" or (decision == "same" and params.accept_on_same):
            accepted = True
            best = candidate
            status_best = status_candidate
        else:
            conflict_count += 1

        # ============================================================
        # Conflict statistics (based on global delta)
        # ============================================================

        degraded, improved = _diff_status(status_before, status_best)

        for cid in degraded:
            if cid != selected:
                cm.record_conflict(selected, cid)

        traces.append(
            TraceStep(
                round_id=t,
                selected_constraint=selected,
                status_before=status_before,
                status_after=copy.deepcopy(status_best),
                degraded_constraints=degraded,
                improved_constraints=improved,
                edit_instruction=local_res.edit_instruction,
                accepted=accepted,
            )
        )

    # ============================================================
    # Summary
    # ============================================================

    final_pass = _all_pass(status_best)

    summary = RunSummary(
        prompt_id=item.prompt_id,
        total_rounds=len(traces),
        final_pass=bool(final_pass),
        conflict_count=int(conflict_count),
        oscillation_detected=False,
        protection_rate=1.0,
    )

    return best, traces, summary
