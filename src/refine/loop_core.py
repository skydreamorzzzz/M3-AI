# -*- coding: utf-8 -*-
"""
src/refine/loop_core.py

Core refinement loop (orchestrator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import copy

from src.io.schemas import PromptItem, Constraint, TraceStep, RunSummary, ConstraintGraph
from src.refine.checker import Checker, CheckResult
from src.refine.editor import Editor, ArtifactHandle
from src.refine.verifier import Verifier, Decision


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class LoopParams:
    max_rounds: int = 8
    early_stop: bool = True
    record_pass_steps: bool = True
    accept_on_same: bool = False


# ============================================================
# Utilities
# ============================================================

def _constraint_map(constraints: List[Constraint]) -> Dict[str, Constraint]:
    return {c.id: c for c in constraints}


def _init_status(constraints: List[Constraint], default_failed: bool = True) -> Dict[str, bool]:
    if default_failed:
        return {c.id: False for c in constraints}
    return {c.id: True for c in constraints}


def _all_pass(status: Dict[str, bool]) -> bool:
    return all(bool(v) for v in status.values()) if status else True


def _diff_status(before: Dict[str, bool], after: Dict[str, bool]) -> Tuple[List[str], List[str]]:
    degraded: List[str] = []
    improved: List[str] = []
    for cid, ok_before in before.items():
        ok_after = bool(after.get(cid, False))
        if bool(ok_before) and (not ok_after):
            degraded.append(cid)
        if (not bool(ok_before)) and ok_after:
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
    oracle_status_by_round: Optional[List[Dict[str, bool]]] = None,
    conflict_risk: Optional[Dict[str, float]] = None,
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

    # IMPORTANT: by default we assume all fail in skeleton mode
    status_best = _init_status(constraints, default_failed=True)

    traces: List[TraceStep] = []
    conflict_count = 0

    for t in range(params.max_rounds):
        if params.early_stop and _all_pass(status_best):
            break

        order = scheduler.schedule(graph=graph, status=status_best, conflict_risk=conflict_risk)
        if not order:
            break

        selected = order[0]
        if selected not in cmap:
            continue

        c = cmap[selected]
        status_before = copy.deepcopy(status_best)

        oracle = None
        if oracle_status_by_round is not None and t < len(oracle_status_by_round):
            oracle = oracle_status_by_round[t]

        # --- check ONE constraint on current best artifact ---
        res: CheckResult = checker.check_one(
            prompt_text=item.text,
            artifact=best,
            constraint=c,
            oracle_status=oracle,
        )

        # Build a "candidate status snapshot" for THIS round
        if oracle is not None:
            status_after_snapshot = copy.deepcopy(oracle)
        else:
            status_after_snapshot = copy.deepcopy(status_before)
            status_after_snapshot[selected] = bool(res.passed)

        edit_instruction: Optional[str] = None
        candidate = best
        accepted = True

        if res.passed:
            # ✅ FIX: when passed, we MUST update status_best
            status_best = status_after_snapshot
            accepted = True
            # best artifact unchanged
        else:
            edit_instruction = res.edit_instruction
            if edit_instruction:
                candidate = editor.edit(best, edit_instruction)
            else:
                candidate = best

            status_candidate = status_after_snapshot

            decision: Decision = verifier.verify(
                prompt_text=item.text,
                best_artifact=best,
                candidate_artifact=candidate,
                status_best=status_best,
                status_candidate=status_candidate,
                extra={"selected": selected, "round": t},
            )

            if decision == "better" or (decision == "same" and params.accept_on_same):
                accepted = True
                best = candidate
                status_best = status_candidate
            else:
                accepted = False
                conflict_count += 1
                # keep best + status_best unchanged

        if res.passed and (not params.record_pass_steps):
            continue

        degraded, improved = _diff_status(status_before, status_best)

        traces.append(
            TraceStep(
                round_id=t,
                selected_constraint=selected,
                status_before=status_before,
                status_after=copy.deepcopy(status_best),
                degraded_constraints=degraded,
                improved_constraints=improved,
                edit_instruction=edit_instruction,
                accepted=accepted,
            )
        )

    final_pass = _all_pass(status_best)
    total_rounds = len(traces)

    summary = RunSummary(
        prompt_id=item.prompt_id,
        total_rounds=total_rounds,
        final_pass=bool(final_pass),
        conflict_count=int(conflict_count),
        oscillation_detected=False,
        protection_rate=1.0,
    )

    return best, traces, summary
