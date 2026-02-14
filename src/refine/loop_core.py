# -*- coding: utf-8 -*-
"""
src/refine/loop_core.py

Core refinement loop (orchestrator).

This module wires together:
- scheduler: chooses which constraint to address next
- checker (+ implicit refiner): checks ONE constraint, produces edit instruction on fail
- editor: applies edit instruction to produce a new candidate artifact
- verifier: decides whether to accept the candidate as the new "best"
- tracing: records TraceStep for each iteration
- summarization: produces RunSummary

Important:
- This loop is backend-agnostic. It can run in dry-run mode (no vision/editing)
  for debugging and ablation of scheduling logic.
- Artifact is handled via ArtifactHandle from src/refine/editor.py.

Inputs:
- PromptItem (prompt_id, text, constraints, graph)
- initial_artifact: optional ArtifactHandle; if not provided, will create a dummy handle
- scheduler must expose: schedule(graph, status, conflict_risk=None) -> List[str]
- checker must expose: check_one(prompt_text, artifact, constraint, oracle_status=None) -> CheckResult
- editor must expose: edit(artifact, edit_instruction) -> ArtifactHandle
- verifier must expose: verify(prompt_text, best_artifact, candidate_artifact, status_best, status_candidate, extra) -> Decision

Outputs:
- traces: List[TraceStep]
- summary: RunSummary
- best_artifact: ArtifactHandle

Notes about "status":
- status dict is our "current belief" pass/fail for each constraint id on the best artifact.
- In dry-run mode without a true judge, status can be driven by oracle_status injected by caller.
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
    """
    Knobs for the refinement loop.
    """
    max_rounds: int = 8
    # If True, stop early when all constraints pass
    early_stop: bool = True
    # If True, when checker says passed, we still record a trace step (recommended)
    record_pass_steps: bool = True
    # If True, accept verifier "same" as non-accept (keep best)
    accept_on_same: bool = False


# ============================================================
# Utilities
# ============================================================

def _constraint_map(constraints: List[Constraint]) -> Dict[str, Constraint]:
    return {c.id: c for c in constraints}


def _init_status(constraints: List[Constraint], default_failed: bool = True) -> Dict[str, bool]:
    """
    Initialize status dict. In absence of a real judge, we assume all fail by default.
    """
    if default_failed:
        return {c.id: False for c in constraints}
    return {c.id: True for c in constraints}


def _all_pass(status: Dict[str, bool]) -> bool:
    return all(bool(v) for v in status.values()) if status else True


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
    """
    Run the refinement loop for ONE prompt item.

    Args:
        item: PromptItem (must contain constraints; graph optional but recommended)
        scheduler: object exposing schedule(graph, status, conflict_risk=None)->List[str]
        checker: Checker instance
        editor: Editor instance
        verifier: Verifier instance
        params: LoopParams
        initial_artifact: starting artifact handle. If None, create a dummy.
        oracle_status_by_round: optional list where oracle_status_by_round[t] provides
            the "true" status dict after round t on the CURRENT artifact. Used for dry-run tests.
            If provided, we use it to:
              - drive checker decisions for selected constraints
              - update status_after snapshot
        conflict_risk: optional risk per constraint id (for conflict-aware schedulers)

    Returns:
        best_artifact, traces, summary
    """
    params = params or LoopParams()

    constraints = item.constraints or []
    cmap = _constraint_map(constraints)

    # Graph is required for DAG scheduler; if missing, build a trivial graph with no edges
    graph: ConstraintGraph
    if item.graph is not None:
        graph = item.graph
    else:
        graph = ConstraintGraph(nodes=constraints, edges=[])

    # initial artifact
    best = initial_artifact or ArtifactHandle(payload=f"artifact://{item.prompt_id}/init", meta={"prompt_id": item.prompt_id})

    # status belief on best
    status_best = _init_status(constraints, default_failed=True)

    traces: List[TraceStep] = []
    conflict_count = 0

    for t in range(params.max_rounds):
        # early stop if all pass
        if params.early_stop and _all_pass(status_best):
            break

        # scheduler proposes an order; pick the first available
        order = scheduler.schedule(graph=graph, status=status_best, conflict_risk=conflict_risk)
        if not order:
            # nothing to do
            break

        selected = order[0]
        if selected not in cmap:
            # scheduler produced unknown id; skip safely
            continue

        c = cmap[selected]

        status_before = copy.deepcopy(status_best)

        # Oracle injection for this round (optional)
        oracle = None
        if oracle_status_by_round is not None and t < len(oracle_status_by_round):
            oracle = oracle_status_by_round[t]

        # --- check ONE constraint ---
        res: CheckResult = checker.check_one(
            prompt_text=item.text,
            artifact=best,  # artifact handle is opaque; checker backend may use it
            constraint=c,
            oracle_status=oracle,
        )

        # If passed: update status_best for this constraint (and optionally snapshot others from oracle)
        status_after = copy.deepcopy(status_before)
        if oracle is not None:
            # oracle provides full snapshot after this round; use it as status_after
            status_after = copy.deepcopy(oracle)
        else:
            status_after[selected] = bool(res.passed)

        degraded: List[str] = []
        improved: List[str] = []

        # infer degraded/improved by comparing before/after
        for cid, ok_before in status_before.items():
            ok_after = bool(status_after.get(cid, False))
            if bool(ok_before) and (not ok_after):
                degraded.append(cid)
            if (not bool(ok_before)) and ok_after:
                improved.append(cid)

        accepted = True
        edit_instruction: Optional[str] = None
        candidate = best

        if res.passed:
            # No edit needed; best remains. We still record if enabled.
            accepted = True
        else:
            edit_instruction = res.edit_instruction
            if edit_instruction:
                candidate = editor.edit(best, edit_instruction)
            else:
                # no instruction -> can't edit
                candidate = best

            # Candidate status: if oracle provided, we already used oracle as status_after.
            # Otherwise, we only know selected constraint's status (still fail in placeholder).
            status_candidate = status_after

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
                # conflict: edit attempt failed to improve or regressed
                conflict_count += 1
                # keep best + status_best

        if (res.passed and not params.record_pass_steps):
            continue

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

    # Protection rate and oscillation detection belong to eval layer;
    # here we keep minimal summary fields.
    summary = RunSummary(
        prompt_id=item.prompt_id,
        total_rounds=total_rounds,
        final_pass=bool(final_pass),
        conflict_count=int(conflict_count),
        oscillation_detected=False,  # can be filled by eval later
        protection_rate=1.0,         # can be filled by eval later
    )

    return best, traces, summary
