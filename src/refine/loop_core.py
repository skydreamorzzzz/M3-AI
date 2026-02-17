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
    quality_threshold: float = 0.8  # Minimum quality score to stop (0-1)


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
    out_dir: Optional[Any] = None,
) -> Tuple[ArtifactHandle, List[TraceStep], RunSummary]:

    params = params or LoopParams()

    # Set default out_dir
    if out_dir is None:
        from pathlib import Path
        out_dir = Path("runs") / item.prompt_id

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
    # Initial global evaluation (constraints + quality)
    # ============================================================

    status_best = checker.check_all(
        prompt_text=item.text,
        artifact=best,
        constraints=constraints,
    )
    
    # Score initial quality
    quality_best = 0.0
    if hasattr(checker.backend, "score_quality"):
        try:
            quality_result = checker.backend.score_quality(
                prompt_text=item.text,
                artifact=best,
            )
            quality_best = float(quality_result.get("quality_score", 0.0))
        except Exception:
            quality_best = 0.0  # Fallback if scoring fails

    # ============================================================
    # Iterative refinement (quality-aware)
    # ============================================================

    for t in range(params.max_rounds):

        # Check stopping conditions
        all_pass = _all_pass(status_best)
        
        # Stop if: all constraints pass AND quality is good
        if params.early_stop and all_pass and quality_best >= params.quality_threshold:
            break

        if conflict_count >= params.max_conflict_count:
            break

        status_before = copy.deepcopy(status_best)
        quality_before = quality_best
        
        # ============================================================
        # Decide what to edit
        # ============================================================
        
        is_quality_round = False
        edit_instruction = ""
        selected = ""
        
        if not all_pass:
            # Case A: Some constraints failed → fix failed constraints
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
            
            # Generate constraint-fixing instruction
            from src.refine.checker import _mk_instruction
            edit_instruction = _mk_instruction(constraint)
            is_quality_round = False
        
        else:
            # Case B: All constraints pass, but quality is low → improve quality
            if quality_best < params.quality_threshold:
                selected = "__quality__"  # Special marker for quality improvement
                
                # Generate quality improvement instruction
                edit_instruction = (
                    "Improve the overall quality of this image: "
                    "enhance composition, refine details, improve lighting and colors. "
                    "Keep all existing content intact, only improve visual quality."
                )
                is_quality_round = True
            else:
                # Should not reach here (early_stop would have triggered)
                break

        # ============================================================
        # Apply edit
        # ============================================================

        candidate = editor.edit(
            artifact=best,
            instruction=edit_instruction,
            round_id=t,
            prompt_id=item.prompt_id,
            out_dir=out_dir,
        )

        # ============================================================
        # Global evaluation of candidate (constraints + quality)
        # ============================================================

        status_candidate = checker.check_all(
            prompt_text=item.text,
            artifact=candidate,
            constraints=constraints,
        )
        
        # Score candidate quality
        quality_candidate = 0.0
        if hasattr(checker.backend, "score_quality"):
            try:
                quality_result = checker.backend.score_quality(
                    prompt_text=item.text,
                    artifact=candidate,
                )
                quality_candidate = float(quality_result.get("quality_score", 0.0))
            except Exception:
                quality_candidate = quality_best  # Fallback to previous score

        # ============================================================
        # Verifier decision (enhanced with quality awareness)
        # ============================================================

        # Simple quality-aware decision (bypass verifier if quality clearly improves)
        n_pass_before = sum(1 for v in status_before.values() if v)
        n_pass_after = sum(1 for v in status_candidate.values() if v)
        
        accepted = False
        
        # Accept if: more constraints pass OR (same constraints + better quality)
        if n_pass_after > n_pass_before:
            accepted = True
        elif n_pass_after == n_pass_before and quality_candidate > quality_before:
            accepted = True
        elif n_pass_after == n_pass_before and quality_candidate == quality_before:
            # Use verifier for tie-breaking
            decision: Decision = verifier.verify(
                prompt_text=item.text,
                best_artifact=best,
                candidate_artifact=candidate,
                status_best=status_best,
                status_candidate=status_candidate,
                extra={"selected": selected, "round": t, "quality_before": quality_before, "quality_after": quality_candidate},
            )
            if decision == "better" or (decision == "same" and params.accept_on_same):
                accepted = True
        
        if accepted:
            best = candidate
            status_best = status_candidate
            quality_best = quality_candidate
        else:
            conflict_count += 1

        # ============================================================
        # Conflict statistics (based on global delta)
        # ============================================================

        degraded, improved = _diff_status(status_before, status_best)

        for cid in degraded:
            if cid != selected:
                cm.record_conflict(selected, cid)

        # Extract error info from candidate artifact (if editor fallback occurred)
        error_type = None
        edit_fallback = False
        if hasattr(candidate, "meta") and isinstance(candidate.meta, dict):
            error_type = candidate.meta.get("error_type")
            edit_fallback = candidate.meta.get("fallback", False)

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
                error_type=error_type,
                edit_fallback=edit_fallback,
                quality_score_before=quality_before,
                quality_score_after=quality_best if accepted else quality_before,
                quality_improvement=is_quality_round,
            )
        )

    # ============================================================
    # Summary
    # ============================================================

    final_pass = _all_pass(status_best)
    quality_improved = any(t.quality_improvement for t in traces)

    summary = RunSummary(
        prompt_id=item.prompt_id,
        total_rounds=len(traces),
        final_pass=bool(final_pass),
        conflict_count=int(conflict_count),
        oscillation_detected=False,
        protection_rate=1.0,
        final_quality_score=quality_best,
        quality_improved=quality_improved,
    )

    return best, traces, summary
