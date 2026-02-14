# -*- coding: utf-8 -*-
"""
src/refine/checker.py

Checker (+ implicit Refiner) for the refinement loop.

Role mapping (per your 5-agent spec):
- Checker: evaluate whether the current artifact satisfies ONE constraint.
- Refiner (implicit): when failed, produce a concrete, executable edit instruction
  (returned as part of the checker result).

This module is intentionally model-agnostic:
- It does NOT require a VLM/LLM to run.
- You can plug in a real checker backend later (e.g., VLM judge) by implementing
  the `JudgeBackend` interface.

Inputs:
- prompt_text: original user prompt (string)
- artifact: "current image/video" handle (path/URI/ID). We treat it as opaque.
- constraint: Constraint (from src/io/schemas.py)

Output:
- CheckResult: passed(bool), reason(str), edit_instruction(Optional[str]), confidence(float)

Deterministic default behavior:
- If no backend is provided, we can run in "placeholder" mode:
  - If an oracle_status dict is provided, use it as truth.
  - Otherwise always FAIL with a best-effort edit instruction template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Any

from src.io.schemas import Constraint, ConstraintType


# ============================================================
# Result schema (local to refine layer)
# ============================================================

@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str
    edit_instruction: Optional[str] = None
    confidence: float = 1.0


# ============================================================
# Backend protocol (pluggable judge)
# ============================================================

class JudgeBackend(Protocol):
    """
    A pluggable judge that can actually look at the artifact (image/video) and decide.
    Implementations could be:
      - VLM judge (prompt + image -> pass/fail + instruction)
      - deterministic rule judge (for synthetic benchmarks)
      - external evaluator

    Must be pure w.r.t. inputs (recommended) and should be robust.
    """

    def judge(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
    ) -> CheckResult:
        ...


# ============================================================
# Default instruction templates (implicit refiner)
# ============================================================

def _mk_instruction(constraint: Constraint) -> str:
    """
    Generate a concrete edit instruction for a failed constraint.
    This is a placeholder "refiner" logic. Later you can replace this with LLM-based
    instruction generation, but keep the output format stable.
    """
    obj = (constraint.object or "").strip()
    ref = (constraint.reference or "").strip()
    rel = (constraint.relation or "").strip()
    val = (constraint.value or "").strip()

    # Keep instructions short, imperative, and localized ("micro-edit" rather than re-generate).
    if constraint.type == ConstraintType.OBJECT:
        # OBJECT: ensure existence
        if obj:
            return f"Add a clearly visible {obj}. Keep existing correct elements unchanged."
        return "Add the missing object mentioned in the prompt. Keep existing correct elements unchanged."

    if constraint.type == ConstraintType.COUNT:
        # COUNT: adjust quantity
        if obj and val:
            return f"Adjust the number of {obj} to exactly {val}. Keep other objects unchanged."
        if obj:
            return f"Adjust the number of {obj} to match the prompt. Keep other objects unchanged."
        return "Adjust object counts to match the prompt. Keep other objects unchanged."

    if constraint.type == ConstraintType.ATTRIBUTE:
        # ATTRIBUTE: color/style/attribute, stored in value
        if obj and val:
            return f"Modify the {obj} so that it has the attribute: {val}. Preserve all other correct details."
        if obj:
            return f"Modify the {obj} attributes to match the prompt. Preserve all other correct details."
        return "Modify the incorrect attribute to match the prompt. Preserve all other correct details."

    if constraint.type == ConstraintType.SPATIAL:
        # SPATIAL: relation + reference
        if obj and rel and ref:
            return f"Reposition {obj} so it is {rel} {ref}. Do not change object identities or counts."
        if obj and ref:
            return f"Reposition {obj} to satisfy the spatial relation with {ref} as described in the prompt."
        if obj:
            return f"Reposition {obj} to satisfy the spatial layout described in the prompt."
        return "Adjust spatial layout to satisfy the prompt. Avoid changing object identities or counts."

    if constraint.type == ConstraintType.RELATION:
        # RELATION: often semantic relation; use relation field + reference
        if obj and rel and ref:
            return f"Edit the scene so that {obj} is {rel} {ref}. Keep other constraints intact."
        if obj and ref:
            return f"Edit the scene so that {obj} has the correct relation with {ref} as described in the prompt."
        if obj:
            return f"Edit the scene so that the relation involving {obj} matches the prompt."
        return "Edit the relational interaction to match the prompt, without breaking other correct elements."

    if constraint.type == ConstraintType.TEXT:
        # TEXT: value is the text content (or a spec)
        if val:
            return f"Correct the rendered text to exactly: “{val}”. Keep typography/style consistent with the scene."
        return "Correct the rendered text to match the prompt. Keep typography/style consistent with the scene."

    # Fallback
    return "Fix this constraint according to the prompt while preserving all other correct elements."


# ============================================================
# Main Checker implementation
# ============================================================

@dataclass(frozen=True)
class CheckerParams:
    """
    Knobs:
    - require_confidence: if constraint.confidence < require_confidence, we can choose to skip or soften.
    - low_confidence_policy:
        - "treat_as_fail": still check and produce instruction
        - "skip": always pass (do not block loop)
        - "warn": pass but include reason
    - always_generate_instruction_on_fail: if True, produce edit_instruction whenever failed.
    """
    require_confidence: float = 0.0
    low_confidence_policy: str = "treat_as_fail"  # treat_as_fail | skip | warn
    always_generate_instruction_on_fail: bool = True


class Checker:
    """
    Checker (+ implicit Refiner).

    Usage:
        ck = Checker(params=CheckerParams(), backend=my_vlm_backend)
        res = ck.check_one(prompt_text, artifact, constraint, oracle_status=None)
    """

    def __init__(self, params: Optional[CheckerParams] = None, backend: Optional[JudgeBackend] = None) -> None:
        self.params = params or CheckerParams()
        self.backend = backend

    def check_one(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
        oracle_status: Optional[Dict[str, bool]] = None,
    ) -> CheckResult:
        """
        Evaluate ONE constraint.

        Priority order:
        1) Low-confidence handling policy (optional gating)
        2) If backend exists: delegate to backend.judge(...)
        3) Else if oracle_status provided: use oracle_status[constraint.id]
        4) Else: placeholder FAIL + template instruction

        Note:
        - oracle_status is extremely useful for unit tests / synthetic benchmarks.
        """
        # --- 1) low-confidence gating ---
        conf = float(getattr(constraint, "confidence", 1.0) or 1.0)
        if conf < float(self.params.require_confidence):
            pol = self.params.low_confidence_policy
            if pol == "skip":
                return CheckResult(
                    passed=True,
                    reason=f"Skipped low-confidence constraint (confidence={conf:.2f} < {self.params.require_confidence:.2f}).",
                    edit_instruction=None,
                    confidence=conf,
                )
            if pol == "warn":
                return CheckResult(
                    passed=True,
                    reason=f"Low-confidence constraint treated as pass with warning (confidence={conf:.2f}).",
                    edit_instruction=None,
                    confidence=conf,
                )
            # default: treat_as_fail continues below

        # --- 2) real backend judge ---
        if self.backend is not None:
            res = self.backend.judge(prompt_text=prompt_text, artifact=artifact, constraint=constraint)
            # Ensure instruction exists on fail if configured
            if (not res.passed) and self.params.always_generate_instruction_on_fail and not res.edit_instruction:
                instr = _mk_instruction(constraint)
                return CheckResult(
                    passed=False,
                    reason=res.reason or "Constraint failed.",
                    edit_instruction=instr,
                    confidence=float(res.confidence),
                )
            return res

        # --- 3) oracle status ---
        if oracle_status is not None and constraint.id in oracle_status:
            passed = bool(oracle_status[constraint.id])
            if passed:
                return CheckResult(passed=True, reason="Constraint passed (oracle).", edit_instruction=None, confidence=conf)
            instr = _mk_instruction(constraint) if self.params.always_generate_instruction_on_fail else None
            return CheckResult(
                passed=False,
                reason="Constraint failed (oracle).",
                edit_instruction=instr,
                confidence=conf,
            )

        # --- 4) placeholder behavior (no vision) ---
        instr = _mk_instruction(constraint) if self.params.always_generate_instruction_on_fail else None
        return CheckResult(
            passed=False,
            reason="Constraint failed (no judge backend configured).",
            edit_instruction=instr,
            confidence=conf,
        )
