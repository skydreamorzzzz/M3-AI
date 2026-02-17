# -*- coding: utf-8 -*-
"""
src/refine/checker.py

Upgraded Checker design:

Separation of concerns:
- check_all(): global evaluation (ALL constraints at once)
- check_one(): local evaluation + precise edit instruction generation

Design goal:
- check_all provides authoritative global status
- check_one only decides whether selected constraint is satisfied
  and produces edit instructions if needed

This supports:
- consistent global scoring
- dynamic scheduling
- conflict tracking
- cost-efficient evaluation (one vision call per round if backend supports batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Any, List

from src.io.schemas import Constraint, ConstraintType


# ============================================================
# Result schema
# ============================================================

@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str
    edit_instruction: Optional[str] = None
    confidence: float = 1.0


# ============================================================
# Backend protocol
# ============================================================

class JudgeBackend(Protocol):
    """
    Backend must support:
      - judge_all(): evaluate all constraints in one call
      - judge_one(): optional (fallback)

    judge_all should return:
        Dict[constraint_id, {"passed": bool, "confidence": float, "reason": str}]
    """

    def judge_all(
        self,
        prompt_text: str,
        artifact: Any,
        constraints: List[Constraint],
    ) -> Dict[str, Dict[str, Any]]:
        ...

    def judge_one(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
    ) -> Dict[str, Any]:
        ...


# ============================================================
# Instruction template (unchanged logic)
# ============================================================

def _mk_instruction(constraint: Constraint) -> str:

    obj = (constraint.object or "").strip()
    ref = (constraint.reference or "").strip()
    rel = (constraint.relation or "").strip()
    val = (constraint.value or "").strip()

    if constraint.type == ConstraintType.OBJECT:
        if obj:
            return f"Add a clearly visible {obj}. Keep existing correct elements unchanged."
        return "Add the missing object mentioned in the prompt."

    if constraint.type == ConstraintType.COUNT:
        if obj and val:
            return f"Adjust the number of {obj} to exactly {val}."
        if obj:
            return f"Adjust the number of {obj} to match the prompt."
        return "Adjust object counts to match the prompt."

    if constraint.type == ConstraintType.ATTRIBUTE:
        if obj and val:
            return f"Modify the {obj} so that it has the attribute: {val}."
        if obj:
            return f"Modify the {obj} attributes to match the prompt."
        return "Modify the incorrect attribute to match the prompt."

    if constraint.type == ConstraintType.SPATIAL:
        if obj and rel and ref:
            return f"Reposition {obj} so it is {rel} {ref}."
        return "Adjust spatial layout to satisfy the prompt."

    if constraint.type == ConstraintType.RELATION:
        if obj and rel and ref:
            return f"Edit the scene so that {obj} is {rel} {ref}."
        return "Fix the relational interaction."

    if constraint.type == ConstraintType.TEXT:
        if val:
            return f"Correct the rendered text to exactly: “{val}”."
        return "Correct the rendered text to match the prompt."

    return "Fix this constraint while preserving correct elements."


# ============================================================
# Checker
# ============================================================

class Checker:

    def __init__(self, backend: Optional[JudgeBackend] = None) -> None:
        self.backend = backend

    # --------------------------------------------------------
    # GLOBAL evaluation
    # --------------------------------------------------------

    def check_all(
        self,
        prompt_text: str,
        artifact: Any,
        constraints: List[Constraint],
        oracle_status: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, bool]:
        """
        Returns authoritative global status dict.

        HARDENING (Step-1 fix):
        - Never crash if backend fails / returns None / returns non-dict.
        - If backend output is malformed, treat as "all failed" (False) to keep loop running.
        """

        # Oracle shortcut
        if oracle_status is not None:
            return {c.id: bool(oracle_status.get(c.id, False)) for c in constraints}

        # Real backend
        if self.backend is not None:
            try:
                result = self.backend.judge_all(
                    prompt_text=prompt_text,
                    artifact=artifact,
                    constraints=constraints,
                )
            except Exception:
                # Backend crashed -> mark all failed (do not crash pipeline)
                return {c.id: False for c in constraints}

            # Defensive: backend must return a dict
            if not isinstance(result, dict):
                return {c.id: False for c in constraints}

            status: Dict[str, bool] = {}
            for c in constraints:
                info = result.get(c.id, {})
                # Defensive: each entry must be dict-like
                if not isinstance(info, dict):
                    status[c.id] = False
                    continue
                status[c.id] = bool(info.get("passed", False))
            return status

        # Placeholder fallback
        return {c.id: False for c in constraints}

    # --------------------------------------------------------
    # LOCAL evaluation + instruction generation
    # --------------------------------------------------------

    def check_one(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
        oracle_status: Optional[Dict[str, bool]] = None,
    ) -> CheckResult:
        """
        Only evaluate ONE constraint.
        Used to decide:
            - whether selected constraint is satisfied
            - what edit instruction to generate
        """

        # Oracle
        if oracle_status is not None and constraint.id in oracle_status:
            passed = bool(oracle_status[constraint.id])
            if passed:
                return CheckResult(True, "Passed (oracle).", None, 1.0)
            return CheckResult(False, "Failed (oracle).", _mk_instruction(constraint), 1.0)

        # Backend single-judge (optional)
        if self.backend is not None and hasattr(self.backend, "judge_one"):
            info = self.backend.judge_one(
                prompt_text=prompt_text,
                artifact=artifact,
                constraint=constraint,
            )
            passed = bool(info.get("passed", False))
            confidence = float(info.get("confidence", 1.0))
            reason = info.get("reason", "")

            if passed:
                return CheckResult(True, reason, None, confidence)

            return CheckResult(False, reason, _mk_instruction(constraint), confidence)

        # Fallback
        return CheckResult(
            False,
            "Failed (no backend).",
            _mk_instruction(constraint),
            1.0,
        )
