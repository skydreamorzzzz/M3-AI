# -*- coding: utf-8 -*-
"""
src/refine/verifier.py

Verifier for the refinement loop.

Role (per 5-agent spec):
- Compare the previous best artifact and the newly edited candidate.
- Decide: "better" / "worse" / "same"
- Prevent regressions: "fix one thing, break others".

Design:
- Backend-agnostic: can be powered by an LLM/VLM judge later.
- Default behavior is deterministic and does not require vision:
    - If provided with status dicts, we can compare by:
        1) number of FAILED constraints (lower is better)
        2) number of newly broken constraints (lower is better)
        3) number of newly fixed constraints (higher is better)
    - If no status information is provided, default to "same" (conservative).

This supports an end-to-end dry-run loop for debugging and ablations of scheduling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal


Decision = Literal["better", "worse", "same"]


# ============================================================
# Backend protocol
# ============================================================

class VerifyBackend(Protocol):
    """
    A pluggable backend that can look at artifacts and judge overall alignment.

    For a VLM-based verifier, typical inputs:
      - prompt_text
      - best_artifact image
      - candidate_artifact image
      - (optional) constraint checklist / status deltas
    """

    def compare(
        self,
        prompt_text: str,
        best_artifact: Any,
        candidate_artifact: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        ...


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class VerifierParams:
    """
    Knobs:
    - mode:
        - "status_only": compare using status dict metrics (no vision)
        - "backend": use backend if provided, else fallback to status_only
        - "conservative": if uncertain, return "same" (default)
    - prefer_non_regression: if candidate breaks any previously passed constraints, penalize strongly
    """
    mode: str = "conservative"  # status_only | backend | conservative
    prefer_non_regression: bool = True


# ============================================================
# Verifier
# ============================================================

class Verifier:
    """
    Verifier wrapper.

    Usage:
      v = Verifier(params=VerifierParams(), backend=vlm_verifier)
      decision = v.verify(prompt_text, best, cand, status_best=..., status_cand=...)
    """

    def __init__(self, params: Optional[VerifierParams] = None, backend: Optional[VerifyBackend] = None) -> None:
        self.params = params or VerifierParams()
        self.backend = backend

    def verify(
        self,
        prompt_text: str,
        best_artifact: Any,
        candidate_artifact: Any,
        status_best: Optional[Dict[str, bool]] = None,
        status_candidate: Optional[Dict[str, bool]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """
        Decide whether candidate is better than best.

        If backend is available and mode requests it, we use backend.
        Otherwise, fallback to status-based comparison.
        """
        mode = (self.params.mode or "conservative").strip().lower()

        if mode == "backend" and self.backend is not None:
            return self.backend.compare(
                prompt_text=prompt_text,
                best_artifact=best_artifact,
                candidate_artifact=candidate_artifact,
                extra=extra,
            )

        if mode in ("status_only", "conservative") or self.backend is None:
            return self._compare_by_status(status_best=status_best, status_candidate=status_candidate)

        # unknown mode -> conservative
        return "same"

    # --------------------------------------------------------
    # Status-based comparison (deterministic)
    # --------------------------------------------------------

    def _compare_by_status(
        self,
        status_best: Optional[Dict[str, bool]],
        status_candidate: Optional[Dict[str, bool]],
    ) -> Decision:
        """
        Compare purely by status dicts.

        Primary objective:
          minimize #failed constraints.

        Secondary:
          avoid regressions (newly broken constraints).

        Tertiary:
          reward newly fixed constraints when failed count ties.

        If missing status info, return "same".
        """
        sb = self._ensure_bool_dict(status_best)
        sc = self._ensure_bool_dict(status_candidate)
        if sb is None or sc is None:
            return "same"

        best_failed = self._count_failed(sb)
        cand_failed = self._count_failed(sc)

        newly_broken = self._newly_broken(sb, sc)
        newly_fixed = self._newly_fixed(sb, sc)

        # Hard anti-regression gate:
        # If candidate breaks anything that was previously passed, it is WORSE.
        if self.params.prefer_non_regression and newly_broken > 0:
            return "worse"

        # No regressions:
        if cand_failed < best_failed:
            return "better"
        if cand_failed > best_failed:
            return "worse"

        # Same failed count:
        if newly_fixed > 0:
            return "better"

        return "same"

    def _ensure_bool_dict(self, status: Optional[Dict[str, bool]]) -> Optional[Dict[str, bool]]:
        """
        Defensive normalization:
        - require dict
        - cast values to bool
        """
        if not status or not isinstance(status, dict):
            return None
        return {str(k): bool(v) for k, v in status.items()}

    def _count_failed(self, status: Dict[str, bool]) -> int:
        return sum(1 for _, ok in status.items() if not bool(ok))

    def _newly_broken(self, before: Dict[str, bool], after: Dict[str, bool]) -> int:
        """
        Count constraints that were PASS in before but FAIL in after.
        Missing in after treated as FAIL (conservative).
        """
        broken = 0
        for cid, ok in before.items():
            if bool(ok) and (not bool(after.get(cid, False))):
                broken += 1
        return broken

    def _newly_fixed(self, before: Dict[str, bool], after: Dict[str, bool]) -> int:
        """
        Count constraints that were FAIL in before but PASS in after.
        Missing in after treated as FAIL.
        """
        fixed = 0
        for cid, ok in before.items():
            if (not bool(ok)) and bool(after.get(cid, False)):
                fixed += 1
        return fixed
