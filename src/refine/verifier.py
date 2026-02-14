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
        3) tie -> "same"
    - If no status information is provided, default to "same" (conservative).

This supports an end-to-end dry-run loop for debugging and ablations of scheduling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal, Tuple


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

        Secondary (anti-regression):
          penalize newly broken constraints (passed -> failed).

        If missing status info, return "same".
        """
        if not status_best or not status_candidate:
            return "same"

        best_failed = self._count_failed(status_best)
        cand_failed = self._count_failed(status_candidate)

        if cand_failed < best_failed:
            # candidate has fewer failures
            if self.params.prefer_non_regression:
                # still check regressions (optional)
                if self._newly_broken(status_best, status_candidate) > 0:
                    # improved failures but broke something: treat as "same" or even "worse"
                    # Here we choose "same" to be conservative.
                    return "same"
            return "better"

        if cand_failed > best_failed:
            return "worse"

        # equal failed count: check regressions if enabled
        if self.params.prefer_non_regression:
            nb = self._newly_broken(status_best, status_candidate)
            if nb > 0:
                return "worse"

        return "same"

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
