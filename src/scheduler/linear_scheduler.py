# -*- coding: utf-8 -*-
"""
src/scheduler/linear_scheduler.py

LinearScheduler (Baseline)

This scheduler represents the paper baseline:
- Fixed linear order
- No topology
- No coupling
- No conflict awareness
- Deterministic

Contract (compatible with loop_core.py):

    schedule(graph, status, conflict_risk=None) -> List[str]

Design:
- Uses the original order of constraints as they appear in graph.nodes
- Optionally prioritizes unpassed constraints (recommended)
- Fully deterministic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class LinearSchedulerParams:
    prefer_unpassed: bool = True
    failed_only: bool = True


# ============================================================
# Scheduler
# ============================================================

class LinearScheduler:

    def __init__(self, params: Optional[LinearSchedulerParams] = None) -> None:
        self.params = params or LinearSchedulerParams()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def schedule(
        self,
        graph: Any,
        status: Dict[str, bool],
        conflict_risk: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Return constraint ids in fixed linear order.

        Ignores:
            - dependency edges
            - coupling edges
            - conflict risk
        """

        node_ids = self._get_node_ids(graph)

        if not node_ids:
            return []

        ordered = list(node_ids)

        # Option 1: only failed constraints
        if self.params.failed_only:
            ordered = [cid for cid in ordered if not bool(status.get(cid, False))]

        # Option 2: failed first, passed later (still deterministic)
        elif self.params.prefer_unpassed:
            unpassed = [cid for cid in ordered if not bool(status.get(cid, False))]
            passed = [cid for cid in ordered if bool(status.get(cid, False))]
            ordered = unpassed + passed

        return ordered

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _get_node_ids(self, graph: Any) -> List[str]:
        """
        Extract node ids in original insertion order.
        """
        nodes = getattr(graph, "nodes", None)
        if not nodes:
            return []

        out: List[str] = []

        for n in nodes:
            if n is None:
                continue
            if isinstance(n, str):
                out.append(n)
            elif hasattr(n, "id"):
                out.append(str(getattr(n, "id")))
            else:
                out.append(str(n))

        return out
