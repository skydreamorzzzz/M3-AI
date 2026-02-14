# -*- coding: utf-8 -*-

"""
Track conflict statistics between constraints.
"""

from typing import Dict, List, Tuple
import numpy as np


class ConflictMatrix:

    def __init__(self):
        self.index: Dict[str, int] = {}
        self.reverse: List[str] = []
        self.matrix = np.zeros((0, 0), dtype=float)

    def _ensure(self, cid: str):
        if cid in self.index:
            return
        idx = len(self.reverse)
        self.index[cid] = idx
        self.reverse.append(cid)

        if self.matrix.shape[0] == 0:
            self.matrix = np.zeros((1, 1), dtype=float)
        else:
            n = self.matrix.shape[0]
            new = np.zeros((n + 1, n + 1), dtype=float)
            new[:n, :n] = self.matrix
            self.matrix = new

    def record_conflict(self, a: str, b: str):
        self._ensure(a)
        self._ensure(b)
        i = self.index[a]
        j = self.index[b]
        self.matrix[i, j] += 1
        self.matrix[j, i] += 1

    def risk_score(self, cid: str) -> float:
        if cid not in self.index:
            return 0.0
        i = self.index[cid]
        return float(self.matrix[i].sum())

    def export_dict(self) -> Dict[str, float]:
        return {cid: self.risk_score(cid) for cid in self.reverse}
