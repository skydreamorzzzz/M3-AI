# -*- coding: utf-8 -*-

from typing import List
from src.io.schemas import TraceStep


def constraint_protection_rate(trace: List[TraceStep]) -> float:
    total = 0
    broken = 0

    for step in trace:
        before = step.status_before
        after = step.status_after
        for cid in before:
            if before[cid] and not after.get(cid, False):
                broken += 1
            total += 1

    if total == 0:
        return 1.0

    return 1.0 - broken / total
