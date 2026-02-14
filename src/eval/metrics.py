# -*- coding: utf-8 -*-

from typing import List
from src.io.schemas import RunSummary


def pass_rate(results: List[RunSummary]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.final_pass) / len(results)


def avg_rounds(results: List[RunSummary]) -> float:
    if not results:
        return 0.0
    return sum(r.total_rounds for r in results) / len(results)


def conflict_rate(results: List[RunSummary]) -> float:
    if not results:
        return 0.0
    return sum(r.conflict_count for r in results) / len(results)
