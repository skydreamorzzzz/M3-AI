# -*- coding: utf-8 -*-
"""
src/eval/aggregate.py

Aggregate metrics from a list of RunSummary.

Field names align with eval/metrics.py:
- pass_rate
- avg_rounds
- conflict_rate
- oscillation_rate
- protection_rate
"""

from __future__ import annotations

from typing import Dict, List

from src.io.schemas import RunSummary


def aggregate_metrics(results: List[RunSummary]) -> Dict[str, float]:
    if not results:
        return {
            "pass_rate": 0.0,
            "avg_rounds": 0.0,
            "conflict_rate": 0.0,
            "oscillation_rate": 0.0,
            "protection_rate": 0.0,
        }

    n = len(results)
    pass_rate = sum(1 for r in results if bool(r.final_pass)) / n
    avg_rounds = sum(float(r.total_rounds) for r in results) / n
    conflict_rate = sum(float(r.conflict_count) for r in results) / n
    oscillation_rate = sum(1 for r in results if bool(r.oscillation_detected)) / n
    protection_rate = sum(float(r.protection_rate) for r in results) / n

    return {
        "pass_rate": float(pass_rate),
        "avg_rounds": float(avg_rounds),
        "conflict_rate": float(conflict_rate),
        "oscillation_rate": float(oscillation_rate),
        "protection_rate": float(protection_rate),
    }
