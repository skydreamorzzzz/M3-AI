# -*- coding: utf-8 -*-
"""
src/utils/error_registry.py

Unified error recording for failed samples.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ErrorRegistry:
    """
    Registry for recording failed samples.
    
    Fields:
        - prompt_id: ID of the failed prompt
        - phase: failure phase (extract_constraints / build_graph / generate_initial / refine_loop / unknown)
        - timestamp: failure timestamp
        - exception_type: exception class name
        - exception_message: exception message
        - traceback: full traceback string
    """
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
    
    def record(
        self,
        prompt_id: str,
        phase: str,
        exception: Exception,
        traceback_str: str,
    ) -> None:
        """Record a failure."""
        self.errors.append({
            "prompt_id": prompt_id,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback_str,
        })
    
    def save(self, path: Path) -> None:
        """Save errors to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_errors": len(self.errors), "errors": self.errors},
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    def count_by_phase(self) -> Dict[str, int]:
        """Count errors by phase."""
        counts = {}
        for e in self.errors:
            phase = e["phase"]
            counts[phase] = counts.get(phase, 0) + 1
        return counts

