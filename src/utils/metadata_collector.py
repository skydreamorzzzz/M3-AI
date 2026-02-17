# -*- coding: utf-8 -*-
"""
src/utils/metadata_collector.py

Collect and save experiment metadata for reproducibility.
"""

import os
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json


def get_git_info() -> Dict[str, Any]:
    """
    Get Git information (commit hash, date, dirty status).
    
    Returns:
        Dict with commit_hash, commit_date, dirty
    """
    try:
        import subprocess
        root = Path(__file__).resolve().parents[2]
        
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        commit_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        
        return {
            "commit_hash": commit_hash[:12],  # Short hash
            "commit_date": commit_date,
            "dirty": dirty,
        }
    except Exception:
        return {
            "commit_hash": "unknown",
            "commit_date": "unknown",
            "dirty": False,
        }


def collect_metadata(
    exp_id: str,
    args: Any,
    start_time: datetime,
    end_time: datetime,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collect experiment metadata.
    
    Args:
        exp_id: Experiment ID
        args: Parsed command-line arguments
        start_time: Experiment start time
        end_time: Experiment end time
        stats: Statistics dict (n_prompts, n_success, n_failed, etc.)
    
    Returns:
        Metadata dict
    """
    root = Path(__file__).resolve().parents[2]
    
    return {
        "exp_id": exp_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": round((end_time - start_time).total_seconds(), 2),
        
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "project_root": str(root),
            **get_git_info(),
        },
        
        "configuration": {
            "backend": args.backend,
            "dry_run": bool(args.dry_run),
            "max_rounds": args.max_rounds,
            "strategies": [s.strip() for s in args.strategies.split(",")],
        },
        
        "statistics": stats,
    }


def save_metadata(metadata: Dict[str, Any], path: Path):
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dict
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

