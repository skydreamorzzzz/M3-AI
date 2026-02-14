# -*- coding: utf-8 -*-
"""
scripts/run_one.py

Run full pipeline for ONE prompt:
    1) Extract constraints (Planner)
    2) Build graph (build_drg)
    3) Validate DAG
    4) Schedule + Refine loop
    5) Save traces + summary

This script connects:
- src/llm/*
- src/graph/*
- src/scheduler/*
- src/refine/*
- src/eval/* (optional, later)

It is designed for:
- single prompt debugging
- sanity check of full 3-layer architecture

Example usage:

    python scripts/run_one.py \
        --prompt "A watercolor painting of five pandas..." \
        --model gpt-4o-mini \
        --backend openai

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.io.schemas import PromptItem, ConstraintGraph
from src.llm.cache import SqliteCache
from src.llm.client import LLMClient
from src.llm.prompts.extract_constraints import extract_constraints
from src.llm.prompts.judge_constraint import LLMJudgeBackend
from src.llm.prompts.verify_pair import LLMVerifyBackend
from src.graph.build_drg import build_drg
from src.graph.validate_graph import ensure_dag
from src.scheduler.dag_topo import DagTopoScheduler
from src.refine.checker import Checker
from src.refine.editor import Editor, EditorParams, ArtifactHandle
from src.refine.verifier import Verifier
from src.refine.loop_core import run_refine_loop, LoopParams


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="mock")
    parser.add_argument("--cache", type=str, default="runs/_cache/llm_cache.sqlite3")
    parser.add_argument("--max_rounds", type=int, default=5)
    args = parser.parse_args()

    # ----------------------------
    # Setup cache + client
    # ----------------------------
    cache = SqliteCache(args.cache)
    client = LLMClient(
        model=args.model,
        cache=cache,
        backend=args.backend,
    )

    # ----------------------------
    # 1) Planner
    # ----------------------------
    constraints = extract_constraints(client, args.prompt)

    # ----------------------------
    # 2) Graph
    # ----------------------------
    graph: ConstraintGraph = build_drg(constraints)
    graph, report = ensure_dag(graph)

    # ----------------------------
    # 3) Setup refine components
    # ----------------------------
    judge_backend = LLMJudgeBackend(client)
    verify_backend = LLMVerifyBackend(client)

    checker = Checker(backend=judge_backend)
    editor = Editor(params=EditorParams(dry_run=True))
    verifier = Verifier(backend=verify_backend)

    scheduler = DagTopoScheduler()

    item = PromptItem(
        prompt_id="demo_prompt",
        text=args.prompt,
        constraints=constraints,
        graph=graph,
    )

    # initial artifact (placeholder)
    initial_artifact = ArtifactHandle(
        payload="https://dummy-image-placeholder",
        meta={"source": "initial"},
    )

    # ----------------------------
    # 4) Run refine loop
    # ----------------------------
    best, traces, summary = run_refine_loop(
        item=item,
        scheduler=scheduler,
        checker=checker,
        editor=editor,
        verifier=verifier,
        params=LoopParams(max_rounds=args.max_rounds),
        initial_artifact=initial_artifact,
    )

    # ----------------------------
    # 5) Save results
    # ----------------------------
    out_dir = Path("runs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([t.__dict__ for t in traces], f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

    print("=== Run Completed ===")
    print("Final pass:", summary.final_pass)
    print("Total rounds:", summary.total_rounds)
    print("Conflicts:", summary.conflict_count)


if __name__ == "__main__":
    main()
