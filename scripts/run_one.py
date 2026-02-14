# -*- coding: utf-8 -*-
"""
scripts/run_one.py

One-click end-to-end pipeline run (works under mock).

Run:
  python scripts/run_one.py --prompt "A watercolor painting of five pandas..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# --- ensure project_root on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.refine.verifier import Verifier, VerifierParams
from src.refine.loop_core import run_refine_loop, LoopParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="mock")
    parser.add_argument("--cache", type=str, default=str(ROOT / "runs/_cache/llm_cache.sqlite3"))
    parser.add_argument("--max_rounds", type=int, default=5)
    args = parser.parse_args()

    cache = SqliteCache(args.cache)
    client = LLMClient(
        model=args.model,
        cache=cache,
        backend=args.backend,
    )

    # 1) Planner
    constraints = extract_constraints(client, args.prompt)

    # 2) Graph
    graph: ConstraintGraph = build_drg(constraints)
    graph, _report = ensure_dag(graph)

    # 3) Refine components
    judge_backend = LLMJudgeBackend(client)
    verify_backend = LLMVerifyBackend(client)

    checker = Checker(backend=judge_backend)
    editor = Editor(params=EditorParams(dry_run=True))
    verifier = Verifier(params=VerifierParams(mode="status_only"), backend=verify_backend)

    scheduler = DagTopoScheduler()

    item = PromptItem(
        prompt_id="demo_prompt",
        text=args.prompt,
        constraints=constraints,
        graph=graph,
    )

    initial_artifact = ArtifactHandle(
        payload="mock://image/init",
        meta={"source": "initial"},
    )

    best, traces, summary = run_refine_loop(
        item=item,
        scheduler=scheduler,
        checker=checker,
        editor=editor,
        verifier=verifier,
        params=LoopParams(max_rounds=args.max_rounds),
        initial_artifact=initial_artifact,
    )

    out_dir = ROOT / "runs/demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([t.__dict__ for t in traces], f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

    print("=== Run Completed ===")
    print("Final pass:", summary.final_pass)
    print("Total rounds:", summary.total_rounds)
    print("Conflicts:", summary.conflict_count)
    print("Output:", str(out_dir))


if __name__ == "__main__":
    main()
