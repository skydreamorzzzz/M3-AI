# -*- coding: utf-8 -*-
"""
scripts/run_batch.py

Batch runner (works under mock).

Input format: data/prompts.json
[
  {"prompt_id":"p1","text":"..."},
  ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.schemas import PromptItem
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


def load_prompts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="mock")
    parser.add_argument("--cache", type=str, default=str(ROOT / "runs/_cache/llm_cache.sqlite3"))
    parser.add_argument("--max_rounds", type=int, default=5)
    args = parser.parse_args()

    prompts_data = load_prompts(args.input)

    cache = SqliteCache(args.cache)
    client = LLMClient(model=args.model, cache=cache, backend=args.backend)

    judge_backend = LLMJudgeBackend(client)
    verify_backend = LLMVerifyBackend(client)

    checker = Checker(backend=judge_backend)
    editor = Editor(params=EditorParams(dry_run=True))
    verifier = Verifier(params=VerifierParams(mode="status_only"), backend=verify_backend)
    scheduler = DagTopoScheduler()

    base_out = ROOT / "runs/batch"
    base_out.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "total_prompts": 0,
        "total_rounds": 0,
        "total_conflicts": 0,
        "final_pass_count": 0,
    }

    for item_data in prompts_data:
        prompt_id = item_data["prompt_id"]
        text = item_data["text"]

        constraints = extract_constraints(client, text)
        graph = build_drg(constraints)
        graph, _ = ensure_dag(graph)

        item = PromptItem(prompt_id=prompt_id, text=text, constraints=constraints, graph=graph)

        initial_artifact = ArtifactHandle(payload="mock://image/init", meta={"source": "initial"})

        best, traces, summary = run_refine_loop(
            item=item,
            scheduler=scheduler,
            checker=checker,
            editor=editor,
            verifier=verifier,
            params=LoopParams(max_rounds=args.max_rounds),
            initial_artifact=initial_artifact,
        )

        out_dir = base_out / prompt_id
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
            json.dump([t.__dict__ for t in traces], f, ensure_ascii=False, indent=2)

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary.__dict__, f, ensure_ascii=False, indent=2)

        aggregate["total_prompts"] += 1
        aggregate["total_rounds"] += summary.total_rounds
        aggregate["total_conflicts"] += summary.conflict_count
        if summary.final_pass:
            aggregate["final_pass_count"] += 1

    with open(base_out / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print("=== Batch Run Completed ===")
    print("Prompts:", aggregate["total_prompts"])
    print("Avg rounds:", aggregate["total_rounds"] / max(aggregate["total_prompts"], 1))
    print("Avg conflicts:", aggregate["total_conflicts"] / max(aggregate["total_prompts"], 1))
    print("Final pass rate:", aggregate["final_pass_count"] / max(aggregate["total_prompts"], 1))


if __name__ == "__main__":
    main()
