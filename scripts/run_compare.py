# -*- coding: utf-8 -*-
"""
scripts/run_compare.py

Compare strategies on same prompt set (works under mock).
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


class SequentialScheduler:
    def schedule(self, graph, status, conflict_risk=None):
        return [cid for cid, ok in status.items() if not ok]


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

    strategies = {
        "sequential": SequentialScheduler(),
        "dag_topo": DagTopoScheduler(),
    }

    base_out = ROOT / "runs/compare"
    base_out.mkdir(parents=True, exist_ok=True)

    aggregate_results = {}

    for strategy_name, scheduler in strategies.items():
        total_rounds = 0
        total_conflicts = 0
        final_pass_count = 0

        for item_data in prompts_data:
            prompt_id = item_data["prompt_id"]
            text = item_data["text"]

            constraints = extract_constraints(client, text)
            graph = build_drg(constraints)
            graph, _ = ensure_dag(graph)

            item = PromptItem(prompt_id=prompt_id, text=text, constraints=constraints, graph=graph)

            initial_artifact = ArtifactHandle(payload="mock://image/init", meta={"source": "initial"})

            _, _, summary = run_refine_loop(
                item=item,
                scheduler=scheduler,
                checker=checker,
                editor=editor,
                verifier=verifier,
                params=LoopParams(max_rounds=args.max_rounds),
                initial_artifact=initial_artifact,
            )

            total_rounds += summary.total_rounds
            total_conflicts += summary.conflict_count
            if summary.final_pass:
                final_pass_count += 1

        n = len(prompts_data)
        aggregate_results[strategy_name] = {
            "avg_rounds": total_rounds / max(n, 1),
            "avg_conflicts": total_conflicts / max(n, 1),
            "final_pass_rate": final_pass_count / max(n, 1),
        }

    with open(base_out / "strategy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, ensure_ascii=False, indent=2)

    print("=== Comparison Results ===")
    for k, v in aggregate_results.items():
        print(k, v)


if __name__ == "__main__":
    main()
