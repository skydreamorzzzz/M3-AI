# -*- coding: utf-8 -*-
"""
scripts/run_one.py

One-click end-to-end pipeline run (works under mock).

What this script does (high level):
1) Planner (Text LLM / mock): extract structured constraints from the prompt.
2) Graph builder (local deterministic): build a Dependency-Relation Graph (DRG).
   - Edges include:
     - "dependency": hard prerequisites for topo scheduling
     - "coupling": soft coupling for analysis (NOT for topo)
3) Graph validation (local): ensure the dependency subgraph is a DAG (break cycles if any).
4) Refinement loop:
   - scheduler picks one constraint each round (topo-guided)
   - checker judges it (LLM backend / mock)
   - editor applies edit instruction (dry-run for skeleton)
   - verifier accepts/rejects candidate
5) Write traces + summary to runs/<prompt_id>/.

Run:
  python scripts/run_one.py --prompt "A watercolor painting of five pandas sitting on a bench." --backend mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# --- ensure project_root on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.schemas import PromptItem, ConstraintGraph  # noqa: E402
from src.llm.cache import SqliteCache  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.llm.prompts.extract_constraints import extract_constraints  # noqa: E402
from src.llm.prompts.judge_constraint import LLMJudgeBackend  # noqa: E402
from src.llm.prompts.verify_pair import LLMVerifyBackend  # noqa: E402
from src.graph.build_drg import build_drg  # noqa: E402
from src.graph.validate_graph import ensure_dag  # noqa: E402
from src.scheduler.dag_topo import DagTopoScheduler  # noqa: E402
from src.refine.checker import Checker  # noqa: E402
from src.refine.editor import Editor, EditorParams, ArtifactHandle  # noqa: E402
from src.refine.verifier import Verifier, VerifierParams  # noqa: E402
from src.refine.loop_core import run_refine_loop, LoopParams  # noqa: E402


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Serialize dataclass/pydantic-like objects to dict safely."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # pydantic v1
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def _filter_dependency_graph(graph: ConstraintGraph) -> ConstraintGraph:
    """
    Keep ONLY dependency edges for topo scheduling.
    Coupling edges are useful for analysis but should not drive topo order
    (they may be bidirectional and create artificial cycles).
    """
    edges = getattr(graph, "edges", []) or []

    dep_edges: List[Any] = []
    for e in edges:
        et = None
        if isinstance(e, dict):
            et = e.get("edge_type", None)
        else:
            et = getattr(e, "edge_type", None)

        # If edge has no type, be conservative: treat it as dependency only if it is a tuple pair.
        if et is None:
            if isinstance(e, (tuple, list)) and len(e) == 2:
                dep_edges.append(e)
            continue

        if str(et) == "dependency":
            dep_edges.append(e)

    return ConstraintGraph(nodes=graph.nodes, edges=dep_edges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompt_id", type=str, default="demo_prompt")
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

    # 1) Planner: extract constraints
    constraints = extract_constraints(client, args.prompt)

    # 2) Graph: build DRG (dependency + coupling)
    full_graph: ConstraintGraph = build_drg(constraints)

    # 3) For scheduling: use ONLY dependency edges, then ensure DAG
    dep_graph = _filter_dependency_graph(full_graph)
    dep_graph, report = ensure_dag(dep_graph)

    # 4) Refine components
    judge_backend = LLMJudgeBackend(client)
    verify_backend = LLMVerifyBackend(client)

    checker = Checker(backend=judge_backend)
    editor = Editor(params=EditorParams(dry_run=True))
    verifier = Verifier(params=VerifierParams(mode="status_only"), backend=verify_backend)

    # Scheduler: topo-guided (uses dep_graph which is DAG now)
    scheduler = DagTopoScheduler()

    item = PromptItem(
        prompt_id=args.prompt_id,
        text=args.prompt,
        constraints=constraints,
        graph=dep_graph,
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

    # 5) Write outputs
    out_dir = ROOT / f"runs/{args.prompt_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([_as_dict(t) for t in traces], f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_as_dict(summary), f, ensure_ascii=False, indent=2)

    # Optional: dump graph report for debugging
    with open(out_dir / "graph_report.json", "w", encoding="utf-8") as f:
        json.dump(_as_dict(report), f, ensure_ascii=False, indent=2)

    print("=== Run Completed ===")
    print("Final pass:", summary.final_pass)
    print("Total rounds:", summary.total_rounds)
    print("Conflicts:", summary.conflict_count)
    print("Output:", str(out_dir))


if __name__ == "__main__":
    main()
