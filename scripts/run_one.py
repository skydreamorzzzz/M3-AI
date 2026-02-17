# -*- coding: utf-8 -*-
"""
scripts/run_one.py

One-click end-to-end pipeline run.

What this script does (high level):
1) Planner (Text LLM / mock): extract structured constraints from the prompt.
2) Graph builder (local deterministic): build a Dependency-Relation Graph (DRG).
   - Edges include:
     - "dependency": hard prerequisites for topo scheduling
     - "coupling": soft coupling for analysis (NOT for topo)
3) Graph validation (local): ensure the dependency subgraph is a DAG (break cycles if any).
4) Refinement loop:
   - scheduler picks one constraint each round (strategy-controlled)
   - checker judges (global check_all + local check_one for edit instruction)
   - editor applies edit instruction (dry-run by default)
   - verifier accepts/rejects candidate
5) Write traces + summary to runs/<prompt_id>_<strategy>/.

Run:
  python scripts/run_one.py --prompt "A watercolor painting of five pandas sitting on a bench." --backend mock --strategy linear
  python scripts/run_one.py --prompt "..." --backend openai --strategy topo_conflict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# --- ensure project_root on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.schemas import PromptItem, ConstraintGraph  # noqa: E402
from src.llm.cache import SqliteCache  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402

from src.config.model_config import TEXT_LLM, VISION_LLM  # noqa: E402

from src.llm.prompts.extract_constraints import extract_constraints  # noqa: E402
from src.llm.prompts.judge_constraint import LLMJudgeBackend  # noqa: E402
from src.llm.prompts.verify_pair import LLMVerifyBackend  # noqa: E402
from src.graph.build_drg import build_drg  # noqa: E402
from src.graph.validate_graph import ensure_dag  # noqa: E402

from src.scheduler.linear_scheduler import LinearScheduler  # noqa: E402
from src.scheduler.dag_topo import DagTopoScheduler, DagTopoParams  # noqa: E402

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

    This is the original conservative logic:
    - accept typed edges with edge_type == "dependency"
    - if edge has no type, accept it only if it's a (u,v) tuple pair
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


def _build_client_from_cfg(
    cfg,
    *,
    backend: str,
    cache: SqliteCache,
    model_override: Optional[str] = None,
) -> LLMClient:
    """
    Create an LLMClient for a given role config.

    - If backend == "mock": run fully offline.
    - Else: use openai-compatible endpoint from cfg.
    """
    if backend == "mock":
        # for mock, model is irrelevant; keep deterministic
        return LLMClient(model="mock", cache=cache, backend="mock")

    model = (model_override or cfg.model).strip()
    api_key = (cfg.api_key or "").strip()  # intentionally left blank by you (fill later)
    base_url = (cfg.base_url or "").strip() or None

    # NOTE: We force openai-compatible mode for real calls (deepseek/qwen compat endpoints)
    return LLMClient(
        model=model,
        cache=cache,
        backend="openai",
        api_key=api_key,
        base_url=base_url,
    )


def build_scheduler(strategy: str):
    """
    Strategies:
      - linear: fixed order baseline
      - topo: topo + tie-break (no conflict risk)
      - topo_conflict: topo + conflict risk penalty
    """
    s = (strategy or "").strip().lower()

    if s == "linear":
        return LinearScheduler()

    if s == "topo":
        return DagTopoScheduler(params=DagTopoParams(use_conflict_risk=False))

    if s in ("topo_conflict", "topo+conflict", "topo-conflict"):
        return DagTopoScheduler(params=DagTopoParams(use_conflict_risk=True))

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompt_id", type=str, default="demo_prompt")

    # unified runner control
    parser.add_argument("--strategy", type=str, default="topo", help="linear | topo | topo_conflict")
    parser.add_argument("--backend", type=str, default="mock", help="mock | openai (openai-compatible endpoints)")

    # optional per-role model overrides (useful for quick debugging)
    parser.add_argument("--text_model", type=str, default="", help="override TEXT_LLM.model")
    parser.add_argument("--vision_model", type=str, default="", help="override VISION_LLM.model")

    parser.add_argument("--cache", type=str, default=str(ROOT / "runs/_cache/llm_cache.sqlite3"))
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--dry_run", type=int, default=1, help="1: dry-run editor (default). 0: real editor backend (if wired).")
    args = parser.parse_args()

    cache = SqliteCache(args.cache)

    text_client = _build_client_from_cfg(
        TEXT_LLM,
        backend=args.backend,
        cache=cache,
        model_override=(args.text_model or "").strip() or None,
    )
    vision_client = _build_client_from_cfg(
        VISION_LLM,
        backend=args.backend,
        cache=cache,
        model_override=(args.vision_model or "").strip() or None,
    )

    # 1) Planner: extract constraints (TEXT LLM)
    constraints = extract_constraints(text_client, args.prompt)

    # 2) Graph: build DRG (dependency + coupling)
    full_graph: ConstraintGraph = build_drg(constraints)

    # 3) For scheduling: use ONLY dependency edges, then ensure DAG
    dep_graph = _filter_dependency_graph(full_graph)
    dep_graph, report = ensure_dag(dep_graph)

    # 4) Refine components (VISION judge/verifier backends)
    judge_backend = LLMJudgeBackend(vision_client)
    verify_backend = LLMVerifyBackend(vision_client)

    checker = Checker(backend=judge_backend)
    editor = Editor(params=EditorParams(dry_run=bool(args.dry_run)))
    verifier = Verifier(params=VerifierParams(mode="status_only"), backend=verify_backend)

    scheduler = build_scheduler(args.strategy)

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

    # 5) Define output directory (before run_refine_loop)
    out_dir = ROOT / f"runs/{args.prompt_id}_{args.strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best, traces, summary = run_refine_loop(
        item=item,
        scheduler=scheduler,
        checker=checker,
        editor=editor,
        verifier=verifier,
        params=LoopParams(max_rounds=args.max_rounds),
        initial_artifact=initial_artifact,
        out_dir=out_dir,
    )

    # 6) Write outputs

    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump([_as_dict(t) for t in traces], f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_as_dict(summary), f, ensure_ascii=False, indent=2)

    # Optional: dump graph report for debugging
    with open(out_dir / "graph_report.json", "w", encoding="utf-8") as f:
        json.dump(_as_dict(report), f, ensure_ascii=False, indent=2)

    print("=== Run Completed ===")
    print("Strategy:", args.strategy)
    print("Final pass:", summary.final_pass)
    print("Total rounds:", summary.total_rounds)
    print("Conflicts:", summary.conflict_count)
    print("Output:", str(out_dir))


if __name__ == "__main__":
    main()
