# -*- coding: utf-8 -*-
"""
scripts/run_experiment.py

Multi-strategy experiment runner (debug-safe version).

Fix:
- In MOCK backend, do NOT call vision judge (LLMJudgeBackend),
  because artifacts are mock://... and not local image paths.
- Use a local MockJudgeBackend that returns deterministic results.

Path structure unchanged.
"""

from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

# --- ensure project_root on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------- Schema / Graph ----------------
from src.io.schemas import PromptItem, Constraint
from src.graph.build_drg import build_drg
from src.graph.validate_graph import ensure_dag

# ---------------- Clients ----------------
from src.config.model_config import (
    get_text_client,
    get_vision_client,
    get_image_gen_client,
    get_image_edit_client,
)

# ---------------- LLM Prompts ----------------
from src.llm.prompts.extract_constraints import extract_constraints
from src.llm.prompts.judge_constraint import LLMJudgeBackend
from src.llm.prompts.verify_pair import LLMVerifyBackend

# ---------------- Refine ----------------
from src.refine.checker import Checker, JudgeBackend
from src.refine.editor import Editor, EditorParams
from src.refine.initial import generate_initial_artifact
from src.refine.verifier import Verifier, VerifierParams
from src.refine.loop_core import run_refine_loop, LoopParams

# ---------------- Scheduler ----------------
from src.scheduler.linear_scheduler import LinearScheduler
from src.scheduler.dag_topo import DagTopoScheduler, DagTopoParams


# ============================================================
# Logging Helper
# ============================================================

def _log(stage: str, msg: str):
    print(f"[{stage}] {msg}")


# ============================================================
# Mock Judge Backend (for --backend mock)
# ============================================================

class MockJudgeBackend(JudgeBackend):
    """
    Deterministic offline judge for MOCK mode.

    Purpose:
      - Keep the pipeline end-to-end runnable
      - Avoid any image path operations
      - Provide stable structure for Checker/check_all/check_one

    Strategy:
      - judge_all: mark all constraints as FAILED (passed=False)
      - judge_one: mark selected constraint as FAILED (passed=False)
    """

    def judge_all(
        self,
        prompt_text: str,
        artifact: Any,
        constraints: List[Constraint],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for c in constraints:
            out[c.id] = {
                "passed": False,
                "confidence": 1.0,
                "reason": "mock judge_all: always fail (offline).",
            }
        return out

    def judge_one(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
    ) -> Dict[str, Any]:
        return {
            "passed": False,
            "confidence": 1.0,
            "reason": "mock judge_one: always fail (offline).",
        }


# ============================================================
# Utils
# ============================================================

def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _load_prompts(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be list of {prompt_id, text}")
    return data


def _build_scheduler(name: str):
    n = (name or "").strip().lower()
    if n == "linear":
        return LinearScheduler()
    if n == "topo":
        return DagTopoScheduler(params=DagTopoParams(use_conflict_risk=False))
    if n in ("topo_conflict", "topo+conflict"):
        return DagTopoScheduler(params=DagTopoParams(use_conflict_risk=True))
    raise ValueError(f"Unknown strategy: {name}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--strategies", type=str, default="linear,topo,topo_conflict")
    parser.add_argument("--backend", type=str, default="mock")   # mock | openai
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--dry_run", type=int, default=1)
    args = parser.parse_args()

    prompts = _load_prompts(args.input)

    exp_id = args.exp_id.strip() or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    base_out = ROOT / "runs" / exp_id
    base_out.mkdir(parents=True, exist_ok=True)

    # ---------------- Build Clients ----------------
    _log("INIT", "Building clients...")

    text_client = get_text_client()
    _log("INIT", f"Text model: {text_client.model}")

    # Vision client only needed for real judging/verifying.
    vision_client = None
    if args.backend != "mock":
        vision_client = get_vision_client()
        _log("INIT", f"Vision model: {vision_client.model}")
    else:
        _log("INIT", "MOCK mode: skip vision client (no real judging).")

    image_gen_client = None
    image_edit_client = None

    if args.backend != "mock":
        image_gen_client = get_image_gen_client()
        image_edit_client = get_image_edit_client()
        _log("INIT", f"Image Gen model: {image_gen_client.model}")
        _log("INIT", f"Image Edit model: {image_edit_client.model}")
    else:
        _log("INIT", "Running in MOCK mode (no real image calls)")

    # ---------------- Backends ----------------
    if args.backend == "mock":
        judge_backend = MockJudgeBackend()
        verify_backend = None  # verifier is status_only anyway
        _log("INIT", "Using MockJudgeBackend (offline).")
    else:
        judge_backend = LLMJudgeBackend(vision_client)   # type: ignore[arg-type]
        verify_backend = LLMVerifyBackend(vision_client) # type: ignore[arg-type]
        _log("INIT", "Using LLMJudgeBackend + LLMVerifyBackend.")

    checker = Checker(backend=judge_backend)

    editor = Editor(
        params=EditorParams(dry_run=bool(args.dry_run)),
        backend=image_edit_client if (args.backend != "mock" and not args.dry_run) else None,
    )

    verifier = Verifier(
        params=VerifierParams(mode="status_only"),
        backend=verify_backend,
    )

    strategy_list = [s.strip() for s in args.strategies.split(",") if s.strip()]

    aggregate_all: Dict[str, Any] = {
        "exp_id": exp_id,
        "n_prompts": len(prompts),
        "backend": args.backend,
        "dry_run": bool(args.dry_run),
        "results": {},
    }

    # ============================================================
    # Run
    # ============================================================

    for strategy_name in strategy_list:

        _log("STRATEGY", f"Running strategy: {strategy_name}")
        scheduler = _build_scheduler(strategy_name)

        trace_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        for item_data in prompts:

            pid = item_data["prompt_id"]
            text = item_data["text"]

            _log("PROMPT", f"Processing prompt_id={pid}")

            # ---------------- Planner ----------------
            constraints = extract_constraints(text_client, text)
            _log("PLANNER", f"{pid}: extracted {len(constraints)} constraints")

            # ---------------- Graph ----------------
            full_graph = build_drg(constraints)
            dep_graph, _ = ensure_dag(full_graph)
            _log("GRAPH", f"{pid}: graph nodes={len(dep_graph.nodes)}")

            prompt_dir = base_out / "_precomputed" / pid
            prompt_dir.mkdir(parents=True, exist_ok=True)

            # ---------------- Initial Image ----------------
            _log("INITIAL", f"{pid}: generating initial image...")
            artifact, _ = generate_initial_artifact(
                prompt_id=pid,
                prompt_text=text,
                out_dir=prompt_dir,
                image_gen_client=image_gen_client,
                backend=args.backend,
            )
            _log("INITIAL", f"{pid}: artifact payload={artifact.payload}")

            # ---------------- Loop ----------------
            prompt_item = PromptItem(
                prompt_id=pid,
                text=text,
                constraints=constraints,
                graph=dep_graph,
            )

            _log("LOOP", f"{pid}: starting refine loop")

            best, traces, summary = run_refine_loop(
                item=prompt_item,
                scheduler=scheduler,
                checker=checker,
                editor=editor,
                verifier=verifier,
                params=LoopParams(max_rounds=args.max_rounds),
                initial_artifact=artifact,
            )

            _log(
                "LOOP",
                f"{pid}: finished. rounds={summary.total_rounds}, "
                f"conflicts={summary.conflict_count}, "
                f"final_pass={summary.final_pass}",
            )

            for t in traces:
                trace_rows.append(t.__dict__)

            summary_rows.append(summary.__dict__)

        # ---------------- Write Outputs ----------------
        out_dir = base_out / strategy_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if trace_rows:
            _write_csv(
                out_dir / "trace_long.csv",
                trace_rows,
                fieldnames=list(trace_rows[0].keys()),
            )

        if summary_rows:
            _write_csv(
                out_dir / "summary.csv",
                summary_rows,
                fieldnames=list(summary_rows[0].keys()),
            )

        aggregate_all["results"][strategy_name] = {
            "n_prompts": len(prompts)
        }

    # ---------------- Final Aggregate ----------------
    (base_out / "aggregate.json").write_text(
        json.dumps(aggregate_all, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _log("DONE", f"Experiment completed. Output at {base_out}")


if __name__ == "__main__":
    main()
