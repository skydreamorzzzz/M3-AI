# -*- coding: utf-8 -*-
"""
src/llm/prompts/extract_constraints.py

Planner prompt builder + response parser.

Role:
- Convert raw user prompt into structured Constraint list.
- This is the "Checklist Planner" agent.

Workflow:
1. Build system + user messages.
2. Call LLMClient.chat(task="extract_constraints", ...)
3. Parse JSON response into List[Constraint].
4. Perform basic validation & normalization.

Expected LLM output format (STRICT JSON, no extra text):

{
  "constraints": [
    {
      "id": "C1",
      "type": "object" | "count" | "attribute" | "spatial" | "relation" | "text",
      "object": "panda",
      "value": null,
      "relation": null,
      "reference": null,
      "confidence": 0.95
    },
    ...
  ]
}

IMPORTANT:
- `type` MUST be one of ConstraintType values (lowercase strings):
  "object", "count", "attribute", "spatial", "relation", "text"
- Use null for fields not applicable.
- confidence is a float between 0 and 1.

Notes:
- This module ONLY extracts constraint NODES.
  Graph edges (dependency/coupling/conflict) are built later (graph/scheduler).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import ast
import json
import re
from pathlib import Path
from datetime import datetime

from src.io.schemas import Constraint, ConstraintType
from src.llm.client import LLMClient, LLMParams


# ============================================================
# Small helpers
# ============================================================

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_json_span(s: str) -> str:
    """
    Extract the outermost JSON-like span.
    Prefer [] if exists earlier than {}.
    """
    s = s.strip()
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return s

    if first_arr != -1 and (first_obj == -1 or first_arr < first_obj):
        start = first_arr
        end = s.rfind("]")
    else:
        start = first_obj
        end = s.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return s
    return s[start : end + 1].strip()


def _remove_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except Exception:
        return Path(".")


def _dump_bad_output(raw: str) -> str:
    """
    Dump raw output to runs/_bad_extract_constraints/ for debugging.
    Returns file path string.
    """
    root = _project_root()
    out_dir = root / "runs" / "_bad_extract_constraints"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = len(list(out_dir.glob("bad_*.txt"))) + 1
    p = out_dir / f"bad_{ts}_{idx:03d}.txt"
    p.write_text(raw, encoding="utf-8")
    return str(p)


def _dump_bad_constraint(obj: Dict[str, Any], *, reason: str, prompt_head: str = "") -> str:
    """
    Dump a single bad constraint object (e.g., unknown type) for quick debugging.
    Returns file path string.
    """
    root = _project_root()
    out_dir = root / "runs" / "_bad_extract_constraints"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = len(list(out_dir.glob("bad_constraint_*.json"))) + 1
    p = out_dir / f"bad_constraint_{ts}_{idx:03d}.json"

    payload = {
        "reason": reason,
        "prompt_head": (prompt_head or "")[:600],
        "constraint_obj": obj,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _try_literal_eval(s: str) -> Optional[Any]:
    """
    Try parsing JSON-ish python literal safely.
    Only used as a last resort when we can still clearly extract object spans.
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _extract_constraints_list_span(s: str) -> Optional[str]:
    """
    Find the [...] span for "constraints": [ ... ] and return that bracket string.
    We do a small bracket-balance scan so we don't depend on perfect JSON.
    """
    m = re.search(r'"constraints"\s*:\s*\[', s)
    if not m:
        return None

    start = m.end() - 1  # index at '['
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _split_top_level_objects(arr_text: str) -> List[str]:
    """
    Given a text that starts with '[' and ends with ']',
    split into top-level '{...}' object strings by brace balancing.
    """
    arr_text = arr_text.strip()
    if not (arr_text.startswith("[") and arr_text.endswith("]")):
        return []

    items: List[str] = []
    i = 0
    n = len(arr_text)
    while i < n:
        ch = arr_text[i]
        if ch == "{":
            depth = 0
            j = i
            while j < n:
                if arr_text[j] == "{":
                    depth += 1
                elif arr_text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        items.append(arr_text[i : j + 1].strip())
                        i = j + 1
                        break
                j += 1
            else:
                # unbalanced braces -> stop
                break
        else:
            i += 1
    return items


def _safe_parse_json(json_str: str):
    """
    Strict parse with controlled salvage:
    1) try json.loads on extracted span
    2) try light repairs and json.loads again
    3) if still fails, try to salvage constraints list by extracting each {..} item
       and parsing each item (json or literal_eval). Only succeed if we can build a valid
       {"constraints":[...]} structure.
    Otherwise raise with a debug dump path (no silent fallback).
    """
    raw0 = (json_str or "").strip()
    raw0 = _strip_code_fences(raw0)
    raw = _extract_json_span(raw0)

    # pass-1: direct
    out = _try_json_loads(raw)
    if out is not None:
        return out

    # pass-2: light repairs
    repaired = _remove_trailing_commas(raw)
    repaired = repaired.replace("None", "null").replace("True", "true").replace("False", "false")
    out = _try_json_loads(repaired)
    if out is not None:
        return out

    # pass-3: controlled salvage ONLY for constraints list
    span = _extract_constraints_list_span(raw0) or _extract_constraints_list_span(raw)
    if span:
        objs = _split_top_level_objects(span)
        parsed_items: List[Dict[str, Any]] = []
        for s_obj in objs:
            s_obj2 = _remove_trailing_commas(s_obj)
            item = _try_json_loads(s_obj2)
            if item is None:
                item = _try_literal_eval(
                    s_obj2.replace("null", "None").replace("true", "True").replace("false", "False")
                )
            if isinstance(item, dict):
                parsed_items.append(item)
            else:
                dump_path = _dump_bad_output(raw0)
                raise ValueError(
                    "Failed to parse one constraint object in salvage mode. "
                    f"See raw dump: {dump_path}"
                )

        if parsed_items:
            return {"constraints": parsed_items}

    dump_path = _dump_bad_output(raw0)
    raise ValueError(
        "Failed to parse JSON from extract_constraints output. "
        f"See raw dump: {dump_path}"
    )


# ============================================================
# Prompt Template
# ============================================================

SYSTEM_PROMPT = """
You are a strict constraint planner.

Your task:
- Read the user prompt.
- Output a checklist of atomic constraints.
- Constraints must be machine-checkable with vision.

Output STRICT JSON only.
No markdown. No explanation outside JSON.

HARD RULES (DO NOT VIOLATE):
- "type" MUST be one of: object, count, attribute, spatial, relation, text
- NEVER invent new types (e.g., "exception", "symmetry", "style", "global", "other", etc.)
- If you need to express an "exception" (e.g., asymmetry), encode it as:
  type="spatial" and describe it in relation/value/reference fields.
- Use null for non-applicable fields. Use double quotes for ALL strings.

JSON format:
{
  "constraints": [
    {
      "id": "C1",
      "type": "object|count|attribute|spatial|relation|text",
      "object": "string",
      "value": "string or null",
      "relation": "string or null",
      "reference": "string or null",
      "confidence": 0.0-1.0
    }
  ]
}
"""

USER_TEMPLATE = """
Extract constraints from the following prompt:

\"\"\"
{prompt}
\"\"\"

Return STRICT JSON only.
"""


# ============================================================
# Planner Interface
# ============================================================

def extract_constraints(
    client: LLMClient,
    prompt_text: str,
    temperature: float = 0.0,
    max_tokens: int = 1400,
) -> List[Constraint]:
    """
    Call LLM to extract structured constraints.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt_text).strip()},
    ]

    raw = client.chat(
        task="extract_constraints",
        messages=messages,
        params=LLMParams(temperature=temperature, max_tokens=max_tokens),
    )

    data = _safe_parse_json(raw)

    if not isinstance(data, dict) or "constraints" not in data:
        dump_path = _dump_bad_output(str(raw))
        raise ValueError(f"LLM output must be a JSON object with field 'constraints'. See: {dump_path}")

    if not isinstance(data["constraints"], list):
        dump_path = _dump_bad_output(str(raw))
        raise ValueError(f"'constraints' must be a list. See: {dump_path}")

    constraints: List[Constraint] = []
    for c in data["constraints"]:
        if not isinstance(c, dict):
            # this is malformed; fail hard with dump
            p = _dump_bad_constraint({"non_dict_item": str(c)}, reason="constraints item is not a dict", prompt_head=prompt_text)
            raise ValueError(f"Constraint item is not a dict. See: {p}")
        constraints.append(_parse_constraint(c, prompt_head=prompt_text))

    constraints = _ensure_unique_ids(constraints)
    return constraints


# ============================================================
# Parsing helpers
# ============================================================

def _normalize_constraint_type(s: str, *, obj: Optional[Dict[str, Any]] = None, prompt_head: str = "") -> ConstraintType:
    s2 = (s or "").strip().lower()
    for t in ConstraintType:
        if s2 == t.value or s2 == t.name.lower():
            return t

    # Hard fail, but with a concrete dump so you can debug the offending item quickly.
    dump_path = _dump_bad_constraint(obj or {"type": s}, reason=f"Unknown constraint type: {s2!r}", prompt_head=prompt_head)
    raise ValueError(f"Unknown constraint type: {s2!r}. See: {dump_path}")


def _coerce_optional_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _coerce_confidence(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 1.0
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def _parse_constraint(obj: Dict[str, Any], *, prompt_head: str = "") -> Constraint:
    cid = str(obj.get("id", "")).strip()
    if not cid:
        dump_path = _dump_bad_constraint(obj, reason="Constraint missing 'id'", prompt_head=prompt_head)
        raise ValueError(f"Constraint missing 'id'. See: {dump_path}")

    raw_type = obj.get("type")
    if raw_type is None:
        dump_path = _dump_bad_constraint(obj, reason=f"Constraint {cid} missing 'type'", prompt_head=prompt_head)
        raise ValueError(f"Constraint {cid} missing 'type'. See: {dump_path}")

    ctype = _normalize_constraint_type(str(raw_type).strip(), obj=obj, prompt_head=prompt_head)

    cobj = obj.get("object")
    cobj_s = str(cobj).strip() if cobj is not None else ""
    if not cobj_s:
        cobj_s = "unknown_object"

    value_s = _coerce_optional_str(obj.get("value"))
    relation_s = _coerce_optional_str(obj.get("relation"))
    reference_s = _coerce_optional_str(obj.get("reference"))
    confidence = _coerce_confidence(obj.get("confidence", 1.0))

    return Constraint(
        id=cid,
        type=ctype,
        object=cobj_s,
        value=value_s,
        relation=relation_s,
        reference=reference_s,
        confidence=confidence,
    )


def _ensure_unique_ids(constraints: List[Constraint]) -> List[Constraint]:
    seen = set()
    out: List[Constraint] = []
    for i, c in enumerate(constraints, start=1):
        cid = c.id
        if cid in seen or not cid:
            cid = f"C{i}"
            c = Constraint(
                id=cid,
                type=c.type,
                object=c.object,
                value=c.value,
                relation=c.relation,
                reference=c.reference,
                confidence=c.confidence,
            )
        seen.add(cid)
        out.append(c)
    return out
