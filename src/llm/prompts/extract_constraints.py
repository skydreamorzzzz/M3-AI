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
      "type": "OBJECT",
      "object": "panda",
      "value": null,
      "relation": null,
      "reference": null,
      "confidence": 0.95
    },
    ...
  ]
}

ConstraintType must match enum in src/io/schemas.py
"""

from __future__ import annotations

from typing import List, Dict, Any
import json
import re

from src.io.schemas import Constraint, ConstraintType
from src.llm.client import LLMClient, LLMParams


# ============================================================
# Prompt Template
# ============================================================

SYSTEM_PROMPT = """
You are a visual constraint planner.

Your task:
- Read a user prompt describing an image.
- Extract ALL explicit and implicit visual constraints.
- Convert them into a structured checklist.
- Output STRICT JSON only.
- Do NOT include explanations or markdown.

Constraint types:
- OBJECT: existence of an object
- COUNT: number of objects
- ATTRIBUTE: color/style/property
- SPATIAL: spatial relation (left of, behind, etc.)
- RELATION: semantic relation between objects
- TEXT: rendered text content

Rules:
- Each constraint must have a unique id like "C1", "C2", ...
- Use null for fields not applicable.
- confidence is a float between 0 and 1.
"""

USER_TEMPLATE = """
User prompt:
\"\"\"
{prompt}
\"\"\"

Extract the constraints.
Return STRICT JSON only.
"""


# ============================================================
# Planner Interface
# ============================================================

def extract_constraints(
    client: LLMClient,
    prompt_text: str,
    temperature: float = 0.0,
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
        params=LLMParams(temperature=temperature),
    )

    data = _safe_parse_json(raw)

    if "constraints" not in data:
        raise ValueError("LLM output missing 'constraints' field.")

    constraints = []
    for c in data["constraints"]:
        constraints.append(_parse_constraint(c))

    return constraints


# ============================================================
# Parsing helpers
# ============================================================

def _safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Extract first JSON block from text and parse.
    """
    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to extract JSON via regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def _parse_constraint(obj: Dict[str, Any]) -> Constraint:
    """
    Convert dict into Constraint dataclass.
    """
    try:
        ctype = ConstraintType[obj["type"]]
    except KeyError:
        raise ValueError(f"Invalid constraint type: {obj.get('type')}")

    return Constraint(
        id=str(obj["id"]),
        type=ctype,
        object=obj.get("object"),
        value=obj.get("value"),
        relation=obj.get("relation"),
        reference=obj.get("reference"),
        confidence=float(obj.get("confidence", 1.0)),
    )
