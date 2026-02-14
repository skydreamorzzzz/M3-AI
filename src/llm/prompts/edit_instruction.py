# -*- coding: utf-8 -*-
"""
src/llm/prompts/edit_instruction.py

Optional: LLM-based Refiner (advanced version).

Why this file exists:
----------------------------------
Right now your Checker already generates edit_instruction.
But if you later want:

    ✔ Stronger instruction generation
    ✔ Multi-constraint reasoning
    ✔ Context-aware micro-editing
    ✔ More precise spatial fix guidance

You can delegate instruction generation to this module.

This is NOT required for pipeline to work.
It is an enhancement layer.

Role:
----------------------------------
Given:
  - original prompt
  - constraint
  - judge reason
Return:
  - high-quality, localized, non-destructive edit instruction

STRICT JSON output:

{
  "edit_instruction": "imperative edit instruction",
  "confidence": 0.0-1.0
}
"""

from __future__ import annotations

from typing import Dict, Any
import json
import re

from src.io.schemas import Constraint
from src.llm.client import LLMClient, LLMParams


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a precise image editing instruction generator.

Goal:
- Given a failed visual constraint,
- Produce a minimal, localized edit instruction.
- Do NOT regenerate the whole image.
- Preserve all correct elements.
- Fix only the specified issue.

Return STRICT JSON only.

Format:
{
  "edit_instruction": "string",
  "confidence": 0.0-1.0
}
"""

USER_TEMPLATE = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Failed constraint:
{constraint_json}

Judge explanation:
{reason}

Generate a minimal edit instruction.
"""


# ============================================================
# API
# ============================================================

def generate_edit_instruction(
    client: LLMClient,
    prompt_text: str,
    constraint: Constraint,
    reason: str,
    temperature: float = 0.2,
) -> str:
    """
    Use LLM to generate refined edit instruction.
    """

    constraint_dict = {
        "id": constraint.id,
        "type": constraint.type.name,
        "object": constraint.object,
        "value": constraint.value,
        "relation": constraint.relation,
        "reference": constraint.reference,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                prompt=prompt_text,
                constraint_json=json.dumps(constraint_dict, ensure_ascii=False),
                reason=reason,
            ).strip(),
        },
    ]

    raw = client.chat(
        task="generate_edit_instruction",
        messages=messages,
        params=LLMParams(temperature=temperature),
    )

    data = _safe_parse_json(raw)

    return data.get("edit_instruction", "")


# ============================================================
# JSON parsing
# ============================================================

def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in edit instruction response.")

    try:
        return json.loads(match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}")
