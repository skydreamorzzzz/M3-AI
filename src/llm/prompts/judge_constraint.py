# -*- coding: utf-8 -*-
"""
src/llm/prompts/judge_constraint.py

Checker backend using LLM/VLM.

Role:
- Given:
    - original prompt text
    - current artifact (image path / URL)
    - ONE constraint
- Ask a vision-language model to judge whether the constraint is satisfied.
- If not satisfied, produce a concrete edit instruction.

Output format (STRICT JSON, no extra text):

{
  "passed": true/false,
  "reason": "short explanation",
  "edit_instruction": "imperative micro-edit instruction or null",
  "confidence": 0.0-1.0
}

This file implements a JudgeBackend-compatible class:
    LLMJudgeBackend

It can be plugged into src/refine/checker.py
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import json
import re

from src.io.schemas import Constraint
from src.llm.client import LLMClient, LLMParams
from src.refine.checker import JudgeBackend, CheckResult


# ============================================================
# Prompt Template
# ============================================================

SYSTEM_PROMPT = """
You are a strict visual constraint judge.

Your task:
- Check whether the given image satisfies ONE specific constraint.
- If satisfied -> passed=true.
- If not satisfied -> passed=false and generate a minimal, targeted edit instruction.
- The edit must preserve correct elements and only fix the specified issue.

Output STRICT JSON only.
No markdown. No explanation outside JSON.

JSON format:
{
  "passed": true/false,
  "reason": "short explanation",
  "edit_instruction": "string or null",
  "confidence": 0.0-1.0
}
"""

USER_TEMPLATE = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Constraint to check:
{constraint_json}

Is this constraint satisfied in the image?
"""


# ============================================================
# Backend Implementation
# ============================================================

class LLMJudgeBackend(JudgeBackend):
    """
    LLM-based judge backend.

    Uses LLMClient to call vision model.
    """

    def __init__(self, client: LLMClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature

    def judge(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
    ) -> CheckResult:
        """
        artifact.payload is expected to be:
          - image URL
          - local path
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
                ).strip(),
            },
        ]

        raw = self.client.chat(
            task="judge_constraint",
            messages=messages,
            params=LLMParams(temperature=self.temperature),
            images=[artifact.payload] if hasattr(artifact, "payload") else None,
        )

        data = _safe_parse_json(raw)

        return CheckResult(
            passed=bool(data.get("passed", False)),
            reason=str(data.get("reason", "")),
            edit_instruction=data.get("edit_instruction"),
            confidence=float(data.get("confidence", 1.0)),
        )


# ============================================================
# JSON parsing helper
# ============================================================

def _safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response.")

    try:
        return json.loads(match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to parse judge JSON: {e}")
