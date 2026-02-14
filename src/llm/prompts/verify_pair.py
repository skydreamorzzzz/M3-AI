# -*- coding: utf-8 -*-
"""
src/llm/prompts/verify_pair.py

Verifier backend using LLM/VLM.

Role:
- Compare previous best artifact and new candidate artifact.
- Decide: "better" | "worse" | "same"
- Prevent regression: avoid fixing one constraint while breaking others.

This implements VerifyBackend-compatible class:
    LLMVerifyBackend

Expected STRICT JSON output (no extra text):

{
  "decision": "better" | "worse" | "same",
  "reason": "short explanation",
  "confidence": 0.0-1.0
}

This backend can be plugged into src/refine/verifier.py
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import json
import re

from src.llm.client import LLMClient, LLMParams
from src.refine.verifier import VerifyBackend, Decision


# ============================================================
# Prompt Template
# ============================================================

SYSTEM_PROMPT = """
You are a strict visual quality verifier.

Your task:
- Compare two images given the original user prompt.
- Decide whether the candidate image is BETTER, WORSE, or SAME compared to the previous best.
- Focus on overall alignment with the prompt.
- Penalize new errors.
- Do not favor changes unless they clearly improve alignment.

Output STRICT JSON only.
No markdown. No extra explanation outside JSON.

Format:
{
  "decision": "better" | "worse" | "same",
  "reason": "short explanation",
  "confidence": 0.0-1.0
}
"""

USER_TEMPLATE = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Image A: previous best
Image B: new candidate

Which image better matches the prompt?
"""


# ============================================================
# Backend Implementation
# ============================================================

class LLMVerifyBackend(VerifyBackend):
    """
    LLM-based verifier backend.
    """

    def __init__(self, client: LLMClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature

    def compare(
        self,
        prompt_text: str,
        best_artifact: Any,
        candidate_artifact: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Decision:

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt_text).strip()},
        ]

        images = []
        if hasattr(best_artifact, "payload"):
            images.append(best_artifact.payload)
        if hasattr(candidate_artifact, "payload"):
            images.append(candidate_artifact.payload)

        raw = self.client.chat(
            task="verify_pair",
            messages=messages,
            params=LLMParams(temperature=self.temperature),
            images=images,
        )

        data = _safe_parse_json(raw)

        decision = data.get("decision", "same")
        if decision not in ("better", "worse", "same"):
            decision = "same"

        return decision


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
        raise ValueError("No JSON object found in verifier response.")

    try:
        return json.loads(match.group(0))
    except Exception as e:
        raise ValueError(f"Failed to parse verifier JSON: {e}")
