# -*- coding: utf-8 -*-
"""
src/llm/prompts/judge_constraint.py

Checker backend using LLM/VLM.

Implements JudgeBackend-compatible class: LLMJudgeBackend

Key contracts (NO "silent fallback"):
- judge_one(...) MUST return a dict:
    {"passed": bool, "reason": str, "confidence": float}
- judge_all(...) MUST return a dict mapping constraint_id to dict:
    {cid: {"passed": bool, "reason": str, "confidence": float}, ...}

If anything goes wrong (API error / invalid image / JSON parse failure),
we RAISE an exception instead of returning None.
This ensures you do NOT "hide errors" behind defaults.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
import base64
import json
import mimetypes
import os
import re

from src.io.schemas import Constraint
from src.llm.client import LLMClient, LLMParams
from src.refine.checker import JudgeBackend


# ============================================================
# Printing helpers (ALWAYS ON)
# ============================================================

def _p(msg: str) -> None:
    # Always print: user requested no DEBUG switch
    print(f"[JUDGE] {msg}")


def _truncate(s: Any, n: int = 800) -> str:
    try:
        t = str(s)
    except Exception:
        return f"<unprintable {type(s)}>"
    if len(t) <= n:
        return t
    return t[:n] + f"...(truncated, total={len(t)})"


# ============================================================
# Prompt Templates
# ============================================================

SYSTEM_PROMPT_ONE = """
You are a strict visual constraint judge.

Task:
- Given the ORIGINAL prompt, ONE constraint, and ONE image:
  decide whether the constraint is satisfied.
- Output STRICT JSON only. No markdown. No extra text.

JSON format:
{
  "passed": true/false,
  "reason": "short explanation",
  "confidence": 0.0-1.0
}
""".strip()

USER_TEMPLATE_ONE = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Constraint to check (JSON):
{constraint_json}

Return STRICT JSON only.
""".strip()


SYSTEM_PROMPT_ALL = """
You are a strict visual constraint judge.

Task:
- Given the ORIGINAL prompt, a LIST of constraints, and ONE image:
  judge each constraint independently.

Output STRICT JSON only. No markdown. No extra text.

JSON format:
{
  "results": {
    "<constraint_id>": {"passed": true/false, "reason": "...", "confidence": 0.0-1.0},
    ...
  }
}
""".strip()

USER_TEMPLATE_ALL = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Constraints list (JSON array):
{constraints_json}

Return STRICT JSON only.
""".strip()


SYSTEM_PROMPT_QUALITY = """
You are an EXTREMELY STRICT image quality evaluator for scientific research and academic publications.

Task:
- Given the ORIGINAL prompt and ONE image:
  rate the overall quality with PROFESSIONAL PUBLICATION STANDARDS.

CRITICAL EVALUATION CRITERIA:
1. **Precision**: Are measurements, counts, positions EXACT as specified?
2. **Composition**: Is the layout clean, balanced, and aesthetically pleasing?
3. **Detail Fidelity**: Are textures, colors, lighting realistic and high-resolution?
4. **Text Quality**: Is text perfectly sharp, legible, and correctly positioned?
5. **Spatial Accuracy**: Are spatial relationships (distances, angles, overlaps) precise?

YOU MUST BE HARSH. Assume the image has flaws unless proven otherwise.

Output STRICT JSON only. No markdown. No extra text.

JSON format:
{
  "quality_score": 0.0-1.0,
  "reason": "detailed explanation with specific issues",
  "weaknesses": ["list", "each", "specific", "flaw", "separately"]
}

Scoring guideline (EXTREMELY STRICT):
- 0.95-1.0: Perfect (ZERO flaws, indistinguishable from professional photography, publication-ready)
- 0.85-0.95: Excellent (1-2 barely noticeable imperfections, requires close inspection to find)
- 0.70-0.85: Good (3-5 minor issues: slightly imperfect spacing, minor lighting issues, small detail inaccuracies)
- 0.50-0.70: Acceptable (multiple noticeable issues: imperfect alignment, suboptimal composition, minor missing details)
- 0.30-0.50: Poor (significant issues: wrong proportions, messy layout, poor color balance, low resolution feel)
- 0.0-0.30: Very poor (major defects: missing elements, severe distortions, unprofessional quality)

MANDATORY PENALTIES (apply cumulatively):
- Any imprecise spacing or alignment: -0.10
- Imperfect text (blur, wrong font, misalignment): -0.15
- Wrong count visible (even if off by 1): -0.25
- Incorrect spatial relationship (wrong distance/angle): -0.15
- Poor lighting or color balance: -0.10
- Low resolution or blurry details: -0.15
- Unnatural composition or awkward framing: -0.10
- Any element looks AI-generated (not photorealistic): -0.10

BASELINE STARTING SCORE: 0.60 (assume imperfect until proven otherwise).
Adjust UP if image exceeds expectations, or DOWN for each flaw found.

NEVER give 0.9+ unless the image is TRULY FLAWLESS at professional photography standards.
""".strip()

USER_TEMPLATE_QUALITY = """
Original prompt:
\"\"\"
{prompt}
\"\"\"

Rate the overall quality of this image.

Return STRICT JSON only.
""".strip()


# ============================================================
# Backend Implementation
# ============================================================

class LLMJudgeBackend(JudgeBackend):
    """
    LLM/VLM-based judge backend.

    image handling:
    - If artifact/payload is http(s) URL -> pass through
    - If artifact/payload is a local file path -> convert to data URL (base64)
    - If artifact is an object with .payload -> use that
    """

    def __init__(self, client: LLMClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = float(temperature)

    # -----------------------------
    # Judge ONE constraint
    # -----------------------------
    def judge_one(
        self,
        prompt_text: str,
        artifact: Any,
        constraint: Constraint,
    ) -> Dict[str, Any]:
        _p("========== judge_one ==========")
        _p(f"model={getattr(self.client, 'model', None)} backend={getattr(self.client, 'backend', None)}")
        _p(f"temperature={self.temperature}")
        _p(f"prompt_len={len(prompt_text) if isinstance(prompt_text, str) else 'NA'}")
        _p(f"constraint_id={getattr(constraint, 'id', None)} constraint_type={getattr(constraint, 'type', None)}")

        img = _extract_image_handle(artifact)
        _p(f"artifact_handle={img!r}")

        img_url = _to_image_url(img)
        _p(f"img_url_prefix={img_url[:40]!r} (len={len(img_url)})")

        constraint_dict = _constraint_to_dict(constraint)
        _p(f"constraint_dict={_truncate(constraint_dict, 500)}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ONE},
            {
                "role": "user",
                "content": USER_TEMPLATE_ONE.format(
                    prompt=prompt_text,
                    constraint_json=json.dumps(constraint_dict, ensure_ascii=False),
                ),
            },
        ]

        _p("calling client.chat(task=judge_constraint_one)...")
        raw = self.client.chat(
            task="judge_constraint_one",
            messages=messages,
            params=LLMParams(temperature=self.temperature),
            images=[img_url],
        )
        _p(f"raw_response={_truncate(raw, 1200)}")

        data = _safe_parse_json(raw)
        _p(f"parsed_json={_truncate(data, 1200)}")

        # STRICT validation (raise, do not hide)
        passed = _require_bool(data, "passed")
        reason = _require_str(data, "reason")
        confidence = _require_float_01(data, "confidence")

        _p(f"judge_one_result: passed={passed} confidence={confidence} reason={_truncate(reason, 240)}")

        return {
            "passed": passed,
            "reason": reason,
            "confidence": confidence,
        }

    # -----------------------------
    # Judge ALL constraints (single call)
    # -----------------------------
    def judge_all(
        self,
        prompt_text: str,
        artifact: Any,
        constraints: List[Constraint],
    ) -> Dict[str, Dict[str, Any]]:
        _p("========== judge_all ==========")
        _p(f"model={getattr(self.client, 'model', None)} backend={getattr(self.client, 'backend', None)}")
        _p(f"temperature={self.temperature}")
        _p(f"prompt_len={len(prompt_text) if isinstance(prompt_text, str) else 'NA'}")
        _p(f"n_constraints={len(constraints or [])}")

        # quick type histogram
        try:
            hist: Dict[str, int] = {}
            for c in (constraints or []):
                ct = getattr(c.type, "name", None)
                if ct is None:
                    ct = str(c.type)
                hist[str(ct).lower()] = hist.get(str(ct).lower(), 0) + 1
            _p(f"constraint_type_hist={hist}")
        except Exception as e:
            _p(f"constraint_type_hist_error={e}")

        img = _extract_image_handle(artifact)
        _p(f"artifact_handle={img!r}")

        img_url = _to_image_url(img)
        _p(f"img_url_prefix={img_url[:40]!r} (len={len(img_url)})")

        cons_list = [_constraint_to_dict(c) for c in (constraints or [])]
        # 只打印前 2 个，避免刷屏
        _p(f"constraints_json_preview(first2)={_truncate(cons_list[:2], 1200)}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ALL},
            {
                "role": "user",
                "content": USER_TEMPLATE_ALL.format(
                    prompt=prompt_text,
                    constraints_json=json.dumps(cons_list, ensure_ascii=False),
                ),
            },
        ]

        _p("calling client.chat(task=judge_constraint_all)...")
        raw = self.client.chat(
            task="judge_constraint_all",
            messages=messages,
            params=LLMParams(temperature=self.temperature),
            images=[img_url],
        )
        _p(f"raw_response={_truncate(raw, 1600)}")

        data = _safe_parse_json(raw)
        _p(f"parsed_json_keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        if "results" not in data or not isinstance(data["results"], dict):
            _p(f"ERROR: missing/invalid 'results' field. parsed_json={_truncate(data, 1200)}")
            raise ValueError("judge_all: response JSON must contain a dict field 'results'.")

        results = data["results"]
        _p(f"results_keys_count={len(results.keys())}")

        out: Dict[str, Dict[str, Any]] = {}

        # For each expected constraint id, strictly validate its entry
        miss: List[str] = []
        for c in (constraints or []):
            cid = str(c.id)
            entry = results.get(cid, None)
            if not isinstance(entry, dict):
                miss.append(cid)
                continue

            passed = _require_bool(entry, "passed")
            reason = _require_str(entry, "reason")
            confidence = _require_float_01(entry, "confidence")

            out[cid] = {
                "passed": passed,
                "reason": reason,
                "confidence": confidence,
            }

        if miss:
            _p(f"ERROR: missing/invalid entries for cids={miss[:20]} (showing up to 20)")
            # 这里保持原有行为：直接 raise
            raise ValueError(f"judge_all: missing/invalid entry for constraint id={miss[0]}")

        # overall pass (for printing only; upstream decides loop stop)
        try:
            overall_pass = all(v.get("passed") is True for v in out.values())
            n_pass = sum(1 for v in out.values() if v.get("passed") is True)
            _p(f"judge_all_summary: passed={n_pass}/{len(out)} overall_pass={overall_pass}")
        except Exception as e:
            _p(f"judge_all_summary_error={e}")

        # print first few results
        try:
            shown = 0
            for cid, v in out.items():
                _p(f"cid={cid} passed={v.get('passed')} conf={v.get('confidence')} reason={_truncate(v.get('reason'), 140)}")
                shown += 1
                if shown >= 5:
                    break
        except Exception as e:
            _p(f"print_first_results_error={e}")

        return out
    
    # -----------------------------
    # Score image quality
    # -----------------------------
    def score_quality(
        self,
        prompt_text: str,
        artifact: Any,
    ) -> Dict[str, Any]:
        """
        Score overall image quality (0-1).
        
        Returns:
            {
                "quality_score": float,
                "reason": str,
                "weaknesses": List[str],
            }
        """
        _p("========== score_quality ==========")
        _p(f"model={getattr(self.client, 'model', None)} backend={getattr(self.client, 'backend', None)}")
        _p(f"temperature={self.temperature}")
        _p(f"prompt_len={len(prompt_text) if isinstance(prompt_text, str) else 'NA'}")
        
        img = _extract_image_handle(artifact)
        _p(f"artifact_handle={img!r}")
        
        img_url = _to_image_url(img)
        _p(f"img_url_prefix={img_url[:40]!r} (len={len(img_url)})")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_QUALITY},
            {
                "role": "user",
                "content": USER_TEMPLATE_QUALITY.format(prompt=prompt_text),
            },
        ]
        
        _p("calling client.chat(task=score_quality)...")
        raw = self.client.chat(
            task="score_quality",
            messages=messages,
            params=LLMParams(temperature=self.temperature),
            images=[img_url],
        )
        _p(f"raw_response={_truncate(raw, 1200)}")
        
        data = _safe_parse_json(raw)
        _p(f"parsed_json={_truncate(data, 1200)}")
        
        # STRICT validation
        quality_score = _require_float_01(data, "quality_score")
        reason = _require_str(data, "reason")
        weaknesses = data.get("weaknesses", [])
        
        _p(f"score_quality_result: quality={quality_score:.2f} reason={_truncate(reason, 240)}")
        
        return {
            "quality_score": quality_score,
            "reason": reason,
            "weaknesses": weaknesses if isinstance(weaknesses, list) else [],
        }


# ============================================================
# Helpers
# ============================================================

def _constraint_to_dict(constraint: Constraint) -> Dict[str, Any]:
    # NOTE: constraint.type may be Enum; support .name and fallback str
    ctype = getattr(constraint.type, "name", None)
    if ctype is None:
        ctype = str(constraint.type)

    return {
        "id": str(constraint.id),
        "type": str(ctype).lower(),
        "object": constraint.object,
        "value": constraint.value,
        "relation": constraint.relation,
        "reference": constraint.reference,
        "confidence": float(getattr(constraint, "confidence", 1.0)),
    }


def _extract_image_handle(artifact: Any) -> str:
    """
    Accept:
    - ArtifactHandle(payload=...)
    - plain string path/URL/dataURL
    """
    if artifact is None:
        raise ValueError("artifact is None (no image provided).")

    if hasattr(artifact, "payload"):
        p = getattr(artifact, "payload")
        if not isinstance(p, str) or not p.strip():
            raise ValueError("artifact.payload must be a non-empty string (path/URL/dataURL).")
        return p.strip()

    if isinstance(artifact, str) and artifact.strip():
        return artifact.strip()

    raise ValueError("artifact must be a string (path/URL/dataURL) or an object with .payload")


def _to_image_url(handle: str) -> str:
    """
    Convert handle to a URL usable by openai-compatible VLM:
    - http(s)://... -> keep
    - data:image/...;base64,... -> keep
    - local file path -> convert to data URL
    """
    h = (handle or "").strip()
    if not h:
        raise ValueError("empty image handle")

    if h.startswith("http://") or h.startswith("https://"):
        return h

    if h.startswith("data:image/"):
        return h

    # Treat as local path
    if not os.path.exists(h):
        raise FileNotFoundError(f"Local image path not found: {h}")

    return _file_to_data_url(h)


def _file_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        # fallback by extension
        if ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            # still try generic image/*
            mime = "image/png"

    with open(path, "rb") as f:
        b = f.read()

    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _safe_parse_json(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("LLM returned None (empty response).")

    t = str(text).strip()

    # First try direct JSON
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Top-level JSON is not an object.")
    except Exception:
        pass

    # Try extract first {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response.")

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
        raise ValueError("Extracted JSON is not an object.")
    except Exception as e:
        raise ValueError(f"Failed to parse judge JSON: {e}")


def _require_bool(d: Dict[str, Any], k: str) -> bool:
    if k not in d:
        raise ValueError(f"Missing key '{k}' in judge response.")
    v = d[k]
    if isinstance(v, bool):
        return v
    # accept 0/1
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise ValueError(f"Key '{k}' must be boolean. Got: {type(v).__name__}={v}")


def _require_str(d: Dict[str, Any], k: str) -> str:
    if k not in d:
        raise ValueError(f"Missing key '{k}' in judge response.")
    v = d[k]
    if v is None:
        return ""
    if not isinstance(v, str):
        return str(v)
    return v


def _require_float_01(d: Dict[str, Any], k: str) -> float:
    if k not in d:
        raise ValueError(f"Missing key '{k}' in judge response.")
    v = d[k]
    try:
        f = float(v)
    except Exception:
        raise ValueError(f"Key '{k}' must be a float in [0,1]. Got: {type(v).__name__}={v}")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"Key '{k}' must be in [0,1]. Got: {f}")
    return f
