# -*- coding: utf-8 -*-
"""
src/llm/client.py

Unified LLM / VLM client wrapper.

Key fixes for "qualified skeleton":
- Mock backend MUST return STRICT JSON for tasks that require JSON parsing.
- chat() keeps returning raw text (string), to match existing prompt parsers.
- Provider backends can be plugged in later.

Supported tasks:
- extract_constraints
- judge_constraint
- verify_pair
- generate_edit_instruction
- general

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import re
import hashlib

from src.llm.cache import SqliteCache, NullCache, make_cache_key


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class LLMParams:
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: Optional[int] = None


# ============================================================
# Client
# ============================================================

class LLMClient:
    """
    Unified client for text and vision models.

    Notes:
- This client returns raw text. Your prompt modules parse JSON from text.
- Mock backend returns strict JSON for JSON-tasks.
    """

    def __init__(
        self,
        model: str,
        cache: Optional[SqliteCache] = None,
        backend: str = "mock",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cache = cache or NullCache()
        self.backend = backend  # "mock", "openai", "gemini", etc.
        self.api_key = api_key
        self.base_url = base_url

    # ============================================================
    # Public API
    # ============================================================

    def chat(
        self,
        task: str,
        messages: List[Dict[str, Any]],
        params: Optional[LLMParams] = None,
        images: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Generic chat entry.

        Args:
            task: logical task name
            messages: chat messages list
            params: decoding params
            images: optional image paths / URIs
            use_cache: enable cache lookup

        Returns:
            model raw text response
        """
        params = params or LLMParams()

        payload = {
            "messages": messages,
            "images": images or [],
        }

        key = make_cache_key(
            task=task,
            model=self.model,
            payload=payload,
            params={
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "top_p": params.top_p,
                "seed": params.seed,
            },
        )

        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.resp_text

        resp_text = self._call_backend(
            task=task,
            messages=messages,
            images=images,
            params=params,
        )

        if use_cache:
            self.cache.set(
                key=key,
                resp_text=resp_text,
                resp_json=None,
                meta={"task": task, "model": self.model, "backend": self.backend},
            )

        return resp_text

    # ============================================================
    # Backend dispatcher
    # ============================================================

    def _call_backend(
        self,
        task: str,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        if self.backend == "mock":
            return self._mock_response(task=task, messages=messages, images=images)

        if self.backend == "openai":
            return self._call_openai(messages, images, params)

        if self.backend == "gemini":
            # keep as explicit error (but skeleton can still run with backend=mock)
            raise NotImplementedError("Gemini backend not implemented yet.")

        raise ValueError(f"Unsupported backend: {self.backend}")

    # ============================================================
    # Mock backend (STRICT JSON)
    # ============================================================

    def _mock_response(self, task: str, messages: List[Dict[str, Any]], images: Optional[List[str]]) -> str:
        """
        Deterministic mock model.

        IMPORTANT:
- For JSON-tasks, MUST return strict JSON (no extra text).
- For general, can return simple text.
        """
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        prompt_text = self._extract_user_prompt_text(last_user)

        if task == "extract_constraints":
            constraints = self._mock_extract_constraints(prompt_text)
            return json.dumps({"constraints": constraints}, ensure_ascii=False)

        if task == "judge_constraint":
            # Always "pass" in mock so refine loop can terminate cleanly without oracle.
            return json.dumps(
                {
                    "passed": True,
                    "reason": "Mock judge: treated as passed.",
                    "edit_instruction": None,
                    "confidence": 0.9,
                },
                ensure_ascii=False,
            )

        if task == "verify_pair":
            return json.dumps(
                {
                    "decision": "same",
                    "reason": "Mock verifier: no vision, keep same.",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )

        if task == "generate_edit_instruction":
            # Provide a minimal placeholder instruction
            return json.dumps(
                {
                    "edit_instruction": "Apply a minimal localized edit to satisfy the failed constraint. Preserve other elements.",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )

        # general fallback
        return f"[MOCK] {prompt_text[:200]}"

    def _extract_user_prompt_text(self, user_block: str) -> str:
        """
        Try to extract the prompt body from our planner templates.
        """
        # planner template uses:
        # User prompt:
        # """
        # {prompt}
        # """
        m = re.search(r'User prompt:\s*"""(.*?)"""', user_block, re.DOTALL)
        if m:
            return m.group(1).strip()
        return user_block.strip()

    def _mock_extract_constraints(self, prompt_text: str) -> List[Dict[str, Any]]:
        """
        Produce a non-empty, stable set of constraints.
        Heuristic: detect common patterns (count, text, style keywords).
        """
        p = prompt_text.lower()

        # subject guess
        subject = "object"
        for cand in ["panda", "cat", "dog", "person", "man", "woman", "car", "tree", "robot"]:
            if cand in p:
                subject = cand
                break

        # count guess (digit)
        count_val = None
        m = re.search(r"\b(\d+)\b", p)
        if m:
            count_val = m.group(1)
        else:
            # word numbers
            word_map = {
                "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            }
            for w, v in word_map.items():
                if re.search(rf"\b{w}\b", p):
                    count_val = v
                    break

        # style attribute guess
        style_val = None
        for style in ["watercolor", "oil painting", "sketch", "cartoon", "photorealistic", "anime"]:
            if style in p:
                style_val = style
                break

        # text rendering guess
        text_val = None
        m2 = re.search(r'"([^"]+)"', prompt_text)
        if m2:
            text_val = m2.group(1).strip()

        out: List[Dict[str, Any]] = []
        cid = 1

        # OBJECT
        out.append(
            {
                "id": f"C{cid}",
                "type": "OBJECT",
                "object": subject,
                "value": None,
                "relation": None,
                "reference": None,
                "confidence": 0.9,
            }
        )
        cid += 1

        # COUNT (optional)
        if count_val is not None:
            out.append(
                {
                    "id": f"C{cid}",
                    "type": "COUNT",
                    "object": subject,
                    "value": str(count_val),
                    "relation": None,
                    "reference": None,
                    "confidence": 0.85,
                }
            )
            cid += 1

        # ATTRIBUTE (optional)
        if style_val is not None:
            out.append(
                {
                    "id": f"C{cid}",
                    "type": "ATTRIBUTE",
                    "object": "overall_style",
                    "value": style_val,
                    "relation": None,
                    "reference": None,
                    "confidence": 0.8,
                }
            )
            cid += 1

        # TEXT (optional)
        if text_val is not None:
            out.append(
                {
                    "id": f"C{cid}",
                    "type": "TEXT",
                    "object": None,
                    "value": text_val,
                    "relation": None,
                    "reference": None,
                    "confidence": 0.8,
                }
            )
            cid += 1

        # Always add one spatial placeholder to keep graph non-trivial
        out.append(
            {
                "id": f"C{cid}",
                "type": "SPATIAL",
                "object": subject,
                "value": None,
                "relation": "center",
                "reference": "image",
                "confidence": 0.6,
            }
        )

        return out

    # ============================================================
    # OpenAI backend (example structure)
    # ============================================================

    def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        """
        Example OpenAI-compatible implementation.

        NOTE:
- This is a skeleton; fill with actual API call when you use backend="openai".
- For skeleton qualification, backend="mock" must be sufficient.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI backend requested but openai package not installed.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        if images:
            content = []
            # merge all user texts
            merged_user = "\n".join([str(m.get("content", "")) for m in messages if m.get("role") == "user"])
            content.append({"type": "text", "text": merged_user})
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
        else:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )

        return resp.choices[0].message.content or ""
