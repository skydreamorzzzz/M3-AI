# -*- coding: utf-8 -*-
"""
src/llm/client.py

Unified LLM / VLM client wrapper.

Design goals:
- One entry point for all model calls in the system:
  - extract constraints (Planner)
  - judge constraint (Checker backend)
  - verify pair (Verifier backend)
- Built-in caching (via src/llm/cache.py)
- Backend-agnostic (OpenAI / Gemini / Qwen / mock)
- Deterministic option (temperature=0 + keep-first cache)

IMPORTANT:
- This file does NOT hardcode a specific provider.
- Provider adapters live inside `_call_backend`.
- For the skeleton stage, "mock" backend MUST be task-aware and MUST return STRICT JSON
  for tasks that require parsing.

Supported tasks (recommended naming):
- "extract_constraints"
- "judge_constraint"
- "verify_pair"
- "general"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import re

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
    """Unified client for text and vision models."""

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
        """Generic chat entry."""
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
        """Dispatch to provider backend."""
        if self.backend == "mock":
            return self._mock_response(task=task, messages=messages, images=images, params=params)
        if self.backend == "openai":
            return self._call_openai(messages, images, params)
        if self.backend == "gemini":
            return self._call_gemini(messages, images, params)
        raise ValueError(f"Unsupported backend: {self.backend}")

    # ============================================================
    # Mock backend (safe for dry-run)
    # ============================================================

    def _mock_response(
        self,
        task: str,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        """
        Deterministic task-aware mock model.

        MUST return STRICT JSON for:
        - extract_constraints
        - judge_constraint
        - verify_pair

        This is critical because prompt parsers are strict JSON parsers.
        """
        # Grab last user content (best-effort)
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        if task == "extract_constraints":
            return self._mock_extract_constraints(last_user)

        if task == "judge_constraint":
            # For skeleton smoke test: default to "passed=true" so loop can converge fast.
            obj = {
                "passed": True,
                "reason": "mock: assume constraint satisfied",
                "edit_instruction": None,
                "confidence": 0.60,
            }
            return json.dumps(obj, ensure_ascii=False)

        if task == "verify_pair":
            obj = {
                "decision": "same",
                "reason": "mock: no real images, treat as same",
                "confidence": 0.60,
            }
            return json.dumps(obj, ensure_ascii=False)

        # Fallback: still JSON (so callers can choose to parse if they want)
        obj = {
            "text": f"mock: {last_user[:200]}",
            "task": task,
        }
        return json.dumps(obj, ensure_ascii=False)

    def _mock_extract_constraints(self, user_text: str) -> str:
        """
        Create a small, deterministic constraint list from the prompt text.

        Output format:
        { "constraints": [ {...}, ... ] }
        """
        prompt = self._extract_prompt_from_template(user_text)
        prompt_l = prompt.lower()

        # naive number extraction (supports "one".."ten" + digits)
        word2num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        count_val: Optional[int] = None
        # digits first
        m = re.search(r"\b(\d+)\b", prompt_l)
        if m:
            try:
                count_val = int(m.group(1))
            except Exception:
                count_val = None
        if count_val is None:
            for w, n in word2num.items():
                if re.search(rf"\b{w}\b", prompt_l):
                    count_val = n
                    break

        # detect a simple object keyword (extend later)
        obj = None
        for k in ["panda", "pandas", "cat", "dog", "person", "people", "man", "woman", "bench", "car"]:
            if k in prompt_l:
                obj = "panda" if "panda" in k else k
                break
        if obj is None:
            obj = "unknown_object"

        constraints: List[Dict[str, Any]] = []
        cid = 1

        # OBJECT existence
        constraints.append({
            "id": f"C{cid}",
            "type": "OBJECT",
            "object": obj,
            "value": None,
            "relation": None,
            "reference": None,
            "confidence": 0.70,
        })
        cid += 1

        # COUNT if we found one
        if count_val is not None:
            constraints.append({
                "id": f"C{cid}",
                "type": "COUNT",
                "object": obj,
                "value": count_val,
                "relation": None,
                "reference": None,
                "confidence": 0.70,
            })
            cid += 1

        # STYLE heuristic
        style = None
        for s in ["watercolor", "oil", "sketch", "photorealistic", "anime", "cartoon"]:
            if s in prompt_l:
                style = s
                break
        if style:
            constraints.append({
                "id": f"C{cid}",
                "type": "ATTRIBUTE",
                "object": None,
                "value": style,
                "relation": None,
                "reference": "global_style",
                "confidence": 0.65,
            })
            cid += 1

        out = {"constraints": constraints}
        return json.dumps(out, ensure_ascii=False)

    @staticmethod
    def _extract_prompt_from_template(user_text: str) -> str:
        """
        Try to recover the original prompt from USER_TEMPLATE formatting.
        Example template contains:
        User prompt:  ...
        """
        # try triple-quoted block
        m = re.search(r'User prompt:\s*"""\s*(.*?)\s*"""', user_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # fallback: entire user_text
        return user_text.strip()

    # ============================================================
    # OpenAI backend (example structure)
    # ============================================================

    def _call_openai(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        """Example OpenAI-compatible implementation (skeleton)."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI backend requested but openai package not installed.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        if images:
            # Vision-style call (pseudo-structure)
            content = []
            for m in messages:
                if m["role"] == "user":
                    content.append({"type": "text", "text": m["content"]})
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

    # ============================================================
    # Gemini backend (skeleton)
    # ============================================================

    def _call_gemini(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        """Skeleton for Gemini-style API."""
        raise NotImplementedError("Gemini backend not implemented yet.")
