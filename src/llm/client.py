# -*- coding: utf-8 -*-
"""
src/llm/client.py

Unified TEXT / VISION chat client (NO image generation here).

Design goals:
- Only handles chat-style models (text + vision)
- Supports:
    - mock
    - openai-compatible endpoints (OpenAI / DeepSeek / Qwen compatible-mode)
- Built-in caching
- Deterministic option (temperature=0 + keep-first cache)

IMPORTANT:
- This client DOES NOT handle Wanx image generation/edit.
- Wanx must use a separate client (wanx_client.py).
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
    """
    Unified client for TEXT and VISION chat models.

    Supported backends:
        - "mock"
        - "openai" (OpenAI-compatible endpoints)

    NOT supported:
        - wanx image generation (moved to separate client)
    """

    def __init__(
        self,
        model: str,
        cache: Optional[SqliteCache] = None,
        backend: str = "mock",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:

        self.model = model
        self.cache = cache or NullCache()
        self.backend = backend  # "mock" | "openai"
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        if self.backend not in {"mock", "openai"}:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                "LLMClient only supports 'mock' or 'openai'. "
                "Use WanxImageClient for image generation/edit."
            )

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
        Generic chat entry for TEXT / VISION models.
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

        # Cache hit
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit.resp_text

        # Call backend
        resp_text = self._call_backend(
            task=task,
            messages=messages,
            images=images,
            params=params,
        )

        # Save cache
        if use_cache:
            self.cache.set(
                key=key,
                resp_text=resp_text,
                resp_json=None,
                meta={
                    "task": task,
                    "model": self.model,
                    "backend": self.backend,
                },
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
            return self._mock_response(task, messages)

        if self.backend == "openai":
            return self._call_openai(task, messages, images, params)

        raise ValueError(f"Unsupported backend: {self.backend}")

    # ============================================================
    # Mock backend
    # ============================================================

    def _mock_response(
        self,
        task: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Mock response that returns parseable structured JSON for each task.
        
        This ensures downstream parsers don't fail in mock mode.
        """

        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        # extract_constraints: must return {"constraints": [...]}
        if task == "extract_constraints":
            return json.dumps(
                {
                    "constraints": [
                        {
                            "id": "C1",
                            "type": "object",
                            "object": "mock_object",
                            "value": None,
                            "relation": None,
                            "reference": None,
                            "confidence": 0.9,
                        },
                        {
                            "id": "C2",
                            "type": "count",
                            "object": "mock_item",
                            "value": "3",
                            "relation": None,
                            "reference": None,
                            "confidence": 0.85,
                        },
                    ]
                },
                ensure_ascii=False,
            )

        # judge_constraint_one: must return {"passed", "confidence", "reason"}
        if task == "judge_constraint_one":
            return json.dumps(
                {
                    "passed": False,
                    "confidence": 1.0,
                    "reason": "mock judge_one: always fail (offline).",
                },
                ensure_ascii=False,
            )

        # judge_constraint_all: must return {"results": {cid: {...}}}
        if task == "judge_constraint_all":
            # Extract constraint IDs from user message (if possible)
            # For mock, we return a generic structure
            return json.dumps(
                {
                    "results": {
                        "C1": {
                            "passed": False,
                            "confidence": 1.0,
                            "reason": "mock judge_all: C1 fail (offline).",
                        },
                        "C2": {
                            "passed": False,
                            "confidence": 1.0,
                            "reason": "mock judge_all: C2 fail (offline).",
                        },
                    }
                },
                ensure_ascii=False,
            )

        # judge_constraint (legacy): same as judge_constraint_one
        if task == "judge_constraint":
            return json.dumps(
                {
                    "passed": False,
                    "confidence": 1.0,
                    "reason": "mock judge: always fail (offline).",
                },
                ensure_ascii=False,
            )

        # verify_pair: must return {"decision", "confidence", "reason"}
        if task == "verify_pair":
            return json.dumps(
                {
                    "decision": "same",
                    "confidence": 1.0,
                    "reason": "mock verify: always same (offline).",
                },
                ensure_ascii=False,
            )

        # score_quality: must return {"quality_score", "reason", "weaknesses"}
        if task == "score_quality":
            return json.dumps(
                {
                    "quality_score": 0.6,
                    "reason": "mock quality: medium quality (offline).",
                    "weaknesses": ["mock limitation", "no real evaluation"],
                },
                ensure_ascii=False,
            )

        # Default fallback
        return json.dumps(
            {
                "text": f"mock: {last_user[:200]}",
                "task": task,
            },
            ensure_ascii=False,
        )

    # ============================================================
    # OpenAI-compatible backend
    # ============================================================

    @staticmethod
    def _needs_strict_json(task: str) -> bool:
        return task in {
            "extract_constraints",
            "judge_constraint",       # Legacy task name (kept for compatibility)
            "judge_constraint_one",   # Used by LLMJudgeBackend.judge_one()
            "judge_constraint_all",   # Used by LLMJudgeBackend.judge_all()
            "verify_pair",
            "score_quality",          # Used by LLMJudgeBackend.score_quality()
        }

    def _call_openai(
        self,
        task: str,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OpenAI-compatible backend requested but openai package not installed."
            ) from e

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        response_format = {"type": "json_object"} if self._needs_strict_json(task) else None

        # Enforce strict JSON mode
        if response_format is not None:
            sys_guard = {
                "role": "system",
                "content": (
                    "You must output a single valid JSON object and NOTHING ELSE. "
                    "No markdown, no extra commentary."
                ),
            }
            _messages = [sys_guard] + list(messages)
        else:
            _messages = list(messages)

        # Vision-style message formatting
        if images:
            content = []

            for m in _messages:
                if m.get("role") in {"system", "user"}:
                    content.append({"type": "text", "text": m.get("content", "")})

            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})

            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                response_format=response_format,
            )

        else:
            resp = client.chat.completions.create(
                model=self.model,
                messages=_messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                response_format=response_format,
            )

        return resp.choices[0].message.content or ""
