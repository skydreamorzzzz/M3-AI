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
This file does NOT hardcode a specific provider.
You must implement provider adapters inside `_call_backend`.

Current implementation:
- Supports a "mock" backend (for dry-run tests)
- Provides a clean structure to plug real APIs

Usage example:

    client = LLMClient(
        model="gpt-4o-mini",
        cache=SqliteCache("runs/_cache/llm_cache.sqlite3"),
    )

    text = client.chat(
        task="extract_constraints",
        messages=[{"role": "user", "content": "..."}],
        temperature=0.0,
    )

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    Supported tasks (recommended naming):
      - "extract_constraints"
      - "judge_constraint"
      - "verify_pair"
      - "general"

    This client does NOT assume a specific provider.
    Implement `_call_backend()` for your API.
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

        # Call actual backend
        resp_text = self._call_backend(
            messages=messages,
            images=images,
            params=params,
        )

        if use_cache:
            self.cache.set(
                key=key,
                resp_text=resp_text,
                resp_json=None,
                meta={"task": task, "model": self.model},
            )

        return resp_text

    # ============================================================
    # Backend dispatcher
    # ============================================================

    def _call_backend(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[str]],
        params: LLMParams,
    ) -> str:
        """
        Dispatch to provider backend.

        You can extend this function to support:
          - OpenAI
          - Gemini
          - Qwen
          - Azure
        """

        if self.backend == "mock":
            return self._mock_response(messages)

        if self.backend == "openai":
            return self._call_openai(messages, images, params)

        if self.backend == "gemini":
            return self._call_gemini(messages, images, params)

        raise ValueError(f"Unsupported backend: {self.backend}")

    # ============================================================
    # Mock backend (safe for dry-run)
    # ============================================================

    def _mock_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Deterministic mock model.
        Always returns a simple echo-style response.
        Useful for pipeline testing without API cost.
        """
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        return f"[MOCK RESPONSE] Received: {last_user[:200]}"

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
        - Requires openai package or http client.
        - This is a skeleton; fill with actual API call.
        """
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI backend requested but openai package not installed.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Basic chat call (vision if images provided)
        if images:
            # Vision style call (pseudo-structure)
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
        """
        Skeleton for Gemini-style API.
        You must fill actual SDK call.
        """
        raise NotImplementedError("Gemini backend not implemented yet.")
