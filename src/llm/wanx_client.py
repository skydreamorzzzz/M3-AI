# -*- coding: utf-8 -*-
"""
src/llm/wanx_client.py

Wanx image generation / editing client.

Key points (per official docs):
- Task submission MUST include header: X-DashScope-Async: enable
- Beijing and Singapore/intl regions use different endpoints and API keys (do NOT mix).
  If auth fails (401/403), we auto-fallback between dashscope.aliyuncs.com and dashscope-intl.aliyuncs.com.
- Text2Image endpoint:
    POST {base}/api/v1/services/aigc/text2image/image-synthesis
- Image2Image (editing) endpoint:
    POST {base}/api/v1/services/aigc/image2image/image-synthesis
- Poll:
    GET  {base}/api/v1/tasks/{task_id}
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests


# ============================================================
# Helpers
# ============================================================

def _swap_region_host(base_url: str) -> str:
    """
    Swap between Beijing and intl endpoints.
    """
    b = base_url.rstrip("/")
    if "dashscope-intl.aliyuncs.com" in b:
        return b.replace("dashscope-intl.aliyuncs.com", "dashscope.aliyuncs.com")
    if "dashscope.aliyuncs.com" in b:
        return b.replace("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com")
    # If user passes some other host, don't change.
    return b


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def _print_http_error(prefix: str, resp: requests.Response) -> None:
    data = _safe_json(resp)
    print(f"{prefix} HTTP {resp.status_code} body: {data}")


# ============================================================
# Base Wanx Client
# ============================================================

class _WanxBaseClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com",
        timeout: int = 120,
        max_retries: int = 1,
        poll_interval: float = 2.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.poll_interval = poll_interval

        if not self.api_key:
            raise ValueError("Wanx requires DASHSCOPE_API_KEY environment variable.")

    def _headers(self, async_enable: bool = False) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_enable:
            # REQUIRED for these endpoints (async-only)
            h["X-DashScope-Async"] = "enable"
        return h

    # ------------------------------------------------------------
    # Task polling
    # ------------------------------------------------------------

    def _poll_task(self, task_id: str, base_url: Optional[str] = None) -> dict:
        """
        Poll async task until completion.
        """
        base = (base_url or self.base_url).rstrip("/")
        url = f"{base}/api/v1/tasks/{task_id}"

        start = time.time()
        while True:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"Wanx task {task_id} timeout after {self.timeout}s.")

            resp = requests.get(url, headers=self._headers(async_enable=False), timeout=30)
            if resp.status_code >= 400:
                _print_http_error("[WANX-POLL]", resp)
                resp.raise_for_status()

            data = resp.json()
            status = data.get("output", {}).get("task_status")

            if status == "SUCCEEDED":
                return data
            if status == "FAILED":
                raise RuntimeError(f"Wanx task failed: {json.dumps(data, ensure_ascii=False)}")

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------
    # Robust POST (with region fallback on 401/403)
    # ------------------------------------------------------------

    def _post_with_fallback(
        self,
        url_path: str,
        payload: Dict[str, Any],
        *,
        timeout: int = 60,
        async_enable: bool = True,
        tag: str = "[WANX]",
    ) -> Dict[str, Any]:
        """
        POST to base_url + url_path.
        If 401/403, auto retry with swapped region host once.
        """
        bases_to_try: List[str] = [self.base_url]
        alt = _swap_region_host(self.base_url)
        if alt != self.base_url:
            bases_to_try.append(alt)

        last_exc: Optional[Exception] = None

        for base in bases_to_try:
            url = f"{base.rstrip('/')}{url_path}"
            try:
                resp = requests.post(url, headers=self._headers(async_enable=async_enable), json=payload, timeout=timeout)
                if resp.status_code in (401, 403):
                    _print_http_error(f"{tag} AUTH", resp)
                    # try next base
                    continue
                if resp.status_code >= 400:
                    _print_http_error(f"{tag} ERR", resp)
                    resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError(f"{tag} request failed with unknown error.")


# ============================================================
# Text → Image
# ============================================================

class WanxImageGenClient(_WanxBaseClient):
    def generate(self, prompt: str, out_path: Path, size: str = "1024*1024") -> Path:
        """
        Generate image from text prompt (async task).
        Returns saved image path.
        """
        if out_path.exists():
            print(f"[WANX-GEN] Cache hit, reuse: {out_path}")
            return out_path

        print(f"[WANX-GEN] Requesting generation: model={self.model}")

        url_path = "/api/v1/services/aigc/text2image/image-synthesis"
        payload = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {
                "n": 1,
                "size": size,
            },
        }

        # retry loop (cost-safe)
        for attempt in range(self.max_retries + 1):
            try:
                data = self._post_with_fallback(url_path, payload, timeout=60, async_enable=True, tag="[WANX-GEN]")
                task_id = data.get("output", {}).get("task_id")
                if not task_id:
                    raise RuntimeError(f"[WANX-GEN] Invalid response: {data}")

                print(f"[WANX-GEN] Task created: {task_id}")

                # IMPORTANT: poll must use SAME region base that created task.
                # The response may include request_id but not base; we infer by trying both in _poll_task via fallback is not safe.
                # Here we poll first with current base_url; if fails auth, user likely has region mismatch.
                final = self._poll_task(task_id, base_url=self.base_url)

                image_url = (final.get("output", {}).get("results", [{}])[0].get("url"))
                if not image_url:
                    raise RuntimeError(f"[WANX-GEN] No image URL in task result: {final}")

                img = requests.get(image_url, timeout=60)
                if img.status_code >= 400:
                    _print_http_error("[WANX-GEN-DL]", img)
                    img.raise_for_status()

                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(img.content)

                print(f"[WANX-GEN] Image saved to: {out_path}")
                return out_path

            except Exception as e:
                if attempt >= self.max_retries:
                    raise
                print(f"[WANX-GEN] retry {attempt+1}/{self.max_retries} after error: {e}")
                time.sleep(2)

        raise RuntimeError("[WANX-GEN] failed after retries.")


# ============================================================
# Image Edit (Image2Image)
# ============================================================

class WanxImageEditClient(_WanxBaseClient):
    def edit(self, image_path: Path, instruction: str, out_path: Path) -> Path:
        """
        Edit image using instruction (async task).
        NOTE: Official HTTP API expects input.images as URL list (not base64).
        So this client currently requires the image to be accessible via URL.

        Practical workaround:
        - If you only run locally, keep dry_run=1 for now.
        - Or upload image to OSS / a temporary HTTP server and pass URL.
        """
        if out_path.exists():
            print(f"[WANX-EDIT] Cache hit, reuse: {out_path}")
            return out_path

        # We cannot send local path directly to official HTTP API (expects URLs).
        # Fail fast with a clear message.
        raise RuntimeError(
            "Wanx image editing HTTP API expects input.images as public URLs. "
            "Your pipeline currently produces local files only. "
            "Options: (1) keep --dry_run 1; (2) add an upload step (OSS / temp server) and pass URL; "
            "(3) switch to Qwen Image Edit (it supports base64 in some modes)."
        )
