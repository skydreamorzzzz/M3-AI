# -*- coding: utf-8 -*-
"""
src/config/model_config.py

配置模型与客户端构建（区分 chat 与 图像 AIGC）。

- TEXT: DeepSeek (OpenAI-compatible)
- VISION: Qwen-VL (OpenAI-compatible via DashScope compatible-mode)
- IMAGE GEN / EDIT: Wanx (DashScope async HTTP API)
"""

from dataclasses import dataclass
from typing import Optional
import os

from src.llm.client import LLMClient
from src.llm.wanx_client import WanxImageGenClient, WanxImageEditClient


# ============================================================
# Model Config Schema
# ============================================================

@dataclass
class ModelConfig:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    backend: str = "openai"  # only for LLMClient
    timeout: int = 60
    max_retries: int = 2


# ============================================================
# 🔵 TEXT LLM（DeepSeek）
# ============================================================

TEXT_LLM = ModelConfig(
    provider="deepseek",
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=None,
    temperature=0.0,
    backend="openai",
    timeout=30,
    max_retries=2,
)

# ============================================================
# 🟣 VISION JUDGE LLM（Qwen-VL via compatible-mode）
# 注意：model 名必须是文档支持的 model id
# 便宜优先：qwen2.5-vl-3b-instruct
# ============================================================

VISION_LLM = ModelConfig(
    provider="qwen",
    model="qwen2.5-vl-3b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=None,
    temperature=0.0,
    backend="openai",
    timeout=60,
    max_retries=2,
)

# ============================================================
# 🟢 IMAGE GENERATION（Wanx async）
# ============================================================

IMAGE_GEN_LLM = ModelConfig(
    provider="qwen",
    model="wanx-v1",
    base_url="https://dashscope.aliyuncs.com",
    api_key=None,
    temperature=0.0,
    backend="",
    timeout=120,
    max_retries=1,
)

# ============================================================
# 🟩 IMAGE EDITING（Qwen Image Edit，compatible-mode）
# 使用 qwen-image-edit 系列，直接支持本地文件输入
# ============================================================

IMAGE_EDIT_LLM = ModelConfig(
    provider="qwen",
    model="qwen-image-edit",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=None,
    temperature=0.0,
    backend="",
    timeout=120,
    max_retries=1,
)


# ============================================================
# Client Builders
# ============================================================

def _env_key_for_provider(provider: str) -> Optional[str]:
    p = (provider or "").lower()
    if p == "deepseek":
        return os.getenv("DEEPSEEK_API_KEY")
    if p == "qwen":
        return os.getenv("DASHSCOPE_API_KEY")
    return os.getenv("LLM_API_KEY")


def _build_llm_client(cfg: ModelConfig) -> LLMClient:
    api_key = (cfg.api_key or "").strip() or _env_key_for_provider(cfg.provider)
    if not api_key:
        raise ValueError(
            f"API 密钥未配置：{cfg.provider}/{cfg.model}。"
            "请设置环境变量（DeepSeek: DEEPSEEK_API_KEY；DashScope: DASHSCOPE_API_KEY）"
        )

    return LLMClient(
        model=cfg.model,
        backend=cfg.backend,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


def get_text_client() -> LLMClient:
    return _build_llm_client(TEXT_LLM)


def get_vision_client() -> LLMClient:
    return _build_llm_client(VISION_LLM)


def get_image_gen_client() -> WanxImageGenClient:
    cfg = IMAGE_GEN_LLM
    api_key = (cfg.api_key or "").strip() or _env_key_for_provider(cfg.provider)
    if not api_key:
        raise ValueError("Wanx 图像生成密钥未配置（DASHSCOPE_API_KEY）。")

    return WanxImageGenClient(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


def get_image_edit_client() -> WanxImageEditClient:
    cfg = IMAGE_EDIT_LLM
    api_key = (cfg.api_key or "").strip() or _env_key_for_provider(cfg.provider)
    if not api_key:
        raise ValueError("Wanx 图像编辑密钥未配置（DASHSCOPE_API_KEY）。")

    return WanxImageEditClient(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
