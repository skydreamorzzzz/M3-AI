# -*- coding: utf-8 -*-
"""
src/config/config_validator.py

Environment configuration validation.
"""

import os
from typing import List


def validate_config(required_keys: List[str]):
    """
    Check if required environment variables are present.
    
    Args:
        required_keys: List of required environment variable names
    
    Raises:
        ValueError: If any required key is missing
    """
    missing = [k for k in required_keys if not os.getenv(k)]
    
    if missing:
        raise ValueError(
            f"缺少必需的环境变量：{', '.join(missing)}\n\n"
            "请按以下步骤配置：\n"
            "  选项 1（推荐）：创建 .env 文件在项目根目录\n"
            "    内容示例：\n"
            "      DEEPSEEK_API_KEY=sk-...\n"
            "      DASHSCOPE_API_KEY=sk-...\n\n"
            "  选项 2：设置系统环境变量\n"
            "    Windows PowerShell:\n"
            "      $env:DEEPSEEK_API_KEY='sk-...'\n"
            "    Linux/macOS:\n"
            "      export DEEPSEEK_API_KEY=sk-...\n\n"
            "  选项 3：使用 --backend mock 模式（无需 API key）"
        )

