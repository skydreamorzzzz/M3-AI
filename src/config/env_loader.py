# -*- coding: utf-8 -*-
"""
src/config/env_loader.py

Unified environment variable loading from .env file.
"""

from pathlib import Path


def load_env(env_file: str = None):
    """
    Load .env file if it exists.
    
    Args:
        env_file: Path to .env file. If None, searches in project root.
    
    Note:
        - Uses python-dotenv if available
        - Falls back to system environment variables if python-dotenv not installed
        - Existing environment variables are NOT overwritten (override=False)
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[ENV] python-dotenv not installed. Skipping .env file.")
        return
    
    if env_file is None:
        # Default: project root / .env
        root = Path(__file__).resolve().parents[2]
        env_file = root / ".env"
    else:
        env_file = Path(env_file)
    
    if not env_file.exists():
        print(f"[ENV] .env file not found: {env_file}. Using system environment only.")
        return
    
    load_dotenv(dotenv_path=env_file, override=False)
    print(f"[ENV] Loaded .env from: {env_file}")

