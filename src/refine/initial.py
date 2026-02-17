# -*- coding: utf-8 -*-
"""
src/refine/initial.py

Initial artifact generation.

Now:
- Chat LLM no longer handles image generation.
- WanxImageGenClient is used for real image generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.refine.editor import ArtifactHandle
from src.llm.wanx_client import WanxImageGenClient


# ============================================================
# Generate Initial Artifact
# ============================================================

def generate_initial_artifact(
    *,
    prompt_id: str,
    prompt_text: str,
    out_dir: Path,
    image_gen_client: Optional[WanxImageGenClient],
    backend: str,
) -> Tuple[ArtifactHandle, Dict[str, Any]]:
    """
    Generate initial image artifact.

    Behavior:
        - mock backend → return mock artifact
        - openai backend → use WanxImageGenClient.generate()
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "initial.png"
    meta_path = out_dir / "initial.json"

    # ------------------------------------------------------------
    # MOCK MODE
    # ------------------------------------------------------------

    if backend == "mock":
        artifact = ArtifactHandle(
            payload="mock://image/init",
            meta={"source": "mock", "prompt_id": prompt_id},
        )
        return artifact, {"mode": "mock"}

    # ------------------------------------------------------------
    # REAL IMAGE GENERATION (Wanx)
    # ------------------------------------------------------------

    if image_gen_client is None:
        raise ValueError("image_gen_client must be provided for real backend.")

    # Avoid duplicate cost
    if img_path.exists():
        meta = {}
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact = ArtifactHandle(
            payload=str(img_path),
            meta={"source": "wanx", "prompt_id": prompt_id},
        )
        return artifact, meta

    # Generate via Wanx
    generated_path = image_gen_client.generate(
        prompt=prompt_text,
        out_path=img_path,
    )

    meta = {
        "prompt_id": prompt_id,
        "model": image_gen_client.model,
        "source": "wanx",
    }

    import json
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifact = ArtifactHandle(
        payload=str(generated_path),
        meta={"source": "wanx", "prompt_id": prompt_id},
    )

    return artifact, meta
