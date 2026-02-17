# -*- coding: utf-8 -*-
"""
src/refine/editor.py

Image editing module.

Refactored:
- Chat LLM no longer edits images
- WanxImageEditClient handles real image editing
- Dry-run supported
- Debug-safe logs added
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Artifact Handle
# ============================================================

@dataclass
class ArtifactHandle:
    """
    Represents an artifact (image or mock payload).
    """
    payload: str
    meta: Optional[dict] = None


# ============================================================
# Editor Params
# ============================================================

@dataclass
class EditorParams:
    dry_run: bool = True


# ============================================================
# Editor
# ============================================================

class Editor:
    """
    Editor applies edit instruction to artifact.

    If dry_run=True:
        - No real editing
        - Returns mock-updated artifact

    If dry_run=False:
        - Calls WanxImageEditClient
    """

    def __init__(
        self,
        params: EditorParams,
        backend: Optional[Any] = None,
    ):
        self.params = params
        self.backend = backend  # WanxImageEditClient

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def edit(
        self,
        artifact: ArtifactHandle,
        instruction: str,
        round_id: int,
        prompt_id: str,
        out_dir: Path,
    ) -> ArtifactHandle:
        """
        Apply edit instruction.

        Returns new ArtifactHandle.
        """

        print(f"[EDITOR] Round {round_id}: editing artifact for prompt {prompt_id}")

        # ------------------------------------------------------------
        # Dry Run
        # ------------------------------------------------------------

        if self.params.dry_run:
            print("[EDITOR] Dry-run mode: no real image edit.")
            return ArtifactHandle(
                payload=artifact.payload,
                meta={
                    "edited": False,
                    "dry_run": True,
                    "instruction": instruction,
                },
            )

        # ------------------------------------------------------------
        # Real Wanx Edit
        # ------------------------------------------------------------

        if self.backend is None:
            raise ValueError("Editor backend is not configured.")

        if not artifact.payload or artifact.payload.startswith("mock://"):
            raise ValueError("Cannot edit mock artifact in real mode.")

        image_path = Path(artifact.payload)

        # Construct output path
        edit_dir = out_dir / "_edited" / prompt_id
        edit_dir.mkdir(parents=True, exist_ok=True)

        new_path = edit_dir / f"round_{round_id}.png"

        print(f"[EDITOR] Calling Wanx edit API...")
        print(f"[EDITOR] Input image: {image_path}")
        print(f"[EDITOR] Output image: {new_path}")

        edited_path = self.backend.edit(
            image_path=image_path,
            instruction=instruction,
            out_path=new_path,
        )

        print(f"[EDITOR] Edit completed: {edited_path}")

        return ArtifactHandle(
            payload=str(edited_path),
            meta={
                "edited": True,
                "round_id": round_id,
                "instruction": instruction,
            },
        )
