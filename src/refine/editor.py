# -*- coding: utf-8 -*-
"""
src/refine/editor.py

Editor for the refinement loop.

Role:
- Execute the edit_instruction on the current artifact (image/video) to produce a new candidate.

Design:
- This module is backend-agnostic. The artifact is treated as an opaque handle (path/URI/ID/object).
- You can plug in a real editor model later (e.g., Qwen-Image-Edit, SDXL inpaint, etc.)
  by implementing the `EditBackend` protocol.

Default behavior (no backend):
- "Dry-run" mode: does not modify the artifact, but produces a new ArtifactHandle that
  records the edit_instruction in metadata (so the loop can run end-to-end deterministically).

This is enough for:
- pipeline wiring
- unit tests (simulate editor)
- trace generation

When integrating a real editor:
- implement EditBackend.apply(...)
- return a new artifact handle (e.g., file path to the edited image)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol
import hashlib
import json
import time


# ============================================================
# Artifact handle (lightweight, opaque-friendly)
# ============================================================

@dataclass(frozen=True)
class ArtifactHandle:
    """
    A lightweight wrapper around your artifact.

    - `payload` can be a path/URI/bytes/object, anything you want.
    - `meta` stores provenance: edit instruction, timestamps, parent linkage, etc.
    """
    payload: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_debug_str(self) -> str:
        p = self.payload
        p_repr = p if isinstance(p, str) else type(p).__name__
        return f"ArtifactHandle(payload={p_repr}, meta_keys={sorted(list(self.meta.keys()))})"


# ============================================================
# Backend protocol
# ============================================================

class EditBackend(Protocol):
    """
    A pluggable backend that actually edits images/videos.

    Contract:
    - Must accept a previous artifact and an edit_instruction.
    - Must return a NEW artifact handle (or other payload) representing the edited result.
    - Should be robust and ideally deterministic given fixed seeds.
    """

    def apply(self, artifact: ArtifactHandle, edit_instruction: str) -> ArtifactHandle:
        ...


# ============================================================
# Params
# ============================================================

@dataclass(frozen=True)
class EditorParams:
    """
    Knobs:
    - dry_run: if True or backend is None, we do not perform real editing.
    - max_instruction_chars: truncate overly long instructions (defensive)
    - attach_instruction_to_meta: store instruction into returned artifact meta
    - attach_parent_to_meta: store parent artifact fingerprint
    """
    dry_run: bool = True
    max_instruction_chars: int = 800
    attach_instruction_to_meta: bool = True
    attach_parent_to_meta: bool = True


# ============================================================
# Editor
# ============================================================

class Editor:
    """
    Editor wrapper.

    Usage:
        ed = Editor(params=EditorParams(dry_run=True))
        new_art = ed.edit(old_art, "Increase panda count to 5...")
    """

    def __init__(self, params: Optional[EditorParams] = None, backend: Optional[EditBackend] = None) -> None:
        self.params = params or EditorParams()
        self.backend = backend

    def edit(self, artifact: ArtifactHandle, edit_instruction: str) -> ArtifactHandle:
        """
        Apply an edit instruction, producing a new artifact.

        If backend is provided AND dry_run is False -> delegate to backend.
        Else -> produce a deterministic "virtual edited" artifact by cloning payload and updating metadata.
        """
        instr = (edit_instruction or "").strip()
        if self.params.max_instruction_chars > 0 and len(instr) > self.params.max_instruction_chars:
            instr = instr[: self.params.max_instruction_chars].rstrip() + "…"

        if (not self.params.dry_run) and (self.backend is not None):
            # Real backend edit
            out = self.backend.apply(artifact=artifact, edit_instruction=instr)
            return out

        # Dry-run: create a new handle with updated metadata
        parent_fp = self._fingerprint_artifact(artifact)
        new_meta = dict(artifact.meta) if artifact.meta else {}

        if self.params.attach_parent_to_meta:
            new_meta["parent_fp"] = parent_fp

        if self.params.attach_instruction_to_meta:
            new_meta["last_edit_instruction"] = instr

        # Add a monotonic timestamp (debug/tracing)
        new_meta["edited_at_unix"] = int(time.time())

        # Create a deterministic "edit_id" that depends on parent + instruction
        edit_id = self._make_edit_id(parent_fp, instr)
        new_meta["edit_id"] = edit_id

        # Payload remains unchanged in dry-run (opaque)
        return ArtifactHandle(payload=artifact.payload, meta=new_meta)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _fingerprint_artifact(self, artifact: ArtifactHandle) -> str:
        """
        Best-effort fingerprint for reproducibility.

        If payload is a string (path/URI), hash it.
        Else hash type name + meta (stable json).
        """
        h = hashlib.sha256()
        if isinstance(artifact.payload, str):
            h.update(artifact.payload.encode("utf-8", errors="ignore"))
        else:
            h.update(type(artifact.payload).__name__.encode("utf-8"))
        h.update(self._stable_json(artifact.meta).encode("utf-8"))
        return h.hexdigest()[:16]

    def _make_edit_id(self, parent_fp: str, instr: str) -> str:
        h = hashlib.sha256()
        h.update(parent_fp.encode("utf-8"))
        h.update(b"|")
        h.update(instr.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def _stable_json(self, obj: Dict[str, Any]) -> str:
        try:
            return json.dumps(obj or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return "{}"
