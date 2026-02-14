# -*- coding: utf-8 -*-
"""
src/llm/normalize.py

Controlled enums / normalization utilities.

Why:
- LLM outputs are messy: "light-blue", "sky blue", "Blue", "#00f"
- Spatial relations may vary: "to the left of", "left", "left_of"
- Text constraints may include quotes, punctuation variations.

This module provides:
- normalize_color: map free-form color strings into a controlled set (or keep raw)
- normalize_relation: map free-form spatial relation strings into controlled enum names
- normalize_object: light normalization for object names (lower, strip, singular-ish)
- normalize_constraint: apply all normalizers to a Constraint (best-effort)

Design goals:
- Deterministic
- Conservative: never hallucinate; if unsure, keep the original normalized token
- Plug-and-play: can be used after Planner extraction, before graph building

It does NOT call LLM.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple
import re

from src.io.schemas import Constraint, ConstraintType, SpatialRelation


# ============================================================
# Controlled vocabularies
# ============================================================

# A small controlled color set (extend if needed).
# You can keep this compact for paper experiments.
CONTROLLED_COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray",
}

# Common synonyms -> controlled color
COLOR_SYNONYMS: Dict[str, str] = {
    # basic variants
    "grey": "gray",
    "light grey": "gray",
    "light gray": "gray",
    "dark grey": "gray",
    "dark gray": "gray",

    "violet": "purple",
    "magenta": "pink",
    "fuchsia": "pink",

    "navy": "blue",
    "sky blue": "blue",
    "light blue": "blue",
    "dark blue": "blue",

    "beige": "brown",
    "tan": "brown",
    "maroon": "red",
}

# Relation synonyms -> SpatialRelation enum value
REL_SYNONYMS: Dict[str, SpatialRelation] = {
    "left": SpatialRelation.LEFT_OF,
    "to the left": SpatialRelation.LEFT_OF,
    "to the left of": SpatialRelation.LEFT_OF,
    "left of": SpatialRelation.LEFT_OF,
    "on the left of": SpatialRelation.LEFT_OF,
    "at left of": SpatialRelation.LEFT_OF,
    "left_of": SpatialRelation.LEFT_OF,

    "right": SpatialRelation.RIGHT_OF,
    "to the right": SpatialRelation.RIGHT_OF,
    "to the right of": SpatialRelation.RIGHT_OF,
    "right of": SpatialRelation.RIGHT_OF,
    "on the right of": SpatialRelation.RIGHT_OF,
    "right_of": SpatialRelation.RIGHT_OF,

    "above": SpatialRelation.ABOVE,
    "over": SpatialRelation.ABOVE,
    "on top of": SpatialRelation.ABOVE,

    "below": SpatialRelation.BELOW,
    "underneath": SpatialRelation.BELOW,
    "beneath": SpatialRelation.BELOW,

    "inside": SpatialRelation.INSIDE,
    "in": SpatialRelation.INSIDE,
    "within": SpatialRelation.INSIDE,

    "on": SpatialRelation.ON,
    "on top": SpatialRelation.ON,

    "under": SpatialRelation.UNDER,

    "between": SpatialRelation.BETWEEN,

    "around": SpatialRelation.AROUND,
    "surrounding": SpatialRelation.AROUND,

    "stacked": SpatialRelation.STACKED,
    "stack": SpatialRelation.STACKED,

    "aligned": SpatialRelation.ALIGNED,
    "in line": SpatialRelation.ALIGNED,
    "in a line": SpatialRelation.ALIGNED,
}


# ============================================================
# Normalizers
# ============================================================

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"^[\"'“”‘’`]+|[\"'“”‘’`]+$")


def _clean_token(s: str) -> str:
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    return s.strip()


def normalize_color(color: Optional[str]) -> Optional[str]:
    """
    Normalize a color string to controlled vocab if possible.

    Returns:
      - controlled color (e.g. "blue") if confidently recognized
      - cleaned original token if not recognized
      - None if input None/empty
    """
    if not color:
        return None
    t = _clean_token(color)

    # strip prefixes like "light-", "dark-" into space variants
    t = t.replace("-", " ")
    t = _WS_RE.sub(" ", t)

    if t in CONTROLLED_COLORS:
        return t
    if t in COLOR_SYNONYMS:
        return COLOR_SYNONYMS[t]

    # hex codes or rgb(): keep as-is (cleaned)
    if re.fullmatch(r"#?[0-9a-f]{3}([0-9a-f]{3})?", t):
        return t
    if t.startswith("rgb(") or t.startswith("rgba("):
        return t

    # try partial matches: "light blue-ish"
    for k, v in COLOR_SYNONYMS.items():
        if k in t:
            return v
    for c in CONTROLLED_COLORS:
        if c in t:
            return c

    return t


def normalize_relation(rel: Optional[str]) -> Optional[str]:
    """
    Normalize relation string to SpatialRelation.value if recognized.
    Otherwise return cleaned string.
    """
    if not rel:
        return None
    t = _clean_token(rel)
    t = t.replace("-", " ")
    t = _WS_RE.sub(" ", t)

    if t in REL_SYNONYMS:
        return REL_SYNONYMS[t].value

    # if already matches enum value
    try:
        # SpatialRelation values are strings like "left_of"
        for r in SpatialRelation:
            if t == r.value:
                return r.value
    except Exception:
        pass

    # try contains match
    for k, v in REL_SYNONYMS.items():
        if k in t:
            return v.value

    return t


def normalize_object(obj: Optional[str]) -> Optional[str]:
    """
    Light normalization for object names:
    - lower
    - strip punctuation
    - collapse spaces
    - naive singularization for trailing 's' (very conservative)
    """
    if not obj:
        return None
    t = _clean_token(obj)
    t = t.replace("-", " ")
    t = _WS_RE.sub(" ", t)

    # conservative singular: only if ends with 's' and not 'ss'
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        # avoid words like "glasses" (often plural-only); keep if ends with "es"
        if not t.endswith("es"):
            t = t[:-1]
    return t


def normalize_text_value(val: Optional[str]) -> Optional[str]:
    """
    Normalize text content for TEXT constraints:
    - strip outer quotes
    - collapse whitespace
    - keep case (text rendering often case-sensitive), but trim
    """
    if val is None:
        return None
    s = str(val).strip()
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s if s else None


def normalize_constraint(c: Constraint) -> Constraint:
    """
    Apply best-effort normalization to one Constraint.

    Heuristics:
    - object: normalize_object
    - relation: normalize_relation (for SPATIAL/RELATION)
    - value:
        - COUNT: keep numeric-like token; strip words like "five" -> keep as-is (LLM should output digits ideally)
        - ATTRIBUTE: if value looks like color -> normalize_color
        - TEXT: normalize_text_value
    """
    obj = normalize_object(c.object)

    rel = c.relation
    if c.type in (ConstraintType.SPATIAL, ConstraintType.RELATION):
        rel = normalize_relation(c.relation)

    val = c.value
    if c.type == ConstraintType.TEXT:
        val = normalize_text_value(c.value)
    elif c.type == ConstraintType.ATTRIBUTE:
        # attempt color normalization if it resembles a color token
        if val is not None:
            val_norm = normalize_color(val)
            val = val_norm
    elif c.type == ConstraintType.COUNT:
        if val is not None:
            val = _clean_token(str(val))
    else:
        if val is not None:
            val = str(val).strip()

    ref = c.reference
    if ref is not None:
        ref = normalize_object(ref)

    return replace(c, object=obj or c.object, relation=rel, value=val, reference=ref)
