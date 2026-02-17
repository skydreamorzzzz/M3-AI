from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ---------- Constraint Types ----------

class ConstraintType(str, Enum):
    OBJECT = "object"
    COUNT = "count"
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    RELATION = "relation"
    TEXT = "text"


# ---------- Relation Enum (受控) ----------

class SpatialRelation(str, Enum):
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    ON = "on"
    UNDER = "under"
    BETWEEN = "between"
    AROUND = "around"
    STACKED = "stacked"
    ALIGNED = "aligned"

@dataclass
class Constraint:
    id: str
    type: ConstraintType
    object: str

    # optional fields
    value: Optional[str] = None         # count / attribute / text content
    relation: Optional[str] = None      # spatial relation
    reference: Optional[str] = None     # reference object (for spatial)
    confidence: float = 1.0             # LLM extraction confidence


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str  # "dependency" or "coupling" or "conflict"
    weight: float = 1.0

@dataclass
class ConstraintGraph:
    nodes: List[Constraint]
    edges: List[GraphEdge]

    def get_adjacency(self) -> Dict[str, List[str]]:
        adj = {}
        for edge in self.edges:
            adj.setdefault(edge.source, []).append(edge.target)
        return adj

@dataclass
class TraceStep:
    round_id: int
    selected_constraint: str

    status_before: Dict[str, bool]  # constraint_id -> pass/fail
    status_after: Dict[str, bool]

    degraded_constraints: List[str]  # 被破坏的约束
    improved_constraints: List[str]  # 被修复的约束

    edit_instruction: Optional[str] = None
    accepted: bool = True
    
    # Stability-related fields (added in PR-7)
    error_type: Optional[str] = None       # Error type if edit failed
    edit_fallback: bool = False            # Whether edit failed and fallback was used
    
    # Quality-related fields (added for quality-aware refinement)
    quality_score_before: Optional[float] = None  # Image quality score before edit (0-1)
    quality_score_after: Optional[float] = None   # Image quality score after edit (0-1)
    quality_improvement: bool = False             # Whether this was a quality improvement round

@dataclass
class PromptItem:
    prompt_id: str
    text: str
    constraints: List[Constraint]
    graph: Optional[ConstraintGraph] = None

@dataclass
class RunSummary:
    prompt_id: str
    total_rounds: int
    final_pass: bool
    conflict_count: int
    oscillation_detected: bool
    protection_rate: float
    
    # Quality-related fields
    final_quality_score: float = 0.0      # Final image quality score (0-1)
    quality_improved: bool = False        # Whether quality improvement was performed
