"""Decision Trace - Records actual reasoning decisions for explainable AI."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class DecisionType(Enum):
    ADD_CLASS = "add_class"
    ADD_FUNCTION = "add_function"
    CHOOSE_ALGORITHM = "choose_algorithm"
    CHOOSE_STRUCTURE = "choose_structure"
    ADD_PIPELINE_STAGE = "add_pipeline_stage"
    MODIFY_CODE = "modify_code"
    APPLY_FIX = "apply_fix"
    SELECT_HYPOTHESIS = "select_hypothesis"
    PLAN_ACTION = "plan_action"


@dataclass
class Alternative:
    """An alternative that was considered but not chosen."""
    option: str
    reason_rejected: str
    confidence: float
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "option": self.option,
            "reason_rejected": self.reason_rejected,
            "confidence": self.confidence,
            "pros": self.pros,
            "cons": self.cons,
        }


@dataclass
class DecisionTrace:
    """Records the actual decision-making process - NO fabrication."""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    chosen_option: str
    alternatives_considered: list[Alternative]
    constraints_applied: list[str]
    evidence: list[str]
    confidence: float
    reasoning_step: str
    related_concepts: list[str] = field(default_factory=list)
    related_operations: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type.value,
            "chosen_option": self.chosen_option,
            "alternatives_considered": [a.to_dict() for a in self.alternatives_considered],
            "constraints_applied": self.constraints_applied,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "reasoning_step": self.reasoning_step,
            "related_concepts": self.related_concepts,
            "related_operations": self.related_operations,
            "related_entities": self.related_entities,
        }