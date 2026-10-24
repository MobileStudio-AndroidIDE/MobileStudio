"""Explanation Events - Extended reasoning events with genuine explanations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

from brain.decision_trace import DecisionTrace


class ExplanationType(Enum):
    WHY_ADDED = "why_added"
    WHY_MODIFIED = "why_modified"
    WHY_STRUCTURE = "why_structure"
    WHY_ALGORITHM = "why_algorithm"
    ALTERNATIVES = "alternatives"
    CONSTRAINT = "constraint"
    VERIFICATION = "verification"
    REPAIR = "repair"
    COMMENT = "comment"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    FILE_START = "file_start"
    CLASS_START = "class_start"
    FUNCTION_START = "function_start"
    PIPELINE_STAGE = "pipeline_stage"
    LINE = "line"
    STRUCTURE_END = "structure_end"


class GenerationEventType(Enum):
    START = "start"
    FILE_START = "file_start"
    CLASS_START = "class_start"
    FUNCTION_START = "function_start"
    PIPELINE_STAGE = "pipeline_stage"
    LINE = "line"
    STRUCTURE_END = "structure_end"
    VERIFY_STRUCTURE = "verify_structure"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ExplanationEvent:
    """An event with a genuine, data-grounded explanation."""
    timestamp: datetime
    event_type: str
    file_path: str
    line_number: int
    code: str
    explanation_type: ExplanationType
    korean_explanation: str
    english_explanation: str
    decision_trace: Optional[DecisionTrace] = None
    structure_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code": self.code,
            "explanation_type": self.explanation_type.value,
            "korean_explanation": self.korean_explanation,
            "english_explanation": self.english_explanation,
            "decision_trace": self.decision_trace.to_dict() if self.decision_trace else None,
            "structure_context": self.structure_context,
        }


@dataclass
class GenerationEvent:
    """A single step in the streaming code generation process."""
    event_type: GenerationEventType
    timestamp: datetime
    file_path: str
    line_number: int
    code: str
    explanation: Optional[ExplanationEvent] = None
    structure_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code": self.code,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "structure_context": self.structure_context,
        }