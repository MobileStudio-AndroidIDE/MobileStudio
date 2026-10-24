"""Generation events and enums for real-time streaming.

Events represent observable system states during generation.
They are emitted as the actual work happens — not faked or deferred.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class EventType(Enum):
    # Pipeline phases
    ANALYZING = "ANALYZING"
    UNDERSTANDING = "UNDERSTANDING"
    KNOWLEDGE = "KNOWLEDGE"
    HYPOTHESIS_STARTED = "HYPOTHESIS_STARTED"
    HYPOTHESIS_GENERATED = "HYPOTHESIS_GENERATED"
    HYPOTHESIS_SELECTED = "HYPOTHESIS_SELECTED"
    PLANNING = "PLANNING"
    IR_GENERATED = "IR_GENERATED"
    SELECTING = "SELECTING"
    # Generation granularity events
    PROJECT_STARTED = "PROJECT_STARTED"
    PROJECT_COMPLETED = "PROJECT_COMPLETED"
    FILE_STARTED = "FILE_STARTED"
    FILE_COMPLETED = "FILE_COMPLETED"
    CLASS_STARTED = "CLASS_STARTED"
    CLASS_COMPLETED = "CLASS_COMPLETED"
    FUNCTION_STARTED = "FUNCTION_STARTED"
    FUNCTION_COMPLETED = "FUNCTION_COMPLETED"
    LINE_GENERATED = "LINE_GENERATED"
    TOKEN_GENERATED = "TOKEN_GENERATED"
    SENTENCE_GENERATED = "SENTENCE_GENERATED"
    PARAGRAPH_GENERATED = "PARAGRAPH_GENERATED"
    # Verification
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    REPAIR_STARTED = "REPAIR_STARTED"
    REPAIR_ANALYZING = "REPAIR_ANALYZING"
    REPAIR_PLAN = "REPAIR_PLAN"
    REPAIR_APPLIED = "REPAIR_APPLIED"
    REVERIFYING = "REVERIFYING"
    REPAIR_SUCCEEDED = "REPAIR_SUCCEEDED"
    REPAIR_FAILED = "REPAIR_FAILED"
    REPAIR_COMPLETED = "REPAIR_COMPLETED"
    # Completion
    GENERATION_COMPLETED = "GENERATION_COMPLETED"
    GENERATION_FAILED = "GENERATION_FAILED"
    # Metadata
    GRANULARITY_SELECTED = "GRANULARITY_SELECTED"
    STRATEGY_SELECTED = "STRATEGY_SELECTED"
    CANCELLED = "CANCELLED"


class Granularity(Enum):
    TOKEN = "token"
    LINE = "line"
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    PROJECT = "project"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    AUTO = "auto"
    # Transactional patch of existing code (see code.ir.patch_ir)
    PATCH = "patch"


class GenerationType(Enum):
    CODE = "code"
    CONVERSATIONAL = "conversational"
    EXPLANATION = "explanation"
    DOCUMENTATION = "documentation"
    PROJECT = "project"


@dataclass
class GenerationEvent:
    """A single event emitted during generation."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    content: str = ""
    metadata: dict = field(default_factory=dict)
    granularity: Optional[Granularity] = None
    index: int = 0
    total: int = 0
    hypothesis_id: Optional[int] = None
    file_path: Optional[str] = None
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    is_final: bool = False

    def to_dict(self) -> dict:
        d = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "content": self.content,
            "metadata": self.metadata,
            "index": self.index,
            "total": self.total,
            "is_final": self.is_final,
        }
        if self.granularity:
            d["granularity"] = self.granularity.value
        if self.hypothesis_id is not None:
            d["hypothesis_id"] = self.hypothesis_id
        if self.file_path:
            d["file_path"] = self.file_path
        if self.class_name:
            d["class_name"] = self.class_name
        if self.function_name:
            d["function_name"] = self.function_name
        return d

    def __str__(self) -> str:
        prefix = f"[{self.event_type.value}]"
        if self.granularity:
            prefix += f" ({self.granularity.value})"
        if self.content:
            return f"{prefix} {self.content}"
        return prefix


@dataclass
class Hypothesis:
    """A single solution hypothesis."""
    id: int
    description: str
    approach: str
    operations: list[str] = field(default_factory=list)
    structure: dict = field(default_factory=dict)
    confidence: float = 0.0
    reasoning_path: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Complete result of a generation pipeline run."""
    success: bool
    output: Any = None
    events: list[GenerationEvent] = field(default_factory=list)
    hypotheses_generated: int = 0
    hypothesis_selected: Optional[int] = None
    granularity_used: Granularity = Granularity.AUTO
    generation_type: GenerationType = GenerationType.CODE
    verification_passed: bool = False
    repair_needed: bool = False
    repair_success: bool = False
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "hypotheses_generated": self.hypotheses_generated,
            "hypothesis_selected": self.hypothesis_selected,
            "granularity_used": self.granularity_used.value,
            "generation_type": self.generation_type.value,
            "verification_passed": self.verification_passed,
            "repair_needed": self.repair_needed,
            "repair_success": self.repair_success,
            "error": self.error,
            "event_count": len(self.events),
            "metadata": self.metadata,
        }
