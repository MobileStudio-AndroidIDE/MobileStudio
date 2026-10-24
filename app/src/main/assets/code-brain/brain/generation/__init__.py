"""Generation module - adaptive, streaming, non-deterministic generation.

Architecture:
    NL → Semantic Understanding → Hypotheses → Non-deterministic Reasoning
    → Planning → IR → Adaptive Generation → Real-time Events → Verification → Repair

Key components:
    - GenerationController: Central orchestrator
    - HypothesisGenerator: Multiple solution candidates
    - GranularitySelector: Adaptive granularity
    - StreamingEmitter: Real-time event emission
"""
from brain.generation.events import (
    GenerationEvent, EventType, Granularity, GenerationType,
    Hypothesis, GenerationResult,
)
from brain.generation.controller import GenerationController, ControllerConfig
from brain.generation.hypotheses import HypothesisGenerator, HypothesisConfig
from brain.generation.strategy import GranularitySelector
from brain.generation.streaming import StreamingEmitter

__all__ = [
    "GenerationController", "ControllerConfig",
    "HypothesisGenerator", "HypothesisConfig",
    "GranularitySelector",
    "StreamingEmitter",
    "GenerationEvent", "EventType", "Granularity", "GenerationType",
    "Hypothesis", "GenerationResult",
]
