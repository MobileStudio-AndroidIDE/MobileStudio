"""Strategy - adaptive granularity selection for generation.

Selects the appropriate generation granularity based on the task structure.
The user should NOT have to manually choose — the system auto-selects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from brain.generation.events import (
    Granularity, GenerationType, EventType, GenerationEvent,
)


@dataclass
class StrategyProfile:
    """Describes the selected strategy for a generation task."""
    granularity: Granularity
    generation_type: GenerationType
    reason: str
    events: list[GenerationEvent]


class GranularitySelector:
    """Selects generation granularity based on task analysis.

    Decision factors:
    - Number of files expected
    - Number of classes/functions
    - Complexity of operations
    - Goal type (function vs project vs conversation)
    - User preference (--stream flag)
    """

    def select(
        self,
        goal_type: str,
        operations: list[str],
        entities: list[dict],
        composition: dict,
        user_preference: Optional[str] = None,
    ) -> StrategyProfile:
        """Select the appropriate granularity and generation type."""
        events = []

        # User override
        if user_preference and user_preference != "auto":
            gran = Granularity(user_preference)
            gen_type = self._infer_generation_type(goal_type)
            # If user selected project granularity, force project type
            if gran == Granularity.PROJECT:
                gen_type = GenerationType.PROJECT
            events.append(GenerationEvent(
                event_type=EventType.GRANULARITY_SELECTED,
                granularity=gran,
                content=f"User selected: {gran.value}",
                metadata={"source": "user_preference"},
            ))
            return StrategyProfile(
                granularity=gran,
                generation_type=gen_type,
                reason="user_preference",
                events=events,
            )

        # Auto-select based on task analysis
        gen_type = self._infer_generation_type(goal_type)
        gran = self._auto_select_granularity(goal_type, operations, entities, composition)

        events.append(GenerationEvent(
            event_type=EventType.GRANULARITY_SELECTED,
            granularity=gran,
            content=f"Auto-selected: {gran.value} (type={gen_type.value})",
            metadata={"source": "auto", "goal": goal_type,
                      "op_count": len(operations)},
        ))

        return StrategyProfile(
            granularity=gran,
            generation_type=gen_type,
            reason="auto_select",
            events=events,
        )

    def _auto_select_granularity(
        self,
        goal_type: str,
        operations: list[str],
        entities: list[dict],
        composition: dict,
    ) -> Granularity:
        """Auto-select granularity based on task structure."""
        # Project-level
        if goal_type == "create_project":
            return Granularity.PROJECT

        # Conversational
        if goal_type in ("chat", "question", "explain"):
            return Granularity.SENTENCE

        # Function-level tasks
        if goal_type in ("create_function", "fix", "analyze"):
            op_count = len(operations)
            if op_count <= 3:
                return Granularity.LINE
            elif op_count <= 6:
                return Granularity.FUNCTION
            else:
                return Granularity.CLASS

        # Class-level
        if goal_type == "create_class":
            return Granularity.CLASS

        # Default: function level
        return Granularity.FUNCTION

    def _infer_generation_type(self, goal_type: str) -> GenerationType:
        """Infer generation type from goal."""
        mapping = {
            "create_function": GenerationType.CODE,
            "create_project": GenerationType.PROJECT,
            "create_class": GenerationType.CODE,
            "fix": GenerationType.CODE,
            "analyze": GenerationType.EXPLANATION,
            "explain": GenerationType.EXPLANATION,
            "question": GenerationType.CONVERSATIONAL,
            "chat": GenerationType.CONVERSATIONAL,
        }
        return mapping.get(goal_type, GenerationType.CODE)
