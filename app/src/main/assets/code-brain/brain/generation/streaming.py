"""Streaming Emitter - real-time event emission during generation.

Events are emitted as the actual work happens.
The content must already exist at the moment its event is emitted.
No fake text animation — only real generated artifacts.
"""
from __future__ import annotations
from typing import Callable, Optional, TextIO
from brain.generation.events import (
    GenerationEvent, EventType, Granularity, GenerationResult,
)
import sys
import json


class StreamingEmitter:
    """Emits generation events in real-time.

    Supports multiple output modes:
    - callback: Call a function for each event
    - stream: Write to a file-like object (default: stdout)
    - silent: Collect events without output
    """

    def __init__(
        self,
        callback: Optional[Callable[[GenerationEvent], None]] = None,
        stream: Optional[TextIO] = None,
        silent: bool = False,
        format: str = "text",
    ):
        self.callback = callback
        self.stream = stream or sys.stdout
        self.silent = silent
        self.format = format
        self._events: list[GenerationEvent] = []
        self._cancelled = False

    def emit(self, event: GenerationEvent) -> None:
        """Emit a single event. Called by the generation controller."""
        self._events.append(event)

        if self._cancelled:
            return

        if self.callback:
            self.callback(event)

        if not self.silent:
            self._write(event)

    def emit_batch(self, events: list[GenerationEvent]) -> None:
        """Emit multiple events."""
        for event in events:
            self.emit(event)

    def cancel(self) -> None:
        """Cancel ongoing generation. Emits CANCELLED event."""
        cancelled_event = GenerationEvent(event_type=EventType.CANCELLED)
        # Emit the cancelled event directly (bypassing the cancelled check)
        self._events.append(cancelled_event)
        if self.callback:
            self.callback(cancelled_event)
        if not self.silent:
            self._write(cancelled_event)
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def events(self) -> list[GenerationEvent]:
        return list(self._events)

    def get_events_by_type(self, event_type: EventType) -> list[GenerationEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def _write(self, event: GenerationEvent) -> None:
        """Write event to output stream."""
        if self.format == "json":
            self.stream.write(json.dumps(event.to_dict()) + "\n")
        elif self.format == "compact":
            self.stream.write(f"{event.event_type.value}\n")
        else:
            line = self._format_text(event)
            self.stream.write(line + "\n")
        self.stream.flush()

    def _format_text(self, event: GenerationEvent) -> str:
        """Format event as human-readable text."""
        etype = event.event_type.value

        if etype == "LINE_GENERATED":
            return f"[LINE] {event.content}"
        if etype == "TOKEN_GENERATED":
            return event.content  # no bracket for tokens
        if etype == "SENTENCE_GENERATED":
            return event.content  # no bracket for sentences
        if etype == "PARAGRAPH_GENERATED":
            return event.content
        if etype == "FUNCTION_STARTED":
            return f"[FUNCTION] {event.function_name or ''}"
        if etype == "FUNCTION_COMPLETED":
            return f"[FUNCTION_COMPLETED] {event.function_name or ''}"
        if etype == "CLASS_STARTED":
            return f"[CLASS] {event.class_name or ''}"
        if etype == "CLASS_COMPLETED":
            return f"[CLASS_COMPLETED] {event.class_name or ''}"
        if etype == "FILE_STARTED":
            return f"[FILE] {event.file_path or ''}"
        if etype == "FILE_COMPLETED":
            return f"[FILE_COMPLETED] {event.file_path or ''}"
        if etype == "PROJECT_STARTED":
            return "[PROJECT_STARTED]"
        if etype == "PROJECT_COMPLETED":
            return "[PROJECT_COMPLETED]"
        if etype == "HYPOTHESIS_GENERATED":
            return f"[HYPOTHESIS #{event.hypothesis_id}] {event.content}"
        if etype == "HYPOTHESIS_SELECTED":
            return f"[SELECTED #{event.hypothesis_id}] {event.content}"
        if etype == "GRANULARITY_SELECTED":
            return f"[GRANULARITY] {event.content}"
        if etype == "VERIFIED":
            return f"[VERIFIED] {event.content}"
        if etype == "VERIFICATION_FAILED":
            return f"[VERIFICATION_FAILED] {event.content}"
        if etype == "REPAIR_STARTED":
            return "[REPAIR_STARTED]"
        if etype == "REPAIR_COMPLETED":
            return f"[REPAIR_COMPLETED] {event.content}"
        if etype == "GENERATION_COMPLETED":
            return "[GENERATION_COMPLETED]"
        if etype == "GENERATION_FAILED":
            return f"[GENERATION_FAILED] {event.content}"

        # Generic format
        if event.content:
            return f"[{etype}] {event.content}"
        return f"[{etype}]"
