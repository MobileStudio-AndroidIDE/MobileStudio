"""Conversation response chunking.

The realized response streams as utterance CHUNKS (sentence-level
units), never as one opaque blob: the receiver renders the reply
incrementally, the same way a conversation unfolds sentence by
sentence.

The split follows the LANGUAGE's sentence terminators and the list
structure — general grammar, never problem-specific rules:
  - sentence terminators: . ! ? 。 ！ ？ and their repetitions
  - list rows: lines starting with a bullet or a dash
  - line structure: a chunk that ends a source line carries its
    newline, so streaming the chunks re-renders the original layout
"""
from __future__ import annotations

import re


def chunk_response(text: str) -> list[str]:
    """Split a realized response into utterance chunks for streaming.

    Each chunk is a complete sentence / list row; whitespace at a
    chunk's head is trimmed; empty lines become their own pause chunk.
    """
    if not text:
        return []
    chunks: list[str] = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        is_last_line = index == len(lines) - 1
        stripped = line.strip()
        if not stripped:
            if not is_last_line:
                chunks.append("\n")
            continue
        line_chunks = _split_line(stripped)
        if not is_last_line and line_chunks:
            line_chunks[-1] = line_chunks[-1] + "\n"
        chunks.extend(line_chunks)
    return chunks


def _split_line(line: str) -> list[str]:
    """Split one line into sentence chunks (a list row is its own
    chunk — the list structure is the pause, not a sentence)."""
    if _is_list_row(line):
        return [line]
    pieces: list[str] = []
    pos = 0
    length = len(line)
    while pos < length:
        match = re.search(r"[.!?。！？]+", line[pos:])
        if not match:
            tail = line[pos:].strip()
            if tail:
                pieces.append(tail)
            break
        end = pos + match.end()
        while end < length and line[end] in "\"'”’)]]":
            end += 1
        piece = line[pos:end].strip()
        if piece:
            pieces.append(piece)
        pos = end
    return pieces or [line]


def _is_list_row(line: str) -> bool:
    """A list row: the bullet/dash marker — its own chunk."""
    return bool(re.match(r"^[-*•]\s+\S", line))


def iter_chunks(text: str):
    """Yield the chunks one at a time (the streaming consumption)."""
    for chunk in chunk_response(text):
        yield chunk
