"""Form segmentation of a submission: instruction vs. evidence.

A submission carries two kinds of material — what the speaker ASKS for
(the instruction / command) and what the speaker SHOWS as support (the
evidence: a code fragment, a stack trace, sample input/output).  When
both are present the form itself separates them: fenced regions, trace
frames and labeled sample I/O are STRUCTURE, recognized by their
delimiters and canonical line shapes, never by their topic or words.

The segmentation is therefore an identity on unstructured text: a
submission with no such structure is its own instruction and yields no
evidence.  An instruction that would be emptied (a submission that is
pure evidence) keeps the whole submission as its instruction, so the
downstream parse never sees less text than the speaker gave.

Shape
-----
InputDocument(raw, instruction, evidence[]) with evidence pieces of
kind: "code" (fenced or an unfenced signature-bearing Python block),
"traceback" (a recognized stack trace), "sample_io" (a labeled
Input/Output section).  The caller attaches the document to the turn's
representation and parses the instruction alone.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_FENCE_BLOCK = re.compile(
    r"(?ms)^[ \t]*```([^\n`]*)\n(.*?)^[ \t]*```[ \t]*$")

_TRACE_STARTS = re.compile(
    r"(?m)^(Traceback \(most recent call last\):"
    r"|Exception in thread [^\n]*"
    r'|File "[^"]+", line \d+, in [^\n]*'
    r"|\w[\w.]*\.(?:Error|Exception):[^\n]*"
    r"|\w+(?:Error|Exception):[^\n]*)")

_TRACE_CONTINUATION = re.compile(
    r"(?i)(traceback|exception|[\w.]*error\b|"
    r"\bat\s+[\w.$]+\(|^\s|file \")")

_SAMPLE_ANY = (
    r"example input|example output|sample input|sample output|"
    r"예제\s*입력|예제\s*출력|샘플\s*입력|샘플\s*출력|입력|출력|input|output")

_SAMPLE_START = re.compile(
    r"(?im)^[ \t]*(?:" + _SAMPLE_ANY + r")[ \t]*"
    r"(?::[ \t]*(?P<c1>[^\n]*)|[:：][ \t]*(?P<c2>[^\n]*)"
    r"|[ \t]+(?P<c3>[^\n]*)|[ \t]*)$")

_DATA_HEAD = "[({'\"0123456789+-"

_DATA_LINE = re.compile(r"[ \t]*[\[({'\"]?[-+]?[\d.]")

_SIGNATURE_LINE = re.compile(
    r"(?m)^[ \t]*(?:async[ \t]+def|def|class|@"
    r"|[ \t]*from[ \t]+\w|[ \t]*import[ \t]+\w)")

_NORMALIZE_BLANKS = re.compile(r"\n{3,}")


@dataclass
class EvidencePiece:
    """One structural region the speaker showed but did not ask about.

    A piece COMPOSES: its span says where in the raw submission it
    stands, its id and parent id say what contains what (a sample
    inside a fenced block is that block's child, never a dropped
    region)."""

    kind: str = ""            # "code" | "traceback" | "sample_io"
    text: str = ""
    language: str = ""
    piece_id: str = ""
    parent_id: str = ""
    span: list = field(default_factory=list)
    children: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "text": self.text}
        if self.language:
            out["language"] = self.language
        if self.piece_id:
            out["piece_id"] = self.piece_id
        if self.parent_id:
            out["parent_id"] = self.parent_id
        if self.span:
            out["span"] = list(self.span)
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


@dataclass
class InputDocument:
    """A submission split into what to do (instruction) and what to
    look at (evidence).  `command` aliases `instruction`."""

    raw: str = ""
    instruction: str = ""
    evidence: list[EvidencePiece] = field(default_factory=list)

    @property
    def command(self) -> str:
        return self.instruction

    @property
    def is_structured(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "evidence": [e.to_dict() for e in self.evidence],
            "raw_length": len(self.raw),
        }


def _overlaps(start: int, end: int, spans: list) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


def _trace_extent(text: str, start: int) -> int:
    """End offset of the stack trace beginning at `start`: through the
    contiguous trace lines, stopping at a blank line or prose."""
    end = start
    for line in text[start:].splitlines(keepends=True):
        if not line.strip():
            break
        if _TRACE_CONTINUATION.search(line):
            end += len(line)
            continue
        break
    return max(end, start)


def _paragraph_spans(text: str):
    pos = 0
    for block in re.split(r"(\n[ \t]*\n)", text):
        if block and not re.fullmatch(r"\n[ \t]*\n", block):
            yield pos, pos + len(block), block
        pos += len(block)


def _is_code_paragraph(block: str) -> bool:
    if not _SIGNATURE_LINE.search(block):
        return False
    try:
        ast.parse(block)
    except (SyntaxError, ValueError):
        return False
    return True


def segment_submission(text: str, split: bool = True) -> InputDocument:
    """Segment a submission into instruction and evidence by structure.

    `split=False` returns the submission whole (no instruction/evidence
    separation is performed) — used when the surrounding pipeline
    already parses the text as a whole, e.g. a structurally verified
    problem statement.
    """
    raw = text or ""
    if not split:
        return InputDocument(raw=raw, instruction=raw, evidence=[])
    spans: list = []
    pieces: list[tuple[int, int, EvidencePiece]] = []
    nested: list[tuple[EvidencePiece, EvidencePiece]] = []

    def _keep(start: int, end: int, piece: EvidencePiece) -> None:
        if end <= start:
            return
        # A region INSIDE an accepted piece is that piece's child: it
        # is composed, never silently dropped and never a second
        # top-level claim on the same text.
        container = next((i for i, (s, e) in enumerate(spans)
                          if s <= start and end <= e), None)
        if container is not None:
            parent = pieces[container][2]
            piece.span = [start, end]
            nested.append((piece, parent))
            parent.children.append(piece)
            return
        # A region that only PARTIALLY overlaps an accepted piece is a
        # cut nobody asked for: it stays out (as it always was).
        if _overlaps(start, end, spans):
            return
        spans.append((start, end))
        pieces.append((start, end, piece))

    for m in _FENCE_BLOCK.finditer(raw):
        info = (m.group(1) or "").strip()
        lang = info.split()[0] if info else ""
        _keep(m.start(), m.end(),
              EvidencePiece(kind="code", text=m.group(0),
                            language=lang))

    for m in _TRACE_STARTS.finditer(raw):
        _keep(m.start(), _trace_extent(raw, m.start()),
              EvidencePiece(kind="traceback", text=""))

    for m in _SAMPLE_START.finditer(raw):
        content = (m.group("c2") or m.group("c1") or m.group("c3")
                   or "").strip()
        if content:
            if content[0] not in _DATA_HEAD:
                continue
            end = m.end()
        else:
            end = m.end()
            if end < len(raw) and raw[end] == "\n":
                end += 1
            for line in raw[end:].splitlines(keepends=True):
                if not line.strip():
                    break
                if _SAMPLE_START.match(line.rstrip("\n")):
                    break
                end += len(line)
            body = raw[m.end():end]
            body_lines = [ln for ln in body.splitlines() if ln.strip()]
            if not body_lines or any(
                    not _DATA_LINE.match(ln) for ln in body_lines):
                continue
        _keep(m.start(), end, EvidencePiece(kind="sample_io", text=""))

    for start, end, block in _paragraph_spans(raw):
        if _overlaps(start, end, spans):
            continue
        if _is_code_paragraph(block):
            _keep(start, end,
                  EvidencePiece(kind="code", text=block.strip("\n")))

    pieces.sort(key=lambda p: p[0])
    for index, (_start, _end, piece) in enumerate(pieces):
        piece.piece_id = f"e{index}"
        piece.span = [_start, _end]
    # the child's parent id is the ID the container ended up with, its
    # own id is the container's, and its text is the span it stands on
    for child, parent in nested:
        child.parent_id = parent.piece_id
        index = parent.children.index(child) \
            if child in parent.children else 0
        child.piece_id = f"{parent.piece_id}.{index}"
        if not child.text:
            child.text = raw[child.span[0]:child.span[1]]
    evidence = []
    instruction = raw
    for start, end, piece in reversed(pieces):
        if not piece.text:
            piece.text = raw[start:end]
        evidence.append(piece)
        instruction = instruction[:start] + instruction[end:]
    evidence.reverse()
    if pieces:
        instruction = _NORMALIZE_BLANKS.sub("\n\n", instruction).strip()
    if not instruction:
        instruction = raw
    return InputDocument(raw=raw, instruction=instruction,
                         evidence=evidence)
