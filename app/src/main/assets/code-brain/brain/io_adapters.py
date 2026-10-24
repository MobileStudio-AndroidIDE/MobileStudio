"""Input/Output adapters: sample I/O evidence ↔ executable checks.

The submission may SHOW its input and expected output as labeled
sample sections ("Input: [1, 2]" / "Output: [2, 1]") — evidence the
segmentation separates from the instruction.  This module adapts that
evidence into executable checks: each Input/Output pair becomes a
sample (parsed value + expected value) that verification runs against
the generated function, and each sample adapts to the function it
meets — arity-aware: a scalar is one argument, a bracketed sequence is
one sequence argument, a column of equal-length scalars is an argument
list.

Parsing is form-tolerant the way a spoken transcript is: labels are
case-insensitive, colons optional, smart quotes are plain quotes, a
bare token is its own string.  A sample that cannot be adapted is
dropped — never guessed; a submission without sample sections yields no
samples.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

_LABEL = re.compile(
    r"(?is)^\s*(example\s+input|example\s+output|sample\s+input|"
    r"sample\s+output|예제\s*입력|예제\s*출력|샘플\s*입력|샘플\s*출력|"
    r"입력|출력|input|output)\s*[:：]?\s*(.*)$")

_INPUT_LABELS = {
    "input", "입력", "example input", "sample input",
    "예제 입력", "샘플 입력",
}

_SCALAR_INT = re.compile(r"^[-+]?\d+$")
_SCALAR_FLOAT = re.compile(r"^[-+]?(?:\d+\.\d*|\d*\.\d+)$")

_SMART_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\uff02": '"', "\u02bc": "'",
}


@dataclass
class IOSample:
    """One speaker-shown Input/Output pair."""

    input: Any = None
    expected: Any = None
    has_input: bool = False
    has_expected: bool = False

    def to_dict(self) -> dict:
        return {"input": self.input, "expected": self.expected,
                "has_input": self.has_input,
                "has_expected": self.has_expected}


def _normalize(text: str) -> str:
    out = text
    for smart, plain in _SMART_QUOTES.items():
        out = out.replace(smart, plain)
    return out.strip()


def parse_value(payload: str) -> tuple[bool, Any]:
    """Parse a sample payload into a value; (False, None) when the
    payload carries nothing parsable — never a guess."""
    text = _normalize(payload)
    if not text:
        return False, None
    if text[0] in "[({\"'" or text.endswith((",", ")", "]", "}")):
        try:
            return True, ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        values = []
        for ln in lines:
            ok, val = _scalar(ln)
            if not ok:
                return False, None
            values.append(val)
        return True, values
    return _scalar(text)


def _scalar(token: str) -> tuple[bool, Any]:
    if _SCALAR_INT.match(token):
        return True, int(token)
    if _SCALAR_FLOAT.match(token):
        return True, float(token)
    if token.lower() in ("true", "false"):
        return True, token.lower() == "true"
    if token.lower() in ("none", "null"):
        return True, None
    try:
        return True, ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return True, token


def samples_from_submission(submission: dict) -> list[IOSample]:
    """Pair the submission's sample_io evidence into executable samples
    (document order: an Input binds to the Output that follows it)."""
    if not isinstance(submission, dict):
        return []
    samples: list[IOSample] = []
    pending: IOSample | None = None
    for piece in submission.get("evidence") or []:
        if not isinstance(piece, dict) or piece.get("kind") != "sample_io":
            continue
        match = _LABEL.match(str(piece.get("text") or ""))
        if not match:
            continue
        label = re.sub(r"\s+", " ", match.group(1).strip().lower())
        ok, value = parse_value(match.group(2) or "")
        if not ok:
            continue
        if label in _INPUT_LABELS:
            if pending is not None:
                samples.append(pending)
            pending = IOSample(input=value, has_input=True)
        else:
            if pending is not None:
                pending.expected = value
                pending.has_expected = True
                samples.append(pending)
                pending = None
            else:
                samples.append(IOSample(expected=value, has_expected=True))
    if pending is not None:
        samples.append(pending)
    return [s for s in samples if s.has_input]


def _adapt_args(value: Any, arity: int) -> tuple | None:
    """Adapt a sample input to the function's parameter count; None
    when the adaptation would be a guess."""
    if arity == 0:
        return ()
    if arity == 1:
        return (value,)
    if isinstance(value, (list, tuple)) and len(value) == arity:
        return tuple(value)
    return None


def _matches(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def execute_samples(code: str, samples: list[IOSample],
                    preferred_names: list[str] | None = None) -> list[str]:
    """Run the samples against the code's own function; returns the
    failure descriptions (empty when every adapted sample passes).

    `preferred_names` names the operation's function (a helper defined
    before it is not the subject) — the same selection rule the repair
    analyzer uses."""
    if not samples:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    defined = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception:
        return []
    preferred = [n for n in defined if n in (preferred_names or [])]
    fn = None
    for name in preferred + defined:
        obj = ns.get(name)
        if callable(obj):
            fn = obj
            break
    if fn is None:
        return []
    import inspect
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        arity = len(params)
    except (TypeError, ValueError):
        arity = 1
    errors: list[str] = []
    for sample in samples:
        args = _adapt_args(sample.input, arity)
        if args is None:
            continue
        try:
            actual = fn(*args)
        except Exception as exc:
            errors.append(
                f"Sample check raised: input={sample.input!r}: "
                f"{type(exc).__name__}: {exc}")
            continue
        if sample.has_expected and not _matches(actual, sample.expected):
            errors.append(
                f"Sample check failed: input={sample.input!r} "
                f"expected={sample.expected!r} got={actual!r}")
    return errors
