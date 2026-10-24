"""Output modes — user-controlled response shape (pure, deterministic).

The user says what shape the answer takes ("코드만", "설명만",
"summary and code", ...); this module parses that request (or the
stored preference value) and reshapes the assembled turn's response
text accordingly — no side effects, no fabrication: when the requested
content is absent the answer says so honestly instead of inventing it.

AUTO (the default) means "the current assembled output": the caller
must leave the response untouched.
"""
from __future__ import annotations

import re
from enum import Enum


class OutputMode(Enum):
    AUTO = "auto"
    EXPLAIN_AND_CODE = "explain_and_code"
    CODE_ONLY = "code_only"
    EXPLANATION_ONLY = "explanation_only"
    SUMMARY_AND_CODE = "summary_and_code"
    DIFF_ONLY = "diff_only"
    PROGRESS_ONLY = "progress_only"


# -- honest notes: absent content is reported, never fabricated -------
NO_CODE_NOTE = "생성된 코드가 없어서 코드만 보여드릴 수 없습니다."
NO_EXPLANATION_NOTE = "설명이 없어서 설명만 보여드릴 수 없습니다."
NO_DIFF_NOTE = "변경사항(diff) 정보가 없어서 변경사항만 보여드릴 수 없습니다."
NO_PROGRESS_NOTE = ("진행 상황이나 검증 정보가 없어서 진행상황만 "
                    "보여드릴 수 없습니다.")
NO_CONTENT_NOTE = "보여드릴 내용이 없습니다."


# -- request parsing ---------------------------------------------------
# Ordered: an explicit negation/short form wins over a longer phrase.
_CODE_ONLY = (
    "코드만", "코드 만", "설명 없이", "설명없이", "해설 없이",
    "코드만 보여", "code only", "only code", "only the code",
    "just code", "just the code", "no explanation", "no explanations",
    "without explanation", "without explanations",
)
_EXPLANATION_ONLY = (
    "설명만", "설명 만", "코드 없이", "코드없이", "코드 제외",
    "explanation only", "only explanation", "only the explanation",
    "just explanation", "no code", "without code", "explain only",
)
_SUMMARY_AND_CODE = (
    "요약이랑 코드", "요약과 코드", "요약 + 코드", "요약이랑코드",
    "summary and code", "summary with code", "code and summary",
    "short summary and code",
)
_EXPLAIN_AND_CODE = (
    "코드랑 설명", "코드와 설명", "코드랑설명", "설명이랑 코드",
    "코드도 설명도", "explain and code", "code and explanation",
    "explanation and code", "code with explanation",
    "explanation with code", "both code and explanation",
)
_DIFF_ONLY = (
    "변경사항만", "변경 사항만", "변경내용만", "변경 내용만", "차이점만",
    "diff만", "diff 만", "diff only", "only the diff", "changes only",
    "only changes", "only the changes", "just the diff",
)
_PROGRESS_ONLY = (
    "진행상황만", "진행 상황만", "진행만", "상태만",
    "progress only", "only progress", "status only", "just the status",
    "just status",
)

_MODE_PHRASES = (
    (OutputMode.CODE_ONLY, _CODE_ONLY),
    (OutputMode.EXPLANATION_ONLY, _EXPLANATION_ONLY),
    (OutputMode.SUMMARY_AND_CODE, _SUMMARY_AND_CODE),
    (OutputMode.EXPLAIN_AND_CODE, _EXPLAIN_AND_CODE),
    (OutputMode.DIFF_ONLY, _DIFF_ONLY),
    (OutputMode.PROGRESS_ONLY, _PROGRESS_ONLY),
)


def parse_output_mode(text: str) -> "OutputMode | None":
    """The output mode the text explicitly asks for, or None.

    Korean and English phrases; None means "no explicit preference"
    (never a guess — AUTO is decided by the caller).
    """
    if not text:
        return None
    low = str(text).lower()
    for mode, phrases in _MODE_PHRASES:
        if any(p in low for p in phrases):
            return mode
    return None


# -- result inspection (multiple key shapes are supported) -------------
def _extract_code(result: dict) -> "tuple[str, str]":
    """The turn's primary code and its language, or ("", "")."""
    if not isinstance(result, dict):
        return "", ""
    direct = result.get("code")
    if isinstance(direct, str) and direct.strip():
        return direct, str(result.get("language") or "")

    def _from_languages(langs: dict) -> "tuple[str, str]":
        if not isinstance(langs, dict):
            return "", ""
        order = ["python"] + [k for k in langs if k != "python"]
        for lang in order:
            spec = langs.get(lang)
            if isinstance(spec, dict):
                code = spec.get("code")
                if isinstance(code, str) and code.strip():
                    return code, str(lang)
            elif isinstance(spec, str) and spec.strip():
                return spec, str(lang)
        return "", ""

    results = result.get("results")
    if isinstance(results, dict):
        for entry in results.values():
            if not isinstance(entry, dict):
                continue
            code, lang = _from_languages(entry.get("languages"))
            if code:
                return code, lang
            entry_code = entry.get("code")
            if isinstance(entry_code, str) and entry_code.strip():
                return entry_code, str(entry.get("language") or "")
    return _from_languages(result.get("languages"))


def _extract_explanation(result: dict) -> str:
    """The turn's explanation text, or ""."""
    if not isinstance(result, dict):
        return ""
    for key in ("explanation", "response", "text"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = result.get("response_content")
    if isinstance(content, dict):
        value = content.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```[^\n]*\n.*?(?:```|$)", "", text, flags=re.S)
    text = re.sub(r"^[ \t]*```[^\n]*$", "", text, flags=re.M)
    return text.strip()


def _format_code(code: str, language: str) -> str:
    lang = str(language or "").strip().lower()
    body = code.rstrip("\n")
    if lang:
        return f"```{lang}\n{body}\n```"
    return f"```\n{body}\n```"


def _short_summary(text: str, limit: int = 160) -> str:
    first = text.strip().split("\n", 1)[0].strip()
    match = re.match(r"^(.+?[.!?。！？])(?:\s|$)", first)
    summary = match.group(1) if match else first
    if len(summary) > limit:
        summary = summary[:limit].rstrip() + "…"
    return summary


def _verification_lines(verification) -> list:
    if isinstance(verification, bool):
        return [f"검증: {'PASS' if verification else 'FAIL'}"]
    if isinstance(verification, dict):
        if "success" in verification:
            status = "PASS" if verification.get("success") else "FAIL"
            line = f"검증: {status}"
            total = verification.get("total_cases")
            if total is not None:
                line += (f" (통과 {verification.get('passed_cases', 0)}"
                         f"/{total})")
            return [line]
        lines = []
        for key in sorted(verification):
            value = verification[key]
            if isinstance(value, (dict, list)):
                continue
            lines.append(f"- {key}: {value}")
        return lines
    if isinstance(verification, str) and verification.strip():
        return [f"검증: {verification.strip()}"]
    return []


def shape_response(result: dict, mode) -> str:
    """The response TEXT honoring ``mode`` for the given result dict.

    Pure string transform: keys read are "response" / "explanation" /
    "text" / "code" / "results" (languages→code) / "steps" /
    "verification" / "diff" / "changes".  Missing content yields an
    honest Korean note — never fabricated text.  AUTO returns the
    assembled response unchanged.
    """
    if not isinstance(result, dict):
        return ""
    try:
        mode = mode if isinstance(mode, OutputMode) else OutputMode(mode)
    except ValueError:
        mode = OutputMode.AUTO

    if mode is OutputMode.AUTO:
        return _extract_explanation(result)

    explanation = _extract_explanation(result)
    code, language = _extract_code(result)

    if mode is OutputMode.CODE_ONLY:
        if not code:
            return NO_CODE_NOTE
        return _format_code(code, language)

    if mode is OutputMode.EXPLANATION_ONLY:
        if not explanation:
            return NO_EXPLANATION_NOTE
        stripped = _strip_code_fences(explanation)
        return stripped if stripped else NO_EXPLANATION_NOTE

    if mode is OutputMode.SUMMARY_AND_CODE:
        summary = _short_summary(explanation) if explanation else ""
        if not code:
            note = NO_CODE_NOTE
            return f"{summary}\n\n{note}" if summary else note
        if summary:
            return f"{summary}\n\n{_format_code(code, language)}"
        return _format_code(code, language)

    if mode is OutputMode.EXPLAIN_AND_CODE:
        parts = []
        if explanation:
            parts.append(explanation)
        if code:
            parts.append(_format_code(code, language))
        return "\n\n".join(parts) if parts else NO_CONTENT_NOTE

    if mode is OutputMode.DIFF_ONLY:
        diff = result.get("diff")
        if not (isinstance(diff, str) and diff.strip()):
            diff = result.get("changes")
        if isinstance(diff, str) and diff.strip():
            return diff.strip()
        if isinstance(diff, dict) and diff:
            sections = []
            for path, value in diff.items():
                text = value if isinstance(value, str) else str(value)
                sections.append(f"--- {path}\n{text}" if path else text)
            joined = "\n".join(sections).strip()
            if joined:
                return joined
        return NO_DIFF_NOTE

    if mode is OutputMode.PROGRESS_ONLY:
        lines = []
        steps = result.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    step = (step.get("description") or step.get("step")
                            or step.get("text") or "")
                text = str(step).strip()
                if text:
                    lines.append(f"- {text}")
        elif isinstance(steps, str) and steps.strip():
            lines.extend(line.rstrip() for line in steps.strip().split("\n")
                         if line.strip())
        lines.extend(_verification_lines(result.get("verification")))
        return "\n".join(lines) if lines else NO_PROGRESS_NOTE

    return _extract_explanation(result)
