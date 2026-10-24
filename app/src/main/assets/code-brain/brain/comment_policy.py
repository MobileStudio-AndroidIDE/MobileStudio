"""Comment policies — user-controlled comment handling (pure, total).

Policies:
    AUTO                 no transformation (the current output).
    WITH_COMMENTS        no transformation (comments are wanted).
    NO_COMMENTS          drop comments AND standalone docstrings.
    DOCUMENTATION_ONLY   keep docstrings, drop ordinary comments
                         (documentation is what remains).
    EXPLANATION_SEPARATE drop comments, keep docstrings (the comments'
                         content belongs in the separate explanation).

Python is transformed through ``tokenize``: strings containing "#"
survive untouched, only real COMMENT tokens (and standalone string
statements, for NO_COMMENTS) are removed.  Other languages use a
conservative per-line state machine for ``//`` and ``#`` aware of
string literals.  Known limits (documented, never silent): block
comments (``/* ... */``) are NOT stripped, multi-line strings are only
tracked when a line opens and closes the same quote, and on a
tokenize failure the Python code is returned unchanged rather than
risking corruption.  Every function is deterministic and total.
"""
from __future__ import annotations

import io
import tokenize
from enum import Enum


class CommentPolicy(Enum):
    AUTO = "auto"
    WITH_COMMENTS = "with_comments"
    NO_COMMENTS = "no_comments"
    DOCUMENTATION_ONLY = "documentation_only"
    EXPLANATION_SEPARATE = "explanation_separate"


# -- request parsing ---------------------------------------------------
_NO_COMMENTS_PHRASES = (
    "주석 없이", "주석없이", "주석 빼", "주석빼", "주석 제거", "주석제거",
    "주석 삭제", "주석삭제", "주석 달지 마", "주석 달지마",
    "no comments", "no comment", "without comments", "remove comments",
    "strip comments", "comments off",
)
_DOCUMENTATION_ONLY_PHRASES = (
    "문서화만", "문서화 만", "문서만", "docstring만", "docstring 만",
    "docstring only", "docstrings only", "documentation only",
    "only documentation",
)
_EXPLANATION_SEPARATE_PHRASES = (
    "설명은 따로", "설명 따로", "설명따로", "설명을 따로", "설명은따로",
    "주석 대신 설명", "explanation separately", "separate explanation",
    "explain separately", "comments in explanation",
)
_WITH_COMMENTS_PHRASES = (
    "주석 달아", "주석달아", "주석 추가", "주석추가", "주석 넣", "주석넣",
    "주석 포함", "주석포함", "주석도", "주석 함께", "주석 같이",
    "add comments", "with comments", "include comments", "comments please",
    "more comments",
)

_POLICY_PHRASES = (
    (CommentPolicy.NO_COMMENTS, _NO_COMMENTS_PHRASES),
    (CommentPolicy.DOCUMENTATION_ONLY, _DOCUMENTATION_ONLY_PHRASES),
    (CommentPolicy.EXPLANATION_SEPARATE, _EXPLANATION_SEPARATE_PHRASES),
    (CommentPolicy.WITH_COMMENTS, _WITH_COMMENTS_PHRASES),
)


def parse_comment_policy(text: str) -> "CommentPolicy | None":
    """The comment policy the text explicitly asks for, or None.

    Korean and English phrases; None means "no explicit preference"
    (the stored preference / AUTO is the caller's decision).
    """
    if not text:
        return None
    low = str(text).lower()
    for policy, phrases in _POLICY_PHRASES:
        if any(p in low for p in phrases):
            return policy
    return None


def policy_from_stored(name: str) -> CommentPolicy:
    """Map a stored density/preference value onto a policy (AUTO when
    the value is unknown — an unknown value never changes output)."""
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "": CommentPolicy.AUTO,
        "auto": CommentPolicy.AUTO,
        "adaptive": CommentPolicy.AUTO,
        "none": CommentPolicy.NO_COMMENTS,
        "no_comments": CommentPolicy.NO_COMMENTS,
        "without_comments": CommentPolicy.NO_COMMENTS,
        "minimal": CommentPolicy.EXPLANATION_SEPARATE,
        "explanation_separate": CommentPolicy.EXPLANATION_SEPARATE,
        "separate": CommentPolicy.EXPLANATION_SEPARATE,
        "documentation_only": CommentPolicy.DOCUMENTATION_ONLY,
        "docstrings_only": CommentPolicy.DOCUMENTATION_ONLY,
        "high": CommentPolicy.WITH_COMMENTS,
        "with_comments": CommentPolicy.WITH_COMMENTS,
    }.get(key, CommentPolicy.AUTO)


# -- application -------------------------------------------------------
_PY_LANG = {"python", "py", "python3"}
_HASH_LANGS = {
    "ruby", "rb", "perl", "shell", "bash", "sh", "zsh", "fish", "r",
    "yaml", "yml", "toml", "makefile", "dockerfile", "powershell", "ps1",
    "julia", "nim", "coffee",
}
_SLASH_LANGS = {
    "javascript", "js", "jsx", "typescript", "ts", "tsx", "java", "kotlin",
    "kts", "c", "cpp", "c++", "cc", "cxx", "h", "hpp", "cs", "csharp",
    "go", "golang", "rust", "rs", "swift", "php", "dart", "scala", "groovy",
    "objective-c", "objc", "m", "mm", "solidity", "zig", "v",
}


def apply_comment_policy(code: str, policy, language: str = "python") -> str:
    """Apply ``policy`` to ``code`` and return the transformed source.

    Deterministic and total: AUTO / WITH_COMMENTS return the input
    unchanged; a value that is not a policy is also left unchanged.
    """
    if not isinstance(code, str) or not code:
        return code
    try:
        policy = policy if isinstance(policy, CommentPolicy) \
            else CommentPolicy(policy)
    except ValueError:
        return code
    if policy in (CommentPolicy.AUTO, CommentPolicy.WITH_COMMENTS):
        return code
    lang = str(language or "python").strip().lower()
    if lang in _PY_LANG:
        return _apply_python(code, policy)
    return _apply_line_based(code, policy, lang)


# -- Python (tokenize) -------------------------------------------------
def _is_standalone_string(tokens: list, index: int) -> bool:
    """The token is an expression statement that IS a string literal
    (a docstring or a string-literal no-op) — the conservative rule for
    NO_COMMENTS; implicit concatenations are left untouched."""
    prev = _prev_significant(tokens, index)
    nxt = _next_significant(tokens, index)
    if nxt is None or nxt.type != tokenize.NEWLINE:
        return False
    return prev is None or prev.type in (tokenize.NEWLINE, tokenize.NL,
                                         tokenize.INDENT, tokenize.DEDENT)


def _prev_significant(tokens: list, index: int):
    j = index - 1
    while j >= 0 and tokens[j].type in (tokenize.COMMENT, tokenize.NL,
                                        tokenize.ENCODING):
        j -= 1
    return tokens[j] if j >= 0 else None


def _next_significant(tokens: list, index: int):
    j = index + 1
    while j < len(tokens) and tokens[j].type in (tokenize.COMMENT,
                                                 tokenize.NL,
                                                 tokenize.ENCODING):
        j += 1
    return tokens[j] if j < len(tokens) else None


def _is_sole_statement(tokens: list, index: int) -> bool:
    """The string is the only statement of its block: removing it would
    empty the suite, so it is replaced with ``pass`` instead."""
    ignored = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)
    j = index + 1
    while j < len(tokens) and tokens[j].type in ignored:
        j += 1
    if j >= len(tokens) or tokens[j].type != tokenize.NEWLINE:
        return False
    j += 1
    while j < len(tokens) and tokens[j].type in ignored:
        j += 1
    return j < len(tokens) and tokens[j].type in (tokenize.DEDENT,
                                                  tokenize.ENDMARKER)


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    return ""


def _apply_python(code: str, policy: CommentPolicy) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return code

    drop_comments = True
    drop_docstrings = policy is CommentPolicy.NO_COMMENTS

    lines = code.splitlines(keepends=True)
    delete_rows: set = set()
    replace_rows: dict = {}

    comment_ops = []
    for index, token in enumerate(tokens):
        if drop_comments and token.type == tokenize.COMMENT:
            comment_ops.append((token.start[0], token.start[1]))
        elif drop_docstrings and token.type == tokenize.STRING \
                and _is_standalone_string(tokens, index):
            start_row, start_col = token.start
            end_row = token.end[0]
            if _is_sole_statement(tokens, index):
                replace_rows[start_row] = (start_col, _line_ending(
                    lines[start_row - 1] if start_row - 1 < len(lines)
                    else ""))
            else:
                delete_rows.add(start_row)
            for row in range(start_row + 1, end_row + 1):
                delete_rows.add(row)

    for row, col in comment_ops:
        if row in replace_rows or row in delete_rows:
            continue
        if row - 1 >= len(lines):
            continue
        line = lines[row - 1]
        prefix = line[:col]
        if not prefix.strip():
            delete_rows.add(row)
        else:
            lines[row - 1] = prefix.rstrip() + _line_ending(line)

    out = []
    for row, line in enumerate(lines, start=1):
        if row in replace_rows:
            start_col, ending = replace_rows[row]
            out.append(line[:start_col] + "pass" + ending)
        elif row in delete_rows:
            continue
        else:
            out.append(line)
    return "".join(out)


# -- non-Python (line-based, string-aware) -----------------------------
def _first_comment_col(line: str, allow_hash: bool,
                       allow_slash: bool) -> int:
    """Column of the first ``#`` / ``//`` comment marker outside string
    literals, or -1 when the line is left untouched (no marker, or an
    unterminated string — conservative)."""
    quote = ""
    escaped = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue
        if allow_slash and ch == "/" and i + 1 < n and line[i + 1] == "/":
            return i
        if allow_hash and ch == "#":
            return i
        i += 1
    return -1


def _apply_line_based(code: str, policy: CommentPolicy,
                      language: str) -> str:
    allow_hash = language in _HASH_LANGS or language not in (
        _HASH_LANGS | _SLASH_LANGS)
    allow_slash = language in _SLASH_LANGS or language not in (
        _HASH_LANGS | _SLASH_LANGS)
    out = []
    for line in code.splitlines(keepends=True):
        col = _first_comment_col(line, allow_hash, allow_slash)
        if col < 0:
            out.append(line)
            continue
        prefix = line[:col]
        if not prefix.strip():
            continue
        out.append(prefix.rstrip() + _line_ending(line))
    return "".join(out)
