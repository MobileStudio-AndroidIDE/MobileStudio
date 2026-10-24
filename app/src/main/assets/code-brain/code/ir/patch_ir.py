"""Transactional Patch IR - atomic, verified edits to existing files.

A PatchTransaction groups Hunk edits that must land together or not at all:

  1. Preconditions: every hunk carries the sha256 of the file BEFORE the
     patch (and optionally an expected snippet / symbol range), so a patch
     built against a stale tree is refused instead of applied blindly.
  2. Validation: validate() reports every problem (missing files, hash
     mismatches, missing anchors, overlapping hunks) before anything runs.
  3. Atomicity: apply() snapshots the original text in memory, writes each
     new file through a temp file in the same directory + os.replace, and
     ast.parse()s new .py content BEFORE replacing it.  On any failure
     mid-transaction every already-written file is restored and no partial
     state is ever left on disk.
  4. Rollback: the restore source is the in-memory snapshot taken by
     apply() at the start of the transaction (documented choice: no
     .patch_txn/*.bak sidecar files are written; the snapshot is
     process-local, so rollback() works within the lifetime of the
     PatchTransaction object).

The diff text produced by dry_run() is deterministic: difflib output with
fixed a/ b/ labels and no timestamps.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TEXT_OPS = ("replace_text", "insert_after", "delete_text")
SYMBOL_OPS = ("replace_symbol",)
ALL_OPS = TEXT_OPS + SYMBOL_OPS


class PatchApplyError(Exception):
    """A hunk could not be applied to the file text."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Hunk:
    """One edit of one file.

    ops and their payloads:
      replace_text   expected_text -> new_text (expected_text must occur once)
      insert_after   new_text inserted immediately after expected_text
      delete_text    expected_text removed (new_text unused)
      replace_symbol symbol's definition lines (start_line..end_line,
                     derived by ast when omitted) -> new_text

    Preconditions (checked against the file BEFORE apply):
      expected_sha256  sha256 of the whole file
      expected_snippet exact substring that must be present
    """

    file: str
    op: str
    expected_text: Optional[str] = None
    new_text: Optional[str] = None
    symbol: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    expected_snippet: Optional[str] = None
    expected_sha256: Optional[str] = None


@dataclass
class PatchTransaction:
    """A group of hunks applied atomically with rollback on failure."""

    id: str = ""
    purpose: str = ""
    hunks: list[Hunk] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    verification_requirements: list[str] = field(default_factory=list)
    created_from: Optional[str] = None
    repo_root: Optional[str] = None
    base_hashes: dict = field(default_factory=dict)
    _originals: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.repo_root:
            self.capture_base_hashes(self.repo_root)

    # ------------------------------------------------------------- helpers

    def capture_base_hashes(self, repo_root) -> dict:
        """Record (file -> sha256) for every hunk file that exists."""
        root = Path(repo_root)
        for hunk in self.hunks:
            path = self._path(root, hunk.file)
            if path is not None and path.is_file():
                self.base_hashes.setdefault(hunk.file, sha256_file(path))
        return self.base_hashes

    @staticmethod
    def _path(root: Path, rel: str) -> Optional[Path]:
        if not rel:
            return None
        candidate = Path(rel)
        if candidate.is_absolute() or ".." in candidate.parts:
            return None
        return root / candidate

    @staticmethod
    def _read(root: Path, rel: str) -> str:
        path = PatchTransaction._path(root, rel)
        if path is None:
            raise PatchApplyError(f"{rel}: unsafe file path (must be repo-relative)")
        if not path.is_file():
            raise PatchApplyError(f"{rel}: file does not exist")
        return path.read_bytes().decode("utf-8")

    def _group_by_file(self) -> dict:
        groups: dict = {}
        for hunk in self.hunks:
            groups.setdefault(hunk.file, []).append(hunk)
        return groups

    @staticmethod
    def symbol_span(source_text: str, name: str) -> tuple:
        """(start_line, end_line) 1-based inclusive span of a def/class."""
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)) and node.name == name:
                start = node.lineno
                if node.decorator_list:
                    start = min(d.lineno for d in node.decorator_list)
                end = getattr(node, "end_lineno", None)
                if end is None:
                    end = max(getattr(n, "lineno", start)
                              for n in ast.walk(node))
                return (int(start), int(end))
        raise ValueError(f"symbol {name!r} not found")

    @classmethod
    def build_replace_symbol_hunk(cls, repo_root, file: str, name: str,
                                  new_code: str,
                                  expected_sha256: Optional[str] = None) -> Hunk:
        """Hunk replacing a function/class by name in an existing file."""
        root = Path(repo_root)
        path = cls._path(root, file)
        if path is None or not path.is_file():
            raise PatchApplyError(f"{file}: file does not exist")
        text = path.read_text(encoding="utf-8")
        start, end = cls.symbol_span(text, name)
        return Hunk(
            file=file,
            op="replace_symbol",
            symbol=name,
            new_text=new_code,
            start_line=start,
            end_line=end,
            expected_sha256=expected_sha256 or sha256_file(path),
        )

    # ------------------------------------------------------------ validate

    def validate(self, repo_root) -> list:
        """Every precondition issue found, in hunk order ([] = clean)."""
        root = Path(repo_root)
        issues: list = []
        current: dict = {}

        for hunk in self.hunks:
            where = f"{hunk.file}: "
            path = self._path(root, hunk.file)
            if path is None:
                issues.append(where + "unsafe or empty file path")
                continue
            if not path.is_file():
                issues.append(where + "file does not exist")
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                issues.append(where + f"unreadable ({exc})")
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                issues.append(where + "file is not valid UTF-8 text")
                continue
            current[hunk.file] = text
            digest = hashlib.sha256(raw).hexdigest()

            expected = hunk.expected_sha256 or self.base_hashes.get(hunk.file)
            if expected and digest != expected:
                issues.append(where + "base hash mismatch "
                              "(file changed since precondition captured)")
            if (hunk.expected_snippet is not None
                    and hunk.expected_snippet not in text):
                issues.append(where + "expected_snippet not present")

            if hunk.op not in ALL_OPS:
                issues.append(where + f"unknown op {hunk.op!r}")
                continue
            if hunk.op in TEXT_OPS:
                if not hunk.expected_text:
                    issues.append(where + f"{hunk.op} requires expected_text")
                else:
                    count = text.count(hunk.expected_text)
                    if count == 0:
                        issues.append(where + "expected_text not found")
                    elif count > 1:
                        issues.append(where + "expected_text is ambiguous "
                                      f"({count} matches)")
                if hunk.op in ("replace_text", "insert_after") \
                        and hunk.new_text is None:
                    issues.append(where + f"{hunk.op} requires new_text")
            else:  # replace_symbol
                if not hunk.symbol:
                    issues.append(where + "replace_symbol requires symbol")
                else:
                    try:
                        span = self.symbol_span(text, hunk.symbol)
                    except (ValueError, SyntaxError):
                        issues.append(where + f"symbol {hunk.symbol!r} not found")
                    else:
                        lines = text.count("\n") + 1
                        if hunk.start_line is not None or hunk.end_line is not None:
                            start = hunk.start_line or span[0]
                            end = hunk.end_line or span[1]
                            if not (1 <= start <= end <= lines):
                                issues.append(where + "symbol range out of bounds")

        self._overlap_issues(current, issues)
        return issues

    def _overlap_issues(self, current: dict, issues: list) -> None:
        ranges: dict = {}
        for hunk in self.hunks:
            text = current.get(hunk.file)
            if text is None:
                continue
            span = self._base_range(text, hunk)
            if span is not None:
                ranges.setdefault(hunk.file, []).append(
                    (span[0], span[1], hunk.op))
        for file, spans in ranges.items():
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    a, b = spans[i], spans[j]
                    if a[0] <= b[1] and b[0] <= a[1]:
                        issues.append(
                            f"{file}: overlapping hunks ({a[2]} "
                            f"[{a[0]}-{a[1]}] and {b[2]} [{b[0]}-{b[1]}])")

    def _base_range(self, text: str, hunk: Hunk):
        """Line interval [start, end] this hunk touches in the base text."""
        if hunk.op == "replace_symbol":
            if hunk.start_line and hunk.end_line:
                return (hunk.start_line, hunk.end_line)
            if hunk.symbol:
                try:
                    return self.symbol_span(text, hunk.symbol)
                except (ValueError, SyntaxError):
                    return None
            return None
        if not hunk.expected_text:
            return None
        idx = text.find(hunk.expected_text)
        if idx < 0:
            return None
        start = text.count("\n", 0, idx) + 1
        end = start + hunk.expected_text.count("\n")
        return (start, end)

    # ------------------------------------------------------------- dry run

    def dry_run(self, repo_root) -> dict:
        """Per-file preview diff text + issues.  Never writes anything."""
        root = Path(repo_root)
        issues = self.validate(repo_root)
        previews: dict = {}
        for file, hunks in self._group_by_file().items():
            try:
                base = self._read(root, file)
            except (PatchApplyError, OSError, UnicodeDecodeError) as exc:
                previews[file] = ""
                if str(exc) not in issues:
                    issues.append(str(exc))
                continue
            patched = base
            ok = True
            for hunk in hunks:
                try:
                    patched = _apply_one(patched, hunk)
                except PatchApplyError as exc:
                    issues.append(f"{file}: {exc}")
                    ok = False
                    break
            previews[file] = _unified_diff(base, patched, file) if ok else ""
        return {"files": previews, "issues": issues, "valid": not issues}

    # --------------------------------------------------------------- apply

    def apply(self, repo_root, require_parse: bool = True) -> dict:
        """Apply every hunk atomically; never leaves partial state.

        Preconditions are re-checked per file against the snapshot taken at
        the start of this call.  A failure anywhere restores every file
        already written and returns success=False with the reasons.
        """
        root = Path(repo_root)
        snapshot: dict = {}
        try:
            for file in self._group_by_file():
                snapshot[file] = self._read(root, file)
        except (PatchApplyError, OSError, UnicodeDecodeError) as exc:
            return {"success": False, "applied": False, "written": [],
                    "rolled_back": [], "failures": [str(exc)], "issues": []}

        self._originals = dict(snapshot)
        written: list = []
        try:
            for file, hunks in self._group_by_file().items():
                base = snapshot[file]
                digest = sha256_text(base)
                for hunk in hunks:
                    expected = (hunk.expected_sha256
                                or self.base_hashes.get(file))
                    if expected and expected != digest:
                        raise PatchApplyError(
                            f"{file}: base hash mismatch "
                            "(file changed since precondition captured)")
                if any(h.expected_snippet is not None
                       and h.expected_snippet not in base for h in hunks):
                    raise PatchApplyError(f"{file}: expected_snippet not present")

                patched = base
                for hunk in hunks:
                    patched = _apply_one(patched, hunk)

                if require_parse and file.endswith(".py"):
                    try:
                        ast.parse(patched, filename=file)
                    except SyntaxError as exc:
                        detail = exc.msg or "invalid syntax"
                        if exc.lineno:
                            detail += f" (line {exc.lineno})"
                        raise PatchApplyError(
                            f"{file}: syntax error in patched content: {detail}")

                _atomic_write(root / file, patched)
                written.append(file)
        except (PatchApplyError, OSError, UnicodeDecodeError) as exc:
            rolled_back = self._restore(root, written, snapshot)
            return {"success": False, "applied": False, "written": written,
                    "rolled_back": rolled_back, "failures": [str(exc)],
                    "issues": []}

        return {"success": True, "applied": True, "written": written,
                "rolled_back": [], "failures": [], "issues": []}

    def rollback(self, repo_root) -> dict:
        """Restore the originals snapshotted by the last apply()."""
        if not self._originals:
            return {"success": False, "restored": [],
                    "issues": ["no snapshot available; apply() must run first"]}
        restored = self._restore(Path(repo_root), list(self._originals),
                                 self._originals)
        return {"success": True, "restored": restored, "issues": []}

    @staticmethod
    def _restore(root: Path, files: list, snapshot: dict) -> list:
        restored: list = []
        for file in files:
            text = snapshot.get(file)
            if text is None:
                continue
            try:
                _atomic_write(root / file, text)
                restored.append(file)
            except OSError:
                continue
        return restored


def _find_anchor(text: str, hunk: Hunk) -> int:
    needle = hunk.expected_text or ""
    if not needle:
        raise PatchApplyError(f"{hunk.op}: missing expected_text")
    count = text.count(needle)
    if count == 0:
        raise PatchApplyError(f"{hunk.op}: expected_text not found")
    if count > 1:
        raise PatchApplyError(
            f"{hunk.op}: expected_text is ambiguous ({count} matches)")
    return text.index(needle)


def _apply_one(text: str, hunk: Hunk) -> str:
    """Apply a single hunk to text; raises PatchApplyError if impossible."""
    if hunk.op == "replace_text":
        idx = _find_anchor(text, hunk)
        end = idx + len(hunk.expected_text or "")
        return text[:idx] + (hunk.new_text or "") + text[end:]
    if hunk.op == "delete_text":
        idx = _find_anchor(text, hunk)
        end = idx + len(hunk.expected_text or "")
        return text[:idx] + text[end:]
    if hunk.op == "insert_after":
        idx = _find_anchor(text, hunk)
        end = idx + len(hunk.expected_text or "")
        return text[:end] + (hunk.new_text or "") + text[end:]
    if hunk.op == "replace_symbol":
        if not hunk.symbol:
            raise PatchApplyError("replace_symbol: missing symbol")
        if hunk.start_line and hunk.end_line:
            start, end = hunk.start_line, hunk.end_line
        else:
            try:
                start, end = PatchTransaction.symbol_span(text, hunk.symbol)
            except (ValueError, SyntaxError):
                raise PatchApplyError(
                    f"replace_symbol: symbol {hunk.symbol!r} not found")
        lines = text.splitlines(keepends=True)
        if not (1 <= start <= end <= len(lines)):
            raise PatchApplyError("replace_symbol: range out of bounds")
        block = hunk.new_text or ""
        if end < len(lines) and not block.endswith("\n"):
            block += "\n"
        result = "".join(lines[:start - 1]) + block + "".join(lines[end:])
        if (end == len(lines) and not text.endswith("\n")
                and result.endswith("\n")):
            result = result[:-1]
        return result
    raise PatchApplyError(f"unknown op {hunk.op!r}")


def _unified_diff(old: str, new: str, file: str) -> str:
    """Deterministic minimal unified diff (no timestamps)."""
    diff = difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"a/{file}", tofile=f"b/{file}", lineterm="",
    )
    return "\n".join(diff)


def _atomic_write(path: Path, text: str) -> None:
    """Temp file in the same directory + os.replace (atomic on POSIX)."""
    directory = path.parent
    fd, tmp = tempfile.mkstemp(prefix=".patch_txn_", suffix=".tmp",
                               dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
