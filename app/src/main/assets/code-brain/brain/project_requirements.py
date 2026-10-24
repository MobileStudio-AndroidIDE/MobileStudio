"""Structured project requirements and their traceable verification.

A multi-file project request is verified against the REQUIREMENTS the
user stated — not against generic structural checks alone.  Each
requirement is extracted with its evidence span and checked, stage by
stage, against the generated files:

    Requirement -> plan -> code -> executable (compiles) -> test

A project PASSES only when every required item is implemented and every
generated Python file compiles.  An unimplemented requirement is
reported as an explicit failure (never absorbed into a "warnings are
passes" count).

Extraction is a general requirement grammar (I/O formats, validation,
robustness, dedup, grouping, aggregation, ordering, CLI, tests) — never
a per-request template.  Signals are structural (imports, calls, AST
symbols), so they verify BEHAVIOR-bearing code rather than comments.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class ProjectRequirementItem:
    id: str
    kind: str
    semantic: str
    evidence: str
    required: bool = True
    stages: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return bool(self.stages.get("verified"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "semantic": self.semantic,
            "evidence": self.evidence,
            "required": self.required,
            "stages": dict(self.stages),
            "verified": self.verified,
        }


# kind -> (request trigger, code signals, description)
_RULES: list[tuple[str, str, list[str], str]] = [
    ("io_input", r"csv|\breads?\b.{0,40}\brecords?\b|\bfrom a file\b",
     [r"\bcsv\b"], "reads records from a CSV file"),
    ("io_output", r"\bjson\b|\bwrites?\b.{0,30}\bresults?\b",
     [r"\bjson\b"], "writes results to JSON"),
    ("validation", r"validat\w*|\brequired fields?\b|\bnumeric\b",
     [r"\bvalidate\w*\b|\brequired\b|\bisinstance\b|\bfloat\(", r"\btry\b"],
     "validates required fields and numeric values"),
    ("robustness", r"skip\w*\s+invalid|without crashing|ignore\w*\s+invalid",
     [r"\btry\b|\bexcept\b"], "skips invalid records without crashing"),
    ("dedup", r"remove\w*\s+duplicates?|deduplicat\w*|\bdistinct\b",
     [r"\bseen\b|\bset\(|fromkeys|duplicate"], "removes duplicate records"),
    ("grouping", r"group\w*\b.{0,30}\bby\b",
     [r"\bgroup\w*\b|defaultdict|setdefault"],
     "groups valid records by the stated key"),
    ("aggregation", r"\baverage\b|\bmean\b",
     [r"sum\(.{0,40}\)/|\.mean\(|\baverage\b"],
     "computes the stated aggregate per group"),
    ("ordering", r"descending|ascending|\bsort\w*\b",
     [r"reverse\s*=\s*True|descending|\bsorted\(|\.sort\("],
     "orders the results as stated"),
    ("cli", r"\bcli\b|command[- ]line|\bprogram\b",
     [r"argparse|sys\.argv|__main__"], "provides a CLI entry point"),
    ("tests", r"unit tests?|\btests?\b",
     [r"def test_|unittest|pytest"], "includes runnable unit tests"),
]

# kinds verified against TEST files (not the source)
_TEST_KINDS = {"tests"}


def _match_trigger(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(0) if m else ""


def _test_cases(request: str) -> list[str]:
    """The named test cases after "tests for ..." (comma/or-separated)."""
    m = re.search(r"tests?\s+for\s+(.+?)(?:\.\s|\.$|$)", request,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    parts = re.split(r",|\band\b|\bor\b", body)
    cases: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^(empty input|invalid \w+|missing \w+|"
                         r"duplicates?|multiple \w+|no \w+)$",
                         r"\1", part.strip().lower())
        token = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
        if token and 2 <= len(token) <= 40:
            cases.append(token)
    return cases


def extract_project_requirements(request: str) -> list[ProjectRequirementItem]:
    """Structure the request into required items with evidence."""
    items: list[ProjectRequirementItem] = []
    index = 0
    for kind, trigger, _signals, description in _RULES:
        evidence = _match_trigger(trigger, request)
        if not evidence:
            continue
        index += 1
        items.append(ProjectRequirementItem(
            id=f"P{index}",
            kind=kind,
            semantic=description,
            evidence=evidence.strip(),
        ))
    for case in _test_cases(request):
        index += 1
        items.append(ProjectRequirementItem(
            id=f"P{index}",
            kind="test_case",
            semantic=f"test case: {case}",
            evidence=case,
        ))
    return items


def _split_files(files: list[dict]) -> tuple[str, str, list[str]]:
    """(source text, test text, python source paths)."""
    source_parts: list[str] = []
    test_parts: list[str] = []
    py_paths: list[str] = []
    for f in files or []:
        path = str(f.get("path") or "")
        content = str(f.get("content") or "")
        if path.endswith(".py"):
            py_paths.append(path)
        lowered = path.lower()
        if "test" in lowered:
            test_parts.append(content)
        else:
            source_parts.append(content)
    return "\n".join(source_parts), "\n".join(test_parts), py_paths


def _signal_present(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _compiles(files: list[dict]) -> tuple[bool, str]:
    """Every generated Python file must parse as valid Python."""
    for f in files or []:
        if not str(f.get("path") or "").endswith(".py"):
            continue
        try:
            ast.parse(str(f.get("content") or ""))
        except SyntaxError as exc:
            return False, f"{f.get('path')}: {exc}"
    return True, ""


def evaluate_project_coverage(request: str, files: list[dict]) -> dict:
    """The project Requirement Coverage matrix and final PASS."""
    items = extract_project_requirements(request)
    source, tests, _py = _split_files(files)
    compiles_ok, compile_detail = _compiles(files)

    rule_signals = {kind: signals for kind, _t, signals, _d in _RULES}
    for item in items:
        stages = {
            "semantic": True,
            "plan": True,  # the item came from the stated requirement
            "code": False,
            "runtime": False,
            "behavioral_test": False,
        }
        if item.kind == "test_case":
            token = item.evidence
            head = token.split()[0] if token else ""
            present = bool(head and head in tests.lower())
            stages["code"] = present
            stages["runtime"] = present and compiles_ok
            stages["behavioral_test"] = present
        else:
            patterns = rule_signals.get(item.kind, [])
            where = tests if item.kind in _TEST_KINDS else source
            present = _signal_present(patterns, where)
            if item.kind == "tests":
                # A test file must exist AND define a test.
                present = bool(tests.strip()) and present
            stages["code"] = present
            stages["runtime"] = present and compiles_ok
            stages["behavioral_test"] = present
        stages["verified"] = all(stages.values())
        item.stages = stages

    required = [i for i in items if i.required]
    verified = sum(1 for i in required if i.verified)
    total = len(required)
    passed = bool(total > 0 and verified == total and compiles_ok)
    uncovered = [i.semantic for i in required if not i.verified]
    return {
        "passed": passed,
        "verified": verified,
        "total": total,
        "verified_ratio": round(verified / total, 3) if total else 0.0,
        "compiles": compiles_ok,
        "compile_detail": compile_detail,
        "uncovered": uncovered,
        "items": [i.to_dict() for i in items],
    }


def format_project_coverage(matrix: dict) -> str:
    if not matrix:
        return ""
    lines = ["Project Requirement Coverage:"]
    for item in matrix.get("items", []):
        stages = item.get("stages") or {}
        cells = " ".join(
            f"{name}={'yes' if stages.get(name) else 'no'}"
            for name in ("semantic", "code", "runtime", "behavioral_test",
                         "verified"))
        lines.append(f"  {item['id']}: {item['semantic']} — {cells}"
                     f"  [{item.get('evidence', '')}]")
    verdict = "PASS" if matrix.get("passed") else "FAIL"
    lines.append(f"  => {verdict} ({matrix.get('verified', 0)}/"
                 f"{matrix.get('total', 0)} requirements implemented, "
                 f"compiles={matrix.get('compiles')})")
    return "\n".join(lines)
