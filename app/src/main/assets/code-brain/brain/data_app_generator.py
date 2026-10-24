"""Data-application project generator (requirements-driven, generic).

A request that names a FILE-FORMAT data flow (read CSV -> validate ->
deduplicate -> group -> aggregate -> order -> write JSON, with an
optional CLI and unit tests) is a DATA APPLICATION.  This module
extracts that structure from the request as a general requirement
grammar (formats, validation, identity fields, group key, measure,
ordering) and renders a working Python project that IMPLEMENTS it:

    src/pipeline.py    load/validate/dedup/group/aggregate/order/write
    src/main.py        argparse CLI entry point
    tests/test_*.py    unit tests for each named case

Nothing here is a per-request template: the same generator serves any
CSV->JSON aggregation request; the field names, group key, measure and
ordering all come from the request's own semantics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from brain.project_requirements import extract_project_requirements


@dataclass
class DataAppSpec:
    """The structured data application the request describes."""

    name: str = "DataPipeline"
    domain: str = ""
    input_format: str = "csv"
    output_format: str = "json"
    required_fields: list[str] = field(default_factory=list)
    numeric_fields: list[str] = field(default_factory=list)
    identity_fields: list[str] = field(default_factory=list)
    group_key: str = ""
    measure: str = ""
    descending: bool = True
    has_cli: bool = False
    test_cases: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "required_fields": list(self.required_fields),
            "numeric_fields": list(self.numeric_fields),
            "identity_fields": list(self.identity_fields),
            "group_key": self.group_key,
            "measure": self.measure,
            "descending": self.descending,
            "has_cli": self.has_cli,
            "test_cases": list(self.test_cases),
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataAppSpec":
        spec = cls()
        for key in ("name", "domain", "input_format", "output_format",
                    "group_key", "measure"):
            setattr(spec, key, str(data.get(key) or ""))
        for key in ("required_fields", "numeric_fields", "identity_fields",
                    "test_cases"):
            setattr(spec, key, [str(v) for v in (data.get(key) or [])])
        spec.descending = bool(data.get("descending", True))
        spec.has_cli = bool(data.get("has_cli", False))
        spec.evidence = dict(data.get("evidence") or {})
        return spec


_STOP_WORDS = {"the", "a", "an", "and", "or", "of", "from", "with", "each"}


def _field_token(word: str) -> Optional[str]:
    word = (word or "").strip().lower()
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", word):
        return None
    if word in _STOP_WORDS or len(word) < 2:
        return None
    if word.endswith("ing") or word.endswith("ed"):
        return None
    return word


def build_data_app_spec(request: str,
                        language: str = "python") -> Optional[DataAppSpec]:
    """Extract the data application the request describes, or None.

    Fires only when the request names a CSV input — the data flow's
    foundational requirement.  Everything else is extracted from the
    request's own wording (fields, keys, measures), never assumed.
    """
    if language and language != "python":
        return None
    lower = request.lower()
    if not re.search(r"\bcsv\b|comma[- ]separated", lower):
        return None

    spec = DataAppSpec()
    evidence: dict = {"input": "csv"}
    spec.evidence = evidence

    # Domain entity: "student records", "records of students"
    domain = ""
    m = re.search(r"([a-z]+)\s+records?\b", lower)
    if m:
        domain = m.group(1)
    if not domain:
        m = re.search(r"records?\s+of\s+([a-z]+)", lower)
        if m:
            domain = m.group(1)
    spec.domain = domain
    if domain:
        spec.name = domain.capitalize() + "Pipeline"

    # Group key: "group ... by class" / "per class" / "for each class"
    m = re.search(r"group\w*\s+(?:\w+\s+){0,3}?by\s+(?:the\s+)?"
                  r"([a-z_][a-z0-9_]*)", lower)
    if m:
        spec.group_key = _field_token(m.group(1)) or ""
        evidence["grouping"] = m.group(0)
    if not spec.group_key:
        m = re.search(r"\bper\s+([a-z_][a-z0-9_]*)|\bfor each\s+"
                      r"([a-z_][a-z0-9_]*)", lower)
        if m:
            spec.group_key = _field_token(m.group(1) or m.group(2)) or ""

    # Measure: "average score" / "average of the scores"
    m = re.search(r"(?:average|mean)\s+(?:of\s+(?:the\s+)?)?"
                  r"([a-z_][a-z0-9_]*)", lower)
    if m:
        measure = _field_token(m.group(1))
        if measure:
            spec.measure = measure
            evidence["aggregation"] = m.group(0)
    if not spec.measure:
        m = re.search(r"(?:average|mean)\s+(?:\w+\s+){0,2}?"
                      r"([a-z_][a-z0-9_]*)\s*(?:per|for each|by)", lower)
        if m:
            measure = _field_token(m.group(1))
            if measure:
                spec.measure = measure

    # Identity: "duplicate ... by name and class"
    m = re.search(r"duplicat\w*[^,.]*?\bby\s+([^,.]*)", lower)
    if m:
        fields: list[str] = []
        for word in re.split(r"\s+and\s+|,|\s*&\s*|\s+", m.group(1)):
            token = _field_token(word)
            if token:
                fields.append(token)
        spec.identity_fields = list(dict.fromkeys(fields))
        evidence["identity"] = m.group(0)

    # Ordering
    if "descending" in lower or "highest to lowest" in lower \
            or "largest first" in lower:
        spec.descending = True
        evidence["ordering"] = "descending"
    elif "ascending" in lower or "lowest to highest" in lower \
            or "smallest first" in lower:
        spec.descending = False
        evidence["ordering"] = "ascending"

    # Required/numeric field contract: the identity fields must be
    # present; the measure must be numeric.  The group key is required
    # for the grouping to be defined.
    required = list(dict.fromkeys(
        spec.identity_fields
        + ([spec.group_key] if spec.group_key else [])
        + (["name"] if not spec.identity_fields and not spec.group_key
           else [])))
    spec.required_fields = required
    spec.numeric_fields = [spec.measure] if spec.measure else []

    # CLI and tests
    spec.has_cli = bool(re.search(r"\bcli\b|command[- ]line", lower))
    if re.search(r"unit tests?|\btests?\b", lower):
        from brain.project_requirements import _test_cases
        spec.test_cases = _test_cases(request)
        if not spec.test_cases:
            spec.test_cases = ["basic behaviour"]
    return spec


# --------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------- #

def _tuple_literal(values: list[str]) -> str:
    if len(values) == 1:
        return f"({values[0]!r},)"
    return "(" + ", ".join(repr(v) for v in values) + ")"


def _render_pipeline(spec: DataAppSpec) -> str:
    return f'''"""Data pipeline: load -> validate -> deduplicate -> group -> aggregate -> order -> JSON."""
import csv
import json
from typing import Any, Dict, List

# The record contract extracted from the requirements
REQUIRED_FIELDS = {_tuple_literal(spec.required_fields)}
NUMERIC_FIELDS = {_tuple_literal(spec.numeric_fields)}
IDENTITY_FIELDS = {_tuple_literal(spec.identity_fields or spec.required_fields)}
GROUP_KEY = {spec.group_key!r}
MEASURE = {spec.measure!r}
DESCENDING = {spec.descending!r}
OUTPUT_PATH_DEFAULT = "results.json"


def validate_record(record: Dict[str, Any]) -> bool:
    """A record is valid when every required field is present and every
    numeric field parses as a number.  Invalid records are never an
    error — the caller skips them."""
    if not isinstance(record, dict):
        return False
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if value is None or str(value).strip() == "":
            return False
    for field in NUMERIC_FIELDS:
        try:
            record[field] = float(record.get(field))
        except (TypeError, ValueError):
            return False
    return True


def load_records(path: str) -> List[Dict[str, Any]]:
    """Read records from a CSV file (the header names the fields)."""
    records: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append({{key: value for key, value in row.items() if key}})
    return records


def clean_records(records) -> List[Dict[str, Any]]:
    """Keep only valid records; invalid ones are skipped without crashing."""
    valid: List[Dict[str, Any]] = []
    for record in records:
        if validate_record(record):
            valid.append(record)
    return valid


def deduplicate(records, fields=IDENTITY_FIELDS) -> List[Dict[str, Any]]:
    """Remove duplicate records by the identity fields, keeping the first."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        key = tuple(str(record.get(field, "")) for field in fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def group_by(records, key=GROUP_KEY) -> Dict[Any, List[Dict[str, Any]]]:
    """Group records by the stated key."""
    groups: Dict[Any, List[Dict[str, Any]]] = {{}}
    for record in records:
        groups.setdefault(record.get(key), []).append(record)
    return groups


def aggregate_average(groups, measure=MEASURE) -> Dict[Any, float]:
    """The average of the measure per group."""
    averages: Dict[Any, float] = {{}}
    for key, group in groups.items():
        values = [float(record.get(measure, 0) or 0) for record in group]
        averages[key] = sum(values) / len(values) if values else 0.0
    return averages


def order_results(results, descending=DESCENDING) -> Dict[Any, float]:
    """Order the groups by their aggregate, descending or ascending."""
    return dict(sorted(results.items(), key=lambda item: item[1],
                       reverse=descending))


def build_result(csv_path: str) -> Dict[Any, float]:
    """The whole pipeline over a CSV file."""
    records = clean_records(load_records(csv_path))
    records = deduplicate(records, IDENTITY_FIELDS)
    groups = group_by(records, GROUP_KEY)
    averages = aggregate_average(groups, MEASURE)
    return order_results(averages, DESCENDING)


def write_json(result, json_path: str = OUTPUT_PATH_DEFAULT) -> str:
    """Write the result to JSON."""
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return json_path


def process(csv_path: str, json_path: str = OUTPUT_PATH_DEFAULT):
    """Run the pipeline and write the JSON output."""
    result = build_result(csv_path)
    write_json(result, json_path)
    return result
'''


def _render_cli(spec: DataAppSpec) -> str:
    subject = spec.domain or "records"
    return f'''"""Command line entry point for the {subject} pipeline."""
import argparse
import sys

from src.pipeline import process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read {subject} from CSV and write group averages to JSON.")
    parser.add_argument("input", help="path to the input CSV file")
    parser.add_argument("--output", default="results.json",
                        help="path to the JSON output file")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = process(args.input, args.output)
    print(f"Wrote {{len(result)}} group(s) to {{args.output}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# Known test categories -> generated test bodies.  The category words
# are a general edge-case vocabulary (empty/invalid/missing/duplicates/
# multiple groups), never a per-request answer.
def _known_case(phrase: str) -> str:
    lowered = phrase.lower()
    if "empty" in lowered:
        return "empty"
    if "invalid" in lowered or "non-numeric" in lowered or "not numeric" in lowered:
        return "invalid"
    if "missing" in lowered:
        return "missing"
    if "field" in lowered:
        return "missing"
    if "duplicat" in lowered:
        return "duplicates"
    if ("multiple" in lowered or "several" in lowered or "classes" in lowered
            or "groups" in lowered):
        return "multiple"
    return "generic"


def _case_method_name(case: str, phrase: str) -> str:
    if case != "generic":
        return "test_" + case
    token = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "case"
    return "test_case_" + token


def _indent(body: str, spaces: int = 8) -> str:
    pad = " " * spaces
    return "\n".join((pad + line) if line else line
                      for line in body.splitlines())


def _test_method(case: str, name: str) -> str:
    bodies = {
        "empty": """
path = _write_csv([])
try:
    self.assertEqual(pipeline.build_result(path), {})
finally:
    os.unlink(path)
""",
        "invalid": """
valid = _record("alice", "A", 90)
invalid = _record("bob", "A", "not-a-number")
self.assertTrue(pipeline.validate_record(valid))
self.assertFalse(pipeline.validate_record(invalid))
self.assertEqual(len(pipeline.clean_records([valid, invalid])), 1)
""",
        "missing": """
complete = _record("alice", "A", 90)
incomplete = _record("bob", "A", 90)
for field in pipeline.REQUIRED_FIELDS:
    incomplete[field] = ""
self.assertTrue(pipeline.validate_record(complete))
self.assertFalse(pipeline.validate_record(incomplete))
self.assertEqual(len(pipeline.clean_records([complete, incomplete])), 1)
""",
        "duplicates": """
first = _record("alice", "A", 90)
duplicate = _record("alice", "A", 50)
unique = pipeline.deduplicate([first, duplicate])
self.assertEqual(len(unique), 1)
self.assertEqual(unique[0][pipeline.NUMERIC_FIELDS[0]], 90)
""",
        "multiple": """
rows = [
    _record("alice", "A", 90),
    _record("bob", "A", 80),
    _record("carol", "B", 70),
]
path = _write_csv(rows)
try:
    result = pipeline.build_result(path)
finally:
    os.unlink(path)
self.assertEqual(len(result), 2)
self.assertAlmostEqual(result["A"], 85.0)
self.assertAlmostEqual(result["B"], 70.0)
keys = list(result.keys())
expected = sorted(keys, key=lambda key: result[key],
                  reverse=pipeline.DESCENDING)
self.assertEqual(keys, expected)
""",
        "generic": """
rows = [_record("alice", "A", 90), _record("bob", "B", 70)]
path = _write_csv(rows)
output = path + ".json"
try:
    result = pipeline.process(path, output)
    self.assertTrue(os.path.exists(output))
    with open(output, encoding="utf-8") as handle:
        self.assertEqual(json.load(handle), result)
    self.assertEqual(len(result), 2)
finally:
    os.unlink(path)
    if os.path.exists(output):
        os.unlink(output)
""",
    }
    body = bodies.get(case, bodies["generic"]).strip("\n")
    return f"    def {name}(self):\n" + _indent(body) + "\n"


def _render_tests(spec: DataAppSpec) -> str:
    methods: list[str] = []
    seen: set[str] = set()
    for phrase in spec.test_cases:
        case = _known_case(phrase)
        # A case the extracted contract cannot express truthfully (no
        # group key / no numeric measure) falls back to the generic
        # end-to-end test — never a test that cannot pass.
        if case == "multiple" and not (spec.group_key and spec.numeric_fields):
            case = "generic"
        if case == "invalid" and not spec.numeric_fields:
            case = "generic"
        if case == "duplicates" and not spec.numeric_fields:
            case = "generic"
        name = _case_method_name(case, phrase)
        if name in seen:
            continue
        seen.add(name)
        methods.append(_test_method(case, name))
    if not methods:
        methods.append(_test_method("generic", "test_basic_behaviour"))
    body = "\n".join(methods)

    header = '''"""Unit tests for the data pipeline."""
import csv
import json
import os
import tempfile
import unittest

from src import pipeline


def _record(name_value="alice", group_value="A", measure_value=1.0):
    record = {}
    for index, field in enumerate(pipeline.REQUIRED_FIELDS):
        if field == pipeline.GROUP_KEY:
            record[field] = group_value
        elif index == 0:
            record[field] = name_value
        else:
            record[field] = group_value
    for field in pipeline.NUMERIC_FIELDS:
        record[field] = measure_value
    return record


def _write_csv(rows):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, newline="", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = list(pipeline.REQUIRED_FIELDS) + list(pipeline.NUMERIC_FIELDS)
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    handle.close()
    return handle.name


class TestPipeline(unittest.TestCase):
'''
    return header + body


def render_data_app_file(content_type: str, file_ir, project_ir) -> str:
    """Render one data-app file (called by ProjectGenerator)."""
    spec = DataAppSpec.from_dict((project_ir.metadata or {}).get("data_app") or {})
    if content_type == "data_pipeline":
        return _render_pipeline(spec)
    if content_type == "data_cli":
        return _render_cli(spec)
    if content_type == "data_tests":
        return _render_tests(spec)
    raise ValueError(f"unknown data-app content type: {content_type}")


def build_data_app_project(spec: DataAppSpec, request: str):
    """Build the ProjectIR and its generated files for a data app."""
    from code.ir.project_ir import (
        ProjectIR, ModuleIR, FileIR, FileType, Platform,
    )
    from code.generator.project_generator import ProjectGenerator

    requirements = extract_project_requirements(request)
    ids_by_kind: dict[str, list[str]] = {}
    for item in requirements:
        ids_by_kind.setdefault(item.kind, []).append(item.id)

    pipeline_ids = []
    for kind in ("io_input", "validation", "robustness", "dedup",
                 "grouping", "aggregation", "ordering", "io_output"):
        pipeline_ids.extend(ids_by_kind.get(kind, []))
    cli_ids = ids_by_kind.get("cli", [])
    test_ids = []
    for kind in ("tests", "test_case"):
        test_ids.extend(ids_by_kind.get(kind, []))

    project = ProjectIR(
        name=spec.name,
        language="python",
        platform=Platform.CLI,
        metadata={
            "data_app": spec.to_dict(),
            "requirements": [
                {"req_id": item.id, "description": item.semantic}
                for item in requirements
            ],
        },
    )
    src = project.add_module(ModuleIR(name="src", path="src"))
    src.add_file(FileIR(
        path="src/__init__.py", file_type=FileType.SOURCE,
        language="python", content_type="init",
        metadata={"requirement_ids": []}))
    src.add_file(FileIR(
        path="src/pipeline.py", file_type=FileType.SOURCE,
        language="python", content_type="data_pipeline",
        imports=["csv", "json"],
        metadata={"requirement_ids": pipeline_ids}))
    if spec.has_cli:
        src.add_file(FileIR(
            path="src/main.py", file_type=FileType.SOURCE,
            language="python", content_type="data_cli",
            imports=["argparse", "sys", "src.pipeline"],
            metadata={"requirement_ids": cli_ids}))
    if test_ids or spec.test_cases:
        tests = project.add_module(ModuleIR(name="tests", path="tests"))
        tests.add_file(FileIR(
            path="tests/__init__.py", file_type=FileType.SOURCE,
            language="python", content_type="init",
            metadata={"requirement_ids": []}))
        tests.add_file(FileIR(
            path="tests/test_pipeline.py", file_type=FileType.TEST,
            language="python", content_type="data_tests",
            imports=["csv", "json", "os", "tempfile", "unittest",
                     "src.pipeline"],
            metadata={"requirement_ids": test_ids}))

    generator = ProjectGenerator()
    result = generator.generate_project(project)
    return project, result
