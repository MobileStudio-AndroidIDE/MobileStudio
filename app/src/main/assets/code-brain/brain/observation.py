"""Observation Layer for RENA-BR NLU pipeline.

Records what each component produces at each stage, enabling:
- Debugging: why was this entity/operation/type selected?
- Audit trail: what evidence supports each inference?
- Regression: did a change improve or worsen results?
- Phase 2+ verification: did retrieved knowledge mutate user requirements?

Usage:
    trace = ObservationTrace(raw_input="소수 판별 함수 만들어줘")
    trace.record_keyword_match("semantic_parser", "생성", "add", confidence=0.6)
    trace.record_entity("semantic_parser", "Number", confidence=0.8)
    trace.record_operation("semantic_parser", "prime", source="OPERATION_PATTERNS")
    trace.finalize()
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import itertools

# --------------------------------------------------------------------------- #
# Execution evidence (verification results linked as evidence)
# --------------------------------------------------------------------------- #
# An ExecutionEvidence record attaches a verification verdict to the
# concrete artifact it judged — the code reference, a short input
# representation, what was expected and what was actually observed.
# A verification stage that did not run MUST say so: the honest verdict
# is "not_run" (never a silent pass).
# --------------------------------------------------------------------------- #

_EVIDENCE_IDS = itertools.count(1)


@dataclass
class ExecutionEvidence:
    """A verification result linked as evidence for generated code."""
    id: str = field(default_factory=lambda: f"exec_ev_{next(_EVIDENCE_IDS)}")
    requirement_id: Optional[str] = None
    code_ref: str = ""
    input_repr: str = ""
    expected_repr: str = ""
    actual_repr: str = ""
    verdict: str = "not_run"  # "pass" | "fail" | "not_run"
    stage: str = "behavior"   # syntax|compile|execute|behavior|edge
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "requirement_id": self.requirement_id,
            "code_ref": self.code_ref,
            "input_repr": self.input_repr,
            "expected_repr": self.expected_repr,
            "actual_repr": self.actual_repr,
            "verdict": self.verdict,
            "stage": self.stage,
            "detail": self.detail,
        }


def not_run_evidence(reason: str, code_ref: str = "",
                     stage: str = "behavior",
                     requirement_id: Optional[str] = None) -> ExecutionEvidence:
    """Evidence for a verification that did NOT run (honest not_run)."""
    return ExecutionEvidence(
        requirement_id=requirement_id,
        code_ref=code_ref,
        verdict="not_run",
        stage=stage,
        detail=str(reason or "not run"),
    )


# --------------------------------------------------------------------------- #
# Canonical pipeline stages (spec #13)
# --------------------------------------------------------------------------- #
# Every full request traverses these 13 stages in this canonical order;
# each stage belongs to one macro phase of the architecture:
# UNDERSTAND -> RETRIEVE -> REASON -> COMPOSE -> VERIFY -> OUTPUT.
# The trace records stages as they EXECUTE (chronological); this tuple
# is the canonical inventory used for coverage and invariants.
# --------------------------------------------------------------------------- #
PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("language_detection", "UNDERSTAND"),
    ("concept_activation", "RETRIEVE"),
    ("relation_traversal", "RETRIEVE"),
    ("dialogue_act", "UNDERSTAND"),
    ("meaning_assembly", "UNDERSTAND"),
    ("target_detection", "UNDERSTAND"),
    ("semantic_composition", "COMPOSE"),
    ("output_type", "COMPOSE"),
    ("reasoning_plan", "REASON"),
    ("ir_generation", "COMPOSE"),
    ("code_generation", "COMPOSE"),
    ("verification", "VERIFY"),
    ("response_output", "OUTPUT"),
)
STAGE_ORDER: tuple[str, ...] = tuple(name for name, _ in PIPELINE_STAGES)
STAGE_MACRO: dict[str, str] = dict(PIPELINE_STAGES)


class EvidenceType(Enum):
    KEYWORD_MATCH = "keyword_match"
    REGEX_MATCH = "regex_match"
    CONCEPT_ACTIVATION = "concept_activation"
    RELATION_TRAVERSAL = "relation_traversal"
    SEMANTIC_COMPOSITION = "semantic_composition"
    CONTEXT_INFERENCE = "context_inference"
    LEGACY_MAPPING = "legacy_mapping"
    USER_EXPLICIT = "user_explicit"


@dataclass
class Evidence:
    """A single piece of evidence supporting an inference."""
    evidence_type: EvidenceType
    source: str            # which module produced this (e.g., "semantic_parser")
    content: str           # what was matched/found (e.g., "생성")
    target: str            # what it mapped to (e.g., "add" operation)
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type.value,
            "source": self.source,
            "content": self.content,
            "target": self.target,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class KeywordMatch:
    """Record of a keyword/pattern match."""
    module: str
    pattern_name: str
    matched_text: str
    mapped_to: str
    confidence: float = 0.0
    was_filtered: bool = False
    filter_reason: str = ""

    def to_dict(self) -> dict:
        d = {
            "module": self.module,
            "pattern": self.pattern_name,
            "matched": self.matched_text,
            "mapped_to": self.mapped_to,
            "confidence": self.confidence,
        }
        if self.was_filtered:
            d["filtered"] = True
            d["filter_reason"] = self.filter_reason
        return d


@dataclass
class EntityRecord:
    """Record of an extracted entity."""
    module: str
    name: str
    type_name: str
    kind: str = "object"
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "name": self.name,
            "type": self.type_name,
            "kind": self.kind,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class OperationRecord:
    """Record of an extracted operation."""
    module: str
    name: str
    source: str  # which list/pattern/composition produced this
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    was_filtered: bool = False
    filter_reason: str = ""

    def to_dict(self) -> dict:
        d = {
            "module": self.module,
            "name": self.name,
            "source": self.source,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }
        if self.was_filtered:
            d["filtered"] = True
            d["filter_reason"] = self.filter_reason
        return d


@dataclass
class TypeRecord:
    """Record of a type inference."""
    module: str
    name: str
    inferred_type: str
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "module": self.module,
            "name": self.name,
            "inferred_type": self.inferred_type,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }
        if self.alternatives:
            d["alternatives"] = self.alternatives
        return d


@dataclass
class PipelineStepRecord:
    """Record of a single pipeline step's input/output."""
    step_name: str
    module: str
    input_summary: str
    output_summary: str
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step_name,
            "module": self.module,
            "input": self.input_summary,
            "output": self.output_summary,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class ObservationTrace:
    """Complete trace of what the NLU pipeline produced for a single request.

    This is the central audit trail. Every component appends its findings here.
    After processing, call finalize() to compute summary statistics.
    """

    def __init__(self, raw_input: str = ""):
        self.raw_input = raw_input
        self.keyword_matches: list[KeywordMatch] = []
        self.entities: list[EntityRecord] = []
        self.operations: list[OperationRecord] = []
        self.types: list[TypeRecord] = []
        self.pipeline_steps: list[PipelineStepRecord] = []
        self.user_semantics_summary: dict = {}
        self.retrieved_knowledge_summary: dict = {}
        self.semantic_composition: dict = {}
        self.errors: list[dict] = []
        self.execution_evidence: list[ExecutionEvidence] = []
        self._finalized = False

    def record_keyword_match(self, module: str, matched_text: str,
                             mapped_to: str, confidence: float = 0.0,
                             pattern_name: str = "", was_filtered: bool = False,
                             filter_reason: str = ""):
        self.keyword_matches.append(KeywordMatch(
            module=module, pattern_name=pattern_name,
            matched_text=matched_text, mapped_to=mapped_to,
            confidence=confidence, was_filtered=was_filtered,
            filter_reason=filter_reason,
        ))

    def record_entity(self, module: str, name: str, type_name: str,
                      kind: str = "object", confidence: float = 0.0,
                      evidence: list[Evidence] = None):
        self.entities.append(EntityRecord(
            module=module, name=name, type_name=type_name,
            kind=kind, confidence=confidence,
            evidence=evidence or [],
        ))

    def record_operation(self, module: str, name: str, source: str,
                         confidence: float = 0.0,
                         evidence: list[Evidence] = None,
                         was_filtered: bool = False,
                         filter_reason: str = ""):
        self.operations.append(OperationRecord(
            module=module, name=name, source=source,
            confidence=confidence, evidence=evidence or [],
            was_filtered=was_filtered, filter_reason=filter_reason,
        ))

    def record_type(self, module: str, name: str, inferred_type: str,
                    confidence: float = 0.0,
                    evidence: list[Evidence] = None,
                    alternatives: list[str] = None):
        self.types.append(TypeRecord(
            module=module, name=name, inferred_type=inferred_type,
            confidence=confidence, evidence=evidence or [],
            alternatives=alternatives or [],
        ))

    def record_pipeline_step(self, step_name: str, module: str,
                             input_summary: str, output_summary: str,
                             confidence: float = 0.0, metadata: dict = None):
        self.pipeline_steps.append(PipelineStepRecord(
            step_name=step_name, module=module,
            input_summary=input_summary, output_summary=output_summary,
            confidence=confidence, metadata=metadata or {},
        ))

    def record_error(self, stage: str, error_type: str, message: str,
                     module: str = ""):
        self.errors.append({
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "module": module,
        })

    def add_evidence(self, evidence: ExecutionEvidence):
        """Append an execution/verification evidence record."""
        self.execution_evidence.append(evidence)

    def add_execution_evidence(self, requirement_id: str = None,
                               code_ref: str = "", input_repr: str = "",
                               expected_repr: str = "", actual_repr: str = "",
                               verdict: str = "not_run",
                               stage: str = "behavior",
                               detail: str = "") -> ExecutionEvidence:
        """Convenience constructor + append for an evidence record."""
        evidence = ExecutionEvidence(
            requirement_id=requirement_id, code_ref=code_ref,
            input_repr=input_repr, expected_repr=expected_repr,
            actual_repr=actual_repr, verdict=verdict, stage=stage,
            detail=detail,
        )
        self.add_evidence(evidence)
        return evidence

    def add_not_run_evidence(self, reason: str, code_ref: str = "",
                             stage: str = "behavior",
                             requirement_id: str = None) -> ExecutionEvidence:
        """Honest NOT_RUN evidence for a verification stage."""
        evidence = not_run_evidence(
            reason, code_ref=code_ref, stage=stage,
            requirement_id=requirement_id)
        self.add_evidence(evidence)
        return evidence

    def set_user_semantics(self, summary: dict):
        self.user_semantics_summary = summary

    def set_retrieved_knowledge(self, summary: dict):
        self.retrieved_knowledge_summary = summary

    def set_semantic_composition(self, composition: dict):
        self.semantic_composition = composition

    def finalize(self):
        """Compute summary statistics."""
        self._finalized = True

    def get_summary(self) -> dict:
        """Return a high-level summary of the trace."""
        if not self._finalized:
            self.finalize()
        recorded = {ps.step_name for ps in self.pipeline_steps}
        stages_recorded = [s for s in STAGE_ORDER if s in recorded]
        stages_missing = [s for s in STAGE_ORDER if s not in recorded]
        return {
            "raw_input": self.raw_input,
            "keyword_matches_count": len(self.keyword_matches),
            "entities_count": len(self.entities),
            "operations_count": len(self.operations),
            "operations": [op.name for op in self.operations if not op.was_filtered],
            "filtered_operations": [
                {"name": op.name, "reason": op.filter_reason}
                for op in self.operations if op.was_filtered
            ],
            "types": [t.inferred_type for t in self.types],
            "errors_count": len(self.errors),
            "execution_evidence_count": len(self.execution_evidence),
            "pipeline_steps": len(self.pipeline_steps),
            "stages_recorded": stages_recorded,
            "stages_missing": stages_missing,
        }

    def check_invariants(self, semantic=None, ir_type_ref: str = None,
                         verification: dict = None,
                         confidence_before: float = None,
                         confidence_after: float = None,
                         check_stage_completion: bool = False) -> list[dict]:
        """Spec #13 invariant checks over the recorded trace.

        Statuses are honest: a check whose inputs are missing reports
        not_applicable (never a silent pass), an unimplemented schema
        reports not_implemented, and a real violation reports fail —
        the trace never hides a violation.
        """
        results: list[dict] = []

        def add(name: str, status: str, detail: str = "") -> None:
            results.append({"invariant": name, "status": status,
                            "detail": detail})

        recorded_names = [ps.step_name for ps in self.pipeline_steps]
        target_stage = next(
            (ps for ps in self.pipeline_steps
             if ps.step_name == "target_detection"), None)

        # INV-1 (#1): the target recorded in the trace and the target
        # carried by the semantic must be the same declaration.
        if semantic is None:
            add("target_preserved", "not_applicable", "no semantic given")
        elif target_stage is None:
            add("target_preserved",
                "fail" if getattr(semantic, "target", {}) else "not_applicable",
                "target declared but target_detection stage not recorded"
                if getattr(semantic, "target", {}) else
                "no target declared, nothing to preserve")
        else:
            sem_target = str(dict(getattr(semantic, "target", {}) or {}))
            stage_target = str(target_stage.output_summary)
            add("target_preserved",
                "pass" if stage_target == sem_target else "fail",
                f"stage={stage_target!r} semantic={sem_target!r}")

        # INV-3 (#3): every composed operation must carry its evidence.
        comp = (getattr(semantic, "composition", None)
                if semantic is not None else None) or {}
        ops = (comp.get("graph") or {}).get("operations") or []
        if semantic is None:
            add("ops_have_evidence", "not_applicable", "no semantic given")
        elif not ops:
            add("ops_have_evidence", "not_applicable",
                "no composed operations")
        else:
            schema_present = any(
                ("semantic_justification" in o) or ("evidence" in o)
                for o in ops)
            if not schema_present:
                add("ops_have_evidence", "not_implemented",
                    "op evidence fields absent from schema (spec #3 pending)")
            else:
                unjustified = [
                    o.get("name", o.get("id", "?")) for o in ops
                    if not (o.get("semantic_justification")
                            or o.get("evidence"))]
                add("ops_have_evidence",
                    "pass" if not unjustified else "fail",
                    "unjustified: " + ", ".join(unjustified)
                    if unjustified else "all ops carry evidence")

        # INV-5 (#5): the semantic output type equals the IR's type_ref.
        if semantic is None or ir_type_ref is None:
            add("output_type_equality", "not_applicable",
                "semantic or IR type missing")
        else:
            sem_type = getattr(semantic, "output_type", "") or ""
            if not sem_type:
                add("output_type_equality", "not_applicable",
                    "semantic declares no output type")
            elif sem_type == ir_type_ref:
                add("output_type_equality", "pass",
                    f"semantic={sem_type} ir={ir_type_ref}")
            else:
                add("output_type_equality", "fail",
                    f"semantic={sem_type} ir={ir_type_ref}")

        # INV-11 (#11): a failed verification must never raise the
        # confidence that was assigned before verification ran.
        if verification is None:
            add("confidence_on_failure", "not_applicable",
                "no verification given")
        else:
            v_ok = (verification.get("success")
                    if isinstance(verification, dict) else bool(verification))
            if v_ok:
                add("confidence_on_failure", "not_applicable",
                    "verification passed")
            elif confidence_before is None or confidence_after is None:
                add("confidence_on_failure", "not_applicable",
                    "confidence values not supplied")
            elif confidence_after > confidence_before:
                add("confidence_on_failure", "fail",
                    f"{confidence_before} -> {confidence_after}")
            else:
                add("confidence_on_failure", "pass",
                    f"{confidence_before} -> {confidence_after}")

        # INV-13 (#13): all 13 canonical stages were traversed.
        if check_stage_completion:
            missing = [s for s in STAGE_ORDER if s not in recorded_names]
            add("all_stages_recorded",
                "pass" if not missing else "fail",
                "complete" if not missing else "missing: " + ", ".join(missing))
        else:
            add("all_stages_recorded", "not_applicable",
                "stage completion not requested for this call site")

        return results

    def to_dict(self) -> dict:
        """Full serialization of the trace."""
        if not self._finalized:
            self.finalize()
        return {
            "raw_input": self.raw_input,
            "keyword_matches": [km.to_dict() for km in self.keyword_matches],
            "entities": [e.to_dict() for e in self.entities],
            "operations": [op.to_dict() for op in self.operations],
            "types": [t.to_dict() for t in self.types],
            "pipeline_steps": [ps.to_dict() for ps in self.pipeline_steps],
            "user_semantics": self.user_semantics_summary,
            "retrieved_knowledge": self.retrieved_knowledge_summary,
            "semantic_composition": self.semantic_composition,
            "errors": self.errors,
            "execution_evidence": [e.to_dict()
                                   for e in self.execution_evidence],
            "summary": self.get_summary(),
        }

    def __repr__(self):
        s = self.get_summary()
        return (f"ObservationTrace(input={s['raw_input']!r}, "
                f"entities={s['entities_count']}, "
                f"ops={s['operations']}, "
                f"types={s['types']}, "
                f"errors={s['errors_count']})")
