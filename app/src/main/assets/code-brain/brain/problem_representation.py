"""ProblemRepresentation: the typed link between a turn's semantic
understanding and the coding problem that gets planned and generated.

A SemanticRepresentation says WHAT THE USER MEANT in this turn; the
ProblemRepresentation says WHAT THE CODE PROBLEM IS — inputs, outputs,
types, transformations, dependencies, algorithm requirements,
verification requirements — derived from the SAME semantics (never
re-read from the request text), and, for a continuation turn, linked
to the PREVIOUS turn's representation and artifact: the conversation's

    Previous Semantic Representation -> Coding Problem

bridge.  An absent field stays empty — never a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The OUTPUT act: where the answer must reach (a return, a print) and
# what it must carry.  The verb's own class decides the source — the
# topic words never do, and no request is looked up.
_OUTPUT_ACT = re.compile(
    r"(?:^|[.!?;]\s+|\band\s+|\bthen\s+|\s)"
    r"(?:please\s+|also\s+)?"
    r"(return|print|output|display|emit|yield|반환|출력)"
    r"(?:s|ed|ing)?\s+([^.;!?\n]{1,80})",
    re.IGNORECASE)

_OUTPUT_SOURCES = {
    "return": "return", "반환": "return",
    "print": "print", "출력": "print", "output": "print",
    "display": "print", "emit": "print", "yield": "yield",
}


def _terminal_operation(semantic) -> str:
    """The composition's LAST operation — the thing the artifact
    finally does, read from the execution order, never from the
    request's wording."""
    comp = getattr(semantic, "composition", None) or {}
    graph = comp.get("graph") or {}
    exec_ids = list(graph.get("execution_order") or [])
    by_id = {o.get("id"): o for o in (graph.get("operations") or [])
             if isinstance(o, dict)}
    names = [by_id[i].get("name", "") for i in exec_ids if i in by_id]
    if not names:
        graph_obj = getattr(semantic, "operation_graph", None)
        names = [op.name for op in
                 (getattr(graph_obj, "get_execution_order", lambda: [])())]
    return str(names[-1]) if names else ""


def output_contract(semantic) -> dict:
    """The OUTPUT contract of an artifact turn: the requested output
    kind, the declared type, the source the answer must reach and the
    operation it must carry.

    Every field comes from the representation: the goal's requested
    output, the derived output type, the composition's terminal
    operation — and the source from the request's own output act, which
    is grammar (return vs print), never a guess.  A turn that carries
    no artifact goal states no contract."""
    goal = getattr(semantic, "goal", None)
    requested = str(getattr(goal, "requested_output", "") or "")
    output_type = str(getattr(semantic, "output_type", "") or "")
    final_operation = _terminal_operation(semantic)
    source = ""
    carry = ""
    if requested == "source_code":
        text = str(getattr(semantic, "raw_text", "") or "")
        match = _OUTPUT_ACT.search(text) if text else None
        if match:
            source = _OUTPUT_SOURCES.get(
                match.group(1).lower(), match.group(1).lower())
            carry = re.sub(r"\s+", " ", match.group(2)).strip(
                " \t,;:!?")
            if len(carry) > 60:
                carry = carry[:60].rstrip()
    return {
        "requested_output": requested,
        "output_type": output_type,
        "output_source": source,
        "final_operation": carry or final_operation,
        "terminal_operation": final_operation,
    }


def check_output_contract(code: str, contract: dict) -> dict:
    """Whether the generated artifact REACHES the contract's output: a
    return act needs a return statement, a print act a print call — and
    a named operation must appear in that statement.  The check reports
    what it found; it never repairs and never rewrites."""
    contract = dict(contract or {})
    source = str(contract.get("output_source") or "")
    carry = str(contract.get("final_operation") or "")
    text = str(code or "")
    report = {
        "requested_output": contract.get("requested_output", ""),
        "output_type": contract.get("output_type", ""),
        "output_source": source,
        "final_operation": carry,
        "satisfied": True,
        "evidence": [],
    }
    if not source:
        report["evidence"].append("the request states no output act")
        return report
    if not text.strip():
        report["satisfied"] = False
        report["evidence"].append(
            "the generation step produced no artifact")
        return report
    if source == "print":
        found = re.search(r"\bprint\s*\(([^)]*)\)", text)
    elif source == "yield":
        found = re.search(r"\byield\b\s*([^\n;]+)", text)
    else:
        found = re.search(r"\breturn\b\s*([^\n;]+)", text)
    if not found:
        report["satisfied"] = False
        report["evidence"].append(
            f"no {source} statement in the generated artifact")
        return report
    report["evidence"].append(f"{source} statement present")
    if carry:
        # A NAMED carry (step_2, total_count) must appear in the
        # statement: an identifier from the request is the request's
        # own vocabulary.  Prose ("the sum of a list") names the
        # MEANING, not a symbol — the generator picks the identifier,
        # so the structure is what can be checked.
        identifiers = [t for t in re.findall(r"[A-Za-z_]\w*", carry)
                       if "_" in t or any(c.isupper() for c in t[1:])]
        if identifiers:
            statement = found.group(0)
            missing = [t for t in identifiers
                       if not re.search(r"\b" + re.escape(t) + r"\b",
                                        statement)]
            if missing:
                report["satisfied"] = False
                report["evidence"].append(
                    f"{source} carries none of " +
                    ", ".join(f"'{t}'" for t in missing))
            else:
                report["evidence"].append(
                    f"{source} carries " +
                    ", ".join(f"'{t}'" for t in identifiers))
        else:
            report["evidence"].append(
                "the carried expression is wording, not a symbol; "
                "the statement's structure is what is checked")
    return report


def _composition_ops(semantic) -> list:
    """The turn's operation list: the composition graph (what the
    generator consumes) when present, else the operation graph."""
    comp = getattr(semantic, "composition", None) or {}
    graph = comp.get("graph") or {}
    ops = graph.get("operations") or []
    if ops:
        return [o for o in ops if isinstance(o, dict) and o.get("name")]
    graph_obj = getattr(semantic, "operation_graph", None)
    return [{"name": o.name, "depends_on": list(getattr(o, "depends_on", [])
                                                or [])}
            for o in (getattr(graph_obj, "operations", None) or [])]


@dataclass
class ProblemRepresentation:
    """The coding problem derived from the turn's semantics."""

    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    types: list = field(default_factory=list)
    transformations: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    algorithm_requirements: list = field(default_factory=list)
    verification_requirements: list = field(default_factory=list)
    # The OUTPUT CONTRACT: the requested output kind, the declared
    # type, the source the answer must reach and the operation it must
    # carry — the same four the plan, the IR and the verification read.
    requested_output: str = ""
    output_type: str = ""
    output_source: str = ""
    final_operation: str = ""
    # The continuation link ({} for a fresh problem): the previous
    # turn's semantic id, the artifact function this problem continues,
    # and why (reference | modify).  The record the trace reads for
    # Conversation State -> Semantic Representation -> Coding Problem.
    continuation: dict = field(default_factory=dict)
    source: str = "semantic"

    # -- construction -------------------------------------------------
    @classmethod
    def from_semantic(cls, semantic, previous=None,
                      artifact=None) -> "ProblemRepresentation":
        """Derive the coding problem from THIS turn's semantics; when
        the turn continues a previous one, carry the link (previous
        semantic id + artifact) and the previous verification
        expectations that must survive the update."""
        inputs = []
        for io in (getattr(semantic, "inputs", None) or []):
            name = str(getattr(io, "name", "") or "")
            type_sig = str(getattr(io, "type_signature", "") or "")
            inputs.append(f"{name}: {type_sig}" if type_sig else name)
        outputs = []
        for io in (getattr(semantic, "outputs", None) or []):
            name = str(getattr(io, "name", "") or "")
            type_sig = str(getattr(io, "type_signature", "") or "")
            outputs.append(f"{name}: {type_sig}" if type_sig else name)

        types = []
        output_type = str(getattr(semantic, "output_type", "") or "")
        if output_type:
            types.append(output_type)
        for io in list(getattr(semantic, "inputs", None) or []) + list(
                getattr(semantic, "outputs", None) or []):
            type_sig = str(getattr(io, "type_signature", "") or "")
            if type_sig and type_sig not in types:
                types.append(type_sig)

        ops = _composition_ops(semantic)
        transformations = [str(o.get("name")) for o in ops]
        dependencies = []
        for o in ops:
            for dep in (o.get("depends_on") or []):
                edge = f"{dep} -> {o.get('name')}"
                if edge not in dependencies:
                    dependencies.append(edge)

        # Algorithm requirements: what the algorithm must realize —
        # the optimization objectives and the declared algorithm hint.
        algorithm_requirements = []
        hint = str(getattr(getattr(semantic, "goal", None),
                           "algorithm_hint", "") or "")
        if hint:
            algorithm_requirements.append(f"algorithm: {hint}")
        for obj in (getattr(semantic, "objectives", None) or []):
            direction = str(getattr(obj, "direction", "") or "")
            measure = str(getattr(obj, "measure", "") or "")
            target = str(getattr(obj, "target", "") or "")
            parts = [p for p in (direction, measure, target) if p]
            if parts:
                text = " ".join(parts)
                if text not in algorithm_requirements:
                    algorithm_requirements.append(text)

        # Verification requirements: the constraints the artifact must
        # satisfy, the declared output type, and (when present) the
        # sample cases the problem statement supplied.
        verification_requirements = []
        for c in (getattr(semantic, "constraints", None) or []):
            name = str(getattr(c, "name", "") or "")
            value = str(getattr(c, "value", "") or "")
            text = f"{name}={value}" if value else name
            if text and text not in verification_requirements:
                verification_requirements.append(text)
        if output_type:
            verification_requirements.append(f"output type: {output_type}")
        # The output contract travels as a VERIFICATION expectation:
        # what the artifact must reach and what it must carry is checked
        # against what was generated, never assumed to have happened.
        contract = output_contract(semantic)
        if contract["output_source"]:
            statement = f"{contract['output_source']}" + (
                f" {contract['final_operation']}"
                if contract["final_operation"] else "")
            verification_requirements.append(f"output source: {statement}")
        patterns = getattr(semantic, "competitive_patterns", None) or {}
        for sample in (patterns.get("sample_cases") or [])[:5]:
            text = str(sample).strip()
            if text and text not in verification_requirements:
                verification_requirements.append(f"sample: {text}")

        problem = cls(
            inputs=inputs,
            outputs=outputs,
            types=types,
            transformations=transformations,
            dependencies=dependencies,
            algorithm_requirements=algorithm_requirements,
            verification_requirements=verification_requirements,
            requested_output=contract["requested_output"],
            output_type=contract["output_type"],
            output_source=contract["output_source"],
            final_operation=contract["final_operation"],
        )
        problem._link(previous, artifact, semantic)
        return problem

    def _link(self, previous, artifact, semantic) -> None:
        """Record the continuation link: which previous representation
        this problem continues and which artifact it must preserve."""
        goal = getattr(semantic, "goal", None)
        parent = dict(getattr(goal, "parent_context", None) or {})
        existing = parent.get("continuation")
        if isinstance(existing, dict) and existing:
            link = dict(existing)
        elif previous is not None:
            artifact = artifact or {}
            function_name = str(artifact.get("function_name") or "")
            if not function_name:
                function_name = str(getattr(
                    getattr(previous, "goal", None), "algorithm_hint",
                    "") or "")
            link = {
                "from": str(getattr(previous, "id", "") or ""),
                "function_name": function_name,
            }
        else:
            link = {}
        if link:
            # the previous turn's verification expectations survive the
            # update unless THIS turn restates them
            if previous is not None:
                prev_problem = getattr(previous, "problem", None)
                if prev_problem is not None:
                    for req in (prev_problem.verification_requirements
                                or []):
                        if req not in self.verification_requirements:
                            self.verification_requirements.append(req)
            self.continuation = link

    # -- serialization ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "types": list(self.types),
            "transformations": list(self.transformations),
            "dependencies": list(self.dependencies),
            "algorithm_requirements": list(self.algorithm_requirements),
            "verification_requirements": list(
                self.verification_requirements),
            "requested_output": self.requested_output,
            "output_type": self.output_type,
            "output_source": self.output_source,
            "final_operation": self.final_operation,
            "continuation": dict(self.continuation),
            "source": self.source,
        }

    @property
    def is_continuation(self) -> bool:
        return bool(self.continuation)
