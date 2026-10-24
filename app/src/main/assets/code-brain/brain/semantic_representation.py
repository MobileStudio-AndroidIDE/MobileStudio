"""Semantic Representation - Language-independent meaning structures for RENA-BR.

This module defines the core data structures for representing meaning without
relying on token prediction or fixed templates. All natural language inputs
are parsed into these structures, enabling cross-lingual understanding and
compositional reasoning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from enum import Enum
from uuid import uuid4

if TYPE_CHECKING:
    from brain.observation import ObservationTrace
    from brain.problem_representation import ProblemRepresentation


class Language(Enum):
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    UNKNOWN = "unknown"


class GoalType(Enum):
    CREATE_FUNCTION = "create_function"
    CREATE_CLASS = "create_class"
    CREATE_PROJECT = "create_project"
    MODIFY = "modify"
    EXPLAIN = "explain"
    FIX = "fix"
    ANALYZE = "analyze"
    QUESTION = "question"
    CHAT = "chat"
    COMPUTER = "computer"
    # Structured dialogue acts: the goal type is inferred from the
    # semantic representation's meaning structure, never from surface
    # keywords.  A low-confidence meaning keeps CANDIDATES (see
    # GoalSpec.alternatives) and the status "ambiguous" instead of
    # guessing one type.
    COMMAND_QUERY = "command_query"        # a shell/platform how-to
    INFORMATION_QUERY = "information_query"  # a knowledge question
    # An INVESTIGATION: the utterance asks the addressee to find out
    # (research / investigate / explore …), with the workflow it
    # states — searched, evaluated, re-searched when thin, reported.
    RESEARCH = "research"
    AMBIGUOUS = "ambiguous"                # several candidates, low confidence
    UNKNOWN = "unknown"


class OperationType(Enum):
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    SORT = "sort"
    IO = "io"
    CONTROL_FLOW = "control_flow"
    COMPARISON = "comparison"
    ARITHMETIC = "arithmetic"


class SpeechAct(Enum):
    INFORM = "inform"
    REQUEST = "request"
    CONFIRM = "confirm"
    APOLOGIZE = "apologize"
    GREET = "greet"
    OFFER_HELP = "offer_help"
    ASK_CLARIFICATION = "ask_clarification"
    EXPLAIN = "explain"
    REJECT = "reject"


class VerbosityLevel(Enum):
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"


class ToneType(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"


@dataclass
class EntityRef:
    """Reference to an entity in the problem domain."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    type_name: str = ""           # Student, Order, Temperature, List, Number
    kind: str = "object"          # object, data, device, resource, collection
    attributes: dict = field(default_factory=dict)  # score: int, name: str
    quantity: str = "single"      # single, list, collection
    is_input: bool = False
    is_output: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_name,
            "kind": self.kind,
            "attributes": self.attributes,
            "quantity": self.quantity,
            "is_input": self.is_input,
            "is_output": self.is_output,
        }


@dataclass
class OperationSpec:
    """A single operation in the computation pipeline."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""                          # remove_duplicates, filter_above
    op_type: OperationType = OperationType.TRANSFORM
    semantic_role: str = ""                 # deduplication, threshold_filter
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # operation ids
    input_entities: list[str] = field(default_factory=list)  # entity ids
    output_entity: str | None = None        # entity id
    description: str = ""
    # ── Grounding (spec #3): an operation enters the graph only with
    # its reasoning and its evidence — semantic_justification says WHY
    # it is here, evidence says WHAT grounds it, input_bindings say
    # what feeds each input, preconditions state the contract it
    # assumes.  Validation rejects an op without its grounding.
    semantic_justification: str = ""
    evidence: list[str] = field(default_factory=list)
    input_bindings: list[dict] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.op_type.value,
            "semantic_role": self.semantic_role,
            "params": self.params,
            "depends_on": self.depends_on,
            "input_entities": self.input_entities,
            "output_entity": self.output_entity,
            "description": self.description,
            "semantic_justification": self.semantic_justification,
            "evidence": self.evidence,
            "input_bindings": self.input_bindings,
            "preconditions": self.preconditions,
        }


@dataclass
class OperationGraph:
    """DAG of operations with dependency ordering."""
    operations: list[OperationSpec] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)  # operation ids
    exit_points: list[str] = field(default_factory=list)   # operation ids

    def add_operation(self, op: OperationSpec) -> None:
        self.operations.append(op)

    def get_execution_order(self) -> list[OperationSpec]:
        """Topological sort for execution order.

        Kahn's algorithm over the dependency edges, with ONE tie-break:
        among the operations that became ready at the same step, the one
        that entered the graph first runs first.  The composer inserts
        operations in PIPELINE STAGE order (source → aggregate →
        comparison → predicate → selection → transform → sort), so the
        ready-set is always read in stage order and the execution order
        states the pipeline rather than the order a ready-queue
        discovered the nodes in.
        """
        import bisect
        in_degree = {op.id: 0 for op in self.operations}
        adj = {op.id: [] for op in self.operations}

        for op in self.operations:
            for dep_id in op.depends_on:
                if dep_id in adj:
                    adj[dep_id].append(op.id)
                    in_degree[op.id] += 1

        op_index = {op.id: i for i, op in enumerate(self.operations)}
        queue = sorted(
            [op_id for op_id, deg in in_degree.items() if deg == 0],
            key=lambda oid: op_index.get(oid, 0)
        )
        result = []

        while queue:
            current = queue.pop(0)
            op = next(o for o in self.operations if o.id == current)
            result.append(op)
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    bisect.insort(queue, neighbor,
                                  key=lambda oid: op_index.get(oid, 0))

        if len(result) != len(self.operations):
            # Cycle detected, return as-is
            return self.operations
        return result

    def to_dict(self) -> dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "entry_points": self.entry_points,
            "exit_points": self.exit_points,
            "execution_order": [op.id for op in self.get_execution_order()],
        }
    
    def get_operation(self, op_id: str) -> OperationSpec | None:
        """Get operation by ID."""
        for op in self.operations:
            if op.id == op_id:
                return op
        return None

    def validate_dataflow(self) -> list[str]:
        """Type/dataflow validation (spec #4).

        Checks the graph's structural and typing contracts without
        guessing: a dependency must resolve, the dependencies must not
        form a cycle, and each operation's consumed types must be
        provided by its producers (external input provides whatever it
        declares).  Returns the violations found — empty means clean.
        """
        from brain.semantic_composer import PRIMITIVE_ONTOLOGY

        violations: list[str] = []
        by_id = {op.id: op for op in self.operations}

        # 1. Every dependency reference must resolve.
        for op in self.operations:
            for dep in op.depends_on:
                if dep != "__input__" and dep not in by_id:
                    violations.append(
                        f"'{op.name}' depends on missing operation '{dep}'")

        # 2. The dependency graph must be acyclic (shared cycle logic).
        if self.operations:
            unresolved = _unresolved_cycle_ops(self.operations)
            if unresolved:
                violations.append(
                    "dependency cycle among: " + ", ".join(unresolved))

        # 3. Every consumed type must be provided: by a dependency's
        # produced type, or by the external input.  An operation wired
        # only to wrong-typed producers is a dataflow violation.
        for op in self.operations:
            prim = PRIMITIVE_ONTOLOGY.get(op.name)
            if prim is None or not prim.consumes:
                continue
            sources = ([b.get("source", "") for b in op.input_bindings]
                       or list(op.depends_on))
            if not sources:
                continue  # an unbound entry op reads the function input
            if "__input__" in sources:
                continue  # external data supplies the declared input
            provided = set()
            for src in sources:
                src_op = by_id.get(src)
                if src_op is None:
                    continue
                src_prim = PRIMITIVE_ONTOLOGY.get(src_op.name)
                if src_prim is not None and src_prim.produces:
                    provided.add(src_prim.produces)
            unsatisfied = [t for t in prim.consumes if t not in provided]
            if unsatisfied:
                violations.append(
                    f"'{op.name}' consumes {unsatisfied} but its "
                    f"dependencies provide {sorted(provided) or 'nothing'}")

        # 4. The TYPED dataflow graph (typed ports, per-binding producer
        # resolution, ontology type compatibility, cycles, unused
        # outputs) runs right after the structural checks.  Findings are
        # appended through the same return mechanism — the contract
        # (list[str], empty means clean) is unchanged.
        violations.extend(self.to_dataflow_graph().validate())
        return violations

    def to_dataflow_graph(
            self, declared_output_type: str | None = None
    ) -> DataflowGraph:
        """Build the typed dataflow graph over this operation graph.

        Ports come from each OperationSpec's input_bindings (the typed
        parameter connections) and outputs (the produced values, typed
        from the primitive ontology).  The external input is a single
        ``__input__`` port: it provides whatever its consumer declares.
        ``declared_output_type``, when given, is checked against the
        terminal operation's produced type by DataflowGraph.validate().
        """
        from brain.semantic_composer import PRIMITIVE_ONTOLOGY

        graph = DataflowGraph(
            nodes={op.id: op for op in self.operations},
            entry_points=list(self.entry_points),
            exit_points=list(self.exit_points),
            declared_output_type=declared_output_type or "",
        )
        external = DFPort(
            id="__input__", kind="input", name="input", type_name="Any",
            producer_op=None, consumer_ops=[],
            evidence=["external function input"])
        graph.ports.append(external)

        output_ids: dict[str, str] = {}
        for op in self.operations:
            prim = PRIMITIVE_ONTOLOGY.get(op.name)
            produces = prim.produces if prim is not None else ""
            out_id = f"{op.id}:out"
            output_ids[op.id] = out_id
            graph.ports.append(DFPort(
                id=out_id, kind="output", name=f"{op.name}_output",
                type_name=produces, producer_op=op.id, consumer_ops=[],
                evidence=list(op.evidence)))

        for op in self.operations:
            prim = PRIMITIVE_ONTOLOGY.get(op.name)
            consumes = list(prim.consumes) if prim is not None else []
            bindings = [b for b in op.input_bindings
                        if isinstance(b, dict)]
            if not bindings and op.depends_on:
                # A declared dependency with no explicit binding is
                # still dataflow: the dependency feeds this operation's
                # first consumed type (or a generic "data" parameter).
                bindings = [
                    {"input": consumes[i] if i < len(consumes) else "data",
                     "source": dep}
                    for i, dep in enumerate(op.depends_on)
                ]
            for i, binding in enumerate(bindings):
                name = str(binding.get("input", "") or "data")
                source = str(binding.get("source", "") or "")
                port_id = f"{op.id}:in:{name}:{i}"
                evidence = binding.get("evidence")
                if not isinstance(evidence, list):
                    evidence = list(op.evidence)
                port = DFPort(
                    id=port_id, kind="op_param", name=name,
                    type_name=name,
                    producer_op=(source if source in graph.nodes
                                 else None),
                    consumer_ops=[op.id], evidence=list(evidence))
                graph.ports.append(port)
                if source == "__input__":
                    external.consumer_ops.append(op.id)
                    graph.edges.append(DFEdge(
                        source_port=external.id, target_port=port_id,
                        type_name=port.type_name or external.type_name,
                        evidence=list(port.evidence)))
                elif source in graph.nodes:
                    src_out = output_ids[source]
                    src_type = next(
                        (p.type_name for p in graph.ports
                         if p.id == src_out), "")
                    graph.edges.append(DFEdge(
                        source_port=src_out, target_port=port_id,
                        type_name=src_type or port.type_name,
                        evidence=list(port.evidence)))

        for edge in graph.edges:
            src = graph.get_port(edge.source_port)
            tgt = graph.get_port(edge.target_port)
            if src is None or tgt is None or src.kind != "output":
                continue
            for cid in tgt.consumer_ops:
                if cid not in src.consumer_ops:
                    src.consumer_ops.append(cid)
        return graph


# ============================================================
# Typed Dataflow Graph
# ============================================================
# The OperationGraph states WHAT runs and in which order; the typed
# dataflow graph states HOW data moves: every operation parameter is a
# port with a declared type, every producer is connected to the
# parameter it feeds, and validation walks those ports — a binding that
# resolves to no producer, a type that contradicts the ontology, a
# cycle, or an output nobody consumes (and that is not the result) are
# findings, not silent assumptions.

#: Types that never constrain a connection: "we do not know" must not
#: be reported as "wrong".
_OPEN_TYPES = frozenset({"", "any", "unknown", "item", "object", "data"})

#: Coercions admitted by the operation ontology: a producer of the key
#: satisfies a consumer of any type in its value set.  The ontology's
#: exact equality rule is the baseline; these are the declared shares.
_TYPE_COERCIONS = {
    "scalar": {"scalar", "number", "value"},
    "sequence": {"sequence", "collection", "list"},
    "predicate": {"predicate", "bool", "boolean"},
    "element": {"element", "scalar"},
    "frequency": {"frequency"},
    "literal": {"literal"},
    "optional": {"optional", "element"},
    "output": {"output"},
}


def _normalize_type(type_name: str) -> str:
    return str(type_name or "").strip().lower()


def _types_compatible(produced: str, consumed: str) -> bool:
    """Does a producer of ``produced`` satisfy a consumer of ``consumed``?

    Unknown on either side passes (never guess a mismatch); otherwise
    the ontology's exact rule applies, plus the declared coercions.
    """
    p = _normalize_type(produced)
    c = _normalize_type(consumed)
    if p in _OPEN_TYPES or c in _OPEN_TYPES:
        return True
    if p == c:
        return True
    return c in _TYPE_COERCIONS.get(p, frozenset())


def _declared_matches(produced_kind: str, declared: str) -> bool:
    """Does the ontology kind (scalar/sequence/…) match a declared
    language type (int, float, list[int], str, bool)?
    """
    p = _normalize_type(produced_kind)
    d = _normalize_type(declared)
    if p in _OPEN_TYPES or d in _OPEN_TYPES:
        return True
    if d.startswith("list") or d.startswith("sequence") or d.startswith("array"):
        return p == "sequence"
    if d == "bool" or d == "boolean":
        return p == "predicate"
    if d == "str" or d == "string":
        return p in ("scalar", "element", "output", "optional", "literal")
    if "int" in d or "float" in d or "number" in d or d == "double":
        return p in ("scalar", "element")
    return True


def _unresolved_cycle_ops(operations: list["OperationSpec"]) -> list[str]:
    """Names of operations left unresolved by a topological sweep.

    The Kahn sweep already used by OperationGraph.validate_dataflow,
    extracted so the typed graph reuses the same cycle logic instead of
    a second implementation.  Empty list means acyclic.
    """
    in_degree = {op.id: 0 for op in operations}
    adj = {op.id: [] for op in operations}
    for op in operations:
        for dep in op.depends_on:
            if dep in adj:
                adj[dep].append(op.id)
                in_degree[op.id] += 1
    queue = [oid for oid, deg in in_degree.items() if deg == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for nb in adj[cur]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    if seen != len(operations):
        return [op.name for op in operations if in_degree[op.id] > 0]
    return []


@dataclass
class DFPort:
    """A typed connection point of the dataflow graph.

    kind:       "input"    — the problem's external input
                "output"   — an operation's produced value
                "op_param" — a parameter an operation consumes
    producer_op: the operation whose output feeds this port (None for
                 the external input and for op_param ports bound to it).
    consumer_ops: operations that read this port.
    """
    id: str = ""
    kind: str = "op_param"
    name: str = ""
    type_name: str = ""
    producer_op: Optional[str] = None
    consumer_ops: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "type": self.type_name,
            "producer_op": self.producer_op,
            "consumer_ops": list(self.consumer_ops),
            "evidence": list(self.evidence),
        }


@dataclass
class DFEdge:
    """A typed connection from a producer port to a consumer port."""
    source_port: str = ""
    target_port: str = ""
    type_name: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_port": self.source_port,
            "target_port": self.target_port,
            "type": self.type_name,
            "evidence": list(self.evidence),
        }


@dataclass
class DataflowGraph:
    """Typed port graph over an OperationGraph, plus its validate()."""
    nodes: dict[str, "OperationSpec"] = field(default_factory=dict)
    ports: list[DFPort] = field(default_factory=list)
    edges: list[DFEdge] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    exit_points: list[str] = field(default_factory=list)
    declared_output_type: str = ""

    def get_port(self, port_id: str) -> DFPort | None:
        for port in self.ports:
            if port.id == port_id:
                return port
        return None

    def consumers_of(self, op_id: str) -> set[str]:
        """Operations reading the output of ``op_id`` — from typed edges
        and from declared dependencies (both are dataflow)."""
        consumers: set[str] = set()
        for edge in self.edges:
            src = self.get_port(edge.source_port)
            tgt = self.get_port(edge.target_port)
            if (src is not None and src.kind == "output"
                    and src.producer_op == op_id
                    and tgt is not None and tgt.kind == "op_param"):
                for cid in tgt.consumer_ops:
                    consumers.add(cid)
            if (tgt is not None and tgt.producer_op == op_id
                    and src is not None and src.kind == "op_param"):
                for cid in src.consumer_ops:
                    consumers.add(cid)
        for op in self.nodes.values():
            if op.id != op_id and op_id in op.depends_on:
                consumers.add(op.id)
        return consumers

    def final_ops(self) -> set[str]:
        """The result operation(s): the declared exit points, or — when
        none were declared — the last operation of the execution order."""
        if self.exit_points:
            return set(self.exit_points)
        if not self.nodes:
            return set()
        ordered = OperationGraph(
            operations=list(self.nodes.values())).get_execution_order()
        return {ordered[-1].id} if ordered else set()

    def validate(self) -> list[str]:
        """Typed dataflow validation. Returns the violations found —
        empty means the graph is clean.

        1. Every edge endpoint and every port must be well-formed.
        2. Every operation input binding must resolve to exactly one
           producer: the external input or an operation output.
        3. Every produced/consumed pair connected by an edge must be
           type-compatible under the operation ontology.
        4. The dependency graph must be acyclic (shared cycle logic).
        5. Every operation output must be consumed or be the final
           result; any other operation is flagged as unused.
        """
        issues: list[str] = []
        port_by_id: dict[str, DFPort] = {}
        for port in self.ports:
            if not port.id:
                issues.append("dataflow port without an id")
                continue
            if port.id in port_by_id:
                issues.append(f"duplicate dataflow port id '{port.id}'")
            port_by_id[port.id] = port
            if port.kind not in ("input", "output", "op_param"):
                issues.append(
                    f"port '{port.id}' has unknown kind '{port.kind}'")
            if port.kind == "output" and port.producer_op not in self.nodes:
                issues.append(
                    f"output port '{port.id}' has no operation producer")
            if port.kind == "op_param":
                for cid in port.consumer_ops:
                    if cid not in self.nodes:
                        issues.append(
                            f"port '{port.id}' is consumed by missing "
                            f"operation '{cid}'")

        for edge in self.edges:
            if edge.source_port not in port_by_id:
                issues.append(
                    f"edge references missing source port "
                    f"'{edge.source_port}'")
                continue
            if edge.target_port not in port_by_id:
                issues.append(
                    f"edge references missing target port "
                    f"'{edge.target_port}'")
                continue
            src = port_by_id[edge.source_port]
            tgt = port_by_id[edge.target_port]
            if not _types_compatible(src.type_name, tgt.type_name):
                issues.append(
                    f"type mismatch: '{src.name}' produces "
                    f"'{src.type_name}' but '{tgt.name}' consumes "
                    f"'{tgt.type_name}'")

        for op in self.nodes.values():
            seen_bindings: set[tuple[str, str]] = set()
            for binding in op.input_bindings:
                if not isinstance(binding, dict):
                    continue
                source = str(binding.get("source", "") or "")
                name = str(binding.get("input", "") or "")
                if not source:
                    issues.append(
                        f"'{op.name}' has an input binding without a "
                        f"producer")
                    continue
                key = (name, source)
                if key in seen_bindings:
                    issues.append(
                        f"'{op.name}' binds input '{name}' to "
                        f"'{source}' more than once")
                seen_bindings.add(key)
                if source == "__input__":
                    continue
                if source not in self.nodes:
                    issues.append(
                        f"'{op.name}' binds input '{name}' to missing "
                        f"producer '{source}'")

        if self.nodes:
            unresolved = _unresolved_cycle_ops(list(self.nodes.values()))
            if unresolved:
                issues.append(
                    "typed dataflow cycle among: " + ", ".join(unresolved))

            finals = self.final_ops()
            for op in self.nodes.values():
                if op.id in finals:
                    continue
                produced = [p for p in self.ports
                            if p.kind == "output" and p.producer_op == op.id]
                if not produced:
                    continue
                consumers = self.consumers_of(op.id)
                consumers.discard(op.id)
                if not consumers:
                    issues.append(
                        f"unused operation '{op.name}': its output is "
                        f"consumed by no operation and it is not the "
                        f"final result")

        if self.declared_output_type:
            for port in self.ports:
                if port.kind != "output":
                    continue
                if port.producer_op not in self.final_ops():
                    continue
                if not _declared_matches(port.type_name,
                                         self.declared_output_type):
                    issues.append(
                        f"declared output type "
                        f"'{self.declared_output_type}' does not match "
                        f"the terminal operation's produced type "
                        f"'{port.type_name}'")

        return issues

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes.keys()),
            "ports": [p.to_dict() for p in self.ports],
            "edges": [e.to_dict() for e in self.edges],
            "entry_points": list(self.entry_points),
            "exit_points": list(self.exit_points),
            "declared_output_type": self.declared_output_type,
            "violations": self.validate(),
        }


@dataclass
class ParameterSpec:
    """A function parameter with name and type."""
    name: str = ""
    type_name: str = "Any"          # int, float, str, list, bool, etc.
    default_value: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type_name,
            "default": self.default_value,
            "description": self.description,
        }


@dataclass
class FunctionSignature:
    """Captures the expected function signature for code generation."""
    name: str = "process_data"
    parameters: list[ParameterSpec] = field(default_factory=list)
    return_type: str = "Any"        # Any, int, float, str, list, bool
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "description": self.description,
        }

    def get_param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def get_param_types(self) -> list[str]:
        return [p.type_name for p in self.parameters]

    def has_typed_params(self) -> bool:
        return any(p.type_name != "Any" for p in self.parameters)


@dataclass
class IOSpec:
    """Input/Output specification."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    type_signature: str = ""        # List<Number>, Student, int
    description: str = ""
    required: bool = True
    entity_ref: str | None = None   # linked entity id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type_signature,
            "description": self.description,
            "required": self.required,
            "entity_ref": self.entity_ref,
        }


@dataclass
class ConstraintSpec:
    """Constraint on the problem or solution."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    constraint_type: str = ""       # language, platform, performance, correctness
    value: str = ""
    description: str = ""
    # Extended fields for richer constraint representation
    subject: str = ""               # what the constraint applies to: "subsequence", "array", etc.
    operator: str = ""              # "==", "!=", "<", ">", "<=", ">=", "contains", etc.
    aggregate_function: str = ""    # "sum", "count", "max", "min", "product", etc.
    confidence: float = 1.0
    evidence: str = ""              # what text evidence supports this constraint

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.constraint_type,
            "value": self.value,
            "description": self.description,
            "subject": self.subject,
            "operator": self.operator,
            "aggregate_function": self.aggregate_function,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


# ============================================================
# Constraint Graph
# ============================================================
# Constraints are not a flat list: each one applies TO something (an
# entity or an operation on the path), implies other constraints when
# it is stronger, and conflicts with a contradictory one on the same
# subject.  The graph materializes those relations so check() can walk
# them and report real conflicts / unsatisfied preconditions.

_COMPARISON_OPERATORS = ("==", "!=", "<", "<=", ">", ">=")
_DIRECTION_OPERATORS = ("maximize", "minimize")


def _as_number(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _implies(op1: str, val1: float, op2: str, val2: float) -> bool:
    """Does ``x op1 val1`` logically imply ``x op2 val2``?"""
    if op1 == "==":
        if op2 == "==":
            return val1 == val2
        if op2 == "!=":
            return val1 != val2
        if op2 == ">":
            return val1 > val2
        if op2 == ">=":
            return val1 >= val2
        if op2 == "<":
            return val1 < val2
        if op2 == "<=":
            return val1 <= val2
    elif op1 == "!=" or op2 == "!=":
        return False
    elif op1 == ">" and op2 in (">", ">="):
        return val1 >= val2
    elif op1 == ">=" and op2 == ">=":
        return val1 >= val2
    elif op1 == ">=" and op2 == ">":
        return val1 > val2
    elif op1 == "<" and op2 in ("<", "<="):
        return val1 <= val2
    elif op1 == "<=" and op2 == "<=":
        return val1 <= val2
    elif op1 == "<=" and op2 == "<":
        return val1 < val2
    return False


def _conflicts(op1: str, val1: float | None, val2: float | None,
               op2: str) -> bool:
    """Are ``x op1 val1`` and ``x op2 val2`` contradictory?"""
    if {op1, op2} == {"maximize", "minimize"}:
        return True
    if val1 is None or val2 is None:
        return False
    if op1 == "==" and op2 == "==":
        return val1 != val2
    if {op1, op2} == {"==", "!="}:
        return val1 == val2
    if op1 == "==":
        return not _implies(op1, val1, op2, val2)
    if op2 == "==":
        return not _implies(op2, val2, op1, val1)
    if op1 == ">" and op2 == "<":
        return val1 >= val2
    if op1 == ">" and op2 == "<=":
        return val1 >= val2
    if op1 == ">=" and op2 == "<":
        return val1 >= val2
    if op1 == ">=" and op2 == "<=":
        return val1 > val2
    if op1 == "<" and op2 == ">":
        return val2 >= val1
    if op1 == "<" and op2 == ">=":
        return val2 >= val1
    if op1 == "<=" and op2 == ">":
        return val2 >= val1
    if op1 == "<=" and op2 == ">=":
        return val2 > val1
    return False


@dataclass
class ConstraintNode:
    """One constraint (or objective) as a node of the constraint graph."""
    id: str = ""
    subject: str = ""
    operator: str = ""
    value: str = ""
    aggregate: str = ""
    source_constraint_id: str = ""
    evidence: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "operator": self.operator,
            "value": self.value,
            "aggregate": self.aggregate,
            "source_constraint_id": self.source_constraint_id,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class ConstraintEdge:
    """applies_to (constraint→entity/op), implies, or conflicts."""
    kind: str = ""
    source_id: str = ""
    target_id: str = ""
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "source": self.source_id,
            "target": self.target_id,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class ConstraintGraph:
    constraint_nodes: list[ConstraintNode] = field(default_factory=list)
    entity_nodes: list[str] = field(default_factory=list)
    op_nodes: list[str] = field(default_factory=list)
    edges: list[ConstraintEdge] = field(default_factory=list)
    op_labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edges and self.constraint_nodes:
            self.infer_edges()

    @classmethod
    def from_semantic(cls, rep: "SemanticRepresentation") -> "ConstraintGraph":
        """Build the graph from a representation's constraints and
        objectives (constraint nodes), entities (entity nodes) and
        operations (op nodes), then derive applies_to / implies /
        conflicts edges."""
        constraints: list[ConstraintNode] = []
        entities: list[str] = []
        op_nodes: list[str] = []
        op_labels: dict[str, str] = {}

        def _add_entity(name: str) -> None:
            name = (name or "").strip()
            if name and name not in entities:
                entities.append(name)

        for e in getattr(rep, "entities", []) or []:
            _add_entity(getattr(e, "name", "") or getattr(e, "id", ""))
        for c in getattr(rep, "constraints", []) or []:
            constraints.append(ConstraintNode(
                id=f"constraint:{getattr(c, 'id', '')}",
                subject=getattr(c, "subject", "") or getattr(c, "name", ""),
                operator=getattr(c, "operator", ""),
                value=getattr(c, "value", ""),
                aggregate=getattr(c, "aggregate_function", ""),
                source_constraint_id=getattr(c, "id", ""),
                evidence=(getattr(c, "evidence", "")
                          or getattr(c, "description", "")),
                confidence=float(getattr(c, "confidence", 1.0) or 0.0),
            ))
            _add_entity(getattr(c, "subject", ""))
        for o in getattr(rep, "objectives", []) or []:
            constraints.append(ConstraintNode(
                id=f"objective:{getattr(o, 'id', '')}",
                subject=(getattr(o, "measure", "")
                         or getattr(o, "target", "")),
                operator=getattr(o, "direction", ""),
                value=getattr(o, "target", ""),
                aggregate=getattr(o, "measure", ""),
                source_constraint_id=getattr(o, "id", ""),
                evidence=getattr(o, "evidence", ""),
                confidence=float(getattr(o, "confidence", 0.0) or 0.0),
            ))
            _add_entity(getattr(o, "target", ""))
            _add_entity(getattr(o, "measure", ""))
        for op in getattr(
                getattr(rep, "operation_graph", None), "operations", []) or []:
            op_id = getattr(op, "id", "")
            if not op_id:
                continue
            op_nodes.append(op_id)
            op_labels[op_id] = getattr(op, "name", "")
        return cls(constraint_nodes=constraints, entity_nodes=entities,
                   op_nodes=op_nodes, op_labels=op_labels)

    def infer_edges(self) -> None:
        """Derive applies_to, implies and conflicts edges from the
        nodes.  Already-inferred edges are never duplicated."""
        existing = {(e.kind, e.source_id, e.target_id) for e in self.edges}

        def _add(kind: str, source: str, target: str, reason: str,
                 confidence: float = 1.0) -> None:
            key = (kind, source, target)
            if key in existing:
                return
            existing.add(key)
            self.edges.append(ConstraintEdge(
                kind=kind, source_id=source, target_id=target,
                reason=reason, confidence=confidence))

        for node in self.constraint_nodes:
            subject = (node.subject or "").strip().lower()
            if not subject:
                continue
            for entity in self.entity_nodes:
                lowered = entity.lower()
                if subject == lowered or subject in lowered or lowered in subject:
                    _add("applies_to", node.id, f"entity:{entity}",
                         f"{node.subject} applies to entity {entity}",
                         node.confidence)
            for op_id in self.op_nodes:
                label = (self.op_labels.get(op_id, "") or "").lower()
                if not label:
                    continue
                if subject == label or subject in label or label in subject:
                    _add("applies_to", node.id, f"op:{op_id}",
                         f"{node.subject} applies to operation "
                         f"{self.op_labels.get(op_id, op_id)}",
                         node.confidence)

        for i, first in enumerate(self.constraint_nodes):
            for second in self.constraint_nodes[i + 1:]:
                if (first.subject or "").strip().lower() != \
                        (second.subject or "").strip().lower():
                    continue
                if not first.subject.strip():
                    continue
                val1 = _as_number(first.value)
                val2 = _as_number(second.value)
                if first.operator in _COMPARISON_OPERATORS and \
                        second.operator in _COMPARISON_OPERATORS and \
                        val1 is not None and val2 is not None:
                    if _conflicts(first.operator, val1, val2,
                                  second.operator):
                        _add("conflicts", first.id, second.id,
                             f"'{first.subject} {first.operator} "
                             f"{first.value}' contradicts "
                             f"'{second.subject} {second.operator} "
                             f"{second.value}'",
                             min(first.confidence, second.confidence))
                        continue
                    if _implies(first.operator, val1, second.operator, val2):
                        _add("implies", first.id, second.id,
                             f"'{first.subject} {first.operator} "
                             f"{first.value}' implies "
                             f"'{second.subject} {second.operator} "
                             f"{second.value}'",
                             min(first.confidence, second.confidence))
                    elif _implies(second.operator, val2,
                                  first.operator, val1):
                        _add("implies", second.id, first.id,
                             f"'{second.subject} {second.operator} "
                             f"{second.value}' implies "
                             f"'{first.subject} {first.operator} "
                             f"{first.value}'",
                             min(first.confidence, second.confidence))
                elif first.operator in _DIRECTION_OPERATORS and \
                        second.operator in _DIRECTION_OPERATORS and \
                        first.operator != second.operator:
                    _add("conflicts", first.id, second.id,
                         f"'{first.subject}' cannot be both "
                         f"{first.operator} and {second.operator}",
                         min(first.confidence, second.confidence))

    def check(self, ops_by_id: dict | None = None) -> list[str]:
        """Walk the graph and report the violations found: contradictory
        constraints and operations whose preconditions are unsatisfied
        because nothing on the path feeds their consumed types."""
        violations: list[str] = []
        for edge in self.edges:
            if edge.kind == "conflicts":
                violations.append(f"constraint conflict: {edge.reason}")
        ops_by_id = ops_by_id or {}
        for op_id in self.op_nodes:
            op = ops_by_id.get(op_id)
            if op is None:
                continue
            if getattr(op, "input_bindings", None) or \
                    getattr(op, "depends_on", None):
                continue
            from brain.semantic_composer import PRIMITIVE_ONTOLOGY
            prim = PRIMITIVE_ONTOLOGY.get(getattr(op, "name", ""))
            consumes = list(prim.consumes) if prim is not None else []
            if not consumes:
                continue
            preconds = list(getattr(op, "preconditions", []) or [])
            if not preconds and prim is not None:
                preconds = list(prim.preconditions) or [
                    f"input provides {t}" for t in consumes]
            detail = preconds[0] if preconds else f"input provides {consumes[0]}"
            violations.append(
                f"unsatisfied precondition for '{getattr(op, 'name', op_id)}'"
                f": {detail}")
        return violations

    def to_dict(self) -> dict:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.kind] = counts.get(edge.kind, 0) + 1
        return {
            "constraint_nodes": [n.to_dict() for n in self.constraint_nodes],
            "entity_nodes": list(self.entity_nodes),
            "op_nodes": list(self.op_nodes),
            "edges": [e.to_dict() for e in self.edges],
            "edge_counts": counts,
        }


@dataclass
class ObjectiveSpec:
    """Optimization objective extracted from natural language.
    
    Examples:
        "longest" → ObjectiveSpec(direction="maximize", measure="length")
        "shortest" → ObjectiveSpec(direction="minimize", measure="length")
        "maximum sum" → ObjectiveSpec(direction="maximize", measure="sum")
        "minimum value" → ObjectiveSpec(direction="minimize", measure="value")
    """
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    direction: str = ""     # "maximize" or "minimize"
    measure: str = ""       # "length", "sum", "count", "value", "product"
    target: str = ""        # what is being optimized: "subarray", "subsequence", "segment"
    confidence: float = 0.0
    evidence: str = ""      # the text that triggered this objective

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "direction": self.direction,
            "measure": self.measure,
            "target": self.target,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class SubstructureSpec:
    """Structural pattern in the data.
    
    Examples:
        "contiguous subarray" → SubstructureSpec(type="contiguous_subsequence")
        "subsequence" → SubstructureSpec(type="subsequence")
        "prefix" → SubstructureSpec(type="prefix")
        "suffix" → SubstructureSpec(type="suffix")
        "pair" → SubstructureSpec(type="pair")
    """
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    structure_type: str = ""    # "contiguous_subsequence", "subsequence", "prefix", "suffix", "pair", "segment"
    scope: str = ""            # "entire_input", "computed", "any"
    element_type: str = ""     # "integer", "string", "any"
    is_contiguous: bool = True
    confidence: float = 0.0
    evidence: str = ""         # the text that triggered this substructure

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "structure_type": self.structure_type,
            "scope": self.scope,
            "element_type": self.element_type,
            "is_contiguous": self.is_contiguous,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class GoalSpec:
    """Goal specification extracted from user input.

    Structured goal inference: the goal is a structured object inferred
    from the semantic representation's meaning structure — the
    performative action, the artifact being created, the requested
    output kind, the parent context, the evidence, and the alternative
    candidates with confidences.  Surface keywords are never the final
    goal-type basis: a keyword occurrence is recorded as a legacy
    candidate and verified by the semantic interpreter.
    """
    goal_type: GoalType = GoalType.UNKNOWN
    target: str = ""                    # what to create/fix/explain
    algorithm_hint: str | None = None   # binary_search, quick_sort
    domain: str = ""                    # search, calculation, data_management
    domain_evidence: dict = field(default_factory=dict)  # evidence for domain inference
    confidence: float = 0.0
    # Extended goal fields for richer semantic representation
    goal_direction: str | None = None   # "maximize", "minimize", None
    goal_measure: str | None = None     # "length", "sum", "count", "value", None
    goal_object: str | None = None      # what is being optimized: "subarray", "subsequence", etc.
    modifiers: list[str] = field(default_factory=list)  # ["contiguous", "non_empty"] etc.
    qualifiers: list[str] = field(default_factory=list)  # additional qualifiers
    # ── Structured goal inference (semantic representation drives) ──
    action: str = ""                    # the performative action: create/modify/explain/query/chat
    artifact: dict = field(default_factory=dict)   # {kind, target, behavior, ordering, domain}
    requested_output: str = ""          # source_code / shell_command / answer / explanation
    parent_context: dict = field(default_factory=dict)  # the conversation context
    evidence: list = field(default_factory=list)        # the goal's evidence items
    alternatives: list = field(default_factory=list)    # [{goal_type, confidence}] candidates
    status: str = ""                    # "" | "ambiguous"
    legacy_candidates: list = field(default_factory=list)  # recorded keyword occurrences, unverified
    # The REQUESTED WORKFLOW: the ordered acts the utterance spells out
    # ("search, and search again if the results are thin, then
    # report").  Read from the utterance's own coordination — never a
    # canned procedure for a particular question.
    workflow: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.goal_type.value,
            "target": self.target,
            "algorithm_hint": self.algorithm_hint,
            "domain": self.domain,
            "domain_evidence": self.domain_evidence,
            "confidence": self.confidence,
            "goal_direction": self.goal_direction,
            "goal_measure": self.goal_measure,
            "goal_object": self.goal_object,
            "modifiers": self.modifiers,
            "qualifiers": self.qualifiers,
            "action": self.action,
            "artifact": self.artifact,
            "requested_output": self.requested_output,
            "parent_context": self.parent_context,
            "evidence": self.evidence,
            "alternatives": self.alternatives,
            "status": self.status,
            "workflow": dict(self.workflow),
        }


@dataclass
class UserContext:
    """Who is speaking in this turn.

    Everything here is OBSERVED from the utterance or carried from the
    discourse — the role and expertise the request itself states, the
    language it is written in, the preferences the speaker expressed.
    Nothing is inferred from the TOPIC: a question about surgery is
    asked by whoever asked it."""
    role: str = ""
    expertise: str = ""
    language: str = ""
    preferences: dict = field(default_factory=dict)
    prior_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "expertise": self.expertise,
            "language": self.language,
            "preferences": dict(self.preferences),
            "prior_topics": list(self.prior_topics),
        }


@dataclass
class ResponseSpecification:
    """What the response must BE.

    Decided from the request's own grammar — an extent adverbial
    ("in detail"), a measure phrase ("under two minutes"), an
    enumerated demand, an audience, a register cue — and carried to
    the generation as the bounds it obeys.  It is never a table of
    question→length: the same grammar yields the same specification
    for any topic."""
    depth: str = ""                # brief | normal | deep
    detail_level: str = ""         # simple | normal | detailed
    target_length: int = 0
    minimum_length: int = 0
    maximum_length: int = 0
    section_count: int = 0
    example_count: int = 0
    explanation_style: str = ""
    complexity: str = ""
    audience: str = ""
    constraints: list[str] = field(default_factory=list)
    warrants_longform: bool = False

    def to_dict(self) -> dict:
        return {
            "depth": self.depth,
            "detail_level": self.detail_level,
            "target_length": self.target_length,
            "minimum_length": self.minimum_length,
            "maximum_length": self.maximum_length,
            "section_count": self.section_count,
            "example_count": self.example_count,
            "explanation_style": self.explanation_style,
            "complexity": self.complexity,
            "audience": self.audience,
            "constraints": list(self.constraints),
            "warrants_longform": self.warrants_longform,
        }


@dataclass
class InformationGoal:
    """One unit of information the utterance asks for, and its place in
    the whole.

    The MAIN goal carries the topic.  a CHILD goal asks for an aspect
    OF it — an example, a mechanism, a report — and names its parent,
    so a reference inside the child ("how IT helps…") resolves against
    the parent instead of the discourse."""
    id: str = ""
    focus: str = ""
    role: str = "main"              # main | child
    parent_id: str = ""
    aspect: str = ""
    asks_for: str = ""              # explanation | example | report | ...
    constraints: list[str] = field(default_factory=list)
    status: str = ""                # "" | answered | gap

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "focus": self.focus,
            "role": self.role,
            "parent_id": self.parent_id,
            "aspect": self.aspect,
            "asks_for": self.asks_for,
            "constraints": list(self.constraints),
            "status": self.status,
        }


@dataclass
class SemanticSegment:
    """One span of the input and its role in the whole.

    `source_span` is the span's position in the RAW text, so a
    downstream stage can always go back to what the speaker actually
    wrote.  `parent_id` composes segments (an instruction's example,
    a submission's stack trace) and `dependencies` records what this
    segment needs before it can be read."""
    segment_id: str = ""
    kind: str = "text"              # instruction | constraint | report | ...
    text: str = ""
    parent_id: str = ""
    source_span: list[int] = field(default_factory=list)
    semantic_roles: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "text": self.text,
            "parent_id": self.parent_id,
            "source_span": list(self.source_span),
            "semantic_roles": list(self.semantic_roles),
            "dependencies": list(self.dependencies),
        }


@dataclass
class SemanticRepresentation:
    """Language-independent meaning representation."""
    # Core identification
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    source_language: Language = Language.UNKNOWN
    raw_text: str = ""
    
    # Goal
    goal: GoalSpec = field(default_factory=GoalSpec)
    
    # Function signature (extracted from natural language)
    function_signature: FunctionSignature = field(default_factory=FunctionSignature)
    
    # Entities
    entities: list[EntityRef] = field(default_factory=list)
    
    # Operations
    operation_graph: OperationGraph = field(default_factory=OperationGraph)
    
    # I/O
    inputs: list[IOSpec] = field(default_factory=list)
    outputs: list[IOSpec] = field(default_factory=list)

    # Declared return type of the composed function (spec #5), derived
    # from the operation graph's terminal contract — e.g. "int",
    # "float", "list[int]", "bool".  The invariant of #5 compares this
    # against the IR's type_ref, the generated signature and the
    # verified behavior.  "" means nothing was declared (no graph or an
    # unknown terminal) — never a guess.
    output_type: str = ""

    # Type/dataflow validation findings (spec #4): violations detected
    # over the final operation graph (dangling dependency, dependency
    # cycle, a consumed type no dependency provides).  Empty means the
    # graph validated clean — never an unchecked claim.
    dataflow_violations: list[str] = field(default_factory=list)
    
    # Constraints
    constraints: list[ConstraintSpec] = field(default_factory=list)
    
    # Objectives (optimization goals)
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    
    # Substructures (structural patterns in data)
    substructures: list[SubstructureSpec] = field(default_factory=list)
    
    # Arithmetic expression parsed from natural language
    # ("square of A + B" → "(A + B) ** 2"); empty string when none
    arithmetic_expression: str = ""
    
    # Meta
    confidence: float = 0.0
    alternatives: list[SemanticRepresentation] = field(default_factory=list)
    
    # Activated concepts/relations (for debugging)
    activated_concept_ids: list[str] = field(default_factory=list)
    relation_paths: list[list[str]] = field(default_factory=list)
    composition: Optional[dict] = None
    # Phase 3: Uncertainty tracking
    uncertainty_status: str = "resolved"  # resolved | partially_resolved | ambiguous | uncertain
    candidate_meanings: list[dict] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    evidence_log: list[dict] = field(default_factory=list)  # what evidence supports each inference
    
    # Competitive programming patterns (detected from problem structure)
    competitive_patterns: dict = field(default_factory=dict)

    # Dialogue act layer (speech-act structure and derived intent).
    # The act describes WHAT THE SPEAKER IS DOING; the intent type is
    # derived from it.  The topic words never decide the intent type.
    dialogue_act: dict = field(default_factory=dict)
    dialogue_intent: dict = field(default_factory=dict)

    # Context / reference resolution: what this turn's referring
    # expression points at (surface_form → referent → confidence →
    # evidence).  {} until the resolver ran — an absent reference stays
    # absent, never a guess.
    reference: dict = field(default_factory=dict)

    # The turn's DISCOURSE operation (e.g. a reset_topic act scoped to
    # the current topic).  {} when the turn performs none.
    discourse_action: dict = field(default_factory=dict)

    # The knowledge requirement decided for this turn: whether external
    # (web) evidence is needed, from what, and why — decided BEFORE the
    # retrieval and recorded with it.
    knowledge_requirement: dict = field(default_factory=dict)

    # Declared output target (spec #1): captured ONCE at parse time from
    # the request's explicit language/platform mention and preserved for
    # every downstream stage.  Shape: {"kind": "language",
    # "language": "kotlin", ...} or {"kind": "platform", "platform":
    # "android", "language": "kotlin", ...}.  {} means the request
    # declared no target — an absent target stays absent, never a guess.
    target: dict = field(default_factory=dict)

    # Observation trace (spec #13): the audit trail of canonical
    # pipeline stages for this request.  Runtime object; to_dict()
    # exposes only its summary (stage coverage + counts), never the
    # full step log.
    observation: Optional["ObservationTrace"] = None

    # The turn's CODING PROBLEM (ProblemRepresentation): the typed link
    # between this understanding and the code problem that gets planned
    # and generated — inputs, outputs, types, transformations,
    # dependencies, algorithm/verification requirements.  Attached on
    # the code routes only; None on a dialogue turn (it has no code
    # problem — an absent link stays absent, never a guess).  On a
    # continuation turn it carries the link to the previous turn's
    # representation (Previous Semantic Representation → Coding
    # Problem).
    problem: Optional["ProblemRepresentation"] = None

    # The turn's SUBMISSION SEGMENTATION (form evidence): the split of
    # the raw submission into its instruction (what was parsed) and its
    # structural evidence — code fragments, stack traces, sample
    # input/output.  {} when the submission carried no such structure
    # (a submission is its own instruction when its form separates
    # nothing).
    submission: dict = field(default_factory=dict)

    # The SPEAKER's context for this turn (UserContext): role,
    # expertise, language and the preferences the request itself
    # states.  Observed, never inferred from the topic.
    user_context: UserContext = field(default_factory=UserContext)

    # The response this request demands (ResponseSpecification): the
    # length bounds, depth, structure and constraints read from the
    # request's own grammar.  The generation obeys these; they are
    # never a question→length table.
    response_spec: ResponseSpecification = field(
        default_factory=ResponseSpecification)

    # The utterance's INFORMATION GOALS: the main topic and the child
    # asks it carries (an example, a mechanism, a report), each naming
    # its parent.  Empty when the utterance asks for no content.
    information_goals: list[InformationGoal] = field(default_factory=list)

    # The input's SEGMENTATION (SemanticSegment): the spans the
    # utterance is made of, their kinds and their dependencies.  The
    # composition (parent/child) is preserved, so a later stage can
    # read the whole structure instead of a flat string.
    segments: list[SemanticSegment] = field(default_factory=list)

    # Lazy cache of the constraint graph (constraint_graph()).  The
    # graph is derived from constraints/objectives/entities/operations,
    # so it is computed on first access and reused afterwards.
    _constraint_graph_cache: Optional["ConstraintGraph"] = field(
        default=None, repr=False, compare=False)

    @property
    def intent(self) -> str:
        """The derived intent type, e.g. 'information_query'."""
        return str(self.dialogue_intent.get("type", "")) or ""

    @property
    def intent_topics(self) -> list[str]:
        return list(self.dialogue_intent.get("topics", []) or [])

    def get_operation_order(self) -> list[str]:
        return [op.name for op in self.operation_graph.get_execution_order()]

    def get_input_types(self) -> list[str]:
        return [inp.type_signature for inp in self.inputs]

    def get_output_types(self) -> list[str]:
        return [out.type_signature for out in self.outputs]

    def constraint_graph(self) -> ConstraintGraph:
        """The typed constraint graph of this meaning: constraint and
        objective nodes, entity and operation nodes, applies_to /
        implies / conflicts edges.  Computed lazily and cached; the
        same object is returned until the representation changes."""
        if self._constraint_graph_cache is None:
            self._constraint_graph_cache = ConstraintGraph.from_semantic(self)
        return self._constraint_graph_cache

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "language": self.source_language.value,
            "detected_language": self.source_language.value,
            "raw_text": self.raw_text,
            "goal": self.goal.to_dict(),
            "function_signature": self.function_signature.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "operation_graph": self.operation_graph.to_dict(),
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "output_type": self.output_type,
            "dataflow_violations": list(self.dataflow_violations),
            "constraints": [c.to_dict() for c in self.constraints],
            "objectives": [o.to_dict() for o in self.objectives],
            "substructures": [s.to_dict() for s in self.substructures],
            "confidence": self.confidence,
            "alternatives_count": len(self.alternatives),
            "activated_concepts": self.activated_concept_ids,
            "relation_paths": self.relation_paths,
            "dialogue_act": dict(self.dialogue_act),
            "dialogue_intent": dict(self.dialogue_intent),
            "reference": dict(self.reference),
            "discourse_action": dict(self.discourse_action),
            "knowledge_requirement": dict(self.knowledge_requirement),
            "composition": self.composition,
            "uncertainty_status": self.uncertainty_status,
            "candidate_meanings": self.candidate_meanings,
            "missing_information": self.missing_information,
            "evidence_log": self.evidence_log,
            "competitive_patterns": self.competitive_patterns,
            "target": dict(self.target),
            "problem": (self.problem.to_dict()
                        if self.problem is not None else None),
            "submission": dict(self.submission),
            "user_context": self.user_context.to_dict(),
            "response_spec": self.response_spec.to_dict(),
            "information_goals": [g.to_dict()
                                  for g in self.information_goals],
            "segments": [s.to_dict() for s in self.segments],
            "trace_summary": (self.observation.get_summary()
                              if self.observation is not None else None),
        }
    
    def check_completeness(self) -> dict:
        """Check if all required semantic components are present."""
        issues = []
        warnings = []
        
        # Goal understanding
        if self.goal.goal_type == GoalType.UNKNOWN:
            issues.append("Goal type not understood")
        elif self.goal.goal_type in (GoalType.CREATE_FUNCTION, GoalType.CREATE_CLASS, GoalType.CREATE_PROJECT):
            if not self.goal.target:
                issues.append("Target (what to create) not specified")
        
        # Entities
        if not self.entities:
            # Entities aren't strictly required when we already have concrete
            # input/output type signatures (e.g. pure algorithm functions that
            # operate on primitives like "sorts numbers").
            if not self.inputs and not self.outputs:
                issues.append("No entities extracted")
            else:
                warnings.append("No named entities extracted")
        else:
            user_entities = [e for e in self.entities if e.is_input]
            if not user_entities:
                warnings.append("No input entities identified")
        
        # Inputs
        if not self.inputs:
            issues.append("Input specification missing")
        else:
            for inp in self.inputs:
                if not inp.type_signature or inp.type_signature == "Item":
                    warnings.append(f"Input type not specific: {inp.type_signature}")
        
        # Outputs
        if not self.outputs:
            issues.append("Output specification missing")
        else:
            for out in self.outputs:
                if not out.type_signature or out.type_signature == "Item":
                    warnings.append(f"Output type not specific: {out.type_signature}")
        
        # Operations
        if not self.operation_graph.operations:
            issues.append("No operations extracted")
        else:
            op_names = [op.name for op in self.operation_graph.operations]
            # Check for dependency violations
            for op in self.operation_graph.operations:
                for dep_id in op.depends_on:
                    dep_op = self.operation_graph.get_operation(dep_id)
                    if dep_op and dep_op.name not in op_names:
                        warnings.append(f"Operation {op.name} depends on missing operation {dep_id}")
        
        # Constraints
        has_language = any(c.constraint_type == "language" for c in self.constraints)
        if not has_language:
            warnings.append("No language constraint specified")

        # Constraint graph: the constraints' contradictions and the
        # preconditions left unsatisfied on the operation path are
        # VALIDATED here, with the other completeness checks — the
        # graph is consumed, never dead code.  Findings are warnings
        # (the completeness verdict's contract is unchanged), and the
        # graph summary is exposed under "constraint_graph".
        constraint_graph_data: dict = {}
        try:
            cg = self.constraint_graph()
            cg_violations = cg.check(
                {op.id: op for op in self.operation_graph.operations})
            constraint_graph_data = cg.to_dict()
            for violation in cg_violations:
                warnings.append(f"constraint graph: {violation}")
        except Exception:
            cg_violations = []
            constraint_graph_data = {}
        
        # Domain
        if not self.goal.domain or self.goal.domain == "unknown":
            warnings.append("Domain could not be confidently determined")
        
        # Requirement coverage
        extracted_count = len(self.entities) + len(self.operation_graph.operations) + len(self.inputs) + len(self.outputs)
        preserved_count = sum(1 for e in self.entities if e.is_input) + len(self.operation_graph.operations)
        lost_count = max(0, extracted_count - preserved_count)
        
        return {
            "complete": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "coverage": {
                "extracted": extracted_count,
                "preserved": preserved_count,
                "lost": lost_count,
                "inferred": len(self.operation_graph.operations) - len([op for op in self.operation_graph.operations if op.name in ("transform", "compute")])
            },
            "constraint_graph": constraint_graph_data,
        }
    
    def debug_output(self) -> str:
        """Generate semantic debugging output."""
        lines = []
        lines.append("=" * 60)
        lines.append("[SEMANTIC DEBUG OUTPUT]")
        lines.append("=" * 60)
        lines.append(f"[Raw Text] {self.raw_text[:200]}")
        lines.append(f"[Language] {self.source_language.value}")
        lines.append(f"[Confidence] {self.confidence:.2f}")
        lines.append("")
        
        # Goal
        lines.append("[Goal]")
        lines.append(f"  type: {self.goal.goal_type.value}")
        lines.append(f"  target: {self.goal.target}")
        lines.append(f"  algorithm_hint: {self.goal.algorithm_hint}")
        lines.append(f"  domain: {self.goal.domain}")
        lines.append(f"  confidence: {self.goal.confidence:.2f}")
        if self.goal.domain_evidence:
            lines.append(f"  domain_evidence: {self.goal.domain_evidence}")
        lines.append("")
        
        # Entities
        lines.append("[Entities]")
        for e in self.entities:
            lines.append(f"  - {e.name} ({e.kind}) type={e.type_name} attrs={e.attributes} is_input={e.is_input}")
        if not self.entities:
            lines.append("  (none)")
        lines.append("")
        
        # Inputs
        lines.append("[Inputs]")
        for i in self.inputs:
            lines.append(f"  - {i.name}: {i.type_signature} ({i.description})")
        if not self.inputs:
            lines.append("  (none)")
        lines.append("")
        
        # Outputs
        lines.append("[Outputs]")
        for o in self.outputs:
            lines.append(f"  - {o.name}: {o.type_signature} ({o.description})")
        if not self.outputs:
            lines.append("  (none)")
        lines.append("")
        
        # Operations
        lines.append("[Operations]")
        for op in self.operation_graph.operations:
            deps = " -> " + ", ".join(op.depends_on) if op.depends_on else ""
            lines.append(f"  - {op.name} ({op.op_type.value}) role={op.semantic_role}{deps}")
        if not self.operation_graph.operations:
            lines.append("  (none)")
        lines.append("")
        
        # Dependencies
        lines.append("[Dependencies]")
        for op in self.operation_graph.operations:
            if op.depends_on:
                lines.append(f"  {op.name} depends on: {', '.join(op.depends_on)}")
        if not any(op.depends_on for op in self.operation_graph.operations):
            lines.append("  (none)")
        lines.append("")
        
        # Constraints
        lines.append("[Constraints]")
        for c in self.constraints:
            lines.append(f"  - {c.name}: {c.value} ({c.constraint_type})")
        if not self.constraints:
            lines.append("  (none)")
        lines.append("")
        
        # User Requirements (extracted from semantic)
        lines.append("[User Requirements]")
        coverage = self.check_completeness()
        lines.append(f"  extracted: {coverage['coverage']['extracted']}")
        lines.append(f"  preserved: {coverage['coverage']['preserved']}")
        lines.append(f"  lost: {coverage['coverage']['lost']}")
        lines.append(f"  inferred: {coverage['coverage']['inferred']}")
        if coverage['issues']:
            lines.append(f"  ISSUES: {coverage['issues']}")
        if coverage['warnings']:
            lines.append(f"  WARNINGS: {coverage['warnings']}")
        lines.append("")
        
        # Retrieved Knowledge
        lines.append("[Retrieved Knowledge]")
        lines.append(f"  activated_concepts: {self.activated_concept_ids[:10]}")
        lines.append(f"  relation_paths: {len(self.relation_paths)}")
        lines.append("")
        
        # Requirement Coverage
        lines.append("[Requirement Coverage]")
        lines.append(f"  complete: {coverage['complete']}")
        lines.append(f"  issues: {len(coverage['issues'])}")
        lines.append(f"  warnings: {len(coverage['warnings'])}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ActivatedConcept:
    """A concept activated during semantic parsing."""
    concept_id: str
    name: str
    category: str
    activation_strength: float
    match_type: str  # lexical, semantic, contextual, structural
    matched_text: str = ""

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "category": self.category,
            "strength": self.activation_strength,
            "match_type": self.match_type,
            "matched_text": self.matched_text,
        }


@dataclass
class RelationPath:
    """A path in the knowledge graph connecting concepts."""
    path: list[str]  # concept ids
    relation_types: list[str]
    total_weight: float
    start_concept: str
    end_concept: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "relations": self.relation_types,
            "weight": self.total_weight,
            "start": self.start_concept,
            "end": self.end_concept,
        }


# ============================================================
# Response Representation Structures
# ============================================================

@dataclass
class Argument:
    """Argument in a predication."""
    role: str           # agent, theme, instrument, location, condition
    value: str
    entity_ref: str | None = None

    def to_dict(self) -> dict:
        return {"role": self.role, "value": self.value, "entity_ref": self.entity_ref}


@dataclass
class Modifier:
    """Modifier on a predication."""
    type: str           # tense, modality, negation, aspect
    value: str

    def to_dict(self) -> dict:
        return {"type": self.type, "value": self.value}


@dataclass
class Predication:
    """Core predicate-argument structure."""
    predicate: str                      # can_help, explain, generate, unknown
    arguments: list[Argument] = field(default_factory=list)
    modifiers: list[Modifier] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "predicate": self.predicate,
            "arguments": [a.to_dict() for a in self.arguments],
            "modifiers": [m.to_dict() for m in self.modifiers],
            "confidence": self.confidence,
        }


@dataclass
class Proposition:
    """A complete propositional content unit."""
    predication: Predication
    modifiers: list[Modifier] = field(default_factory=list)
    focus: str | None = None            # what's new/important

    def to_dict(self) -> dict:
        return {
            "predication": self.predication.to_dict(),
            "modifiers": [m.to_dict() for m in self.modifiers],
            "focus": self.focus,
        }


class RhetoricalRelation(Enum):
    ELABORATION = "elaboration"
    CONTRAST = "contrast"
    SEQUENCE = "sequence"
    CAUSE = "cause"
    CONDITION = "condition"
    PURPOSE = "purpose"
    CONCESSION = "concession"


@dataclass
class DiscourseNode:
    """Node in a discourse tree."""
    relation: RhetoricalRelation | None = None
    nucleus: 'DiscourseNode | None' = None
    satellites: list['DiscourseNode'] = field(default_factory=list)
    content: Proposition | None = None
    is_multinuclear: bool = False

    def to_dict(self) -> dict:
        return {
            "relation": self.relation.value if self.relation else None,
            "nucleus": self.nucleus.to_dict() if self.nucleus else None,
            "satellites": [s.to_dict() for s in self.satellites],
            "content": self.content.to_dict() if self.content else None,
            "is_multinuclear": self.is_multinuclear,
        }


@dataclass
class DiscourseTree:
    """Rhetorical structure of the response."""
    root: DiscourseNode | None = None

    def to_dict(self) -> dict:
        return {"root": self.root.to_dict() if self.root else None}


@dataclass
class FocusInfo:
    """Information about information focus."""
    new_info: list[str] = field(default_factory=list)
    given_info: list[str] = field(default_factory=list)
    contrast_set: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "new": self.new_info,
            "given": self.given_info,
            "contrast": self.contrast_set,
        }


@dataclass
class ToneSpec:
    """Tone and style specification."""
    tone_type: ToneType = ToneType.FRIENDLY
    formality: float = 0.5          # 0=casual, 1=formal
    technicality: float = 0.5       # 0=layman, 1=technical
    warmth: float = 0.5             # 0=cold, 1=warm

    def to_dict(self) -> dict:
        return {
            "tone": self.tone_type.value,
            "formality": self.formality,
            "technicality": self.technicality,
            "warmth": self.warmth,
        }


@dataclass
class AudienceModel:
    """Estimated user knowledge level."""
    programming_level: float = 0.5    # 0=beginner, 1=expert
    domain_familiarity: float = 0.5   # 0=unfamiliar, 1=expert
    prefers_examples: bool = True
    prefers_conciseness: bool = False

    def to_dict(self) -> dict:
        return {
            "programming_level": self.programming_level,
            "domain_familiarity": self.domain_familiarity,
            "prefers_examples": self.prefers_examples,
            "prefers_conciseness": self.prefers_conciseness,
        }


@dataclass
class EvidenceRef:
    """Reference to supporting evidence."""
    type: str           # concept, relation, memory, reasoning, rule
    source_id: str
    description: str
    strength: float = 1.0

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "source_id": self.source_id,
            "description": self.description,
            "strength": self.strength,
        }


@dataclass
class ResponseRepresentation:
    """Complete semantic response representation."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    
    # Communicative intent
    speech_act: SpeechAct = SpeechAct.INFORM
    propositional_content: Proposition | None = None
    
    # Structure
    discourse_structure: DiscourseTree = field(default_factory=DiscourseTree)
    focus: FocusInfo = field(default_factory=FocusInfo)
    
    # Style
    tone: ToneSpec = field(default_factory=ToneSpec)
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    audience: AudienceModel = field(default_factory=AudienceModel)
    
    # Grounding
    evidence: list[EvidenceRef] = field(default_factory=list)
    certainty: float = 0.8
    
    # Language
    target_language: Language = Language.ENGLISH
    source_semantic: SemanticRepresentation | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "speech_act": self.speech_act.value,
            "content": self.propositional_content.to_dict() if self.propositional_content else None,
            "discourse": self.discourse_structure.to_dict(),
            "focus": self.focus.to_dict(),
            "tone": self.tone.to_dict(),
            "verbosity": self.verbosity.value,
            "audience": self.audience.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "certainty": self.certainty,
            "language": self.target_language.value,
        }


# ============================================================
# Problem Model (Extended for Code Generation)
# ============================================================

@dataclass
class DataModelSpec:
    """Data model specification for code generation."""
    name: str
    fields: list[dict] = field(default_factory=list)  # name, type, description
    methods: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)


@dataclass
class APISpec:
    """API endpoint specification."""
    name: str
    method: str = "GET"
    path: str = ""
    params: list[dict] = field(default_factory=list)
    response_type: str = ""
    description: str = ""


@dataclass
class AlgorithmSpec:
    """Algorithm specification."""
    name: str
    type: str = ""           # search, sort, graph, dp, math
    complexity: str = ""     # O(n log n), O(n), etc.
    description: str = ""
    pseudocode: str = ""


@dataclass
class UISpec:
    """UI component specification."""
    name: str
    type: str = ""           # screen, form, list, chart
    fields: list[dict] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class VerificationCriterion:
    """Verification criterion for generated code."""
    criterion_type: str       # has_operation, sort_order, has_model, has_entry_point
    params: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class TestCaseSpec:
    """Test case specification."""
    name: str
    inputs: dict = field(default_factory=dict)
    expected_output: str = ""
    description: str = ""


@dataclass
class ProblemModel:
    """Complete problem model for code generation."""
    # Semantic core
    semantic: SemanticRepresentation
    
    # Target
    target_language: Language | None = None
    target_platform: str | None = None
    architecture_hints: list[str] = field(default_factory=list)
    
    # Derived specifications
    data_models: list[DataModelSpec] = field(default_factory=list)
    apis: list[APISpec] = field(default_factory=list)
    algorithms: list[AlgorithmSpec] = field(default_factory=list)
    ui_components: list[UISpec] = field(default_factory=list)
    
    # Verification
    success_criteria: list[VerificationCriterion] = field(default_factory=list)
    test_cases: list[TestCaseSpec] = field(default_factory=list)
    
    # Learning context
    similar_problems: list[str] = field(default_factory=list)  # problem ids
    applicable_patterns: list[str] = field(default_factory=list)  # pattern ids

    def to_dict(self) -> dict:
        return {
            "semantic": self.semantic.to_dict(),
            "target_language": self.target_language.value if self.target_language else None,
            "target_platform": self.target_platform,
            "architecture_hints": self.architecture_hints,
            "data_models": [{"name": d.name, "fields": d.fields} for d in self.data_models],
            "apis": [{"name": a.name, "method": a.method, "path": a.path} for a in self.apis],
            "algorithms": [{"name": a.name, "type": a.type} for a in self.algorithms],
            "ui_components": [{"name": u.name, "type": u.type} for u in self.ui_components],
            "success_criteria": [{"type": c.criterion_type, "params": c.params} for c in self.success_criteria],
            "test_cases": [{"name": t.name, "inputs": t.inputs} for t in self.test_cases],
            "similar_problems": self.similar_problems,
            "applicable_patterns": self.applicable_patterns,
        }