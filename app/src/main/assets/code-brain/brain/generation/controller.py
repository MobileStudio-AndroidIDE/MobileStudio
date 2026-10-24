"""Generation Controller - central orchestrator for adaptive generation.

Responsibilities:
1. Select generation strategy and granularity
2. Generate hypotheses
3. Select or combine hypotheses
4. Produce semantic response/code representation
5. Emit real-time events
6. Invoke verification
7. Invoke repair if needed
8. Emit final completion state

The controller adapts to the semantic structure of each request.
It does NOT follow a single fixed generation sequence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from brain.generation.events import (
    GenerationEvent, EventType, Granularity, GenerationType,
    Hypothesis, GenerationResult,
)
from brain.generation.hypotheses import HypothesisGenerator, HypothesisConfig
from brain.generation.strategy import GranularitySelector
from brain.generation.streaming import StreamingEmitter


@dataclass
class ControllerConfig:
    stream_format: str = "text"
    stream_mode: str = "auto"
    max_hypotheses: int = 3
    enable_verification: bool = True
    enable_repair: bool = True
    silent: bool = False


class GenerationController:
    """Central orchestrator for adaptive generation.

    Adapts its behavior based on the semantic structure of each request.
    No fixed generation sequence — the pipeline is driven by the task.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.hypothesis_gen = HypothesisGenerator(
            HypothesisConfig(max_hypotheses=self.config.max_hypotheses)
        )
        self.strategy_selector = GranularitySelector()

    def generate(
        self,
        request: str,
        semantic: Any = None,
        context: Optional[dict] = None,
        emitter: Optional[StreamingEmitter] = None,
    ) -> GenerationResult:
        """Run the full generation pipeline with real-time events.

        This is the main entry point. It orchestrates:
        analysis → hypotheses → selection → planning → generation → verification
        → error analysis → repair plan → regeneration → re-verification,
        rotating through structurally unique hypotheses until one passes
        or the budget is exhausted.
        """
        if emitter is None:
            emitter = StreamingEmitter(silent=self.config.silent,
                                       format=self.config.stream_format)

        events: list[GenerationEvent] = []

        try:
            # Phase 1: Analysis
            self._emit_phase(emitter, events, EventType.ANALYZING,
                           "Analyzing request structure")

            # Phase 2: Understanding
            ops, entities, goal_type, composition = self._extract_semantics(semantic)
            self._resolved_ops = self._resolve_ops(ops, composition)
            self._emit_phase(emitter, events, EventType.UNDERSTANDING,
                           f"Goal: {goal_type}, Operations: {self._resolved_ops}")

            # Phase 3: Knowledge retrieval (if available)
            self._emit_phase(emitter, events, EventType.KNOWLEDGE,
                           "Knowledge retrieval")

            # Phase 4: Hypothesis generation (non-deterministic)
            hypotheses, hyp_events = self.hypothesis_gen.generate(
                self._resolved_ops, entities, goal_type, composition, context
            )
            emitter.emit_batch(hyp_events)
            events.extend(hyp_events)

            # Phase 5: Strategy selection
            granularity_pref = self.config.stream_mode
            strategy = self.strategy_selector.select(
                goal_type, self._resolved_ops, entities, composition, granularity_pref
            )
            emitter.emit_batch(strategy.events)
            events.extend(strategy.events)

            # Phase 6+: hypothesis rotation loop.
            # For each ranked valid hypothesis: generate → verify →
            # on failure: analyze → plan → repair → reverify → next hypothesis
            func_name = self._infer_function_name(None) if not hypotheses else None
            # The think budget caps how many candidates this request may
            # rotate through before it reports failure honestly.
            from brain.think_budget import current_budget
            budget = current_budget()
            hypotheses = hypotheses[:budget.max_hypotheses]
            tried = 0
            verified = False
            output = None
            repair_success = False
            selected = None

            for hypothesis in hypotheses:
                if func_name is None:
                    func_name = self._infer_function_name(hypothesis)
                selected = hypothesis
                tried += 1

                sel_event = GenerationEvent(
                    event_type=EventType.HYPOTHESIS_SELECTED,
                    hypothesis_id=hypothesis.id,
                    content=f"Trying hypothesis #{hypothesis.id}: {hypothesis.approach}",
                )
                emitter.emit(sel_event)
                events.append(sel_event)

                # Planning + IR events
                self._emit_phase(emitter, events, EventType.PLANNING,
                               f"Planning {strategy.granularity.value} generation")
                self._emit_phase(emitter, events, EventType.IR_GENERATED,
                               "Intermediate representation created")

                # Generation
                try:
                    output = self._emit_generation(
                        emitter, events, hypothesis, strategy, context
                    )
                except Exception as gen_err:
                    output = None
                    fail_event = GenerationEvent(
                        event_type=EventType.VERIFICATION_FAILED,
                        content=f"Generation error: {gen_err}",
                    )
                    emitter.emit(fail_event)
                    events.append(fail_event)
                    continue

                # Verification
                if self.config.enable_verification:
                    verified, errors = self._emit_verification(
                        emitter, events, output, strategy
                    )
                else:
                    verified, errors = True, []

                if verified:
                    repair_success = False
                    break

                # Repair loop for this hypothesis
                if self.config.enable_repair:
                    output, repair_ok, repair_events_ok = self._repair_cycle(
                        emitter, events, output, hypothesis
                    )
                    events.extend(repair_events_ok)
                    if repair_ok:
                        verified = True
                        repair_success = True
                        break

            # Completion
            final_event = GenerationEvent(
                event_type=EventType.GENERATION_COMPLETED if verified else EventType.GENERATION_FAILED,
                content="Generation complete" if verified else
                        f"All {tried} hypotheses failed verification",
                is_final=True,
                metadata={
                    "hypotheses": len(hypotheses),
                    "selected": selected.id if selected else None,
                    "granularity": strategy.granularity.value,
                    "verified": verified,
                },
            )
            emitter.emit(final_event)
            events.append(final_event)

            return GenerationResult(
                success=verified,
                output=output,
                events=events,
                hypotheses_generated=len(hypotheses),
                hypothesis_selected=selected.id if selected else None,
                granularity_used=strategy.granularity,
                generation_type=strategy.generation_type,
                verification_passed=verified,
                repair_needed=not verified,
                repair_success=repair_success,
            )

        except Exception as e:
            error_event = GenerationEvent(
                event_type=EventType.GENERATION_FAILED,
                content=str(e),
                is_final=True,
            )
            emitter.emit(error_event)
            events.append(error_event)
            return GenerationResult(
                success=False,
                error=str(e),
                events=events,
            )

    def _extract_semantics(self, semantic: Any) -> tuple:
        """Extract operations, entities, goal_type, composition from semantic."""
        if semantic is None:
            return [], [], "unknown", {}

        ops = []
        entities = []
        goal_type = "unknown"
        composition = {}

        # Handle dict input (for testing and direct invocation)
        if isinstance(semantic, dict):
            ops = semantic.get("operations", [])
            goal_type = semantic.get("goal_type", "unknown")
            entities = semantic.get("entities", [])
            composition = semantic.get("composition", {})
            return ops, entities, goal_type, composition

        # Extract from SemanticRepresentation (parser)
        if hasattr(semantic, 'goal') and hasattr(semantic.goal, 'goal_type'):
            goal_type = semantic.goal.goal_type.value
        if hasattr(semantic, 'operation_graph'):
            ops = [op.name for op in semantic.operation_graph.operations]
        if hasattr(semantic, 'entities'):
            entities = [{"type": e.kind, "name": e.name} for e in semantic.entities]
        if hasattr(semantic, 'composition'):
            composition = semantic.composition or {}

        return ops, entities, goal_type, composition

    def _emit_phase(
        self,
        emitter: StreamingEmitter,
        events: list,
        event_type: EventType,
        content: str,
    ) -> None:
        """Emit a pipeline phase event."""
        event = GenerationEvent(event_type=event_type, content=content)
        emitter.emit(event)
        events.append(event)

    def _emit_generation(
        self,
        emitter: StreamingEmitter,
        events: list,
        hypothesis: Optional[Hypothesis],
        strategy: Any,
        context: Optional[dict],
    ) -> Any:
        """Emit generation events based on selected granularity and hypothesis."""
        gran = strategy.granularity
        gen_type = strategy.generation_type

        if gen_type == GenerationType.CONVERSATIONAL:
            return self._emit_conversational(emitter, events, hypothesis)
        elif gen_type == GenerationType.EXPLANATION:
            return self._emit_explanation(emitter, events, hypothesis)
        elif gen_type == GenerationType.PROJECT:
            return self._emit_project(emitter, events, hypothesis, context)
        else:
            return self._emit_code(emitter, events, hypothesis, gran)

    def _emit_code(
        self,
        emitter: StreamingEmitter,
        events: list,
        hypothesis: Optional[Hypothesis],
        granularity: Granularity,
    ) -> dict:
        """Emit code generation events at the appropriate granularity."""
        ops = hypothesis.operations if hypothesis else []
        lines = self._generate_code_lines(ops, hypothesis)

        if granularity == Granularity.TOKEN:
            return self._emit_token_stream(emitter, events, lines)
        elif granularity == Granularity.LINE:
            return self._emit_line_stream(emitter, events, lines)
        elif granularity == Granularity.FUNCTION:
            return self._emit_function_stream(emitter, events, lines, hypothesis)
        elif granularity == Granularity.CLASS:
            return self._emit_class_stream(emitter, events, lines, hypothesis)
        else:
            return self._emit_line_stream(emitter, events, lines)

    def _emit_token_stream(
        self, emitter: StreamingEmitter, events: list, lines: list[str]
    ) -> dict:
        """Stream token-by-token."""
        for i, line in enumerate(lines):
            for char in line:
                event = GenerationEvent(
                    event_type=EventType.TOKEN_GENERATED,
                    content=char,
                    index=i, total=len(lines),
                )
                emitter.emit(event)
                events.append(event)
        return {"lines": lines, "stream_mode": "token"}

    def _emit_line_stream(
        self, emitter: StreamingEmitter, events: list, lines: list[str]
    ) -> dict:
        """Stream line-by-line."""
        for i, line in enumerate(lines):
            event = GenerationEvent(
                event_type=EventType.LINE_GENERATED,
                content=line,
                index=i, total=len(lines),
            )
            emitter.emit(event)
            events.append(event)
        return {"lines": lines, "stream_mode": "line"}

    def _emit_function_stream(
        self, emitter: StreamingEmitter, events: list,
        lines: list[str], hypothesis: Optional[Hypothesis]
    ) -> dict:
        """Stream function-by-function."""
        func_name = self._infer_function_name(hypothesis)

        start = GenerationEvent(
            event_type=EventType.FUNCTION_STARTED,
            function_name=func_name,
        )
        emitter.emit(start)
        events.append(start)

        result = self._emit_line_stream(emitter, events, lines)

        end = GenerationEvent(
            event_type=EventType.FUNCTION_COMPLETED,
            function_name=func_name,
        )
        emitter.emit(end)
        events.append(end)

        result["stream_mode"] = "function"
        result["function_name"] = func_name
        return result

    def _emit_class_stream(
        self, emitter: StreamingEmitter, events: list,
        lines: list[str], hypothesis: Optional[Hypothesis]
    ) -> dict:
        """Stream class-by-class."""
        class_name = self._infer_class_name(hypothesis)

        start = GenerationEvent(
            event_type=EventType.CLASS_STARTED,
            class_name=class_name,
        )
        emitter.emit(start)
        events.append(start)

        result = self._emit_function_stream(emitter, events, lines, hypothesis)

        end = GenerationEvent(
            event_type=EventType.CLASS_COMPLETED,
            class_name=class_name,
        )
        emitter.emit(end)
        events.append(end)

        result["stream_mode"] = "class"
        result["class_name"] = class_name
        return result

    def _emit_project(
        self, emitter: StreamingEmitter, events: list,
        hypothesis: Optional[Hypothesis], context: Optional[dict]
    ) -> dict:
        """Emit project-level generation events."""
        files = self._infer_project_files(hypothesis, context)

        start = GenerationEvent(event_type=EventType.PROJECT_STARTED)
        emitter.emit(start)
        events.append(start)

        results = []
        for file_path, file_content in files:
            file_start = GenerationEvent(
                event_type=EventType.FILE_STARTED,
                file_path=file_path,
            )
            emitter.emit(file_start)
            events.append(file_start)

            # Generate content for this file
            if isinstance(file_content, dict) and "classes" in file_content:
                for cls in file_content["classes"]:
                    cls_start = GenerationEvent(
                        event_type=EventType.CLASS_STARTED,
                        class_name=cls["name"],
                        file_path=file_path,
                    )
                    emitter.emit(cls_start)
                    events.append(cls_start)

                    for func in cls.get("functions", []):
                        func_start = GenerationEvent(
                            event_type=EventType.FUNCTION_STARTED,
                            function_name=func["name"],
                            file_path=file_path,
                            class_name=cls["name"],
                        )
                        emitter.emit(func_start)
                        events.append(func_start)

                        for line in func.get("lines", []):
                            line_event = GenerationEvent(
                                event_type=EventType.LINE_GENERATED,
                                content=line,
                                file_path=file_path,
                                class_name=cls["name"],
                                function_name=func["name"],
                            )
                            emitter.emit(line_event)
                            events.append(line_event)

                        func_end = GenerationEvent(
                            event_type=EventType.FUNCTION_COMPLETED,
                            function_name=func["name"],
                            file_path=file_path,
                        )
                        emitter.emit(func_end)
                        events.append(func_end)

                    cls_end = GenerationEvent(
                        event_type=EventType.CLASS_COMPLETED,
                        class_name=cls["name"],
                        file_path=file_path,
                    )
                    emitter.emit(cls_end)
                    events.append(cls_end)

            file_end = GenerationEvent(
                event_type=EventType.FILE_COMPLETED,
                file_path=file_path,
            )
            emitter.emit(file_end)
            events.append(file_end)
            results.append({"file": file_path, "content": file_content})

        end = GenerationEvent(event_type=EventType.PROJECT_COMPLETED)
        emitter.emit(end)
        events.append(end)

        return {"files": results, "stream_mode": "project"}

    def _emit_conversational(
        self, emitter: StreamingEmitter, events: list,
        hypothesis: Optional[Hypothesis]
    ) -> dict:
        """Emit conversational response as sentences."""
        sentences = self._generate_conversational(hypothesis)
        for i, sentence in enumerate(sentences):
            event = GenerationEvent(
                event_type=EventType.SENTENCE_GENERATED,
                content=sentence,
                index=i, total=len(sentences),
            )
            emitter.emit(event)
            events.append(event)
        return {"sentences": sentences, "stream_mode": "sentence"}

    def _emit_explanation(
        self, emitter: StreamingEmitter, events: list,
        hypothesis: Optional[Hypothesis]
    ) -> dict:
        """Emit explanation as paragraphs."""
        paragraphs = self._generate_explanation(hypothesis)
        for i, para in enumerate(paragraphs):
            event = GenerationEvent(
                event_type=EventType.PARAGRAPH_GENERATED,
                content=para,
                index=i, total=len(paragraphs),
            )
            emitter.emit(event)
            events.append(event)
        return {"paragraphs": paragraphs, "stream_mode": "paragraph"}

    def _emit_verification(
        self,
        emitter: StreamingEmitter,
        events: list,
        output: Any,
        strategy: Any,
    ) -> tuple[bool, list[dict]]:
        """Emit verification events; return (passed, errors)."""
        verify_start = GenerationEvent(
            event_type=EventType.VERIFYING,
            content="Verifying generated output",
        )
        emitter.emit(verify_start)
        events.append(verify_start)

        passed, errors = self._run_verification(output, strategy)

        meta = {"passed": passed}
        if errors:
            meta["error_types"] = [e.get("type") for e in errors]
            meta["error_count"] = len(errors)
        verify_event = GenerationEvent(
            event_type=EventType.VERIFIED if passed else EventType.VERIFICATION_FAILED,
            content="Verification passed" if passed else
                    f"Verification failed: {len(errors)} error(s) - "
                    + "; ".join(e.get("cause", e.get("type", ""))
                                for e in errors[:3]),
            metadata=meta,
        )
        emitter.emit(verify_event)
        events.append(verify_event)

        return passed, errors

    def _repair_cycle(
        self,
        emitter: StreamingEmitter,
        events: list,
        output: Any,
        hypothesis: Optional[Hypothesis],
    ) -> tuple[Any, bool, list]:
        """Run structural error analysis → repair plan → regeneration → reverify.

        Returns (repaired_output_or_original, success, emitted_events).
        """
        from brain.generation.repair import RepairEngine

        code = self._extract_code_from_output(output)
        required_ops = self._resolved_ops or (hypothesis.operations if hypothesis else [])
        func_name = self._infer_function_name(hypothesis)
        current_variants = {}
        if hypothesis and hypothesis.structure:
            current_variants = hypothesis.structure.get("strategies", {})

        engine = RepairEngine()
        emitted: list[GenerationEvent] = []

        # The think budget: how many repair cycles this request may
        # spend on ONE candidate before rotating to the next hypothesis.
        from brain.think_budget import current_budget
        max_attempts = current_budget().max_repair_attempts

        # 1. REPAIR_STARTED
        e = GenerationEvent(event_type=EventType.REPAIR_STARTED,
                            content="Attempting structural repair")
        emitter.emit(e); emitted.append(e)

        # 2. REPAIR_ANALYZING
        errors = engine.analyze(code, required_ops)
        e = GenerationEvent(
            event_type=EventType.REPAIR_ANALYZING,
            content=f"Found {len(errors)} verification error(s)",
            metadata={"errors": [{"type": err.get("type"),
                                   "operation": err.get("operation"),
                                   "cause": err.get("cause")} for err in errors]},
        )
        emitter.emit(e); emitted.append(e)

        if not errors:
            e = GenerationEvent(event_type=EventType.REPAIR_SUCCEEDED,
                                content="No errors found after analysis")
            emitter.emit(e); emitted.append(e)
            return output, True, emitted

        # 3-6. The repair ATTEMPTS: plan → regenerate → reverify, as
        # many cycles as the budget paid for. A cycle that cannot plan
        # or regenerate ends the attempts for this candidate honestly.
        attempt = 0
        current_code = code
        while attempt < max_attempts:
            attempt += 1
            plan = engine.plan(errors, required_ops, current_variants)
            if plan is None:
                e = GenerationEvent(event_type=EventType.REPAIR_FAILED,
                                    content="No applicable repair plan")
                emitter.emit(e); emitted.append(e)
                return output, False, emitted

            e = GenerationEvent(
                event_type=EventType.REPAIR_PLAN,
                content=f"Plan: {plan.describe()}",
                metadata={"action": plan.action,
                          "ops": plan.ops,
                          "switch_ops": plan.switch_ops,
                          "notes": plan.notes},
            )
            emitter.emit(e); emitted.append(e)

            new_code = engine.apply(plan, func_name)
            if not new_code:
                e = GenerationEvent(event_type=EventType.REPAIR_FAILED,
                                    content="Repair regeneration failed")
                emitter.emit(e); emitted.append(e)
                return output, False, emitted

            new_output = {"lines": new_code.split("\n"), "stream_mode": "line",
                          "function_name": func_name}
            e = GenerationEvent(event_type=EventType.REPAIR_APPLIED,
                                content="Repair applied: regenerated pipeline from IR",
                                metadata={"lines": len(new_code.split("\n")),
                                          "attempt": attempt})
            emitter.emit(e); emitted.append(e)

            for i, line in enumerate(new_code.split("\n")):
                le = GenerationEvent(event_type=EventType.LINE_GENERATED,
                                     content=line, index=i,
                                     total=len(new_code.split("\n")))
                emitter.emit(le); emitted.append(le)

            e = GenerationEvent(event_type=EventType.REVERIFYING,
                                content="Running verification on repaired output")
            emitter.emit(e); emitted.append(e)

            errors = engine.verify(new_code, required_ops)
            if not errors:
                e = GenerationEvent(event_type=EventType.REPAIR_SUCCEEDED,
                                    content="Repaired output passes verification")
                emitter.emit(e); emitted.append(e)
                return new_output, True, emitted
            current_code = new_code

        e = GenerationEvent(
            event_type=EventType.REPAIR_FAILED,
            content=f"Repair budget exhausted after {attempt} attempt(s): "
                    + "; ".join(er.get("cause", er.get("type", ""))
                                for er in errors[:3]),
            metadata={"errors": [{"type": er.get("type"),
                                   "operation": er.get("operation"),
                                   "cause": er.get("cause")} for er in errors]},
        )
        emitter.emit(e); emitted.append(e)
        return output, False, emitted

    # ------------------------------------------------------------------ #
    # Code generation helpers (produce actual content)
    # ------------------------------------------------------------------ #

    def _resolve_ops(self, ops: list[str], composition: dict) -> list[str]:
        """Resolve required operations from semantic graph + composition.

        The operation graph preserves user-stated ordering; composition
        graph provides fine-grained primitive decomposition when the
        operation graph is empty.
        """
        from brain.generation.op_semantics import normalize_ops

        comp_ops = []
        try:
            comp_ops = [o["name"] for o in composition.get("graph", {}).get("operations", [])]
        except AttributeError:
            comp_ops = []

        if ops:
            resolved = normalize_ops(list(ops), comp_ops)
        elif comp_ops:
            resolved = normalize_ops([], comp_ops) or normalize_ops(comp_ops)
        else:
            resolved = []

        # Compose fine-grained composition ops into pipeline form too
        if not resolved and comp_ops:
            resolved = normalize_ops(comp_ops)
        if not resolved and comp_ops:
            # last resort: use normalized composition directly
            resolved = self._normalize_from_composition(comp_ops)
        return resolved

    @staticmethod
    def _normalize_from_composition(comp_ops: list[str]) -> list[str]:
        from brain.generation.op_semantics import normalize_ops
        return normalize_ops(comp_ops)

    def _build_code(self, hypothesis: Optional[Hypothesis]) -> list[str]:
        """Build code from the semantic IR (op graph). Raises on unknown ops."""
        from brain.generation.op_semantics import build_pipeline

        ops = self._resolved_ops or (hypothesis.operations if hypothesis else [])
        func_name = self._infer_function_name(hypothesis)
        variants = {}
        if hypothesis and hypothesis.structure:
            variants = hypothesis.structure.get("strategies", {})
        return build_pipeline(ops, func_name, variants=variants)

    def _generate_code_lines(
        self, ops: list[str], hypothesis: Optional[Hypothesis]
    ) -> list[str]:
        """Generate real, dataflow-chained code lines from the op graph."""
        return self._build_code(hypothesis)

    def _generate_conversational(
        self, hypothesis: Optional[Hypothesis]
    ) -> list[str]:
        """Generate conversational response sentences."""
        return [
            "Hello! I'm RENA-BR, your AI coding assistant.",
            "I can help you generate code, debug errors, and explain concepts.",
            "What would you like to work on today?",
        ]

    def _generate_explanation(
        self, hypothesis: Optional[Hypothesis]
    ) -> list[str]:
        """Generate explanation paragraphs."""
        return [
            "This is a code analysis request.",
            "The system will examine the code structure and provide insights.",
        ]

    def _infer_project_files(
        self, hypothesis: Optional[Hypothesis], context: Optional[dict]
    ) -> list[tuple[str, Any]]:
        """Infer project file structure."""
        return [
            ("Main.kt", {
                "classes": [{
                    "name": "MainActivity",
                    "functions": [{
                        "name": "onCreate",
                        "lines": [
                            "class MainActivity : AppCompatActivity() {",
                            "    override fun onCreate(savedInstanceState: Bundle?) {",
                            "        super.onCreate(savedInstanceState)",
                            "        setContentView(R.layout.activity_main)",
                            "    }",
                            "}",
                        ],
                    }],
                }],
            }),
        ]

    def _infer_function_name(self, hypothesis: Optional[Hypothesis]) -> str:
        """Derive a deterministic function name from the hypothesis / IR.

        Single-operation pipelines use the operation name; multi-operation
        pipelines use 'process_data'.  If the hypothesis itself carries a
        function_name in its structure, that wins.
        """
        if hypothesis is not None:
            fn = hypothesis.structure.get("function_name") if hypothesis.structure else None
            if fn:
                return fn
        ops = self._resolved_ops or []
        if len(ops) == 1:
            return ops[0]
        if ops:
            return "process_data"
        return "process_data"

    def _infer_class_name(self, hypothesis: Optional[Hypothesis]) -> str:
        return "Solution"

    def _extract_code_from_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, dict):
            lines = output.get("lines", [])
            if lines:
                return "\n".join(lines)
            code = output.get("code", "")
            if code:
                return code
        if isinstance(output, str):
            return output
        return ""

    def _run_verification(self, output: Any, strategy: Any) -> tuple[bool, list[dict]]:
        """Return (passed, structured_errors) using ErrorAnalyzer."""
        from brain.generation.repair import ErrorAnalyzer

        code = self._extract_code_from_output(output)
        required_ops = self._resolved_ops or []

        if not code.strip():
            return False, [{"type": "EMPTY_OUTPUT",
                            "cause": "No code was generated"}]

        analyzer = ErrorAnalyzer()
        errors = analyzer.analyze(code, required_ops)
        return (len(errors) == 0, errors)

    def _run_repair(
        self, output: Any, strategy: Any, context: Optional[dict]
    ) -> Optional[Any]:
        return None
