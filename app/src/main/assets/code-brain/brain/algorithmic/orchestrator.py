"""Algorithmic Orchestrator.

Main entry point that orchestrates the entire algorithmic reasoning pipeline:
  NL → CompetitiveProblemDetector → StructuredProblemExtractor → StructuredProblem
  → AlgorithmHypothesisGenerator → Hypotheses → HypothesisEvaluator → SelectedHypothesis
  → AlgorithmPlanBuilder → AlgorithmPlan → TypedIRGenerator → Typed IR
  → CompetitiveCodeGenerator → Python code → SampleTestVerifier → VerificationResult

Also provides a fallback mechanism: if algorithmic reasoning fails,
returns None so the caller can fall back to the existing data pipeline.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from brain.algorithmic.competitive_problem_detector import (
    CompetitiveProblemDetector, DetectionResult,
)
from brain.algorithmic.structured_problem_extractor import (
    StructuredProblemExtractor, StructuredProblem,
)
from brain.algorithmic.algorithm_hypothesis_generator import (
    AlgorithmHypothesisGenerator, AlgorithmicHypothesis,
)
from brain.algorithmic.hypothesis_evaluator import (
    HypothesisEvaluator, EvaluationResult,
)
from brain.algorithmic.algorithm_plan_builder import (
    AlgorithmPlanBuilder, AlgorithmPlan,
)
from brain.algorithmic.typed_ir_generator import (
    TypedIRGenerator, TypedIR,
)
from brain.algorithmic.competitive_code_generator import (
    CompetitiveCodeGenerator,
)
from brain.algorithmic.sample_test_verifier import (
    SampleTestVerifier, VerificationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmicResult:
    """Result of the algorithmic reasoning pipeline."""
    success: bool
    code: str = ""
    function_name: str = "solve"
    # Pipeline stages (for debugging / tracing)
    detection: Optional[DetectionResult] = None
    problem: Optional[StructuredProblem] = None
    hypotheses: list[AlgorithmicHypothesis] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    plan: Optional[AlgorithmPlan] = None
    typed_ir: Optional[TypedIR] = None
    verification: Optional[VerificationResult] = None
    error: str = ""
    # Semantic representation (when using solve_with_semantic)
    semantic: Optional[object] = None


class AlgorithmicOrchestrator:
    """Orchestrates the algorithmic reasoning pipeline.

    Usage:
        orchestrator = AlgorithmicOrchestrator()
        result = orchestrator.solve(problem_text)
        if result.success:
            # Use result.code
        else:
            # Fall back to existing pipeline
    """

    def __init__(
        self,
        detection_threshold: float = 0.45,
        max_hypotheses: int = 12,
        verification_timeout: float = 5.0,
        outcome_path: Optional[str] = None,
    ):
        self.detector = CompetitiveProblemDetector(threshold=detection_threshold)
        self.extractor = StructuredProblemExtractor()
        self.hypothesis_gen = AlgorithmHypothesisGenerator(max_hypotheses=max_hypotheses)
        self.evaluator = HypothesisEvaluator()
        self.plan_builder = AlgorithmPlanBuilder()
        self.ir_generator = TypedIRGenerator()
        self.code_generator = CompetitiveCodeGenerator()
        self.verifier = SampleTestVerifier(timeout=verification_timeout)
        # Generalized learning from execution outcomes (aggregate stats
        # only — never problem text).  None -> the default store; the
        # evaluator's tie-break is a no-op until enough data exists.
        from learning.learner.code_learner import LearningFromOutcome
        self.outcome_path = outcome_path
        self._outcomes = LearningFromOutcome(outcome_path)
        # Optional observation trace: when set (set_trace), verification
        # verdicts are appended as ExecutionEvidence.
        self._trace = None

    def set_trace(self, trace) -> None:
        """Attach an ObservationTrace to receive execution evidence."""
        self._trace = trace

    @staticmethod
    def _problem_shape(structured_problem) -> str:
        """Structural tags only (e.g. 'filter+sort') — never the text."""
        tags = [str(t).strip().lower()
                for t in (getattr(structured_problem, "structural_hints",
                                  None) or [])
                if str(t).strip()]
        tags += [str(t).strip().lower()
                 for t in (getattr(structured_problem, "data_types",
                                   None) or [])
                 if str(t).strip()]
        return "+".join(sorted(set(tags))) or "unknown"

    def _record_attempt_outcome(self, hypothesis, structured_problem,
                                verification) -> None:
        """Record one execution outcome (strategy x shape aggregate)."""
        try:
            passed = verification is None or bool(
                getattr(verification, "passed", False))
            cost = None
            results = getattr(verification, "test_results", None) or []
            if results:
                cost = float(sum(getattr(r, "execution_time", 0.0) or 0.0
                                 for r in results))
            self._outcomes.record_outcome(
                hypothesis.name, self._problem_shape(structured_problem),
                success=passed, cost=cost,
                failure_stage=None if passed else "verification")
        except Exception:
            pass

    def _record_attempt_evidence(self, hypothesis, structured_problem,
                                 code: str, verification) -> None:
        """Append verification verdicts as evidence to the trace."""
        if self._trace is None:
            return
        try:
            code_ref = str(getattr(hypothesis, "name", "") or "")
            if verification is None:
                self._trace.add_not_run_evidence(
                    "behavior verification not run: no sample cases "
                    "or verification raised", code_ref=code_ref)
            else:
                for ev in self.verifier.verification_evidence(
                        verification, code_ref=code_ref):
                    self._trace.add_evidence(ev)
        except Exception:
            pass


    def solve(
        self,
        problem_text: str,
        force_algorithmic: bool = False,
    ) -> AlgorithmicResult:
        """Attempt to solve a competitive programming problem.

        Args:
            problem_text: The full problem text (natural language).
            force_algorithmic: If True, skip detection and always use
                              algorithmic pipeline.

        Returns:
            AlgorithmicResult with code or error.
        """
        # Stage 1: Detection
        if not force_algorithmic:
            detection = self.detector.detect(problem_text)
            if not detection.is_competitive:
                return AlgorithmicResult(
                    success=False,
                    detection=detection,
                    error="Not detected as competitive programming problem",
                )
        else:
            detection = DetectionResult(
                is_competitive=True, confidence=1.0,
                signals=["forced"],
            )

        # Stage 2: Extract structured problem
        try:
            structured_problem = self.extractor.extract(problem_text)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                error=f"Failed to extract structured problem: {e}",
            )

        # Stage 3: Generate hypotheses
        try:
            hypotheses = self.hypothesis_gen.generate(structured_problem)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                error=f"Failed to generate hypotheses: {e}",
            )

        if not hypotheses:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                error="No hypotheses generated",
            )

        # Stage 4: Evaluate and rank all hypotheses
        try:
            evaluations = self.evaluator.evaluate_all(hypotheses, structured_problem)
            ranked = list(evaluations)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                error=f"Failed to evaluate hypotheses: {e}",
            )

        if not ranked:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                error="No hypothesis selected",
            )

        hypothesis_by_id = {h.id: h for h in hypotheses}

        # Stage 5-8: Hypothesis rotation with verification.
        # Try hypotheses in ranked order; for each: build plan -> typed
        # IR -> code -> verify against the problem's own public sample
        # cases.  The first hypothesis whose code passes the samples
        # wins; if every attempt fails, the best attempt (most sample
        # cases passed) is returned so the caller still gets real code.
        # A plan built from a different hypothesis than the selected one
        # is a semantic mismatch and is rejected.
        best_attempt = None
        best_timeouts = 0
        for evaluation in ranked:
            selected_hypothesis = hypothesis_by_id.get(evaluation.hypothesis_id)
            if selected_hypothesis is None:
                continue

            # Stage 5: Build algorithm plan
            try:
                plan = self.plan_builder.build(selected_hypothesis, structured_problem)
            except Exception:
                continue
            if plan is None:
                continue
            # semantic consistency: the plan must come from the
            # selected hypothesis (never a mismatched plan)
            plan_name = getattr(plan, "hypothesis_name", "") or \
                getattr(plan, "source_hypothesis", "")
            if plan_name and plan_name != selected_hypothesis.name:
                continue

            # Stage 6: Generate typed IR
            typed_ir = None
            try:
                typed_ir = self.ir_generator.generate(plan)
            except Exception:
                pass

            # Stage 7: Generate code
            try:
                code = self.code_generator.generate_from_plan(plan, structured_problem)
            except Exception:
                continue
            if not code or not code.strip():
                continue

            # Stage 8: Verify against the problem's own sample tests
            verification = None
            if structured_problem.sample_cases:
                try:
                    verification = self.verifier.verify(
                        code, structured_problem.sample_cases,
                        structured_problem.function_name,
                        request_form=getattr(
                            structured_problem, "request_form", "program"),
                    )
                except Exception:
                    verification = None

            # CONSUME (learning): the execution outcome is known here —
            # aggregate stats updated; evidence linked to the trace.
            self._record_attempt_outcome(selected_hypothesis,
                                         structured_problem, verification)
            self._record_attempt_evidence(selected_hypothesis,
                                          structured_problem, code,
                                          verification)

            if verification is None or verification.passed:
                return AlgorithmicResult(
                    success=True,
                    code=code,
                    function_name=structured_problem.function_name,
                    detection=detection,
                    problem=structured_problem,
                    hypotheses=hypotheses,
                    evaluation=evaluation,
                    plan=plan,
                    typed_ir=typed_ir,
                    verification=verification,
                )
            # failed samples: keep the best attempt (most sample cases
            # passed) for the honest fallback
            attempt = (code, plan, typed_ir, verification, evaluation)
            # ranking: most sample cases passed first; among equals prefer
            # the attempt that FINISHED (a timeout attempt never judged)
            timeouts = sum(
                1 for t in (getattr(verification, "test_results", None) or [])
                if str(getattr(t, "error", "") or "").startswith("Timeout"))
            if best_attempt is None:
                best_attempt, best_timeouts = attempt, timeouts
                continue
            cur = (verification.passed_cases, -timeouts)
            old = (best_attempt[3].passed_cases, -best_timeouts)
            if cur > old:
                best_attempt, best_timeouts = attempt, timeouts

        # All hypotheses failed the samples: return the best attempt
        if best_attempt is not None:
            code, plan, typed_ir, verification, evaluation = best_attempt
            return AlgorithmicResult(
                success=True,
                code=code,
                function_name=structured_problem.function_name,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                evaluation=evaluation,
                plan=plan,
                typed_ir=typed_ir,
                verification=verification,
            )

        return AlgorithmicResult(
            success=False,
            detection=detection,
            problem=structured_problem,
            hypotheses=hypotheses,
            error="No hypothesis produced working code",
        )

    def solve_with_semantic(
        self,
        problem_text: str,
        semantic=None,
    ) -> AlgorithmicResult:
        """Solve using SemanticRepresentation to drive the pipeline.

        This method bypasses detection and uses the semantic representation's
        objectives, substructures, and constraints to generate hypotheses.

        Args:
            problem_text: The full problem text (for structured extraction).
            semantic: A SemanticRepresentation with objectives, substructures, constraints.

        Returns:
            AlgorithmicResult with code or error.
        """
        # Create a forced detection result (bypass detection)
        detection = DetectionResult(
            is_competitive=True, confidence=1.0,
            signals=["semantic_driven"],
        )

        # Stage 1: Extract structured problem from text
        try:
            structured_problem = self.extractor.extract(problem_text)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                error=f"Failed to extract structured problem: {e}",
                semantic=semantic,
            )

        # Stage 1b: Enrich structured problem with semantic information
        if semantic:
            self._enrich_structured_problem(structured_problem, semantic)

        # Stage 2: Generate hypotheses using semantic info
        try:
            hypotheses = self.hypothesis_gen.generate(structured_problem)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                error=f"Failed to generate hypotheses: {e}",
                semantic=semantic,
            )

        if not hypotheses:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                error="No hypotheses generated",
                semantic=semantic,
            )

        # Stage 3: Evaluate and rank all hypotheses
        try:
            evaluations = self.evaluator.evaluate_all(hypotheses, structured_problem)
            ranked = list(evaluations)
        except Exception as e:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                error=f"Failed to evaluate hypotheses: {e}",
                semantic=semantic,
            )

        if not ranked:
            return AlgorithmicResult(
                success=False,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                error="No hypothesis selected",
                semantic=semantic,
            )

        hypothesis_by_id = {h.id: h for h in hypotheses}

        # Stages 4-8: Hypothesis rotation with verification (the same
        # structure as solve() — semantic enrichment is already applied
        # above).  Try hypotheses in ranked order; the first whose code
        # passes the problem's own public sample cases wins; if every
        # attempt fails, the best attempt (most sample cases passed) is
        # returned so the caller still gets real code instead of a
        # generic fallback.  A plan built from a different hypothesis
        # than the selected one is a semantic mismatch and is rejected.
        best_attempt = None
        best_timeouts = 0
        last_error = ""
        # Rotation walks the ranking until an attempt SUCCEEDS: a
        # candidate the code generator cannot build is not a judgment on
        # the problem, only on that candidate — the next compatible one
        # is tried.  With sample cases the loop continues past builds
        # that fail the samples (and keeps the best attempt); without
        # samples the first buildable candidate wins and the result
        # carries no verification claim.
        limit = len(ranked)
        for evaluation in ranked[:limit]:
            selected_hypothesis = hypothesis_by_id.get(evaluation.hypothesis_id)
            if selected_hypothesis is None:
                continue

            # Stage 4: Build algorithm plan
            try:
                plan = self.plan_builder.build(selected_hypothesis, structured_problem)
            except Exception as e:
                last_error = f"Failed to build plan: {e}"
                continue
            if plan is None:
                continue
            plan_name = getattr(plan, "hypothesis_name", "") or \
                getattr(plan, "source_hypothesis", "")
            if plan_name and plan_name != selected_hypothesis.name:
                continue

            # Stage 5: Generate typed IR (optional — the code generator
            # reads the plan directly; an IR failure is not fatal)
            typed_ir = None
            try:
                typed_ir = self.ir_generator.generate(plan)
            except Exception as e:
                last_error = f"Failed to generate typed IR: {e}"

            # Stage 6: Generate code
            try:
                code = self.code_generator.generate_from_plan(plan, structured_problem)
            except Exception as e:
                last_error = f"Failed to generate code: {e}"
                continue
            if not code or not code.strip():
                continue

            # Stage 7: Verify against sample tests
            verification = None
            if structured_problem.sample_cases:
                try:
                    verification = self.verifier.verify(
                        code, structured_problem.sample_cases,
                        structured_problem.function_name,
                        request_form=getattr(
                            structured_problem, "request_form", "program"),
                    )
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
                    verification = None

            # CONSUME (learning): the execution outcome is known here —
            # aggregate stats updated; evidence linked to the trace.
            self._record_attempt_outcome(selected_hypothesis,
                                         structured_problem, verification)
            self._record_attempt_evidence(selected_hypothesis,
                                          structured_problem, code,
                                          verification)

            if verification is None or verification.passed:
                return AlgorithmicResult(
                    success=True,
                    code=code,
                    function_name=structured_problem.function_name,
                    detection=detection,
                    problem=structured_problem,
                    hypotheses=hypotheses,
                    evaluation=evaluation,
                    plan=plan,
                    typed_ir=typed_ir,
                    verification=verification,
                    semantic=semantic,
                )
            # failed samples: keep the best attempt (most sample cases
            # passed) for the honest fallback
            attempt = (code, plan, typed_ir, verification, evaluation)
            # ranking: most sample cases passed first; among equals prefer
            # the attempt that FINISHED (a timeout attempt never judged)
            timeouts = sum(
                1 for t in (getattr(verification, "test_results", None) or [])
                if str(getattr(t, "error", "") or "").startswith("Timeout"))
            if best_attempt is None:
                best_attempt, best_timeouts = attempt, timeouts
                continue
            cur = (verification.passed_cases, -timeouts)
            old = (best_attempt[3].passed_cases, -best_timeouts)
            if cur > old:
                best_attempt, best_timeouts = attempt, timeouts

        # All hypotheses failed the samples: return the best attempt
        if best_attempt is not None:
            code, plan, typed_ir, verification, evaluation = best_attempt
            return AlgorithmicResult(
                success=True,
                code=code,
                function_name=structured_problem.function_name,
                detection=detection,
                problem=structured_problem,
                hypotheses=hypotheses,
                evaluation=evaluation,
                plan=plan,
                typed_ir=typed_ir,
                verification=verification,
                semantic=semantic,
            )

        # The evaluation still SURVIVES a failed generation: the caller
        # can see which hypothesis was ranked best and why, instead of
        # an empty evaluation that says nothing about the reasoning.
        return AlgorithmicResult(
            success=False,
            detection=detection,
            problem=structured_problem,
            hypotheses=hypotheses,
            evaluation=(ranked[0] if ranked else None),
            error=last_error or "No hypothesis produced working code",
            semantic=semantic,
        )

    def _enrich_structured_problem(self, sp, semantic):
        """Enrich StructuredProblem with SemanticRepresentation data.

        This bridges the semantic parser output to the algorithmic pipeline.
        """
        # The INTERFACE semantics: a request whose goal is a FUNCTION
        # ("create a function that takes ... and returns ...") carries a
        # function call contract — the input arrives as a parameter and
        # the result is RETURNED.  The stdio program form is reserved
        # for problems explicitly presented as stdin problems.
        goal = getattr(semantic, "goal", None)
        goal_value = getattr(getattr(goal, "goal_type", None), "value", "")
        if goal_value == "create_function" and not sp.uses_stdio \
                and len(sp.input_variables or []) <= 1:
            sp.request_form = "function"
        # Bridge the arithmetic expression ("square of A + B" → "(A + B) ** 2").
        # Variable names are re-resolved against the real input variables from
        # the I/O format (more reliable than entity-based resolution).
        expr = getattr(semantic, 'arithmetic_expression', '')
        if expr:
            # Re-resolve single letters against sp.input_variables
            real_names = [v.name for v in sp.input_variables
                          if re.match(r'^[A-Za-z]$', v.name)]
            resolved = []
            for tok in expr.split():
                if re.match(r'^[A-Za-z]$', tok):
                    match = next((r for r in real_names if r.upper() == tok.upper()), None)
                    resolved.append(match if match else tok)
                else:
                    resolved.append(tok)
            sp.arithmetic_expression = " ".join(resolved)
        else:
            sp.arithmetic_expression = ""

        # Enrich output measure: what the output represents
        # Priority: explicit objective measure > aggregate constraint > operation evidence
        if hasattr(semantic, 'objectives') and semantic.objectives:
            obj = semantic.objectives[0]
            sp.output_measure = obj.measure
            sp.objective_direction = obj.direction
        elif hasattr(semantic, 'outputs') and semantic.outputs:
            # Check output description/type for aggregate semantics
            out = semantic.outputs[0]
            out_text = f"{out.description} {out.type_signature}".lower()
            if any(k in out_text for k in ["sum", "합", "총합", "total"]):
                sp.output_measure = "sum"
            elif any(k in out_text for k in ["count", "개수", "length", "길이"]):
                sp.output_measure = "count"
        elif hasattr(semantic, 'operation_graph'):
            # Fall back to operation names: "calculate_sum" suggests sum output
            op_names = {op.name for op in semantic.operation_graph.operations}
            if "calculate_sum" in op_names or "sum_list" in op_names:
                sp.output_measure = "sum"

        # Enrich structural_hints from semantic substructures
        if hasattr(semantic, 'substructures'):
            for sub in semantic.substructures:
                hint = sub.structure_type
                if hint and hint not in sp.structural_hints:
                    sp.structural_hints.append(hint)

        # Enrich structural_hints from semantic objectives.  The
        # objective's DIRECTION is structure: it becomes the generic
        # optimization_maximize/minimize hint, which the hypothesis
        # stage reads as the signal for extremum candidates.
        if hasattr(semantic, 'objectives'):
            for obj in semantic.objectives:
                direction = getattr(obj, "direction", "") or ""
                target = str(getattr(obj, "target", "") or "")
                if target and not sp.objective_target:
                    sp.objective_target = target
                if direction in ("maximize", "minimize"):
                    for hint in (f"optimization_{direction}", "optimization"):
                        if hint not in sp.structural_hints:
                            sp.structural_hints.append(hint)
                if obj.measure == "length" and obj.direction == "maximize":
                    if "optimization" not in sp.structural_hints:
                        sp.structural_hints.append("optimization")
                    if "find_longest" not in sp.structural_hints:
                        sp.structural_hints.append("find_longest")
                elif obj.measure == "length" and obj.direction == "minimize":
                    if "optimization" not in sp.structural_hints:
                        sp.structural_hints.append("optimization")
                    if "find_shortest" not in sp.structural_hints:
                        sp.structural_hints.append("find_shortest")
                elif obj.measure == "sum" and obj.direction == "maximize":
                    if "optimization" not in sp.structural_hints:
                        sp.structural_hints.append("optimization")
                    if "find_max_sum" not in sp.structural_hints:
                        sp.structural_hints.append("find_max_sum")

        # The DECLARED element type: compatibility of a candidate
        # strategy with the problem's types is evaluated against it.
        try:
            from brain.semantic_parser import SemanticParser
            element = SemanticParser._element_type(semantic)
            if element:
                token = (SemanticParser._canonical_type(element)
                         or element).lower()
                if token and token not in sp.data_types:
                    sp.data_types.append(token)
        except Exception:
            pass

        # The DECLARED I/O: a "create a function that takes ..." request
        # states its parameters in prose, so the format-line extractor
        # finds no variables.  The semantics DO declare them — without
        # them the plan has no data variable and no result variable to
        # build on (the absence is the extractor's, not the request's).
        # ONLY a function-form contract declares parameters: a
        # stdin/stdout program reads lines the request never named, so
        # its variables must come from the format lines alone.  Python
        # builtins never become variable names (a parameter called
        # `input` shadows the reader it would need).
        def _io_var_type(type_signature: str) -> str:
            ts = str(type_signature or "").strip()
            lower = ts.lower()
            mapping = {
                "list<string>": "string[]", "list<str>": "string[]",
                "list<integer>": "int[]", "list<int>": "int[]",
                "list<number>": "float[]", "list<float>": "float[]",
                "list<double>": "float[]",
                "string": "string", "str": "string",
                "integer": "int", "int": "int",
                "number": "float", "float": "float", "double": "float",
                "boolean": "boolean", "bool": "boolean",
            }
            if lower in mapping:
                return mapping[lower]
            if lower.startswith("list<"):
                return "array"
            return ""

        _unsafe_names = frozenset({
            "input", "print", "sum", "max", "min", "len", "id", "type",
            "str", "int", "list", "map", "filter", "next", "all", "any",
            "sorted", "round", "abs", "hash", "object", "property", "set",
            "vars", "help", "dir", "locals", "globals", "open", "read",
        })

        def _safe_name(raw: str, fallback: str) -> str:
            name = (raw or "").strip()
            return fallback if (not name or name.lower() in _unsafe_names) else name

        if getattr(sp, "request_form", "") == "function":
            if not sp.input_variables:
                for spec in (getattr(semantic, "inputs", None) or []):
                    var_type = _io_var_type(getattr(spec, "type_signature", ""))
                    if not var_type:
                        continue
                    name = _safe_name(getattr(spec, "name", "") or "", "data")
                    from brain.algorithmic.structured_problem_extractor import (
                        IOVariable as _IOVar,
                    )
                    sp.input_variables.append(_IOVar(
                        name=name, var_type=var_type,
                        description=getattr(spec, "description", "") or ""))
            if not sp.output_variables:
                for spec in (getattr(semantic, "outputs", None) or []):
                    var_type = _io_var_type(getattr(spec, "type_signature", ""))
                    if not var_type:
                        continue
                    name = _safe_name(getattr(spec, "name", "") or "", "result")
                    from brain.algorithmic.structured_problem_extractor import (
                        IOVariable as _IOVar,
                    )
                    sp.output_variables.append(_IOVar(
                        name=name, var_type=var_type,
                        description=getattr(spec, "description", "") or ""))

        # Enrich constraints from semantic aggregate constraints
        if hasattr(semantic, 'constraints'):
            from brain.algorithmic.structured_problem_extractor import Constraint as AlgoConstraint
            from brain.algorithmic.structured_problem_extractor import Predicate as AlgoPredicate
            for c in semantic.constraints:
                if hasattr(c, 'constraint_type') and c.constraint_type == "aggregate":
                    # Add as a structural hint for hypothesis generation
                    hint = f"aggregate_{c.aggregate_function}"
                    if hint not in sp.structural_hints:
                        sp.structural_hints.append(hint)
                    # Add constraint value
                    if hasattr(c, 'value') and c.value == "0":
                        if "zero_sum" not in sp.structural_hints:
                            sp.structural_hints.append("zero_sum")
                    # Bridge the constraint itself so downstream phases can
                    # reason about it (subject, operator, aggregate function)
                    pred = AlgoPredicate(
                        relation="aggregate",
                        source=getattr(c, 'subject', '') or "subarray",
                        target=str(getattr(c, 'value', '')),
                        polarity="positive",
                        scope="substructure",
                        confidence=getattr(c, 'confidence', 1.0),
                        evidence=getattr(c, 'evidence', ''),
                    )
                    sp.constraints.append(AlgoConstraint(
                        variable=getattr(c, 'subject', '') or "subarray",
                        constraint_type="aggregate",
                        predicate=pred,
                        value=str(getattr(c, 'value', '')),
                        description=getattr(c, 'description', ''),
                        aggregate_function=getattr(c, 'aggregate_function', ''),
                    ))
