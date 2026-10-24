"""Pipeline Trace — captures real data at each stage of the algorithmic pipeline.

Usage:
    from brain.algorithmic.pipeline_trace import trace_algorithmic_pipeline

    tracer = PipelineTracer()
    result = tracer.trace(orchestrator, problem_text)
    tracer.print_trace(result)
"""
from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from io import StringIO

from brain.algorithmic.orchestrator import AlgorithmicOrchestrator, AlgorithmicResult


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    """Safe string representation that won't crash."""
    if obj is None:
        return "None"
    if isinstance(obj, str):
        s = obj
    elif isinstance(obj, (int, float, bool)):
        s = str(obj)
    elif isinstance(obj, dict):
        try:
            s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
    elif isinstance(obj, (list, tuple)):
        try:
            s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
    elif hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()
            s = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
    elif hasattr(obj, "__dict__"):
        try:
            d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            s = json.dumps(d, indent=2, ensure_ascii=False, default=str)
        except Exception:
            s = str(obj)
    else:
        s = str(obj)
    if len(s) > max_len and max_len > 0:
        s = s[:max_len] + f"\n... ({len(s)} chars total)"
    return s


@dataclass
class TraceEntry:
    """One stage of the trace."""
    stage_num: int
    stage_name: str
    elapsed: float  # seconds
    data: dict  # stage-specific data


@dataclass
class TraceResult:
    """Full trace of a pipeline run."""
    raw_input: str
    entries: list[TraceEntry] = field(default_factory=list)
    total_elapsed: float = 0.0


class PipelineTracer:
    """Wraps AlgorithmicOrchestrator and captures real data at each stage."""

    def trace(self, orchestrator: AlgorithmicOrchestrator, problem_text: str) -> TraceResult:
        result = TraceResult(raw_input=problem_text)
        t_total_start = time.time()

        # ── Stage 1: Detection ──
        t0 = time.time()
        detection = orchestrator.detector.detect(problem_text)
        t1 = time.time()
        result.entries.append(TraceEntry(
            stage_num=1,
            stage_name="DETECTION",
            elapsed=t1 - t0,
            data={
                "is_competitive": detection.is_competitive,
                "confidence": detection.confidence,
                "signals": detection.signals,
            },
        ))

        if not detection.is_competitive:
            result.total_elapsed = time.time() - t_total_start
            return result

        # ── Stage 2: Structured Problem Extraction ──
        t0 = time.time()
        structured_problem = orchestrator.extractor.extract(problem_text)
        t1 = time.time()

        sp_data = {}
        try:
            sp_data["function_name"] = structured_problem.function_name
        except Exception:
            sp_data["function_name"] = "N/A"
        try:
            sp_data["input_variables"] = [
                {"name": v.name, "var_type": v.var_type, "description": v.description}
                for v in structured_problem.input_variables
            ]
        except Exception:
            sp_data["input_variables"] = "N/A"
        try:
            sp_data["output_variables"] = [
                {"name": v.name, "var_type": v.var_type, "description": v.description}
                for v in structured_problem.output_variables
            ]
        except Exception:
            sp_data["output_variables"] = "N/A"
        try:
            sp_data["sample_cases"] = [
                {"input": c.input_data, "output": c.expected_output, "explanation": c.explanation}
                for c in structured_problem.sample_cases
            ]
        except Exception:
            sp_data["sample_cases"] = "N/A"
        try:
            sp_data["constraints"] = [
                {"type": c.constraint_type, "description": c.description, "value": c.value}
                for c in structured_problem.constraints
            ]
        except Exception:
            sp_data["constraints"] = "N/A"
        try:
            sp_data["structural_hints"] = list(structured_problem.structural_hints)
        except Exception:
            sp_data["structural_hints"] = "N/A"
        try:
            sp_data["predicates"] = [
                {
                    "relation": p.relation,
                    "source": p.source,
                    "target": p.target,
                    "polarity": p.polarity,
                    "scope": p.scope,
                    "confidence": p.confidence,
                    "evidence": p.evidence,
                }
                for p in structured_problem.predicates
            ]
        except Exception:
            sp_data["predicates"] = "N/A"

        result.entries.append(TraceEntry(
            stage_num=2,
            stage_name="STRUCTURED PROBLEM",
            elapsed=t1 - t0,
            data=sp_data,
        ))

        # ── Stage 3: Hypothesis Generation ──
        t0 = time.time()
        hypotheses = orchestrator.hypothesis_gen.generate(structured_problem)
        t1 = time.time()

        hyp_data = []
        for h in hypotheses:
            hyp_entry = {
                "id": h.id,
                "name": h.name,
                "description": h.description,
                "confidence": h.confidence,
                "algorithmic_primitives": h.algorithmic_primitives,
            }
            try:
                hyp_entry["reasoning"] = h.reasoning
            except Exception:
                pass
            hyp_data.append(hyp_entry)

        result.entries.append(TraceEntry(
            stage_num=3,
            stage_name="HYPOTHESES",
            elapsed=t1 - t0,
            data={"count": len(hyp_data), "hypotheses": hyp_data},
        ))

        if not hypotheses:
            result.total_elapsed = time.time() - t_total_start
            return result

        # ── Stage 4: Hypothesis Evaluation ──
        t0 = time.time()
        evaluations = orchestrator.evaluator.evaluate_all(hypotheses, structured_problem)
        best = orchestrator.evaluator.select_best(evaluations)
        t1 = time.time()

        eval_data = []
        for ev in evaluations:
            eval_entry = {
                "hypothesis_id": ev.hypothesis_id,
                "total_score": ev.total_score,
                "structure_match": ev.structure_match_score,
                "constraint_coverage": ev.constraint_coverage_score,
                "output_compatibility": ev.output_compatibility_score,
                "knowledge_support": ev.knowledge_support_score,
            }
            eval_data.append(eval_entry)

        selected_id = best.hypothesis_id if best else None
        result.entries.append(TraceEntry(
            stage_num=4,
            stage_name="HYPOTHESIS EVALUATION",
            elapsed=t1 - t0,
            data={"evaluations": eval_data, "selected_hypothesis_id": selected_id},
        ))

        if best is None:
            result.total_elapsed = time.time() - t_total_start
            return result

        # Find selected hypothesis
        selected_hypothesis = None
        for h in hypotheses:
            if h.id == best.hypothesis_id:
                selected_hypothesis = h
                break
        if selected_hypothesis is None:
            selected_hypothesis = hypotheses[0]

        # ── Stage 5: Plan Building ──
        t0 = time.time()
        plan = orchestrator.plan_builder.build(selected_hypothesis, structured_problem)
        t1 = time.time()

        plan_data = {
            "function_name": plan.function_name,
            "input_parsing": [],
            "algorithm_steps": [],
            "output_step": None,
            "variables": plan.variables if hasattr(plan, "variables") else [],
        }
        for step in plan.input_parsing:
            plan_data["input_parsing"].append({
                "step_id": step.step_id,
                "operation": step.operation,
                "description": step.description,
                "input_vars": step.input_vars,
                "output_vars": step.output_vars,
                "details": step.details,
            })
        for step in plan.algorithm_steps:
            plan_data["algorithm_steps"].append({
                "step_id": step.step_id,
                "operation": step.operation,
                "description": step.description,
                "input_vars": step.input_vars,
                "output_vars": step.output_vars,
                "details": step.details,
            })
        if plan.output_step:
            step = plan.output_step
            plan_data["output_step"] = {
                "step_id": step.step_id,
                "operation": step.operation,
                "description": step.description,
                "input_vars": step.input_vars,
                "output_vars": step.output_vars,
                "details": step.details,
            }

        result.entries.append(TraceEntry(
            stage_num=5,
            stage_name="PLAN",
            elapsed=t1 - t0,
            data=plan_data,
        ))

        # ── Stage 6: Typed IR ──
        t0 = time.time()
        typed_ir = orchestrator.ir_generator.generate(plan)
        t1 = time.time()

        ir_data = {}
        if typed_ir and typed_ir.root:
            ir_data["root"] = typed_ir.root.to_dict()

        result.entries.append(TraceEntry(
            stage_num=6,
            stage_name="IR",
            elapsed=t1 - t0,
            data=ir_data,
        ))

        # ── Stage 7: Code Generation ──
        t0 = time.time()
        code = orchestrator.code_generator.generate_from_plan(plan, structured_problem)
        t1 = time.time()

        result.entries.append(TraceEntry(
            stage_num=7,
            stage_name="CODE GENERATION",
            elapsed=t1 - t0,
            data={"code": code},
        ))

        # ── Stage 8: Verification ──
        verification = None
        if structured_problem.sample_cases:
            t0 = time.time()
            try:
                verification = orchestrator.verifier.verify(
                    code, structured_problem.sample_cases,
                    structured_problem.function_name,
                )
            except Exception as e:
                verification = None
            t1 = time.time()

            v_data = {}
            if verification:
                v_data["passed"] = verification.passed
                v_data["total_cases"] = verification.total_cases
                v_data["passed_cases"] = verification.passed_cases
                v_data["failed_cases"] = verification.failed_cases
                if hasattr(verification, "errors"):
                    v_data["errors"] = verification.errors
                if hasattr(verification, "outputs"):
                    v_data["outputs"] = verification.outputs
            else:
                v_data["error"] = "Verification failed or not applicable"

            result.entries.append(TraceEntry(
                stage_num=8,
                stage_name="VERIFICATION",
                elapsed=t1 - t0,
                data=v_data,
            ))

        result.total_elapsed = time.time() - t_total_start
        return result

    def print_trace(self, trace: TraceResult, file=None):
        """Print trace in the requested format."""
        out = file or __import__("sys").stdout
        sep = "=" * 60

        print(sep, file=out)
        print("RENA-BR TRACE", file=out)
        print(sep, file=out)
        print(file=out)

        # [01] RAW INPUT
        print(f"[01] RAW INPUT", file=out)
        print(f"    {_safe_repr(trace.raw_input, 0)}", file=out)
        print(file=out)

        # Remaining stages
        for entry in trace.entries:
            num = entry.stage_num + 1  # shift: detection=02, etc.
            label = entry.stage_name
            print(f"[{num:02d}] {label}", file=out)
            print(f"    elapsed: {entry.elapsed:.3f}s", file=out)
            text = _safe_repr(entry.data, 0)
            for line in text.split("\n"):
                print(f"    {line}", file=out)
            print(file=out)

        print(sep, file=out)
        print("TRACE END", file=out)
        print(sep, file=out)
