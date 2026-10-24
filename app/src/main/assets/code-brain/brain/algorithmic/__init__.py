"""Algorithmic Reasoning Layer for competitive programming problems.

Non-LLM pipeline that solves competitive programming problems via:
  NL -> CompetitiveProblemDetector -> StructuredProblemExtractor -> StructuredProblem
  -> AlgorithmHypothesisGenerator -> Hypotheses -> HypothesisEvaluator -> SelectedHypothesis
  -> AlgorithmPlanBuilder -> AlgorithmPlan -> TypedIRGenerator -> Typed IR
  -> CompetitiveCodeGenerator -> Python code -> SampleTestVerifier -> VerificationResult

All reasoning is rule-based + KG traversal + composition.
No hardcoded problem_id-to-algorithm or word-to-algorithm mappings.
"""
from brain.algorithmic.competitive_problem_detector import CompetitiveProblemDetector
from brain.algorithmic.structured_problem_extractor import StructuredProblemExtractor
from brain.algorithmic.algorithm_hypothesis_generator import AlgorithmHypothesisGenerator
from brain.algorithmic.hypothesis_evaluator import HypothesisEvaluator
from brain.algorithmic.algorithm_plan_builder import AlgorithmPlanBuilder
from brain.algorithmic.typed_ir_generator import TypedIRGenerator
from brain.algorithmic.competitive_code_generator import CompetitiveCodeGenerator
from brain.algorithmic.sample_test_verifier import SampleTestVerifier
from brain.algorithmic.orchestrator import AlgorithmicOrchestrator

__all__ = [
    "CompetitiveProblemDetector",
    "StructuredProblemExtractor",
    "AlgorithmHypothesisGenerator",
    "HypothesisEvaluator",
    "AlgorithmPlanBuilder",
    "TypedIRGenerator",
    "CompetitiveCodeGenerator",
    "SampleTestVerifier",
    "AlgorithmicOrchestrator",
]
