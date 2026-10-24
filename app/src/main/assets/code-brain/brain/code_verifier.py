"""Code Verification & Repair - Phase 7.

Executes generated code, classifies errors by layer (Semantic, Type, Plan,
IR, Code, Runtime), and suggests repairs at the appropriate layer.
"""
from __future__ import annotations
import ast
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorLayer(Enum):
    SEMANTIC = "semantic"        # Wrong meaning interpretation
    TYPE = "type"                # Type inference error
    KNOWLEDGE = "knowledge"      # Wrong knowledge retrieved
    HYPOTHESIS = "hypothesis"    # Wrong algorithm chosen
    PLAN = "plan"                # Wrong operation sequence
    IR = "ir"                    # Wrong intermediate representation
    CODE = "code"                # Syntax/code generation error
    RUNTIME = "runtime"          # Runtime execution error
    TEST = "test"                # Test interpretation error
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    CRITICAL = "critical"    # Blocks execution
    MAJOR = "major"          # Wrong result
    MINOR = "minor"          # Style/efficiency issue


@dataclass
class ErrorAnalysis:
    """Analysis of a single error."""
    error_layer: ErrorLayer
    severity: ErrorSeverity
    message: str
    original_error: str = ""
    stack_trace: str = ""
    suggested_repair: str = ""
    repair_target: str = ""  # which layer to repair at
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "error_layer": self.error_layer.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_error": self.original_error,
            "suggested_repair": self.suggested_repair,
            "repair_target": self.repair_target,
            "confidence": self.confidence,
        }


@dataclass
class VerificationResult:
    """Result of code verification."""
    success: bool
    code: str = ""
    language: str = "python"
    compilation_passed: bool = False
    execution_passed: bool = False
    test_results: list[dict] = field(default_factory=list)
    error_analyses: list[ErrorAnalysis] = field(default_factory=list)
    execution_output: str = ""
    execution_error: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "language": self.language,
            "compilation_passed": self.compilation_passed,
            "execution_passed": self.execution_passed,
            "test_results": self.test_results,
            "error_analyses": [e.to_dict() for e in self.error_analyses],
            "execution_output": self.execution_output[:500],
            "execution_error": self.execution_error[:500],
        }


class CodeVerifier:
    """Verifies generated code by execution and classifies errors."""

    def verify_python(self, code: str, test_cases: list[dict] = None,
                      expected_output: Any = None) -> VerificationResult:
        """Verify Python code by compilation + execution + test cases."""
        result = VerificationResult(code=code, language="python")

        # Step 1: Compilation check
        try:
            ast.parse(code)
            result.compilation_passed = True
        except SyntaxError as e:
            result.error_analyses.append(ErrorAnalysis(
                error_layer=ErrorLayer.CODE,
                severity=ErrorSeverity.CRITICAL,
                message=f"Syntax error: {e.msg}",
                original_error=str(e),
                suggested_repair="Fix syntax at line {}".format(e.lineno),
                repair_target="code",
                confidence=0.95,
            ))
            return result

        # Step 2: Execution check (try to run the code)
        ns = {}
        try:
            exec(code, ns)
            result.execution_passed = True
        except Exception as e:
            tb = traceback.format_exc()
            analysis = self._classify_runtime_error(e, tb)
            result.error_analyses.append(analysis)
            result.execution_error = tb
            return result

        # Step 3: Test case execution
        if test_cases:
            for tc in test_cases:
                tc_result = self._run_test_case(ns, tc)
                result.test_results.append(tc_result)
                if not tc_result.get("passed"):
                    analysis = self._classify_test_failure(tc_result)
                    result.error_analyses.append(analysis)

        # Step 4: Verify expected output
        if expected_output is not None:
            fn_name = self._find_main_function(ns)
            if fn_name:
                try:
                    actual = ns[fn_name]([])
                    if actual != expected_output:
                        result.test_results.append({
                            "name": "expected_output",
                            "passed": False,
                            "expected": str(expected_output),
                            "actual": str(actual),
                        })
                        result.error_analyses.append(ErrorAnalysis(
                            error_layer=ErrorLayer.TEST,
                            severity=ErrorSeverity.MAJOR,
                            message=f"Expected {expected_output}, got {actual}",
                            suggested_repair="Check logic for correct output",
                            repair_target="plan",
                            confidence=0.7,
                        ))
                except Exception as e:
                    result.error_analyses.append(ErrorAnalysis(
                        error_layer=ErrorLayer.RUNTIME,
                        severity=ErrorSeverity.CRITICAL,
                        message=f"Execution failed: {e}",
                        original_error=str(e),
                        repair_target="code",
                        confidence=0.8,
                    ))

        result.success = (result.compilation_passed and result.execution_passed
                         and not any(e.severity == ErrorSeverity.CRITICAL
                                    for e in result.error_analyses))
        return result

    def _classify_runtime_error(self, error: Exception, tb: str) -> ErrorAnalysis:
        """Classify a runtime error to its root cause layer."""
        error_type = type(error).__name__
        error_msg = str(error)

        if error_type == "TypeError":
            if "unsupported operand" in error_msg or "argument" in error_msg:
                return ErrorAnalysis(
                    error_layer=ErrorLayer.TYPE,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Type error: {error_msg}",
                    original_error=error_msg,
                    stack_trace=tb,
                    suggested_repair="Check input types and operator compatibility",
                    repair_target="type",
                    confidence=0.8,
                )
        elif error_type == "NameError":
            return ErrorAnalysis(
                error_layer=ErrorLayer.CODE,
                severity=ErrorSeverity.CRITICAL,
                message=f"Undefined name: {error_msg}",
                original_error=error_msg,
                stack_trace=tb,
                suggested_repair="Define the missing variable or function",
                repair_target="code",
                confidence=0.9,
            )
        elif error_type == "IndexError":
            return ErrorAnalysis(
                error_layer=ErrorLayer.CODE,
                severity=ErrorSeverity.MAJOR,
                message=f"Index out of range: {error_msg}",
                original_error=error_msg,
                stack_trace=tb,
                suggested_repair="Add bounds checking",
                repair_target="code",
                confidence=0.85,
            )
        elif error_type == "ZeroDivisionError":
            return ErrorAnalysis(
                error_layer=ErrorLayer.CODE,
                severity=ErrorSeverity.MAJOR,
                message=f"Division by zero: {error_msg}",
                original_error=error_msg,
                stack_trace=tb,
                suggested_repair="Add zero-check before division",
                repair_target="code",
                confidence=0.9,
            )
        elif error_type == "RecursionError":
            return ErrorAnalysis(
                error_layer=ErrorLayer.PLAN,
                severity=ErrorSeverity.CRITICAL,
                message=f"Infinite recursion: {error_msg}",
                original_error=error_msg,
                stack_trace=tb,
                suggested_repair="Add base case or use iterative approach",
                repair_target="plan",
                confidence=0.85,
            )

        return ErrorAnalysis(
            error_layer=ErrorLayer.RUNTIME,
            severity=ErrorSeverity.CRITICAL,
            message=f"Runtime error ({error_type}): {error_msg}",
            original_error=error_msg,
            stack_trace=tb,
            suggested_repair="Debug runtime error",
            repair_target="code",
            confidence=0.6,
        )

    def _classify_test_failure(self, tc_result: dict) -> ErrorAnalysis:
        """Classify a test case failure."""
        return ErrorAnalysis(
            error_layer=ErrorLayer.TEST,
            severity=ErrorSeverity.MAJOR,
            message=f"Test '{tc_result.get('name', '?')}': "
                    f"expected {tc_result.get('expected')}, got {tc_result.get('actual')}",
            suggested_repair="Check algorithm logic",
            repair_target="plan",
            confidence=0.7,
        )

    def _run_test_case(self, ns: dict, tc: dict) -> dict:
        """Run a single test case against the code."""
        fn_name = self._find_main_function(ns)
        if not fn_name:
            return {"name": tc.get("name", "?"), "passed": False,
                    "error": "No function found"}

        fn = ns[fn_name]
        input_val = tc.get("input")
        expected = tc.get("expected")
        try:
            actual = fn(input_val) if input_val is not None else fn()
            passed = actual == expected
            return {"name": tc.get("name", "?"), "passed": passed,
                    "input": str(input_val), "expected": str(expected),
                    "actual": str(actual)}
        except Exception as e:
            return {"name": tc.get("name", "?"), "passed": False,
                    "error": str(e)}

    def _find_main_function(self, ns: dict) -> Optional[str]:
        """Find the main function in the namespace."""
        for name, obj in ns.items():
            if callable(obj) and not name.startswith("_"):
                return name
        return None
