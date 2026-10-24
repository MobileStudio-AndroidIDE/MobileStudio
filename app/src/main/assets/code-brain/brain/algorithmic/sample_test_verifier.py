"""Sample Test Verifier.

Verifies generated code against sample test cases.
Runs the code with sample inputs and checks if outputs match.
No LLM involved.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional

from brain.observation import ExecutionEvidence, not_run_evidence


def _short(text, limit: int = 200) -> str:
    s = str(text if text is not None else "")
    return s if len(s) <= limit else s[:limit] + "..."



@dataclass
class TestCaseResult:
    """Result of a single test case."""
    input_text: str
    expected_output: str
    actual_output: str
    passed: bool
    error: str = ""
    execution_time: float = 0.0


@dataclass
class VerificationResult:
    """Complete verification result."""
    passed: bool
    total_cases: int
    passed_cases: int
    test_results: list[TestCaseResult] = field(default_factory=list)
    error: str = ""


class SampleTestVerifier:
    """Verifies generated code against sample test cases.

    Runs the code in a subprocess with timeouts and captures output.
    """

    def __init__(self, timeout: float = 5.0):
        """
        Args:
            timeout: Maximum execution time per test case in seconds.
        """
        self.timeout = timeout

    def verify(self, code: str, sample_cases: list, function_name: str = "solve",
               request_form: str = "program") -> VerificationResult:
        """Verify code against sample test cases.

        Args:
            code: The generated Python code.
            sample_cases: List of SampleCase objects with input_text and output_text.
            function_name: The name of the function to call.
            request_form: the artifact's INTERFACE contract — "program"
                (stdin/stdout) or "function" (parameter in, result out).

        Returns:
            VerificationResult with pass/fail status and details.
        """
        if not sample_cases:
            return VerificationResult(
                passed=True,
                total_cases=0,
                passed_cases=0,
                error="No sample cases to verify against",
            )

        results = []
        for i, case in enumerate(sample_cases):
            result = self._run_test(code, case.input_text, case.output_text,
                                    i + 1, function_name,
                                    request_form=request_form)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        all_passed = passed_count == len(results)

        return VerificationResult(
            passed=all_passed,
            total_cases=len(results),
            passed_cases=passed_count,
            test_results=results,
        )

    @staticmethod
    def verification_evidence(verification: VerificationResult,
                              code_ref: str = "",
                              requirement_id: Optional[str] = None
                              ) -> list[ExecutionEvidence]:
        """Link each test verdict as ExecutionEvidence.

        A pass/fail per sample case — the verdict is whatever actually
        happened, never inferred.  Powers ObservationTrace.add_evidence
        at the orchestrator call site.
        """
        evidence: list[ExecutionEvidence] = []
        for i, r in enumerate(verification.test_results or [], 1):
            verdict = "pass" if r.passed else "fail"
            evidence.append(ExecutionEvidence(
                requirement_id=requirement_id,
                code_ref=code_ref,
                input_repr=_short(r.input_text),
                expected_repr=_short(r.expected_output),
                actual_repr=_short(r.actual_output),
                verdict=verdict,
                stage="behavior",
                detail=(f"case {i}: {r.error}" if r.error else f"case {i}"),
            ))
        return evidence

    @staticmethod
    def not_run_evidence(reason: str, code_ref: str = "",
                         stage: str = "behavior",
                         requirement_id: Optional[str] = None
                         ) -> ExecutionEvidence:
        """Honest NOT_RUN evidence when verification could not run."""
        return not_run_evidence(reason, code_ref=code_ref, stage=stage,
                                requirement_id=requirement_id)

    def _run_test(self, code: str, input_text: str, expected_output: str, case_num: int,
                  function_name: str = "solve",
                  request_form: str = "program") -> TestCaseResult:
        """Run a single test case against the artifact's INTERFACE.

        The harness exercises the form the request declared: a program
        reads stdin and prints; a function takes its input as a
        parameter and its RETURN value is the output.
        """
        # Create a complete test script
        test_script = code + "\n\n"
        test_script += "# Run the function with the provided input\n"
        test_script += "import sys\n"
        test_script += "from io import StringIO\n"
        test_script += f"sys.stdin = StringIO({repr(input_text)})\n"
        test_script += "old_stdout = sys.stdout\n"
        test_script += "sys.stdout = StringIO()\n"
        test_script += "try:\n"
        if request_form == "function":
            # The function contract: the sample input is parsed the same
            # way the stdio form's own parser would (ints when they
            # parse, tokens otherwise), passed as the parameter, and
            # the RETURN value is the output.
            test_script += "    _raw = sys.stdin.read().split()\n"
            test_script += "    try:\n"
            test_script += "        data = [int(x) for x in _raw]\n"
            test_script += "    except ValueError:\n"
            test_script += "        data = _raw\n"
            test_script += f"    result = {function_name}(data)\n"
            test_script += "    print(result)\n"
        else:
            test_script += f"    {function_name}()\n"
        test_script += "    actual = sys.stdout.getvalue().strip()\n"
        test_script += "finally:\n"
        test_script += "    sys.stdout = old_stdout\n"
        test_script += "print(actual)\n"

        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir='/tmp'
            ) as f:
                f.write(test_script)
                tmp_path = f.name

            import time
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            execution_time = time.time() - start_time

            os.unlink(tmp_path)

            actual_output = result.stdout.strip()

            if result.returncode != 0:
                return TestCaseResult(
                    input_text=input_text,
                    expected_output=expected_output,
                    actual_output="",
                    passed=False,
                    error=f"Runtime error (exit code {result.returncode}): {result.stderr[:500]}",
                    execution_time=execution_time,
                )

            # Normalize outputs for comparison: line endings differ
            # between problem statements (CRLF) and program output (LF)
            expected_normalized = expected_output.replace(
                "\r\n", "\n").replace("\r", "\n").strip()
            actual_normalized = actual_output.replace(
                "\r\n", "\n").replace("\r", "\n").strip()
            # trailing whitespace per line is not semantic
            expected_normalized = "\n".join(
                l.rstrip() for l in expected_normalized.split("\n"))
            actual_normalized = "\n".join(
                l.rstrip() for l in actual_normalized.split("\n"))

            passed = actual_normalized == expected_normalized

            return TestCaseResult(
                input_text=input_text,
                expected_output=expected_output,
                actual_output=actual_output,
                passed=passed,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired:
            return TestCaseResult(
                input_text=input_text,
                expected_output=expected_output,
                actual_output="",
                passed=False,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return TestCaseResult(
                input_text=input_text,
                expected_output=expected_output,
                actual_output="",
                passed=False,
                error=f"Exception: {str(e)}",
            )
