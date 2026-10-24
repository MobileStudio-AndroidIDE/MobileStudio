"""Competitive Problem Detector.

Detects whether a natural-language input is a competitive programming problem
using STRUCTURAL heuristics only (no keyword dictionaries per problem).
The detection is based on general, stable structural markers:
  - Presence of Input/Output sections
  - Presence of Constraints section
  - Sample input/output examples
  - Platform indicators (Codeforces, AtCoder, BOJ, etc.)
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class DetectionResult:
    is_competitive: bool
    confidence: float  # 0.0 - 1.0
    signals: list[str]  # which signals fired


# Structural patterns that indicate competitive programming.
# Each pattern is (regex, signal_name, weight).
_DETECTION_SIGNALS: list[tuple[str, str, float]] = [
    # Standard sections
    (r"(?:Input|입력)\s*\n", "input_section", 0.25),
    (r"(?:Output|출력)\s*\n", "output_section", 0.25),
    (r"(?:Constraint|제약|Constraints)\s*\n", "constraints_section", 0.20),
    # Sample / Example sections
    (r"(?:Sample\s+Input|예시\s+입력|Sample\s+Input\s*\d)", "sample_input", 0.20),
    (r"(?:Sample\s+Output|예시\s+출력|Sample\s+Output\s*\d)", "sample_output", 0.20),
    # stdin/stdout references
    (r"(?:standard\s+input|standard\s+output|stdin|stdout)", "stdio_ref", 0.15),
    # Time/memory limits
    (r"(?:time\s+limit|memory\s+limit|시간\s+제한|메모리\s+제한)", "resource_limits", 0.15),
    # Platform indicators
    (r"(?:Codeforces|AtCoder|BOJ|백준|LeetCode|HackerRank|SPOJ|USACO|CSES)", "platform_ref", 0.10),
    # Format descriptions
    (r"(?:The\s+input\s+is\s+given|입력은|첫\s+째\s+줄|첫 번째\s+줄)", "format_description", 0.10),
    (r"(?:print\s+the|출력하라|출력해야|출력하세요)", "print_instruction", 0.10),
    # Number of test cases
    (r"(?:number\s+of\s+test\s+cases|테스트\s+케이스의?\s+수)", "test_cases_count", 0.10),
]


class CompetitiveProblemDetector:
    """Rule-based detector for competitive programming problems.

    Uses general structural signals, NOT per-problem heuristics.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Args:
            threshold: Minimum confidence score to classify as competitive.
        """
        self.threshold = threshold

    def detect(self, text: str) -> DetectionResult:
        """Detect if the text is a competitive programming problem.

        Args:
            text: The full problem text (may include all sections).

        Returns:
            DetectionResult with is_competitive, confidence, and signals.
        """
        fired_signals: list[str] = []
        total_weight = 0.0

        for pattern, signal_name, weight in _DETECTION_SIGNALS:
            if re.search(pattern, text, re.IGNORECASE):
                fired_signals.append(signal_name)
                total_weight += weight

        # Bonus for having BOTH input AND output sections (strong combo)
        has_input = "input_section" in fired_signals
        has_output = "output_section" in fired_signals
        if has_input and has_output:
            total_weight += 0.15
            fired_signals.append("input_output_combo")

        # Bonus for having constraints section
        if "constraints_section" in fired_signals:
            total_weight += 0.10

        # Bonus for sample input + output
        has_sample_in = "sample_input" in fired_signals
        has_sample_out = "sample_output" in fired_signals
        if has_sample_in and has_sample_out:
            total_weight += 0.10
            fired_signals.append("sample_io_combo")

        # Clamp confidence to [0.0, 1.0]
        confidence = min(total_weight, 1.0)

        return DetectionResult(
            is_competitive=confidence >= self.threshold,
            confidence=confidence,
            signals=fired_signals,
        )
