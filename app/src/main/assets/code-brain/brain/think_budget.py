"""The Think budget: how much reasoning a single request may spend.

Think is NOT "repeat the same reasoning louder" — it is a budget of
genuinely additional reasoning work:

  - more candidate hypotheses explored,
  - alternative plans formulated (not just the winner's),
  - more counterexamples sought against the selected hypothesis,
  - more verification tests executed,
  - more repair attempts allowed before giving up.

The budget is a property of the REQUEST, not of the engine: the same
CodeBrain answers a simple greeting with a normal budget and a hard
algorithmic problem with a deep-think budget.  The budget is carried in
a ContextVar so the whole pipeline (ThinkEngine, generation
controller, verifier, repair) reads ONE value without threading it
through every signature.

The numbers are measured against the current engine's candidate space
(the ThinkEngine's strategy space yields at most 5 base candidates; the
generation-side HypothesisGenerator explores 2-3 strategy variants per
operation), not arbitrary.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThinkMode(str, Enum):
    NORMAL = "normal"
    THINK = "think"
    DEEP_THINK = "deep_think"


@dataclass(frozen=True)
class ThinkBudget:
    """The reasoning budget for one request.

    max_hypotheses          candidates the hypothesize step may consider
    max_alternative_plans   losing hypotheses whose plans are also
                            formulated (recorded, not executed)
    max_reasoning_iterations extra evaluate→plan→critique passes
    max_counterexamples     adversarial inputs sought against the
                            selected hypothesis
    max_verification_tests  test inputs the verification stage may run
    max_repair_attempts     repair cycles before the hypothesis is
                            abandoned for the next candidate
    """

    mode: ThinkMode
    max_hypotheses: int
    max_alternative_plans: int
    max_reasoning_iterations: int
    max_counterexamples: int
    max_verification_tests: int
    max_repair_attempts: int


# The measured budgets. NORMAL never clips the base strategy space (the
# ThinkEngine's five base strategies all stay), so a normal request is
# byte-for-byte what the engine did before budgets existed.
BUDGETS: dict[ThinkMode, ThinkBudget] = {
    ThinkMode.NORMAL: ThinkBudget(
        mode=ThinkMode.NORMAL,
        max_hypotheses=5,           # the ThinkEngine's full base strategy space
        max_alternative_plans=0,
        max_reasoning_iterations=0,
        max_counterexamples=0,     # the base verification input set only
        max_verification_tests=8,  # the full base input set (8 inputs)
        max_repair_attempts=1,
    ),
    ThinkMode.THINK: ThinkBudget(
        mode=ThinkMode.THINK,
        max_hypotheses=8,
        max_alternative_plans=1,
        max_reasoning_iterations=1,
        max_counterexamples=1,
        max_verification_tests=9,
        max_repair_attempts=2,
    ),
    ThinkMode.DEEP_THINK: ThinkBudget(
        mode=ThinkMode.DEEP_THINK,
        max_hypotheses=12,
        max_alternative_plans=2,
        max_reasoning_iterations=2,
        max_counterexamples=3,
        max_verification_tests=11,
        max_repair_attempts=3,
    ),
}

_NORMAL_BUDGET = BUDGETS[ThinkMode.NORMAL]

# The active budget.  None means "normal" — the default request spends
# the normal budget without any explicit setup.
_active_budget: ContextVar[Optional[ThinkBudget]] = ContextVar(
    "rena_think_budget", default=None
)


def budget_for(mode: ThinkMode | str) -> ThinkBudget:
    """The budget of a mode (a ThinkMode, or its string value)."""
    if isinstance(mode, ThinkMode):
        return BUDGETS[mode]
    try:
        return BUDGETS[ThinkMode(str(mode).strip().lower())]
    except ValueError:
        return _NORMAL_BUDGET


def current_budget() -> ThinkBudget:
    """The budget this request runs under."""
    return _active_budget.get() or _NORMAL_BUDGET


def current_mode() -> ThinkMode:
    """The mode this request runs under."""
    return current_budget().mode


def set_think_mode(mode: ThinkMode | str) -> None:
    """Set the budget for all subsequent requests in this context."""
    _active_budget.set(budget_for(mode))


def clear_think_mode() -> None:
    """Return to the normal budget."""
    _active_budget.set(None)


def resolve_mode(mode: Optional[str] = None, *,
                 think: bool = False, deep_think: bool = False) -> ThinkMode:
    """Resolve the mode from explicit flags (CLI style):
    --deep-think wins over --think, an explicit mode name wins over both."""
    if deep_think:
        return ThinkMode.DEEP_THINK
    if think:
        return ThinkMode.THINK
    if mode:
        stripped = str(mode).strip().lower().replace("-", "_")
        try:
            return ThinkMode(stripped)
        except ValueError:
            return ThinkMode.NORMAL
    return ThinkMode.NORMAL


class think_context:
    """Context manager: run a block under a think mode.

        with think_context("deep_think"):
            brain.process(request)

    The mode is scoped to the block — requests outside it run normal.
    """

    def __init__(self, mode: ThinkMode | str):
        self._budget = budget_for(mode)
        self._token = None

    def __enter__(self) -> ThinkBudget:
        self._token = _active_budget.set(self._budget)
        return self._budget

    def __exit__(self, exc_type, exc, tb) -> None:
        _active_budget.reset(self._token)


# Backwards-friendly alias matching common naming.
think_mode = think_context
