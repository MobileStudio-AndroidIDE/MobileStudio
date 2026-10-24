"""Configuration and resource bounds for the cognitive subsystem.

Every loop in this package is bounded by an explicit cycle count, a
wall-clock budget, and an attempt budget.  Nothing here retries
forever: a budget exhaustion is a recorded, reportable outcome.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


def default_data_dir() -> str:
    return os.environ.get(
        "RENA_COGNITION_DIR",
        os.path.join(os.path.expanduser("~"), ".rena_cognition"))


@dataclass
class GrowthConfig:
    # Representation
    dimension: int = 2048
    vector_seed: int = 1729
    working_capacity: int = 7
    episode_capacity: int = 4000

    # Learning / verification gates
    gate_min_successes: int = 3
    gate_min_rate: float = 0.75
    gate_min_holdout: int = 1
    validation_successes: int = 3
    replay_every: int = 5
    rehearsal_keep: int = 120
    consolidation_interval: int = 5
    contradiction_threshold: float = 0.55

    # Search budgets
    max_cycles: int = 24
    time_budget_seconds: float = 90.0
    cycle_seconds: float = 12.0
    max_attempts_per_task: int = 4
    max_synthesis_candidates: int = 20000
    synthesis_depth: int = 3
    max_tasks_per_goal: int = 4
    practice_per_cycle: int = 2

    # Reuse / transfer
    abstraction_enabled: bool = True
    abstraction_min_depth: int = 2
    abstraction_max_per_skill: int = 8
    augmentation_enabled: bool = True
    augmentation_trials: int = 3
    augmentation_seconds: float = 20.0
    augmentation_candidates: int = 6
    invention_enabled: bool = True
    invention_seconds: float = 20.0
    invention_candidates: int = 6
    sensor_invention_enabled: bool = True
    sensor_candidates: int = 12
    transfer_min_rate: float = 0.6
    transfer_trials: int = 3
    skill_similarity_floor: float = 0.08

    # Safety
    allow_self_modification: bool = False
    sandbox_timeout: float = 5.0
    sandbox_memory_mb: int = 512
    allowed_read_roots: list[str] = field(default_factory=list)
    allowed_write_roots: list[str] = field(default_factory=list)
    allow_network: bool = False

    # Paths
    data_dir: str = field(default_factory=default_data_dir)
    artifact_dir: str = ""

    # Determinism
    seed: int = 13
    journal_limit: int = 4000

    def __post_init__(self) -> None:
        self.data_dir = os.path.abspath(self.data_dir)
        if not self.artifact_dir:
            self.artifact_dir = os.path.join(self.data_dir, "artifacts")
        else:
            self.artifact_dir = os.path.abspath(self.artifact_dir)
        if not self.allowed_read_roots:
            self.allowed_read_roots = [self.data_dir]
        if not self.allowed_write_roots:
            self.allowed_write_roots = [self.data_dir]
        self.dimension = max(256, int(self.dimension))
        self.working_capacity = max(1, int(self.working_capacity))

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.artifact_dir):
            os.makedirs(path, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GrowthConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})
