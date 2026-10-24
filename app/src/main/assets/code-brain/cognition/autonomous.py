"""Bounded autonomous developmental loop over RenaBrain and growth."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.config import GrowthConfig
from cognition.rena import RenaBrain
from cognition.util import atomic_write_json, stable_hash


@dataclass
class AutonomousReport:
    session_id: str
    started_at: float
    cycles_requested: int
    cycles_completed: int = 0
    growth_cycles: int = 0
    growth_tasks: int = 0
    growth_successes: int = 0
    probes: int = 0
    probe_successes: int = 0
    augmentations: int = 0
    augmentation_candidates: int = 0
    augmentation_adopted: int = 0
    inventions: int = 0
    invention_candidates: int = 0
    primitives_invented: int = 0
    consolidations: int = 0
    domain_counts: dict = field(default_factory=dict)
    per_cycle: list[dict] = field(default_factory=list)
    stopped_reason: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "cycles_requested": self.cycles_requested,
            "cycles_completed": self.cycles_completed,
            "growth_cycles": self.growth_cycles,
            "growth_tasks": self.growth_tasks,
            "growth_successes": self.growth_successes,
            "probes": self.probes,
            "probe_successes": self.probe_successes,
            "augmentations": self.augmentations,
            "augmentation_candidates": self.augmentation_candidates,
            "augmentation_adopted": self.augmentation_adopted,
            "inventions": self.inventions,
            "invention_candidates": self.invention_candidates,
            "primitives_invented": self.primitives_invented,
            "consolidations": self.consolidations,
            "domain_counts": self.domain_counts,
            "per_cycle": self.per_cycle,
            "stopped_reason": self.stopped_reason,
            "limitations": self.limitations,
        }


class AutonomousAgent:
    def __init__(self, config: Optional[GrowthConfig] = None,
                 brain: Optional[RenaBrain] = None,
                 seed: Optional[int] = None):
        self.config = config or GrowthConfig()
        self.config.ensure_dirs()
        self.brain = brain or RenaBrain(self.config)
        self.seed = self.config.seed if seed is None else int(seed)
        self.reports_dir = os.path.join(self.config.data_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.report: Optional[AutonomousReport] = None
        self.rng = __import__("random").Random(self.seed)

    def _select_domain(self) -> str:
        status = self.brain.status()
        growth = status.get("growth") or {}
        domains = growth.get("domains") or {}
        if not domains:
            return "math"
        rows = []
        for name, data in domains.items():
            active = int(data.get("active_skills", 0))
            candidate = int(data.get("candidate_skills", 0))
            meta = self.brain.meta.reflect().get("domains", {}).get(
                name, {})
            rate = float(meta.get("success_rate", 0.5))
            rows.append((active + candidate, -rate, name))
        rows.sort()
        return rows[0][2]

    def _probe(self, domain: str) -> dict:
        tasks = self.brain.curiosity.micro_tasks(domain, count=1)
        if not tasks:
            return {"ok": False, "reason": "no_probe"}
        task = tasks[0]
        if domain == "math":
            result = self.brain.act(
                "math", task["prompt"], force=True)
            expected = task.get("expected")
            actual = None
            if result.get("ok"):
                output = result.get("output") or {}
                actual = output.get("value", output.get("derivative"))
            ok = bool(result.get("ok")) and (
                expected is None or str(actual) == str(expected)
                or _numeric_equal(actual, expected)
                or _expression_equal(actual, expected))
            self.brain.curiosity.observe(
                task["task"], 0.2 if ok else 0.8, ok, domain=domain)
            return {
                "ok": ok,
                "domain": domain,
                "task": task["prompt"],
                "expected": expected,
                "actual": actual,
                "result": result,
            }
        return {
            "ok": False,
            "domain": domain,
            "task": task["prompt"],
            "reason": "probe_not_implemented_for_domain",
        }

    def run(self, cycles: int = 4,
            time_budget: Optional[float] = None,
            consolidate_every: int = 2) -> dict:
        budget = float(time_budget if time_budget is not None
                       else min(120.0, 25.0 * max(1, cycles)))
        started = time.monotonic()
        report = AutonomousReport(
            session_id="auto_" + stable_hash(time.time(), self.seed,
                                             length=12),
            started_at=time.time(),
            cycles_requested=max(0, int(cycles)))
        report.limitations = [
            "Probes beyond math use the growth domains' own independent "
            "verification; this loop does not claim general natural-language "
            "task solving.",
            "All cycles are bounded by wall-clock time and explicit cycle "
            "counts; budget exhaustion is reported, not retried forever.",
            "Self-augmentation recombines the typed operator/macro vocabulary. "
            "Primitive invention abstracts constants/subexpressions from "
            "verified programs into new parametric primitives; it does not "
            "invent a new physical sensor or arbitrary code.",
            "Virtual sensor invention combines existing measurement channels "
            "and is invoked explicitly with samples; it is not a hardware "
            "transducer.",
        ]
        for index in range(max(0, int(cycles))):
            if time.monotonic() - started >= budget:
                report.stopped_reason = "time_budget"
                break
            before = self.brain.status()
            domain = self._select_domain()
            report.domain_counts[domain] = report.domain_counts.get(
                domain, 0) + 1
            growth = self.brain.grow(cycles=1)
            report.growth_cycles += int(growth.get("cycles", 0))
            report.growth_tasks += int(growth.get("tasks_attempted", 0))
            report.growth_successes += int(
                growth.get("tasks_verified", 0))
            probe = self._probe(domain)
            report.probes += 1
            if probe.get("ok"):
                report.probe_successes += 1
            augmentation = {}
            remaining = budget - (time.monotonic() - started)
            if (getattr(self.config, "augmentation_enabled", True)
                    and remaining > 2.0):
                try:
                    augmentation = self.brain.augment(
                        cycles=1,
                        candidates=min(3, int(getattr(
                            self.config, "augmentation_candidates", 6))),
                        time_budget=min(8.0, remaining - 1.0),
                        domain=domain)
                    report.augmentations += int(
                        augmentation.get("cycles", 0))
                    report.augmentation_candidates += int(
                        augmentation.get("candidates_generated", 0))
                    report.augmentation_adopted += int(
                        augmentation.get("candidates_adopted", 0))
                except Exception as exc:
                    augmentation = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}
            invention = {}
            remaining = budget - (time.monotonic() - started)
            if (getattr(self.config, "invention_enabled", True)
                    and remaining > 2.0):
                try:
                    invention = self.brain.invent_primitives(
                        cycles=1,
                        candidates=min(3, int(getattr(
                            self.config, "invention_candidates", 6))),
                        time_budget=min(6.0, remaining - 1.0))
                    report.inventions += 1
                    report.invention_candidates += int(
                        invention.get("candidates_generated", 0))
                    report.primitives_invented += int(
                        invention.get("candidates_adopted", 0))
                except Exception as exc:
                    invention = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}
            consolidation = {}
            if (index + 1) % max(1, int(consolidate_every)) == 0:
                consolidation = self.brain.consolidate()
                report.consolidations += 1
            after = self.brain.status()
            report.per_cycle.append({
                "cycle": index + 1,
                "domain": domain,
                "growth": growth,
                "probe": probe,
                "augmentation": augmentation,
                "invention": invention,
                "consolidation": consolidation,
                "concepts_before": before["concepts"]["concepts"],
                "concepts_after": after["concepts"]["concepts"],
                "learning_stage": after["learning"]["stage"],
                "metacognition": after["metacognition"],
            })
            report.cycles_completed += 1
            self.brain.save()
        if not report.stopped_reason:
            report.stopped_reason = "cycles_completed"
        self.report = report
        data = report.to_dict()
        atomic_write_json(
            os.path.join(self.reports_dir,
                         report.session_id + ".json"), data)
        self.brain.history.append({"event": "autonomous", "report": data})
        return data

    def status(self) -> dict:
        return {
            "brain": self.brain.status(),
            "last_report": self.report.to_dict() if self.report else {},
        }


def _numeric_equal(actual, expected) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= \
            1e-6 * max(1.0, abs(float(expected)))
    except (TypeError, ValueError):
        return False


def _expression_equal(actual, expected) -> bool:
    try:
        from cognition.math_engine import evaluate, parse_expression
        left = parse_expression(str(actual))
        right = parse_expression(str(expected))
        variables = sorted(set(left.variables()) | set(right.variables()))
        sample = {name: 1.7 + index for index, name in
                  enumerate(variables)}
        for offset in (0.0, 0.37, 1.11):
            env = {name: value + offset
                   for name, value in sample.items()}
            if abs(evaluate(left, env) - evaluate(right, env)) > 1e-6:
                return False
        return True
    except Exception:
        return False
