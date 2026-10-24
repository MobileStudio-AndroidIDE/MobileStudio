"""Calibrated self-monitoring: confidence, error diagnosis and budgets.

Metacognition here is measurable: historical success rates, calibration
error (Brier/ECE), latency and an explicit error taxonomy.  It is not a
hidden claim of consciousness.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.util import atomic_write_json, read_json


@dataclass
class TaskStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    confidence_sum: float = 0.0
    brier_sum: float = 0.0
    latency_sum: float = 0.0
    surprise_sum: float = 0.0
    streak: int = 0
    best_streak: int = 0
    last_error: str = ""
    last_outcome: Optional[bool] = None
    domain: str = "general"
    errors: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "confidence_sum": self.confidence_sum,
            "brier_sum": self.brier_sum,
            "latency_sum": self.latency_sum,
            "surprise_sum": self.surprise_sum,
            "streak": self.streak,
            "best_streak": self.best_streak,
            "last_error": self.last_error,
            "last_outcome": self.last_outcome,
            "domain": self.domain,
            "errors": self.errors,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskStats":
        return cls(
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            confidence_sum=float(data.get("confidence_sum", 0.0)),
            brier_sum=float(data.get("brier_sum", 0.0)),
            latency_sum=float(data.get("latency_sum", 0.0)),
            surprise_sum=float(data.get("surprise_sum", 0.0)),
            streak=int(data.get("streak", 0)),
            best_streak=int(data.get("best_streak", 0)),
            last_error=str(data.get("last_error", "")),
            last_outcome=data.get("last_outcome"),
            domain=str(data.get("domain", "general")),
            errors=dict(data.get("errors") or {}),
            calibration=dict(data.get("calibration") or {}),
        )


class MetaCognition:
    def __init__(self, data_dir: str, exploration: float = 0.2):
        self.data_dir = str(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.path = os.path.join(self.data_dir, "metacognition.json")
        self.exploration = max(0.0, min(1.0, float(exploration)))
        self.tasks: dict[str, TaskStats] = {}
        self.total_attempts = 0
        self.total_successes = 0
        self._load()

    def _key(self, task: str) -> str:
        return str(task).strip() or "unknown"

    def record(self, task: str, success: bool, confidence: float = 0.5,
               latency: float = 0.0, domain: str = "",
               error: Optional[str] = None, surprise: float = 0.0) -> dict:
        key = self._key(task)
        confidence = max(0.0, min(1.0, float(confidence)))
        stats = self.tasks.setdefault(key, TaskStats())
        stats.attempts += 1
        stats.confidence_sum += confidence
        stats.brier_sum += (confidence - (1.0 if success else 0.0)) ** 2
        stats.latency_sum += max(0.0, float(latency))
        stats.surprise_sum += max(0.0, float(surprise))
        if domain:
            stats.domain = str(domain)
        bucket = str(min(9, int(confidence * 10)))
        calibration = stats.calibration.setdefault(
            bucket, {"count": 0, "confidence": 0.0, "success": 0.0})
        calibration["count"] += 1
        calibration["confidence"] += confidence
        calibration["success"] += 1.0 if success else 0.0
        if success:
            stats.successes += 1
            stats.streak = max(0, stats.streak) + 1
            stats.best_streak = max(stats.best_streak, stats.streak)
            stats.last_outcome = True
            self.total_successes += 1
        else:
            stats.failures += 1
            stats.streak = min(0, stats.streak) - 1
            stats.last_outcome = False
            kind = self.classify_error(error or "")
            stats.errors[kind] = stats.errors.get(kind, 0) + 1
            stats.last_error = str(error or "")[:500]
        self.total_attempts += 1
        self.save()
        return {
            "task": key,
            "success": bool(success),
            "confidence": confidence,
            "error_kind": self.classify_error(error or "") if not success
            else "",
            "streak": stats.streak,
        }

    @staticmethod
    def classify_error(error: str) -> str:
        text = str(error or "").lower()
        if not text:
            return "unspecified"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "syntax" in text or "parse" in text:
            return "syntax"
        if "import" in text or "module" in text:
            return "dependency"
        if "verif" in text or "holdout" in text or "test" in text:
            return "verification"
        if "http" in text or "network" in text or "connection" in text:
            return "network"
        if "permission" in text or "denied" in text:
            return "permission"
        return "runtime"

    def success_rate(self, task: str, prior: float = 0.5,
                     strength: float = 2.0) -> float:
        stats = self.tasks.get(self._key(task))
        if stats is None or stats.attempts == 0:
            return float(prior)
        return (stats.successes + prior * strength) / \
            (stats.attempts + strength)

    def confidence(self, task: str) -> float:
        stats = self.tasks.get(self._key(task))
        if stats is None or stats.attempts == 0:
            return 0.5
        empirical = self.success_rate(task)
        mean_confidence = stats.confidence_sum / stats.attempts
        return max(0.0, min(1.0, 0.65 * empirical +
                              0.35 * mean_confidence))

    def should_attempt(self, task: str, threshold: float = 0.25,
                       explore: bool = True) -> dict:
        stats = self.tasks.get(self._key(task))
        if stats is None or stats.attempts == 0:
            return {
                "attempt": bool(explore),
                "reason": "novelty" if explore else "insufficient_evidence",
                "confidence": 0.5,
            }
        confidence = self.confidence(task)
        recent_failures = max(0, -stats.streak)
        if confidence >= threshold:
            return {"attempt": True, "reason": "expected_success",
                    "confidence": confidence}
        if explore and recent_failures < 3:
            return {"attempt": True, "reason": "exploration",
                    "confidence": confidence}
        return {"attempt": False, "reason": "low_confidence",
                "confidence": confidence}

    def event_budget(self, task: str, base: int = 3,
                     maximum: int = 12) -> int:
        stats = self.tasks.get(self._key(task))
        if stats is None or stats.attempts == 0:
            return max(1, min(int(maximum), int(base)))
        confidence = self.confidence(task)
        uncertainty = 1.0 - abs(confidence - 0.5) * 2.0
        failures = max(0, -stats.streak)
        budget = int(round(base + 3.0 * uncertainty - failures))
        return max(1, min(int(maximum), budget))

    def calibration(self) -> dict:
        total = 0
        brier = 0.0
        ece = 0.0
        for stats in self.tasks.values():
            for bucket, row in stats.calibration.items():
                count = int(row["count"])
                if not count:
                    continue
                confidence = float(row["confidence"]) / count
                accuracy = float(row["success"]) / count
                total += count
                ece += count * abs(accuracy - confidence)
        for stats in self.tasks.values():
            brier += stats.brier_sum
        if total == 0:
            return {"brier": 0.0, "ece": 0.0, "samples": 0}
        return {
            "brier": round(brier / total, 6),
            "ece": round(ece / total, 6),
            "samples": total,
        }

    def reflect(self, minimum_attempts: int = 2) -> dict:
        domain_rows: dict[str, dict] = {}
        weakest = []
        strongest = []
        for task, stats in self.tasks.items():
            if stats.attempts < minimum_attempts:
                continue
            row = domain_rows.setdefault(
                stats.domain, {"attempts": 0, "successes": 0})
            row["attempts"] += stats.attempts
            row["successes"] += stats.successes
            rate = stats.successes / stats.attempts
            if rate < 0.5:
                weakest.append((rate, task))
            elif rate >= 0.8:
                strongest.append((rate, task))
        weakest.sort()
        strongest.sort(reverse=True)
        calibration = self.calibration()
        recommendations = []
        if calibration["samples"] >= 5:
            if calibration["ece"] > 0.25:
                recommendations.append(
                    "confidence is poorly calibrated; lower estimates")
            elif calibration["ece"] < 0.1:
                recommendations.append(
                    "confidence is well calibrated; keep current gate")
        if weakest:
            recommendations.append(
                "practice: " + ", ".join(task for _, task in weakest[:3]))
        if strongest:
            recommendations.append(
                "reuse: " + ", ".join(task for _, task in strongest[:3]))
        return {
            "tasks_tracked": len(self.tasks),
            "total_attempts": self.total_attempts,
            "success_rate": round(
                self.total_successes / max(1, self.total_attempts), 6),
            "calibration": calibration,
            "domains": {
                name: {
                    **row,
                    "success_rate": round(
                        row["successes"] / max(1, row["attempts"]), 6),
                }
                for name, row in sorted(domain_rows.items())
            },
            "weakest": [task for _, task in weakest[:6]],
            "strongest": [task for _, task in strongest[:6]],
            "recommendations": recommendations,
        }

    def prioritise(self, candidates: list[str]) -> list[dict]:
        ranked = []
        for task in candidates:
            stats = self.tasks.get(self._key(task))
            attempts = stats.attempts if stats else 0
            rate = self.success_rate(task)
            novelty = 1.0 / (1.0 + attempts)
            uncertainty = 1.0 - abs(rate - 0.5) * 2.0
            ranked.append({
                "task": task,
                "priority": round(0.45 * uncertainty + 0.4 * novelty
                                  + 0.15 * (1.0 - rate), 6),
                "attempts": attempts,
                "success_rate": round(rate, 6),
            })
        ranked.sort(key=lambda item: item["priority"], reverse=True)
        return ranked

    def propose_curriculum(self, candidates: list[str],
                           limit: int = 8) -> list[dict]:
        rows = self.prioritise(candidates)[:max(1, int(limit))]
        for row in rows:
            competence = self.success_rate(row["task"])
            row["difficulty"] = round(1.0 - competence, 6)
            row["reason"] = ("novel" if row["attempts"] == 0
                             else "weak" if competence < 0.5
                             else "consolidate")
        return rows

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "tasks": {key: stats.to_dict()
                      for key, stats in self.tasks.items()},
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        self.total_attempts = int(data.get("total_attempts", 0))
        self.total_successes = int(data.get("total_successes", 0))
        for key, row in (data.get("tasks") or {}).items():
            if isinstance(row, dict):
                self.tasks[str(key)] = TaskStats.from_dict(row)

    def stats(self) -> dict:
        return {
            "tasks": len(self.tasks),
            "attempts": self.total_attempts,
            "successes": self.total_successes,
            "success_rate": round(
                self.total_successes / max(1, self.total_attempts), 6),
            "calibration": self.calibration(),
        }
