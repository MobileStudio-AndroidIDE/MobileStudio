"""Curiosity as expected learning progress, not constant novelty seeking."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

from cognition.util import atomic_write_json, read_json


@dataclass
class Interest:
    attempts: int = 0
    successes: int = 0
    surprise_ema: float = 0.5
    progress_ema: float = 0.0
    last_surprise: float = 0.5
    domain: str = "general"

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "surprise_ema": self.surprise_ema,
            "progress_ema": self.progress_ema,
            "last_surprise": self.last_surprise,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Interest":
        return cls(
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
            surprise_ema=float(data.get("surprise_ema", 0.5)),
            progress_ema=float(data.get("progress_ema", 0.0)),
            last_surprise=float(data.get("last_surprise", 0.5)),
            domain=str(data.get("domain", "general")),
        )


class CuriosityEngine:
    def __init__(self, data_dir: str, seed: int = 23):
        self.data_dir = str(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.path = os.path.join(self.data_dir, "curiosity.json")
        self.rng = random.Random(int(seed))
        self.interests: dict[str, Interest] = {}
        self._load()

    def competence(self, task: str, prior: float = 0.5,
                   strength: float = 2.0) -> float:
        interest = self.interests.get(str(task))
        if interest is None or interest.attempts == 0:
            return float(prior)
        return (interest.successes + prior * strength) / \
            (interest.attempts + strength)

    def surprise(self, task: str, default: float = 0.5) -> float:
        interest = self.interests.get(str(task))
        return float(default) if interest is None else interest.surprise_ema

    def novelty(self, task: str) -> float:
        interest = self.interests.get(str(task))
        attempts = interest.attempts if interest else 0
        return 1.0 / (1.0 + attempts) ** 0.75

    def score(self, task: str, predicted_surprise: float = 0.5,
              domain: str = "") -> dict:
        interest = self.interests.get(str(task))
        attempts = interest.attempts if interest else 0
        competence = self.competence(task)
        prior_surprise = self.surprise(task, predicted_surprise)
        expected_surprise = 0.65 * prior_surprise + \
            0.35 * max(0.0, min(1.0, float(predicted_surprise)))
        progress = interest.progress_ema if interest else 0.0
        uncertainty = 1.0 - abs(competence - 0.5) * 2.0
        value = (0.35 * expected_surprise + 0.25 * self.novelty(task)
                 + 0.25 * uncertainty + 0.15 * progress)
        return {
            "task": str(task),
            "domain": domain or (interest.domain if interest else "general"),
            "score": round(float(value), 6),
            "attempts": attempts,
            "competence": round(float(competence), 6),
            "expected_surprise": round(float(expected_surprise), 6),
            "novelty": round(float(self.novelty(task)), 6),
            "progress": round(float(progress), 6),
        }

    def observe(self, task: str, surprise: float, success: bool,
                domain: str = "") -> dict:
        key = str(task)
        interest = self.interests.setdefault(key, Interest(domain=domain))
        if domain:
            interest.domain = str(domain)
        before = interest.surprise_ema
        surprise = max(0.0, float(surprise))
        interest.attempts += 1
        interest.successes += 1 if success else 0
        interest.last_surprise = surprise
        interest.surprise_ema = 0.8 * interest.surprise_ema + 0.2 * surprise
        delta = max(-1.0, min(1.0, before - surprise))
        interest.progress_ema = 0.8 * interest.progress_ema + 0.2 * delta
        self.save()
        return {
            "task": key,
            "learning_progress": round(delta, 6),
            "surprise": round(surprise, 6),
            "competence": round(self.competence(key), 6),
        }

    def choose(self, candidates: Iterable, top_k: int = 1) -> list[dict]:
        scored = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                task = candidate.get("task") or candidate.get("prompt") \
                    or candidate.get("name") or str(candidate)
                domain = candidate.get("domain", "")
                predicted = float(candidate.get(
                    "predicted_surprise",
                    candidate.get("difficulty", 0.5)))
            else:
                task, domain, predicted = str(candidate), "", 0.5
            row = self.score(task, predicted, domain)
            row["candidate"] = candidate
            scored.append(row)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:max(1, int(top_k))]

    def forget(self, task: str) -> bool:
        removed = self.interests.pop(str(task), None) is not None
        if removed:
            self.save()
        return removed

    def micro_tasks(self, domain: str = "mixed",
                    count: int = 5) -> list[dict]:
        generators = {
            "math": self._math_tasks,
            "sequence": self._sequence_tasks,
            "logic": self._logic_tasks,
            "code": self._code_tasks,
            "graph": self._graph_tasks,
            "text": self._text_tasks,
        }
        if domain == "mixed":
            names = list(generators)
            self.rng.shuffle(names)
        else:
            names = [domain] if domain in generators else list(generators)
        tasks = []
        for index in range(max(0, int(count))):
            generator = generators[names[index % len(names)]]
            tasks.extend(generator(1))
        return tasks[:max(0, int(count))]

    def _math_tasks(self, count: int) -> list[dict]:
        out = []
        for _ in range(count):
            a = self.rng.randint(1, 9)
            b = self.rng.randint(-9, 9)
            x = self.rng.randint(-6, 6)
            if self.rng.random() < 0.5:
                prompt = f"solve {a}*x+{b}={a * x + b}"
                expected = x
            else:
                prompt = f"differentiate {a}*x^2+{b}*x"
                expected = f"{2 * a}*x+{b}"
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "math",
                "expected": expected,
                "difficulty": round(self.rng.uniform(0.2, 0.8), 4),
            })
        return out

    def _sequence_tasks(self, count: int) -> list[dict]:
        out = []
        for _ in range(count):
            start = self.rng.randint(1, 9)
            step = self.rng.randint(1, 6)
            values = [start + step * index for index in range(5)]
            prompt = "next in sequence: " + ", ".join(map(str, values))
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "sequence",
                "expected": start + step * 5,
                "difficulty": round(self.rng.uniform(0.1, 0.5), 4),
            })
        return out

    def _logic_tasks(self, count: int) -> list[dict]:
        names = ["A", "B", "C", "D"]
        self.rng.shuffle(names)
        out = []
        for _ in range(count):
            first, second = names[:2]
            prompt = (f"if {first} implies {second}; {first} is true; "
                      f"what follows?")
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "logic",
                "expected": second,
                "difficulty": round(self.rng.uniform(0.1, 0.4), 4),
            })
        return out

    def _code_tasks(self, count: int) -> list[dict]:
        out = []
        for _ in range(count):
            number = self.rng.randint(2, 6)
            prompt = f"write a function that returns the square of {number}"
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "code",
                "expected": number * number,
                "difficulty": round(self.rng.uniform(0.3, 0.7), 4),
            })
        return out

    def _graph_tasks(self, count: int) -> list[dict]:
        out = []
        for _ in range(count):
            edges = []
            names = ["n0", "n1", "n2"]
            for index in range(2):
                edges.append([names[index], names[index + 1]])
            prompt = "path from n0 to n2 in " + str(edges)
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "graph",
                "expected": ["n0", "n1", "n2"],
                "difficulty": round(self.rng.uniform(0.1, 0.4), 4),
            })
        return out

    def _text_tasks(self, count: int) -> list[dict]:
        pairs = [("king", "queen"), ("man", "woman"),
                 ("cat", "kitten"), ("dog", "puppy")]
        out = []
        for _ in range(count):
            left, right = self.rng.choice(pairs)
            prompt = f"analogy: {left} is to {right} as father is to ?"
            out.append({
                "task": prompt,
                "prompt": prompt,
                "domain": "text",
                "expected": "mother",
                "difficulty": round(self.rng.uniform(0.2, 0.6), 4),
            })
        return out

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "interests": {key: value.to_dict()
                          for key, value in self.interests.items()},
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for key, row in (data.get("interests") or {}).items():
            if isinstance(row, dict):
                self.interests[str(key)] = Interest.from_dict(row)

    def stats(self) -> dict:
        if not self.interests:
            return {"interests": 0, "mean_surprise": 0.0,
                    "mean_progress": 0.0}
        return {
            "interests": len(self.interests),
            "mean_surprise": round(
                sum(row.surprise_ema for row in self.interests.values())
                / len(self.interests), 6),
            "mean_progress": round(
                sum(row.progress_ema for row in self.interests.values())
                / len(self.interests), 6),
        }
