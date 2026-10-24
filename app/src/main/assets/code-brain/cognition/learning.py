"""Developmental learning: replay, contrastive association, meta-strategy.

No gradients are used.  Replay re-observes stored experiences, contrastive
association binds context to outcome, and a small UCB bandit learns which
strategy works for which kind of task.  Stages are earned from measured
experience, not declared in advance.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cognition.util import atomic_write_json, read_json
from cognition.vsa import VSA


@dataclass
class Experience:
    context: np.ndarray
    action: str
    outcome: np.ndarray
    task: str = ""
    success: bool = False
    reward: float = 0.0
    surprise: float = 0.0
    metadata: dict = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int = 512, seed: int = 17):
        self.capacity = max(1, int(capacity))
        self.items: list[Experience] = []
        self.rng = random.Random(int(seed))
        self.replays = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, experience: Experience) -> None:
        self.items.append(experience)
        if len(self.items) > self.capacity:
            del self.items[0:len(self.items) - self.capacity]

    def sample(self, count: int = 8, recent_bias: float = 0.5
               ) -> list[Experience]:
        if not self.items:
            return []
        count = max(1, min(int(count), len(self.items)))
        if self.rng.random() < recent_bias:
            pool = self.items[-max(count, len(self.items) // 2):]
        else:
            pool = self.items
        return self.rng.sample(pool, count)

    def stats(self) -> dict:
        if not self.items:
            return {"size": 0, "success_rate": 0.0, "mean_surprise": 0.0}
        return {
            "size": len(self.items),
            "success_rate": round(
                sum(1 for item in self.items if item.success)
                / len(self.items), 6),
            "mean_surprise": round(
                sum(item.surprise for item in self.items)
                / len(self.items), 6),
            "replays": self.replays,
        }


class ContrastiveMemory:
    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.association = np.zeros(vsa.dimension, dtype=np.float32)
        self.targets: dict[str, np.ndarray] = {}
        self.pairs = 0
        self.margin_hits = 0

    def learn(self, context, target, label: str = "") -> dict:
        context = self.vsa.normalize(np.asarray(context, dtype=np.float32))
        target = self.vsa.normalize(np.asarray(target, dtype=np.float32))
        bound = self.vsa.bind(context, target)
        if self.pairs == 0:
            self.association = bound
        else:
            self.association = self.vsa.bundle(
                [self.association, bound], weights=[0.9, 0.1])
        self.pairs += 1
        if not label:
            label = "target:%d" % len(self.targets)
        self.targets[str(label)] = target
        return {"pairs": self.pairs, "label": str(label)}

    def retrieve(self, context, k: int = 3) -> list[tuple[str, float]]:
        if not self.targets or self.pairs == 0:
            return []
        context = self.vsa.normalize(np.asarray(context, dtype=np.float32))
        probe = self.vsa.unbind(self.association, context)
        scored = [(label, self.vsa.similarity(probe, target))
                  for label, target in self.targets.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max(1, int(k))]

    def contrastive_loss(self, context, positive, negatives) -> float:
        context = np.asarray(context, dtype=np.float32)
        positive = np.asarray(positive, dtype=np.float32)
        pos = self.vsa.similarity(
            self.vsa.bind(context, positive), self.association)
        losses = [-math.log(max(1e-9, (pos + 1.0) / 2.0))]
        for negative in negatives:
            negative = np.asarray(negative, dtype=np.float32)
            neg = self.vsa.similarity(
                self.vsa.bind(context, negative), self.association)
            losses.append(-math.log(max(1e-9, (1.0 - neg) / 2.0)))
        return float(np.mean(losses))

    def stats(self) -> dict:
        return {"pairs": self.pairs, "targets": len(self.targets)}


class StrategySelector:
    def __init__(self):
        self.strategies: dict[str, dict] = {}

    def ensure(self, name: str) -> dict:
        return self.strategies.setdefault(
            str(name), {"pulls": 0, "successes": 0, "reward": 0.0})

    def choose(self, available, exploration: float = 1.2) -> str:
        available = [str(name) for name in available]
        if not available:
            return ""
        total = max(1, sum(row["pulls"] for row in self.strategies.values()))
        best_name = available[0]
        best_score = -float("inf")
        for name in available:
            row = self.ensure(name)
            if row["pulls"] == 0:
                return name
            mean = row["reward"] / row["pulls"]
            bonus = exploration * math.sqrt(
                math.log(total + 1) / row["pulls"])
            score = mean + bonus
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    def update(self, name: str, reward: float, success: bool) -> dict:
        row = self.ensure(name)
        row["pulls"] += 1
        row["successes"] += 1 if success else 0
        row["reward"] += float(reward)
        return dict(row)

    def stats(self) -> dict:
        return {
            name: {
                **row,
                "success_rate": round(
                    row["successes"] / max(1, row["pulls"]), 6),
                "mean_reward": round(
                    row["reward"] / max(1, row["pulls"]), 6),
            }
            for name, row in sorted(self.strategies.items())
        }


class DevelopmentalLearner:
    STAGES = ("sensorimotor", "relational", "symbolic", "metacognitive")

    def __init__(self, data_dir: str, vsa: VSA,
                 replay_capacity: int = 512, seed: int = 17):
        self.data_dir = str(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.vsa = vsa
        self.buffer = ReplayBuffer(replay_capacity, seed)
        self.contrastive = ContrastiveMemory(vsa)
        self.strategies = StrategySelector()
        self.stage = 0
        self.experiences = 0
        self.successes = 0
        self.consolidations = 0
        self.path = os.path.join(self.data_dir, "learning.json")
        self.state_path = os.path.join(self.data_dir, "learning_state.npz")
        self._load()

    def strategy_for(self, task: str, available) -> str:
        return self.strategies.choose(available)

    def record_strategy(self, name: str, reward: float,
                        success: bool) -> dict:
        return self.strategies.update(name, reward, success)

    def learn(self, context, action: str, outcome, task: str = "",
              success: bool = False, reward: Optional[float] = None,
              surprise: float = 0.0, metadata: Optional[dict] = None,
              label: str = "") -> dict:
        context = np.asarray(context, dtype=np.float32)
        outcome = np.asarray(outcome, dtype=np.float32)
        if reward is None:
            reward = 1.0 if success else -0.25
        experience = Experience(
            context=context, action=str(action), outcome=outcome,
            task=str(task), success=bool(success), reward=float(reward),
            surprise=float(surprise), metadata=dict(metadata or {}))
        self.buffer.add(experience)
        self.contrastive.learn(context, outcome, label or action)
        self.experiences += 1
        if success:
            self.successes += 1
        if success and len(self.buffer) >= 4 and \
                self.experiences % 4 == 0:
            self.consolidate()
        self._advance_stage()
        self.save()
        return {
            "experiences": self.experiences,
            "stage": self.STAGES[self.stage],
            "replay": self.buffer.stats(),
        }

    def consolidate(self, count: int = 8) -> dict:
        sample = self.buffer.sample(count, recent_bias=0.35)
        for experience in sample:
            self.contrastive.learn(
                experience.context, experience.outcome,
                experience.action)
            self.buffer.replays += 1
        self.consolidations += 1
        return {
            "consolidations": self.consolidations,
            "replayed": len(sample),
            "contrastive": self.contrastive.stats(),
        }

    def recall(self, context, k: int = 3) -> list[tuple[str, float]]:
        return self.contrastive.retrieve(context, k=k)

    def _advance_stage(self) -> None:
        success_rate = self.successes / max(1, self.experiences)
        if self.stage == 0 and self.experiences >= 10:
            self.stage = 1
        if self.stage == 1 and self.experiences >= 30 and \
                success_rate >= 0.5:
            self.stage = 2
        if self.stage == 2 and self.experiences >= 60 and \
                success_rate >= 0.6 and self.consolidations >= 3:
            self.stage = 3

    def stats(self) -> dict:
        return {
            "stage": self.STAGES[self.stage],
            "experiences": self.experiences,
            "success_rate": round(
                self.successes / max(1, self.experiences), 6),
            "replay": self.buffer.stats(),
            "contrastive": self.contrastive.stats(),
            "strategies": self.strategies.stats(),
            "consolidations": self.consolidations,
        }

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "stage": self.stage,
            "experiences": self.experiences,
            "successes": self.successes,
            "consolidations": self.consolidations,
            "strategies": self.strategies.strategies,
        })
        labels = list(self.contrastive.targets)
        targets = np.stack([self.contrastive.targets[label]
                            for label in labels]).astype(np.float32) \
            if labels else np.zeros((0, self.vsa.dimension),
                                    dtype=np.float32)
        np.savez_compressed(
            self.state_path,
            association=self.contrastive.association.astype(np.float32),
            labels=np.array(labels, dtype=object),
            targets=targets,
            pairs=np.array(self.contrastive.pairs, dtype=np.int64))

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        self.stage = max(0, min(len(self.STAGES) - 1,
                                int(data.get("stage", 0))))
        self.experiences = int(data.get("experiences", 0))
        self.successes = int(data.get("successes", 0))
        self.consolidations = int(data.get("consolidations", 0))
        for name, row in (data.get("strategies") or {}).items():
            if isinstance(row, dict):
                self.strategies.strategies[str(name)] = {
                    "pulls": int(row.get("pulls", 0)),
                    "successes": int(row.get("successes", 0)),
                    "reward": float(row.get("reward", 0.0)),
                }
        if not os.path.exists(self.state_path):
            return
        try:
            with np.load(self.state_path, allow_pickle=True) as state:
                association = np.asarray(state["association"],
                                         dtype=np.float32)
                if association.shape == (self.vsa.dimension,):
                    self.contrastive.association = association
                labels = list(state["labels"])
                targets = state["targets"]
                for index, label in enumerate(labels):
                    self.contrastive.targets[str(label)] = np.asarray(
                        targets[index], dtype=np.float32)
                self.contrastive.pairs = int(state["pairs"])
        except Exception:
            return
