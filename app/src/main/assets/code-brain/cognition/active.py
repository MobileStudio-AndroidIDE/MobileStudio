"""Active inference over concept-space beliefs and learned transitions."""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Optional

import numpy as np

from cognition.concept_space import ConceptSpace
from cognition.predictive import PredictiveEngine
from cognition.vsa import VSA


class ActiveInference:
    def __init__(self, vsa: VSA, concepts: ConceptSpace,
                 predictive: Optional[PredictiveEngine] = None,
                 temperature: float = 0.12):
        self.vsa = vsa
        self.concepts = concepts
        self.predictive = predictive
        self.temperature = max(1e-4, float(temperature))
        self.beliefs: dict[str, float] = {}
        self.goal: Optional[str] = None
        self.history: deque[dict] = deque(maxlen=256)
        self.free_energy_history: deque[float] = deque(maxlen=256)

    @staticmethod
    def _normalize(distribution: dict) -> dict:
        total = sum(max(0.0, float(v)) for v in distribution.values())
        if total <= 1e-12:
            count = max(1, len(distribution))
            return {k: 1.0 / count for k in distribution}
        return {k: max(0.0, float(v)) / total
                for k, v in distribution.items()}

    @staticmethod
    def entropy(distribution: dict) -> float:
        values = [float(v) for v in distribution.values() if float(v) > 0]
        return -sum(v * math.log2(v) for v in values)

    def likelihood(self, observation, labels: Optional[Iterable[str]] = None
                   ) -> dict[str, float]:
        if isinstance(observation, str):
            if observation in self.concepts.concepts and \
                    self.concepts.get(observation) is not None:
                observation = self.concepts.prototype(observation)
            else:
                observation = self.vsa.symbol("active-observation:" +
                                              observation)
        vector = np.asarray(observation, dtype=np.float32)
        names = list(labels) if labels is not None else \
            list(self.concepts.concepts)
        if not names:
            return {}
        similarities = np.array([
            self.vsa.similarity(vector, self.concepts.prototype(name))
            for name in names], dtype=np.float32)
        scaled = similarities / self.temperature
        scaled -= float(scaled.max())
        weights = np.exp(scaled)
        total = float(weights.sum())
        if total <= 1e-12:
            return {name: 1.0 / len(names) for name in names}
        return {name: float(weight / total)
                for name, weight in zip(names, weights)}

    def observe(self, state: str, observation=None,
                learn: bool = False) -> dict:
        state = str(state)
        if learn:
            vector = observation if observation is not None else \
                self.concepts.prototype(state)
            self.concepts.add(state, vector, "active")
        if observation is None:
            observation = self.concepts.prototype(state)
        observed = self.likelihood(observation)
        prior = self.beliefs or observed or {state: 1.0}
        names = set(prior) | set(observed)
        posterior = {
            name: float(prior.get(name, 1.0 / max(1, len(names))))
            * float(observed.get(name, 0.0))
            for name in names
        }
        posterior = self._normalize(posterior)
        surprise = -math.log2(max(float(observed.get(state, 0.0)), 1e-9))
        cross_entropy = -sum(
            posterior.get(name, 0.0)
            * math.log2(max(float(observed.get(name, 0.0)), 1e-9))
            for name in posterior)
        free_energy = cross_entropy - self.entropy(posterior)
        self.beliefs = posterior
        record = {
            "state": state,
            "surprise": round(surprise, 6),
            "free_energy": round(free_energy, 6),
            "entropy": round(self.entropy(posterior), 6),
            "prediction_error": round(1.0
                                      - float(observed.get(state, 0.0)), 6),
            "confidence": round(float(posterior.get(state, 0.0)), 6),
        }
        self.history.append(record)
        self.free_energy_history.append(free_energy)
        if self.predictive is not None:
            self.predictive.observe_sequence([state])
        return record

    def predict(self, context, top_k: int = 3) -> dict:
        if self.predictive is None:
            return {"symbol": "", "confidence": 0.0,
                    "distribution": {}, "surprise": 1.0}
        prediction = self.predictive.predict(context, top_k=top_k)
        if prediction["symbol"] and prediction["symbol"] not in \
                self.concepts.concepts:
            self.concepts.add(prediction["symbol"],
                              self.vsa.symbol("predicted:" +
                                              prediction["symbol"]),
                              "predicted")
        return prediction

    def expected_free_energy(self, candidate, goal: Optional[str] = None,
                             epistemic: float = 0.5,
                             pragmatic: float = 1.0) -> dict:
        goal = goal or self.goal
        scores = self.likelihood(candidate)
        if not scores:
            return {"efe": 1.0, "risk": 1.0, "ambiguity": 0.0,
                    "confidence": 0.0}
        distribution = self._normalize(scores)
        confidence = max(distribution.values())
        ambiguity = self.entropy(distribution)
        if goal and goal in self.concepts.concepts:
            vector = candidate if not isinstance(candidate, str) else \
                self.concepts.prototype(goal)
            target = self.concepts.prototype(goal)
            risk = max(0.0, 1.0 - self.vsa.similarity(vector, target))
        else:
            risk = 1.0 - confidence
        efe = pragmatic * risk - epistemic * ambiguity
        return {
            "efe": round(efe, 6),
            "risk": round(risk, 6),
            "ambiguity": round(ambiguity, 6),
            "confidence": round(confidence, 6),
        }

    def rank_actions(self, actions: Iterable, goal: Optional[str] = None,
                     epistemic: float = 0.5,
                     pragmatic: float = 1.0) -> list[dict]:
        ranked = []
        for action in actions:
            if isinstance(action, dict):
                name = action.get("name", action.get("action", str(action)))
                vector = action.get("vector")
            else:
                name, vector = str(action), None
            if vector is None:
                vector = self.concepts.prototype(name) \
                    if name in self.concepts.concepts else \
                    self.vsa.symbol("action:" + name)
            score = self.expected_free_energy(
                vector, goal=goal, epistemic=epistemic,
                pragmatic=pragmatic)
            score["action"] = name
            ranked.append(score)
        ranked.sort(key=lambda item: item["efe"])
        return ranked

    def set_goal(self, goal: Optional[str]) -> None:
        self.goal = str(goal) if goal is not None else None

    def progress(self) -> float:
        if not self.goal or not self.beliefs:
            return 0.0
        return float(self.beliefs.get(self.goal, 0.0))

    def stats(self) -> dict:
        mean_fe = float(np.mean(self.free_energy_history)) \
            if self.free_energy_history else 0.0
        mean_surprise = float(np.mean(
            [item["surprise"] for item in self.history])) \
            if self.history else 0.0
        return {
            "beliefs": len(self.beliefs),
            "goal": self.goal,
            "goal_belief": round(self.progress(), 6),
            "mean_free_energy": round(mean_fe, 6),
            "mean_surprise": round(mean_surprise, 6),
            "observations": len(self.history),
        }
