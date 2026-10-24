"""RenaBrain: one orchestrator joining perception, concepts and action."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from cognition.active import ActiveInference
from cognition.actuators import ActuatorRegistry, ActuatorResult
from cognition.concept_space import ConceptSpace
from cognition.config import GrowthConfig
from cognition.curiosity import CuriosityEngine
from cognition.learning import DevelopmentalLearner
from cognition.memory import (ConceptMemory, Consolidator, EpisodicMemory,
                              WorkingMemory)
from cognition.metacognition import MetaCognition
from cognition.predictive import PredictiveEngine
from cognition.semantic import SemanticEncoder
from cognition.sensors import SensorForge, SensorRegistry
from cognition.util import json_safe, stable_hash
from cognition.vsa import VSA


class RenaBrain:
    def __init__(self, config: Optional[GrowthConfig] = None,
                 brain=None, load: bool = True):
        self.config = config or GrowthConfig()
        self.config.ensure_dirs()
        self.vsa = VSA(self.config.dimension, self.config.vector_seed)
        self.encoder = SemanticEncoder(self.vsa)
        self.concept_path = os.path.join(self.config.data_dir,
                                         "concept_space.npz")
        self.concepts = ConceptSpace(self.vsa)
        if load and os.path.exists(self.concept_path):
            self.concepts = ConceptSpace.load(self.concept_path, self.vsa)
        self.semantic = ConceptMemory(
            os.path.join(self.config.data_dir, "concepts"),
            self.vsa, self.config)
        self.working = WorkingMemory(self.vsa,
                                     self.config.working_capacity)
        self.episodes = EpisodicMemory(
            os.path.join(self.config.data_dir, "episodes"),
            self.config.episode_capacity)
        self.predictive = PredictiveEngine(
            os.path.join(self.config.data_dir, "predictive"), self.vsa)
        self.active = ActiveInference(self.vsa, self.concepts,
                                      self.predictive)
        self.meta = MetaCognition(self.config.data_dir)
        self.curiosity = CuriosityEngine(self.config.data_dir,
                                         seed=self.config.seed)
        self.learning = DevelopmentalLearner(
            self.config.data_dir, self.vsa,
            replay_capacity=min(512, self.config.episode_capacity),
            seed=self.config.seed)
        self.actuators = ActuatorRegistry(self.config, brain=brain)
        self.sensors = SensorRegistry(self.config.data_dir)
        self.consolidator = Consolidator(
            self.semantic, self.episodes, self.working, self.predictive,
            self.config)
        self.brain = brain
        self._last_vector = self.vsa.symbol("rena:start")
        self._growth = None
        self._augmentation = None
        self.history: list[dict] = []

    def feedback(self, value: float, episode_id: Optional[str] = None,
                 note: str = "") -> bool:
        """Record a prediction-error signal for the last turn.

        Positive reward reinforces the last context->action association;
        negative reward weakens it.  No backpropagation — only Hebbian /
        contrastive VSA updates through the developmental learner.
        """
        try:
            context = self._last_vector
            task = "turn:" + (note or str(episode_id or "unknown"))
            outcome = self.vsa.symbol("feedback:" + str(
                1.0 if value > 0 else -1.0))
            self.learning.learn(
                context, task, outcome, task=task, success=value > 0,
                reward=float(value), surprise=0.2 if value > 0 else 0.8,
                label=note)
            self.meta.record(
                task, value > 0, confidence=abs(float(value)),
                domain="turn", error=note if value < 0 else "",
                surprise=0.2 if value > 0 else 0.8)
            return True
        except Exception:
            return False

    def observe(self, modality: str, data, label: str = "",
                **kwargs) -> dict:
        vector = self.encoder.encode(modality, data, **kwargs)
        label = label or self.auto_label(modality, data)
        self.concepts.add(label, vector, modality)
        self.semantic.add(
            label, kind="perception", source="rena",
            attributes={"modality": modality, "auto": not bool(label)})
        self.working.add(label, kind=modality, vector=vector)
        active = self.active.observe(label, vector)
        tokens = []
        if str(modality).lower() in ("text", "language", "chat"):
            tokens = self.encoder.tokenize(str(data))
            if tokens:
                self.predictive.observe_sequence(tokens)
        self.episodes.record(
            domain="perception", action=label,
            outcome={"modality": modality, "label": label},
            success=True, surprise=active["surprise"],
            context={"subject": label, "relation": "observed_as",
                     "modality": modality, "symbol_sequence": tokens})
        learning = self.learning.learn(
            self._last_vector, "observe", vector,
            task="perception:" + str(modality), success=True,
            reward=0.1, surprise=active["surprise"], label=label)
        self._last_vector = vector
        record = {
            "modality": str(modality),
            "label": label,
            "active": active,
            "learning": learning,
            "concepts": self.concepts.stats(),
        }
        self.history.append({"event": "observe", **record})
        return record

    def think(self, query, modality: str = "text", **kwargs) -> dict:
        data = query
        if isinstance(query, dict):
            modality = str(query.get("modality", modality))
            data = query.get("data", query.get("query", ""))
        vector = self.encoder.encode(modality, data, **kwargs)
        concept_rank = self.concepts.classify(vector, k=5)
        prediction = {}
        if str(modality).lower() in ("text", "language", "chat"):
            tokens = self.encoder.tokenize(str(data))
            if tokens:
                prediction = self.predictive.predict(tokens[-3:], top_k=5)
        likelihood = self.active.likelihood(vector)
        active_rank = sorted(likelihood.items(), key=lambda item: -item[1])
        confidence = 0.0
        if concept_rank:
            confidence = max(0.0, min(1.0, (concept_rank[0][1] + 1.0) / 2.0))
        best = concept_rank[0][0] if concept_rank else ""
        best_score = concept_rank[0][1] if concept_rank else 0.0
        response = ""
        if best and best_score >= 0.2:
            response = "closest concept: %s (%.2f)" % (best, best_score)
        elif prediction.get("symbol"):
            response = "predicted next: %s" % prediction["symbol"]
        else:
            response = "no confident concept match"
        self.working.add(str(data)[:200], kind="query", vector=vector)
        self.active.beliefs = {
            label: score for label, score in active_rank[:10]}
        result = {
            "modality": str(modality),
            "concept": best,
            "concept_score": round(float(best_score), 6),
            "concept_rank": [(label, round(float(score), 6))
                             for label, score in concept_rank],
            "prediction": prediction,
            "active": {
                "beliefs": {label: round(float(score), 6)
                            for label, score in active_rank[:10]},
                "free_energy": self.active.stats()["mean_free_energy"],
            },
            "confidence": round(confidence, 6),
            "response": response,
        }
        self.history.append({"event": "think", **result})
        return result

    def act(self, name: str, request, **kwargs) -> dict:
        operation = request.get("operation", "") \
            if isinstance(request, dict) else "execute"
        task = "%s:%s" % (name, operation or "execute")
        confidence = self.meta.confidence(task)
        decision = self.meta.should_attempt(task)
        if not decision["attempt"] and not kwargs.pop("force", False):
            return {
                "ok": False,
                "actuator": name,
                "operation": operation,
                "error": "metacognitive_refusal: " + decision["reason"],
                "confidence": confidence,
            }
        started = time.time()
        result: ActuatorResult = self.actuators.execute(
            name, request, **kwargs)
        latency = time.time() - started
        self.meta.record(
            task, result.ok, confidence=confidence, latency=latency,
            domain=name, error=result.error,
            surprise=0.2 if result.ok else 0.8)
        self.curiosity.observe(
            task, 0.2 if result.ok else 0.8, result.ok, domain=name)
        context = self.encoder.encode_text(str(request))
        outcome = self.encoder.encode_text(str(result.output))
        self.learning.learn(
            context, task, outcome, task=task, success=result.ok,
            reward=1.0 if result.ok else -0.5,
            surprise=0.2 if result.ok else 0.8, label=task)
        self.episodes.record(
            domain=name, action=task,
            outcome={"ok": result.ok, "artifacts": result.artifacts,
                     "error": result.error},
            success=result.ok, reward=1.0 if result.ok else -0.5,
            surprise=0.2 if result.ok else 0.8,
            context={"subject": name, "relation": "acted_with",
                     "skill": task, "failure": result.error})
        result_record = result.to_dict()
        result_record["task"] = task
        result_record["metacognition"] = self.meta.stats()
        self.history.append({"event": "act", **result_record})
        return result_record

    def _ensure_growth(self, domains: Optional[list] = None):
        from cognition.growth import GrowthEngine
        if self._growth is None:
            self._growth = GrowthEngine(self.config, domains=domains)
        return self._growth

    def grow(self, cycles: Optional[int] = None,
             domains: Optional[list] = None) -> dict:
        growth = self._ensure_growth(domains=domains)
        report = growth.grow(cycles=cycles)
        data = report.to_dict()
        self.history.append({"event": "grow", "report": data})
        return data

    def augment(self, cycles: int = 1, candidates: Optional[int] = None,
                trials: Optional[int] = None,
                time_budget: Optional[float] = None,
                domain: str = "math") -> dict:
        from cognition.augmentation import SelfAugmentationEngine
        growth = self._ensure_growth()
        if trials is not None:
            self.config.augmentation_trials = max(1, int(trials))
        engine = SelfAugmentationEngine(
            growth, self.config, seed=self.config.seed, domain=domain)
        report = engine.run(
            cycles=cycles,
            candidates_per_cycle=(self.config.augmentation_candidates
                                  if candidates is None else candidates),
            time_budget=(self.config.augmentation_seconds
                         if time_budget is None else time_budget))
        self._augmentation = engine
        data = report.to_dict()
        self.history.append({"event": "augment", "report": data})
        return data

    def invent_primitives(self, cycles: int = 1, candidates: int = 6,
                          time_budget: Optional[float] = None) -> dict:
        from cognition.invention import PrimitiveInventor
        growth = self._ensure_growth()
        inventor = PrimitiveInventor(
            growth, self.config, seed=self.config.seed)
        report = inventor.run(
            cycles=cycles, candidates=candidates,
            time_budget=(self.config.invention_seconds
                         if time_budget is None else time_budget))
        data = report.to_dict()
        self.history.append({"event": "invent_primitives",
                             "report": data})
        return data

    def invent_sensor(self, modality: str, samples: list,
                      cycles: int = 1, candidates: int = 12,
                      time_budget: Optional[float] = None) -> dict:
        forge = SensorForge(
            self.config, self.sensors, seed=self.config.seed)
        report = forge.run(
            modality=modality, samples=samples, cycles=cycles,
            candidates=candidates, time_budget=time_budget)
        data = report.to_dict()
        self.history.append({"event": "invent_sensor", "report": data})
        return data

    def read_sensor(self, name: str, source) -> dict:
        return self.sensors.read(name, source)

    def observe_sensor(self, name: str, source,
                       label: str = "") -> dict:
        reading = self.sensors.read(name, source)
        automatic = not bool(label)
        label = label or ("sensor:" + name + ":" +
                          stable_hash(reading["features"], length=8))
        features = {"feature:%d" % index: value
                    for index, value in enumerate(reading["features"])}
        vector = self.vsa.encode_features(features)
        self.concepts.add(label, vector, "sensor:" + name)
        self.semantic.add(
            label, kind="perception", source="sensor_forge",
            attributes={"modality": "sensor:" + name,
                        "sensor": name, "auto": automatic})
        self.working.add(label, kind="sensor:" + name, vector=vector)
        active = self.active.observe(label, vector)
        record = {
            "sensor": name,
            "label": label,
            "reading": reading,
            "active": active,
            "concepts": self.concepts.stats(),
        }
        self.history.append({"event": "observe_sensor", **record})
        return record

    def propose_curriculum(self, count: int = 6,
                           domain: str = "mixed") -> list[dict]:
        tasks = self.curiosity.micro_tasks(domain, count=count)
        candidates = [task.get("task", task.get("prompt", ""))
                      for task in tasks]
        return self.meta.propose_curriculum(candidates, limit=count)

    def consolidate(self) -> dict:
        merged = self.concepts.merge_similar()
        learning = self.learning.consolidate()
        memory = self.consolidator.consolidate()
        reflection = self.meta.reflect()
        result = {
            "merged_concepts": merged,
            "learning": learning,
            "memory": memory,
            "reflection": reflection,
        }
        self.history.append({"event": "consolidate", **result})
        return result

    @staticmethod
    def auto_label(modality: str, data) -> str:
        text = str(data)
        if str(modality).lower() in ("text", "language", "chat"):
            compact = " ".join(text.strip().lower().split())
            return compact[:80] or "text:empty"
        if str(modality).lower() in ("image", "video", "file", "files"):
            base = os.path.basename(text)
            if base:
                return "%s:%s" % (modality, base)
        if str(modality).lower() in ("code", "program"):
            first = text.strip().splitlines()[0] if text.strip() else ""
            return "code:" + (first[:60] or stable_hash(text, length=8))
        return "%s:%s" % (modality, stable_hash(text, length=10))

    def status(self) -> dict:
        return {
            "data_dir": self.config.data_dir,
            "dimension": self.vsa.dimension,
            "concepts": self.concepts.stats(),
            "semantic": self.semantic.stats(),
            "working": self.working.snapshot(),
            "episodes": self.episodes.stats(),
            "predictive": self.predictive.stats(),
            "active": self.active.stats(),
            "metacognition": self.meta.stats(),
            "curiosity": self.curiosity.stats(),
            "learning": self.learning.stats(),
            "actuators": self.actuators.capabilities(),
            "sensors": self.sensors.stats(),
            "growth": self._growth.status() if self._growth else {},
            "augmentation": (self._augmentation.status()
                             if self._augmentation else {}),
        }

    def save(self) -> dict:
        self.concepts.save(self.concept_path)
        self.semantic.save()
        self.episodes.save()
        self.predictive.save()
        self.meta.save()
        self.curiosity.save()
        self.learning.save()
        self.sensors.save()
        return {"saved": True, "concepts": len(self.concepts)}

    def close(self) -> None:
        self.save()
        if self._growth is not None:
            self._growth.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
