"""Cognitive memory: concepts, working memory, episodes, consolidation.

The three stores have different jobs and different lifetimes:

* :class:`ConceptMemory` — durable semantic knowledge (concepts,
  typed relations, confidence, evidence, contradiction history).
* :class:`WorkingMemory` — a small, capacity-limited active set with a
  goal stack and activation decay.
* :class:`EpisodicMemory` — timestamped experienced events with
  surprise/importance, replay sampling and bounded forgetting.

:class:`Consolidator` turns episodes into durable knowledge: experience
replay, concept generalization, contradiction resolution and protective
rehearsal (so repeated useful knowledge is not decayed away).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from cognition.config import GrowthConfig
from cognition.util import (atomic_write_json, clamp, read_json, stable_hash)
from cognition.vsa import VSA

FUNCTIONAL_RELATIONS = {
    "is_a", "part_of", "has_property", "located_in", "equals", "favorite",
    "color", "shape", "role",
}


# --------------------------------------------------------------------- #
# Semantic memory
# --------------------------------------------------------------------- #
@dataclass
class Concept:
    id: str
    name: str
    kind: str = "concept"
    attributes: dict = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)
    confidence: float = 0.5
    evidence: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    superseded: list[dict] = field(default_factory=list)
    source: str = "growth"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "attributes": dict(self.attributes),
            "relations": [dict(r) for r in self.relations],
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "superseded": [dict(s) for s in self.superseded],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            kind=str(data.get("kind") or "concept"),
            attributes=dict(data.get("attributes") or {}),
            relations=[dict(r) for r in data.get("relations") or []],
            confidence=clamp(data.get("confidence", 0.5)),
            evidence=int(data.get("evidence", 1)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            superseded=list(data.get("superseded") or []),
            source=str(data.get("source") or "growth"),
        )


class ConceptMemory:
    """Durable concept/relation memory with VSA retrieval."""

    FILE = "concepts.json"

    def __init__(self, data_dir: str, vsa: VSA, config: Optional[GrowthConfig] = None):
        self.data_dir = data_dir
        self.vsa = vsa
        self.config = config or GrowthConfig(data_dir=data_dir)
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, self.FILE)
        self.concepts: dict[str, Concept] = {}
        self.contradictions: list[dict] = []
        self.generalizations: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Concept lifecycle
    # ------------------------------------------------------------------
    def _vector(self, concept: Concept) -> np.ndarray:
        pairs = [("kind", concept.kind)]
        for key in sorted(concept.attributes):
            pairs.append(("attr:" + key, concept.attributes[key]))
        for relation in concept.relations:
            pairs.append(("rel:" + str(relation.get("relation", "")),
                          relation.get("target", "")))
        base = self.vsa.encode_text(concept.name)
        return self.vsa.bundle([base, self.vsa.encode_role_fillers(pairs)])

    def add(self, name: str, kind: str = "concept",
            attributes: Optional[dict] = None,
            relations: Optional[Iterable[tuple]] = None,
            source: str = "growth", confidence: float = 0.6) -> Concept:
        key = _concept_key(name)
        now = time.time()
        concept = self.concepts.get(key)
        if concept is None:
            concept = Concept(
                id="c_" + stable_hash(key, length=12),
                name=str(name), kind=str(kind or "concept"),
                attributes={}, relations=[], confidence=clamp(confidence),
                evidence=1, created_at=now, updated_at=now,
                source=source)
            self.concepts[key] = concept
        else:
            concept.evidence += 1
            concept.confidence = clamp(
                1.0 - (1.0 - concept.confidence) * 0.85)
            concept.updated_at = now
            if kind and kind != "concept":
                concept.kind = str(kind)
        for attr, value in (attributes or {}).items():
            concept.attributes[str(attr)] = value
        for item in relations or []:
            if isinstance(item, dict):
                relation, target = item.get("relation"), item.get("target")
                weight = float(item.get("weight", 0.8))
            else:
                relation, target = item[0], item[1]
                weight = float(item[2]) if len(item) > 2 else 0.8
            self.add_relation(key, relation, target, weight=weight,
                              source_tag=source)
        self._register(key)
        return concept

    def add_relation(self, source: str, relation: str, target: str,
                     weight: float = 0.8,
                     source_tag: str = "growth") -> Optional[dict]:
        source_key = _concept_key(source)
        concept = self._ensure(source_key, source)
        relation_key = _relation_key(relation)
        target_text = str(target)
        existing = [r for r in concept.relations
                    if _relation_key(r.get("relation")) == relation_key]
        for relation_row in existing:
            if _value_key(relation_row.get("target")) == _value_key(target_text):
                relation_row["weight"] = clamp(
                    max(float(relation_row.get("weight", 0.8)), weight))
                relation_row["evidence"] = int(
                    relation_row.get("evidence", 1)) + 1
                concept.updated_at = time.time()
                return relation_row
        conflict = None
        if relation_key in FUNCTIONAL_RELATIONS and existing:
            old = max(existing, key=lambda r: (float(r.get("weight", 0)),
                                               int(r.get("evidence", 0))))
            old_weight = float(old.get("weight", 0.5))
            if abs(old_weight - weight) >= self.config.contradiction_threshold:
                conflict = {
                    "source": source_key,
                    "relation": relation_key,
                    "old": old.get("target"),
                    "new": target_text,
                    "resolved": "new" if weight >= old_weight else "old",
                    "at": time.time(),
                }
                self.contradictions.append(conflict)
                if weight >= old_weight:
                    concept.superseded.append({
                        "relation": relation_key,
                        "target": old.get("target"),
                        "weight": old_weight,
                        "at": conflict["at"],
                    })
                    concept.relations.remove(old)
                else:
                    return None
        row = {
            "relation": relation_key,
            "target": target_text,
            "weight": clamp(weight),
            "evidence": 1,
            "source": source_tag,
            "at": time.time(),
        }
        concept.relations.append(row)
        concept.updated_at = time.time()
        if conflict:
            conflict["kept"] = target_text if conflict["resolved"] == "new" \
                else (existing[0].get("target") if existing else None)
        return row

    def _ensure(self, key: str, name: str) -> Concept:
        concept = self.concepts.get(key)
        if concept is None:
            concept = self.add(name, source="implicit", confidence=0.4)
        return concept

    def _register(self, key: str) -> None:
        concept = self.concepts.get(key)
        if concept:
            self.vsa.register("concept:" + concept.name)
            self.vsa.register("concept:" + key)

    def get(self, name: str) -> Optional[Concept]:
        return self.concepts.get(_concept_key(name))

    def get_or_create(self, name: str, kind: str = "concept") -> Concept:
        return self.get(name) or self.add(name, kind=kind)

    def all(self) -> list[Concept]:
        return list(self.concepts.values())

    def count(self) -> int:
        return len(self.concepts)

    # ------------------------------------------------------------------
    # Retrieval and reasoning
    # ------------------------------------------------------------------
    def vector(self, name: str) -> np.ndarray:
        concept = self.get(name)
        if concept is None:
            return self.vsa.encode_text(name)
        return self._vector(concept)

    def similar(self, query: str, k: int = 5) -> list[tuple[Concept, float]]:
        concepts = list(self.concepts.values())
        if not concepts:
            return []
        query_vector = (self.vsa.encode_text(query)
                        if isinstance(query, str) else np.asarray(query))
        items = [self._vector(c) for c in concepts]
        sims = self.vsa.similarity_matrix([query_vector], items)[0]
        order = np.argsort(-sims)[:max(1, k)]
        return [(concepts[int(i)], float(sims[int(i)])) for i in order]

    def query_relation(self, source: str, relation: str) -> list[tuple[str, float]]:
        concept = self.get(source)
        if concept is None:
            return []
        relation_key = _relation_key(relation)
        rows = [r for r in concept.relations
                if _relation_key(r.get("relation")) == relation_key]
        rows.sort(key=lambda r: -float(r.get("weight", 0.0)))
        return [(str(r.get("target")), float(r.get("weight", 0.0)))
                for r in rows]

    def transitive(self, source: str, relation: str,
                   max_depth: int = 4) -> list[str]:
        relation_key = _relation_key(relation)
        seen = {_concept_key(source)}
        frontier = [source]
        path: list[str] = []
        for _ in range(max(1, int(max_depth))):
            next_frontier = []
            for node in frontier:
                for target, _weight in self.query_relation(node, relation_key):
                    key = _concept_key(target)
                    if key in seen:
                        continue
                    seen.add(key)
                    path.append(target)
                    next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break
        return path

    def attribute(self, name: str, attribute: str,
                  default=None) -> object:
        concept = self.get(name)
        if concept is None:
            return default
        return concept.attributes.get(attribute, default)

    def contradicted(self) -> list[dict]:
        return list(self.contradictions)

    # ------------------------------------------------------------------
    # Consolidation operations
    # ------------------------------------------------------------------
    def generalize(self, min_support: int = 3) -> list[dict]:
        """Induce shared-attribute abstractions for groups of concepts.

        Purely structural: same kind + same attribute value shared by at
        least ``min_support`` concepts.  The abstract concept inherits
        the attribute and links members with ``generalizes``.
        """
        groups: dict[tuple, list[Concept]] = {}
        for concept in self.concepts.values():
            for attr, value in concept.attributes.items():
                key = (concept.kind, _value_key(attr), _value_key(value))
                groups.setdefault(key, []).append(concept)
        created: list[dict] = []
        for (kind, attr, value), members in groups.items():
            if len(members) < max(2, int(min_support)):
                continue
            name = f"{kind}:{attr}={value}"
            if _concept_key(name) in self.concepts:
                continue
            abstract = self.add(
                name, kind="abstraction",
                attributes={"shared_attribute": attr,
                            "shared_value": value,
                            "support": len(members)},
                source="consolidation", confidence=0.7)
            for member in members:
                self.add_relation(name, "generalizes", member.name,
                                  weight=0.7, source_tag="consolidation")
            record = {"name": name, "members": [m.name for m in members],
                      "support": len(members)}
            self.generalizations.append(record)
            created.append(record)
        return created

    def rehearse(self, names: Iterable[str], delta: float = 0.08) -> int:
        count = 0
        for name in names:
            concept = self.get(name)
            if concept is None:
                continue
            concept.confidence = clamp(concept.confidence + delta)
            concept.evidence += 1
            concept.updated_at = time.time()
            count += 1
        return count

    def decay(self, factor: float = 0.985,
              protective_evidence: int = 3) -> int:
        changed = 0
        for concept in self.concepts.values():
            if concept.evidence >= protective_evidence:
                concept.confidence = clamp(
                    concept.confidence - 0.01 * (1.0 - concept.confidence))
            else:
                concept.confidence = clamp(concept.confidence * factor)
            changed += 1
        return changed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "concepts": [c.to_dict() for c in self.concepts.values()],
            "contradictions": self.contradictions[-200:],
            "generalizations": self.generalizations[-200:],
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("concepts") or []:
            if not isinstance(row, dict):
                continue
            concept = Concept.from_dict(row)
            if concept.name:
                self.concepts[_concept_key(concept.name)] = concept
                self._register(_concept_key(concept.name))
        self.contradictions = list(data.get("contradictions") or [])
        self.generalizations = list(data.get("generalizations") or [])

    def stats(self) -> dict:
        return {
            "concepts": len(self.concepts),
            "relations": sum(len(c.relations) for c in self.concepts.values()),
            "abstractions": sum(1 for c in self.concepts.values()
                                if c.kind == "abstraction"),
            "contradictions": len(self.contradictions),
            "generalizations": len(self.generalizations),
        }


# --------------------------------------------------------------------- #
# Working memory
# --------------------------------------------------------------------- #
@dataclass
class WorkingSlot:
    id: str
    content: object
    vector: np.ndarray
    activation: float = 1.0
    kind: str = "item"
    created_at: float = field(default_factory=time.time)
    ttl: float = 0.0
    use_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content if isinstance(
                self.content, (str, int, float, bool, list, dict, type(None))
            ) else str(self.content),
            "activation": round(float(self.activation), 4),
            "kind": self.kind,
            "created_at": self.created_at,
            "use_count": self.use_count,
        }


class WorkingMemory:
    """Capacity-limited active memory with decay and a goal stack."""

    def __init__(self, vsa: VSA, capacity: int = 7):
        self.vsa = vsa
        self.capacity = max(1, int(capacity))
        self.slots: dict[str, WorkingSlot] = {}
        self.goals: list[dict] = []
        self.focus_id: str = ""
        self._counter = 0

    def add(self, content: object, kind: str = "item",
            activation: float = 1.0, vector=None,
            ttl: float = 0.0) -> WorkingSlot:
        self._counter += 1
        slot_id = f"wm_{self._counter}"
        if vector is None:
            vector = (self.vsa.encode_text(str(content))
                      if isinstance(content, str)
                      else self.vsa.encode_features({"content": str(content)}))
        slot = WorkingSlot(id=slot_id, content=content,
                           vector=np.asarray(vector, dtype=np.float32),
                           activation=clamp(activation), kind=kind,
                           ttl=float(ttl))
        self.slots[slot_id] = slot
        self._enforce_capacity()
        return slot

    def _enforce_capacity(self) -> list[str]:
        evicted: list[str] = []
        if len(self.slots) <= self.capacity:
            return evicted
        order = sorted(self.slots.values(),
                       key=lambda s: (s.activation, s.created_at))
        while len(self.slots) > self.capacity and order:
            victim = order.pop(0)
            if victim.id == self.focus_id and len(order):
                order.append(victim)
                continue
            evicted.append(victim.id)
            self.slots.pop(victim.id, None)
        return evicted

    def get(self, slot_id: str) -> Optional[WorkingSlot]:
        slot = self.slots.get(slot_id)
        if slot:
            slot.use_count += 1
            slot.activation = clamp(slot.activation + 0.05)
        return slot

    def focus(self, query: str, k: int = 3) -> list[WorkingSlot]:
        if not self.slots:
            return []
        q = self.vsa.encode_text(query)
        scored = []
        for slot in self.slots.values():
            scored.append((self.vsa.similarity(q, slot.vector), slot))
        scored.sort(key=lambda pair: -pair[0])
        result = []
        for _score, slot in scored[:max(1, k)]:
            slot.activation = clamp(slot.activation + 0.1)
            slot.use_count += 1
            result.append(slot)
        if result:
            self.focus_id = result[0].id
        return result

    def push_goal(self, goal: str, priority: float = 0.5,
                  parent: str = "", metadata: Optional[dict] = None) -> dict:
        row = {
            "id": "g_" + stable_hash(goal, len(self.goals), time.time(),
                                     length=10),
            "goal": str(goal),
            "priority": clamp(priority),
            "parent": parent,
            "created_at": time.time(),
            "metadata": dict(metadata or {}),
            "status": "open",
        }
        self.goals.append(row)
        self.goals.sort(key=lambda g: -float(g.get("priority", 0.0)))
        return row

    def active_goal(self) -> Optional[dict]:
        for goal in self.goals:
            if goal.get("status") == "open":
                return goal
        return None

    def complete_goal(self, goal_id: str, status: str = "done") -> bool:
        for goal in self.goals:
            if goal.get("id") == goal_id:
                goal["status"] = status
                goal["closed_at"] = time.time()
                return True
        return False

    def decay(self, factor: float = 0.9) -> list[str]:
        removed: list[str] = []
        now = time.time()
        for slot in list(self.slots.values()):
            expired = slot.ttl > 0 and now - slot.created_at > slot.ttl
            slot.activation = clamp(slot.activation * factor)
            if expired or slot.activation < 0.02:
                removed.append(slot.id)
                self.slots.pop(slot.id, None)
        return removed

    def contents(self) -> list[dict]:
        return [slot.to_dict() for slot in
                sorted(self.slots.values(),
                       key=lambda s: -s.activation)]

    def snapshot(self) -> dict:
        return {
            "capacity": self.capacity,
            "slots": self.contents(),
            "focus": self.focus_id,
            "goals": [dict(g) for g in self.goals[-20:]],
        }


# --------------------------------------------------------------------- #
# Episodic memory
# --------------------------------------------------------------------- #
@dataclass
class Episode:
    id: str
    domain: str
    action: str
    outcome: dict
    success: bool
    reward: float = 0.0
    surprise: float = 0.0
    importance: float = 0.5
    context: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    replayed: int = 0
    source: str = "growth"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "action": self.action,
            "outcome": self.outcome,
            "success": bool(self.success),
            "reward": self.reward,
            "surprise": self.surprise,
            "importance": self.importance,
            "context": self.context,
            "tags": list(self.tags),
            "timestamp": self.timestamp,
            "replayed": self.replayed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        return cls(
            id=str(data.get("id") or ""),
            domain=str(data.get("domain") or ""),
            action=str(data.get("action") or ""),
            outcome=dict(data.get("outcome") or {}),
            success=bool(data.get("success")),
            reward=float(data.get("reward", 0.0)),
            surprise=float(data.get("surprise", 0.0)),
            importance=clamp(data.get("importance", 0.5)),
            context=dict(data.get("context") or {}),
            tags=list(data.get("tags") or []),
            timestamp=float(data.get("timestamp", time.time())),
            replayed=int(data.get("replayed", 0)),
            source=str(data.get("source") or "growth"),
        )


class EpisodicMemory:
    """Bounded episode store with prioritized replay and protection."""

    FILE = "episodes.json"

    def __init__(self, data_dir: str, capacity: int = 4000):
        self.data_dir = data_dir
        self.capacity = max(32, int(capacity))
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, self.FILE)
        self.episodes: list[Episode] = []
        self._load()

    def store(self, episode: Episode) -> Episode:
        if not episode.id:
            episode.id = "e_" + stable_hash(
                episode.domain, episode.action, episode.timestamp,
                len(self.episodes), length=12)
        self.episodes.append(episode)
        self._enforce_capacity()
        return episode

    def record(self, domain: str, action: str, outcome: dict,
               success: bool, reward: float = 0.0,
               surprise: float = 0.0, importance: Optional[float] = None,
               context: Optional[dict] = None,
               tags: Optional[Iterable[str]] = None,
               source: str = "growth") -> Episode:
        if importance is None:
            importance = clamp(0.35 + abs(reward) * 0.4 + surprise * 0.35
                               + (0.15 if not success else 0.0))
        return self.store(Episode(
            id="", domain=domain, action=action, outcome=outcome,
            success=success, reward=float(reward),
            surprise=float(surprise), importance=clamp(importance),
            context=dict(context or {}), tags=list(tags or []),
            source=source))

    def _enforce_capacity(self) -> None:
        if len(self.episodes) <= self.capacity:
            return
        # Keep successful high-importance episodes; drop low-value ones.
        ranked = sorted(
            self.episodes,
            key=lambda e: (e.importance + (0.25 if e.success else 0.0)
                           + min(e.replayed, 3) * 0.05
                           + _recency_bonus(e.timestamp)))
        drop = len(self.episodes) - self.capacity
        victims = {id(e) for e in ranked[:drop]}
        self.episodes = [e for e in self.episodes if id(e) not in victims]

    def recent(self, limit: int = 20, domain: str = "",
               success: Optional[bool] = None) -> list[Episode]:
        rows = [e for e in self.episodes
                if (not domain or e.domain == domain)
                and (success is None or e.success == success)]
        rows.sort(key=lambda e: -e.timestamp)
        return rows[:max(1, limit)]

    def sample_replay(self, n: int = 8, rng=None,
                      seed: Optional[int] = None) -> list[Episode]:
        if not self.episodes:
            return []
        if rng is None:
            rng = np.random.default_rng(seed if seed is not None else 7)
        priorities = np.array([
            1e-4 + e.importance + e.surprise * 0.5
            - min(e.replayed, 5) * 0.02 + (0.1 if e.success else 0.2)
            for e in self.episodes], dtype=np.float64)
        priorities = priorities / priorities.sum()
        indices = rng.choice(len(self.episodes), size=min(n, len(self.episodes)),
                             replace=False, p=priorities)
        out = []
        for index in indices:
            episode = self.episodes[int(index)]
            episode.replayed += 1
            out.append(episode)
        return out

    def failure_counts(self, limit: int = 12) -> dict[str, int]:
        counts: dict[str, int] = {}
        for episode in self.episodes:
            if episode.success:
                continue
            key = episode.context.get("failure", "") or episode.domain
            counts[str(key)] = counts.get(str(key), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:limit])

    def stats(self) -> dict:
        successes = sum(1 for e in self.episodes if e.success)
        return {
            "episodes": len(self.episodes),
            "successes": successes,
            "failures": len(self.episodes) - successes,
            "capacity": self.capacity,
            "domains": sorted({e.domain for e in self.episodes}),
        }

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "episodes": [e.to_dict() for e in self.episodes],
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("episodes") or []:
            if isinstance(row, dict):
                episode = Episode.from_dict(row)
                if episode.domain or episode.action:
                    self.episodes.append(episode)
        self._enforce_capacity()


# --------------------------------------------------------------------- #
# Consolidation
# --------------------------------------------------------------------- #
class Consolidator:
    """Experience replay -> durable knowledge, with forgetting defense."""

    def __init__(self, concept_memory: ConceptMemory,
                 episodic_memory: EpisodicMemory,
                 working_memory: WorkingMemory,
                 predictive_engine=None,
                 config: Optional[GrowthConfig] = None):
        self.concepts = concept_memory
        self.episodes = episodic_memory
        self.working = working_memory
        self.predictive = predictive_engine
        self.config = config or GrowthConfig(data_dir=concept_memory.data_dir)
        self.cycles = 0

    def consolidate(self, replay_batch: int = 12,
                    seed: Optional[int] = None) -> dict:
        self.cycles += 1
        rng = np.random.default_rng(seed if seed is not None else
                                    self.config.seed + self.cycles)
        replayed = self.episodes.sample_replay(replay_batch, rng=rng)
        rehearsed = 0
        contradictions = 0
        predictive_updates = 0
        for episode in replayed:
            if episode.success:
                context = episode.context or {}
                subject = context.get("subject") or episode.domain
                relation = context.get("relation") or "has_skill"
                target = context.get("skill") or episode.action
                self.concepts.add(subject, kind=context.get("kind", "skill"),
                                  attributes={"domain": episode.domain,
                                              "action": episode.action},
                                  relations=[(relation, target)],
                                  source="replay", confidence=0.65)
                rehearsed += self.concepts.rehearse([subject], 0.04)
            else:
                failure = episode.context.get("failure") or "unknown"
                self.concepts.add(
                    "failure:" + str(failure), kind="failure_mode",
                    attributes={"domain": episode.domain,
                                "action": episode.action},
                    source="replay", confidence=0.55)
            if self.predictive is not None:
                sequence = (episode.context or {}).get("symbol_sequence") or []
                if sequence:
                    self.predictive.observe_sequence(sequence)
                    predictive_updates += 1
        newly_general = self.concepts.generalize(
            self.config.gate_min_successes)
        contradictions = len(self.concepts.contradicted())
        decayed = self.concepts.decay(0.99)
        self.episodes.save()
        self.concepts.save()
        if self.predictive is not None:
            self.predictive.save()
        return {
            "cycle": self.cycles,
            "replayed": len(replayed),
            "rehearsed": rehearsed,
            "generalizations": len(newly_general),
            "contradictions_seen": contradictions,
            "predictive_updates": predictive_updates,
            "decayed": decayed,
        }


def _concept_key(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _relation_key(name: object) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _value_key(value: object) -> str:
    return str(value).strip().lower()


def _recency_bonus(timestamp: float) -> float:
    age_days = max(0.0, (time.time() - float(timestamp)) / 86400.0)
    return 0.2 * math.exp(-age_days / 30.0)
