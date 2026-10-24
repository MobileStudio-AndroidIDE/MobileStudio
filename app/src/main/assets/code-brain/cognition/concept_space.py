"""Conceptual spaces: prototypes, exemplars and similarity-based concepts."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cognition.vsa import VSA


@dataclass
class Concept:
    label: str
    prototype: np.ndarray
    modality: str = "text"
    count: int = 1
    spread: float = 0.0
    attributes: dict = field(default_factory=dict)
    exemplars: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "modality": self.modality,
            "count": self.count,
            "spread": self.spread,
            "attributes": self.attributes,
        }


class ConceptSpace:
    def __init__(self, vsa: VSA, exemplar_limit: int = 8,
                 merge_threshold: float = 0.997):
        self.vsa = vsa
        self.exemplar_limit = int(exemplar_limit)
        self.merge_threshold = float(merge_threshold)
        self.concepts: dict[str, Concept] = {}

    def __len__(self) -> int:
        return len(self.concepts)

    def __contains__(self, label) -> bool:
        return str(label) in self.concepts

    def add(self, label, vector, modality: str = "text",
            attributes: Optional[dict] = None) -> Concept:
        label = str(label)
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            vector = self.vsa.symbol("empty-concept")
        else:
            vector = vector / norm
        concept = self.concepts.get(label)
        if concept is None:
            concept = Concept(label=label, prototype=vector.copy(),
                              modality=str(modality), count=1,
                              spread=0.0,
                              attributes=dict(attributes or {}))
            self.concepts[label] = concept
        else:
            concept.count += 1
            concept.prototype = self.vsa.normalize(
                concept.prototype * (concept.count - 1) + vector)
            similarity = float(np.dot(concept.prototype, vector))
            concept.spread = 0.8 * concept.spread + 0.2 * (1.0 - similarity)
            concept.modality = str(modality)
            if attributes:
                concept.attributes.update(attributes)
        if self.exemplar_limit:
            concept.exemplars.append(vector.copy())
            if len(concept.exemplars) > self.exemplar_limit:
                del concept.exemplars[0]
        return concept

    def add_text(self, label, text, encoder) -> Concept:
        return self.add(label, encoder.encode_text(text), "text",
                        {"last_text": str(text)})

    def get(self, label) -> Optional[Concept]:
        return self.concepts.get(str(label))

    def prototype(self, label) -> np.ndarray:
        concept = self.concepts.get(str(label))
        if concept is None:
            return self.vsa.symbol(str(label))
        return concept.prototype

    def similarity(self, left, right) -> float:
        left, right = str(left), str(right)
        if left in self.concepts and right in self.concepts:
            return self.vsa.similarity(
                self.concepts[left].prototype,
                self.concepts[right].prototype)
        return self.vsa.similarity(self.vsa.symbol(str(left)),
                                   self.vsa.symbol(str(right)))

    def classify(self, vector, k: int = 5, predicate=None
                 ) -> list[tuple[str, float]]:
        vector = np.asarray(vector, dtype=np.float32)
        scored = []
        for label, concept in self.concepts.items():
            if predicate is not None and not predicate(concept):
                continue
            scored.append((label, self.vsa.similarity(
                vector, concept.prototype)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max(1, int(k))]

    def best(self, vector, predicate=None
             ) -> Optional[tuple[str, float]]:
        ranked = self.classify(vector, k=1, predicate=predicate)
        return ranked[0] if ranked else None

    def within(self, vector, threshold: float = 0.35,
               predicate=None) -> list[tuple[str, float]]:
        return [(label, score)
                for label, score in self.classify(
                    vector, k=len(self.concepts), predicate=predicate)
                if score >= threshold]

    def interpolate(self, left, right, alpha: float = 0.5) -> np.ndarray:
        a = self.prototype(left)
        b = self.prototype(right)
        alpha = max(0.0, min(1.0, float(alpha)))
        return self.vsa.normalize(a * (1.0 - alpha) + b * alpha)

    def merge_similar(self, threshold: Optional[float] = None) -> int:
        threshold = self.merge_threshold if threshold is None else threshold
        labels = list(self.concepts)
        merged = 0
        for index, left in enumerate(labels):
            if left not in self.concepts:
                continue
            for right in labels[index + 1:]:
                if right not in self.concepts:
                    continue
                score = self.vsa.similarity(
                    self.concepts[left].prototype,
                    self.concepts[right].prototype)
                if score < threshold:
                    continue
                source = self.concepts.pop(right)
                target = self.concepts[left]
                total = target.count + source.count
                target.prototype = self.vsa.normalize(
                    target.prototype * target.count
                    + source.prototype * source.count)
                target.count = total
                target.spread = (target.spread * (total - source.count)
                                 + source.spread * source.count) / total
                target.exemplars.extend(source.exemplars)
                target.exemplars = target.exemplars[-self.exemplar_limit:] \
                    if self.exemplar_limit else []
                target.attributes.update(
                    {("merged:" + k): v for k, v in
                     source.attributes.items()})
                merged += 1
        return merged

    def cluster(self, vectors, k: int = 4, iterations: int = 8,
                seed: int = 0) -> list[list[int]]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) == 0:
            return []
        k = max(1, min(int(k), len(vectors)))
        rng = np.random.default_rng(seed)
        centers = vectors[rng.choice(len(vectors), size=k,
                                     replace=False)].copy()
        for _ in range(max(1, iterations)):
            assignments = np.argmax(vectors @ centers.T, axis=1)
            new_centers = centers.copy()
            for index in range(k):
                members = vectors[assignments == index]
                if len(members):
                    new_centers[index] = members.mean(axis=0)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        assignments = np.argmax(vectors @ centers.T, axis=1)
        return [np.where(assignments == index)[0].tolist()
                for index in range(k)]

    def save(self, path) -> None:
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        labels = list(self.concepts)
        vectors = np.stack([self.concepts[label].prototype
                            for label in labels]).astype(np.float32) \
            if labels else np.zeros((0, self.vsa.dimension),
                                    dtype=np.float32)
        counts = np.array([self.concepts[label].count for label in labels],
                          dtype=np.int64)
        spreads = np.array([self.concepts[label].spread
                            for label in labels], dtype=np.float32)
        modalities = np.array([self.concepts[label].modality
                               for label in labels], dtype=object)
        attributes = np.array([json.dumps(self.concepts[label].attributes)
                               for label in labels], dtype=object)
        np.savez_compressed(path, labels=np.array(labels, dtype=object),
                            vectors=vectors, counts=counts,
                            spreads=spreads, modalities=modalities,
                            attributes=attributes)

    @classmethod
    def load(cls, path, vsa: VSA) -> "ConceptSpace":
        path = str(path)
        space = cls(vsa)
        if not os.path.exists(path):
            return space
        with np.load(path, allow_pickle=True) as data:
            labels = list(data["labels"])
            vectors = data["vectors"]
            counts = data["counts"]
            spreads = data["spreads"]
            modalities = list(data["modalities"])
            attributes = list(data["attributes"])
        for index, label in enumerate(labels):
            try:
                attrs = json.loads(attributes[index])
            except Exception:
                attrs = {}
            space.concepts[str(label)] = Concept(
                label=str(label),
                prototype=np.asarray(vectors[index], dtype=np.float32),
                modality=str(modalities[index]),
                count=int(counts[index]),
                spread=float(spreads[index]),
                attributes=attrs)
        return space

    def stats(self) -> dict:
        counts = [concept.count for concept in self.concepts.values()]
        return {
            "concepts": len(self.concepts),
            "observations": int(sum(counts)),
            "modalities": sorted({concept.modality
                                  for concept in self.concepts.values()}),
            "mean_spread": float(np.mean(
                [concept.spread for concept in self.concepts.values()]))
            if counts else 0.0,
        }
