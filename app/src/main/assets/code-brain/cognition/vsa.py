"""Vector Symbolic Architecture (VSA) core — MAP-style bipolar vectors.

No pretrained model and no gradient descent are involved.  A symbol is
a deterministic pseudo-random bipolar vector (``-1/+1``); meaning is
constructed from three algebraically invertible operations:

* ``bind(a, b)`` — elementwise product.  Self-inverse for bipolar
  vectors, so ``unbind(bind(a, b), a) == b`` (up to noise).
* ``bundle([a, b, ...])`` — majority sign of the sum.  Superposition
  with graceful degradation rather than destructive overwrite.
* ``permute(a, k)`` — cyclic shift.  Encodes order/roles without a
  learned position embedding.

Composite structures (role-filler pairs, sequences, trees, sentences)
are vectors; cleanup memory maps a noisy vector back to the nearest
stored symbol.  Capacity and noise robustness are *measured* in
``cognition.benchmark`` — they are not assumed to be unbounded.

This is a representation and retrieval substrate, not a claim that
vector similarity alone constitutes understanding.  Reasoning lives in
``compositional.py`` / ``predictive.py`` and is verified by tests.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional, Sequence

import numpy as np

from cognition.util import clamp


class VSA:
    """Deterministic MAP VSA over ``{-1, +1}^dimension``."""

    def __init__(self, dimension: int = 2048, seed: int = 1729):
        self.dimension = max(64, int(dimension))
        self.seed = int(seed)
        self._cache: dict[str, np.ndarray] = {}
        self._item_names: list[str] = []
        self._item_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True

    # ------------------------------------------------------------------
    # Symbol generation
    # ------------------------------------------------------------------
    def _seed_for(self, *parts: object) -> int:
        h = hashlib.sha256()
        h.update(str(self.seed).encode("ascii"))
        for part in parts:
            h.update(b"\x1e")
            h.update(str(part).encode("utf-8", "ignore"))
        return int.from_bytes(h.digest()[:8], "big")

    def symbol(self, name: object) -> np.ndarray:
        """The deterministic vector for a symbol name."""
        key = str(name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rng = np.random.default_rng(self._seed_for("symbol", key))
        vector = rng.integers(0, 2, size=self.dimension,
                              dtype=np.int8).astype(np.float32) * 2.0 - 1.0
        self._cache[key] = vector
        return vector

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        """A fresh bipolar vector from an explicit generator."""
        return rng.integers(0, 2, size=self.dimension,
                            dtype=np.int8).astype(np.float32) * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------
    @staticmethod
    def _as(array: Iterable[float]) -> np.ndarray:
        return np.asarray(array, dtype=np.float32)

    def bind(self, a, b) -> np.ndarray:
        """Binding by elementwise product (self-inverse on bipolar)."""
        return self._as(a) * self._as(b)

    def unbind(self, composite, key) -> np.ndarray:
        return self.bind(composite, key)

    def bundle(self, vectors: Sequence[Iterable[float]],
               weights: Optional[Sequence[float]] = None) -> np.ndarray:
        """Superposition: weighted sum, then majority sign."""
        if not vectors:
            return np.zeros(self.dimension, dtype=np.float32)
        matrix = np.stack([self._as(v) for v in vectors])
        if weights is not None:
            w = np.asarray(weights, dtype=np.float32).reshape((-1, 1))
            total = (matrix * w).sum(axis=0)
        else:
            total = matrix.sum(axis=0)
        # Zero only occurs on exact ties; resolve deterministically to +1
        # rather than introducing a position-dependent bias.
        return np.where(total >= 0.0, 1.0, -1.0).astype(np.float32)

    def permute(self, vector, shift: int = 1) -> np.ndarray:
        """Invertible cyclic shift encoding order."""
        return np.roll(self._as(vector), int(shift))

    def invert_permute(self, vector, shift: int = 1) -> np.ndarray:
        return np.roll(self._as(vector), -int(shift))

    def similarity(self, a, b) -> float:
        va, vb = self._as(a), self._as(b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def normalize(self, vector) -> np.ndarray:
        v = self._as(vector)
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            return v
        return v / norm

    # ------------------------------------------------------------------
    # Structure encoding
    # ------------------------------------------------------------------
    def encode_pair(self, role: object, filler: object) -> np.ndarray:
        return self.bind(self.symbol(role), self.symbol(filler))

    def encode_role_fillers(self, pairs: Iterable[tuple]) -> np.ndarray:
        bound = [self.encode_pair(role, filler) for role, filler in pairs]
        return self.bundle(bound)

    def encode_sequence(self, items: Iterable[object]) -> np.ndarray:
        vectors = []
        for index, item in enumerate(items):
            base = (self.symbol(item) if isinstance(item, (str, int, float))
                    else self._as(item))
            vectors.append(self.permute(base, index))
        return self.bundle(vectors)

    def decode_position(self, sequence_vector, position: int) -> np.ndarray:
        return self.invert_permute(sequence_vector, position)

    def encode_text(self, text: str, bigrams: bool = True) -> np.ndarray:
        tokens = (text or "").lower().split()
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)
        parts = [self.symbol("tok:" + token) for token in tokens]
        if bigrams and len(tokens) > 1:
            parts.extend(self.bind(self.symbol("tok:" + tokens[i]),
                                   self.permute(self.symbol("tok:" +
                                                            tokens[i + 1]), 1))
                            for i in range(len(tokens) - 1))
        return self.bundle(parts)

    def encode_features(self, features: dict) -> np.ndarray:
        parts = []
        for key in sorted(features or {}):
            value = features[key]
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    parts.append(self.encode_pair("feature:" + str(key),
                                                  item))
            elif isinstance(value, bool):
                parts.append(self.encode_pair("feature:" + str(key),
                                              "true" if value else "false"))
            elif isinstance(value, (int, float)):
                bucket = _numeric_bucket(float(value))
                parts.append(self.encode_pair("feature:" + str(key), bucket))
            else:
                parts.append(self.encode_pair("feature:" + str(key), value))
        return self.bundle(parts)

    # ------------------------------------------------------------------
    # Item / cleanup memory
    # ------------------------------------------------------------------
    def register(self, name: object) -> None:
        key = str(name)
        if key not in self._cache:
            self.symbol(key)
        if key not in self._item_names:
            self._item_names.append(key)
            self._matrix_dirty = True

    def register_many(self, names: Iterable[object]) -> None:
        for name in names:
            self.register(name)

    def _item_matrix_for_cleanup(self) -> np.ndarray:
        if self._matrix_dirty or self._item_matrix is None:
            if not self._item_names:
                self._item_matrix = np.zeros((0, self.dimension),
                                             dtype=np.float32)
            else:
                self._item_matrix = np.stack(
                    [self.symbol(n) for n in self._item_names])
            self._matrix_dirty = False
        return self._item_matrix

    def cleanup(self, query, k: int = 1,
                exclude: Optional[set] = None) -> list[tuple[str, float]]:
        """Nearest registered symbols by cosine similarity."""
        matrix = self._item_matrix_for_cleanup()
        if matrix.shape[0] == 0:
            return []
        q = self._as(query)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12:
            return []
        scores = matrix @ (q / norm) / np.sqrt(self.dimension)
        if exclude:
            for index, name in enumerate(self._item_names):
                if name in exclude:
                    scores[index] = -np.inf
        order = np.argsort(-scores)[:max(1, int(k))]
        return [(self._item_names[int(i)], float(scores[int(i)]))
                for i in order if np.isfinite(scores[int(i)])]

    def similarity_matrix(self, queries, items) -> np.ndarray:
        q = np.stack([self._as(x) for x in queries])
        m = np.stack([self._as(x) for x in items])
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        return q @ m.T

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def capacity_probe(self, items: int = 40, noise: float = 0.25,
                       seed: int = 3) -> dict:
        """Store ``items`` pairs in superposition and decode them.

        Returns measured accuracy at the requested noise level.  This is
        the physical capacity check the architecture documentation
        cites; it is not a theoretical promise.
        """
        rng = np.random.default_rng(seed)
        keys = [f"k{i}" for i in range(items)]
        values = [f"v{i}" for i in range(items)]
        memory = self.bundle([
            self.bind(self.symbol(k), self.symbol(v))
            for k, v in zip(keys, values)
        ])
        self.register_many(values)
        correct = 0
        for key, value in zip(keys, values):
            probe = self.unbind(memory, self.symbol(key))
            if noise:
                flip = rng.random(self.dimension) < noise
                probe = probe * np.where(flip, -1.0, 1.0).astype(np.float32)
            guess = self.cleanup(probe, k=1, exclude={"key:" + key})
            if guess and guess[0][0] == value:
                correct += 1
        return {
            "items": items,
            "noise": noise,
            "accuracy": round(correct / max(1, items), 4),
            "dimension": self.dimension,
        }

    def stats(self) -> dict:
        return {
            "dimension": self.dimension,
            "symbols_cached": len(self._cache),
            "cleanup_items": len(self._item_names),
            "implementation": "MAP bipolar (no learned weights)",
        }


def _numeric_bucket(value: float) -> str:
    """Coarse numeric bucket — representation, not a learned scale."""
    if value != value or value in (float("inf"), float("-inf")):
        return "nan"
    magnitude = abs(value)
    if magnitude == 0:
        return "zero"
    sign = "neg" if value < 0 else "pos"
    if magnitude < 1:
        scale = "lt1"
    elif magnitude < 10:
        scale = "lt10"
    elif magnitude < 100:
        scale = "lt100"
    elif magnitude < 10000:
        scale = "lt10k"
    else:
        scale = "big"
    return f"{sign}:{scale}"
