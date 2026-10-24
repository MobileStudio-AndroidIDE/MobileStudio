"""Prediction-based reasoning with online, non-backprop updates.

Two complementary predictive memories are updated by observation:

1. bounded n-gram transition counts (interpretable, fast, sparse),
2. a VSA context->next associative memory (generalizes across contexts
   that share structure).

There is no gradient, no loss backpropagation and no pretrained
sequence model.  Learning is counting plus Hebbian bundling.  Surprise
is measured, not hidden: a novel transition produces high surprise and
is explicitly reported.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict, deque
from typing import Iterable, Optional

import numpy as np

from cognition.util import atomic_write_json, read_json
from cognition.vsa import VSA

MAX_ORDER = 3


class PredictiveEngine:
    def __init__(self, data_dir: str, vsa: VSA,
                 max_order: int = MAX_ORDER):
        self.data_dir = data_dir
        self.vsa = vsa
        self.max_order = max(1, int(max_order))
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "predictive.json")
        self.counts: dict[int, dict[tuple, Counter]] = {
            order: defaultdict(Counter)
            for order in range(1, self.max_order + 1)
        }
        self.vsa_associations: dict[str, np.ndarray] = {}
        self.vsa_contexts: dict[str, tuple] = {}
        self.observations = 0
        self.history: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Online learning (Hebbian / counting, no gradients)
    # ------------------------------------------------------------------
    def observe_sequence(self, symbols: Iterable[object]) -> int:
        sequence = [str(s) for s in symbols]
        if not sequence:
            return 0
        updated = 0
        for index, symbol in enumerate(sequence):
            self.history.append(symbol)
            for order in range(1, self.max_order + 1):
                if index - order < 0:
                    continue
                context = tuple(sequence[index - order:index])
                self.counts[order][context][symbol] += 1
                updated += 1
            self.observations += 1
        for index in range(self.max_order, len(sequence)):
            for order in range(1, self.max_order + 1):
                context = tuple(sequence[index - order:index])
                key = "ctx:" + "|".join(context)
                bound = self.vsa.encode_pair(
                    self.vsa.symbol("ctxobj:" + "|".join(context)),
                    self.vsa.symbol("next:" + sequence[index]))
                previous = self.vsa_associations.get(key)
                if previous is None:
                    self.vsa_associations[key] = bound
                else:
                    self.vsa_associations[key] = self.vsa.bundle(
                        [previous, bound], weights=[0.9, 0.1])
                self.vsa_contexts[key] = context
                self.vsa.register("next:" + sequence[index])
            self.vsa.register("ctxobj:" + "|".join(context))
        if len(self.history) > 4096:
            self.history = self.history[-2048:]
        return updated

    def observe_transition(self, context: Iterable[object],
                           symbol: object) -> None:
        ctx = tuple(str(c) for c in context)[-self.max_order:]
        self.observe_sequence([*ctx, str(symbol)])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _distribution(self, context: Iterable[object]) -> tuple[Counter, int]:
        ctx = tuple(str(c) for c in context)
        for order in range(min(self.max_order, len(ctx)), 0, -1):
            key = ctx[-order:]
            counter = self.counts[order].get(key)
            if counter:
                return counter, order
        return Counter(), 0

    def _vsa_distribution(self, context: Iterable[object]) -> dict[str, float]:
        ctx = tuple(str(c) for c in context)
        best: tuple[float, str, float] | None = None
        for order in range(min(self.max_order, len(ctx)), 0, -1):
            key = "ctx:" + "|".join(ctx[-order:])
            memory = self.vsa_associations.get(key)
            if memory is None:
                continue
            base = self.vsa.symbol("ctxobj:" + "|".join(ctx[-order:]))
            probe = self.vsa.unbind(memory, base)
            guess = self.vsa.cleanup(probe, k=1)
            if guess:
                name, score = guess[0]
                if best is None or score > best[2]:
                    best = (name, score, order)
            break
        if best is None:
            return {}
        name, score, order = best
        return {name.removeprefix("next:"): max(0.0, score)}

    def predict(self, context: Iterable[object],
                top_k: int = 3) -> dict:
        counter, order = self._distribution(context)
        total = sum(counter.values())
        distribution: dict[str, float] = {}
        if total:
            distribution = {symbol: count / total
                            for symbol, count in counter.most_common()}
        vsa_dist = self._vsa_distribution(context)
        for symbol, value in vsa_dist.items():
            distribution.setdefault(symbol, 0.0)
            distribution[symbol] = max(distribution[symbol],
                                       0.25 * value)
        merged_total = sum(distribution.values()) or 1.0
        distribution = {k: v / merged_total
                        for k, v in sorted(distribution.items(),
                                           key=lambda kv: -kv[1])[:max(1, top_k)]}
        if distribution:
            symbol = next(iter(distribution))
            confidence = distribution[symbol]
            surprise = -math.log2(max(confidence, 1e-6))
        else:
            symbol, confidence, surprise = "", 0.0, 1.0
        return {
            "symbol": symbol,
            "confidence": round(float(confidence), 6),
            "distribution": {k: round(float(v), 6)
                             for k, v in distribution.items()},
            "surprise": round(float(surprise), 6),
            "order": order,
            "backoff": "counts" if order else "vsa",
        }

    def surprise(self, actual: object, context: Iterable[object]) -> float:
        prediction = self.predict(context, top_k=16)
        probability = prediction["distribution"].get(str(actual), 0.0)
        if probability <= 0.0:
            # smoothing: unseen but possible
            parameter_pool = max(2, len(prediction["distribution"]) + 1)
            probability = 1.0 / (self.observations + parameter_pool)
        return round(-math.log2(max(probability, 1e-6)), 6)

    def predict_sequence(self, context: Iterable[object],
                         steps: int = 3) -> list[str]:
        ctx = deque([str(c) for c in context], maxlen=self.max_order)
        out = []
        for _ in range(max(0, int(steps))):
            prediction = self.predict(ctx)
            symbol = prediction["symbol"]
            if not symbol:
                break
            out.append(symbol)
            ctx.append(symbol)
        return out

    # ------------------------------------------------------------------
    # Planning over learned transitions
    # ------------------------------------------------------------------
    def plan(self, start: str, goal: str,
             max_steps: int = 6) -> dict:
        """Breadth-first search over learned transitions (bounded).

        This is planning, not memorization: the transition graph is the
        model learned from observations, and the search visits each
        state at most once.
        """
        start, goal = str(start), str(goal)
        if start == goal:
            return {"found": True, "path": [start], "expanded": 0}
        frontier = deque([(start, [start])])
        seen = {start}
        expanded = 0
        while frontier and expanded < max(1, int(max_steps)) * 8:
            state, path = frontier.popleft()
            expanded += 1
            candidates = self.predict(state, top_k=8)["distribution"]
            for nxt in candidates:
                if nxt in seen:
                    continue
                if nxt == goal:
                    return {"found": True, "path": path + [nxt],
                            "expanded": expanded}
                seen.add(nxt)
                if len(path) < max_steps:
                    frontier.append((nxt, path + [nxt]))
        return {"found": False, "path": [], "expanded": expanded}

    # ------------------------------------------------------------------
    # Persistence / metrics
    # ------------------------------------------------------------------
    def save(self) -> None:
        counts = {
            str(order): {
                "|".join(context): dict(counter)
                for context, counter in table.items()
            }
            for order, table in self.counts.items()
        }
        atomic_write_json(self.path, {
            "version": 1,
            "max_order": self.max_order,
            "counts": counts,
            "associations": {key: memory.tolist()
                             for key, memory in self.vsa_associations.items()},
            "observations": self.observations,
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        self.max_order = max(1, int(data.get("max_order", self.max_order)))
        self.counts = {order: defaultdict(Counter)
                       for order in range(1, self.max_order + 1)}
        for order_text, table in (data.get("counts") or {}).items():
            try:
                order = int(order_text)
            except ValueError:
                continue
            if order not in self.counts:
                continue
            for context_text, counter in table.items():
                context = tuple(context_text.split("|")) if context_text else ()
                self.counts[order][context] = Counter(
                    {str(k): int(v) for k, v in counter.items()})
        for key, raw in (data.get("associations") or {}).items():
            try:
                vector = np.asarray(raw, dtype=np.float32)
            except Exception:
                continue
            if vector.shape == (self.vsa.dimension,):
                self.vsa_associations[key] = vector
        self.observations = int(data.get("observations", 0))

    def stats(self) -> dict:
        contexts = sum(len(table) for table in self.counts.values())
        return {
            "observations": self.observations,
            "n_contexts": contexts,
            "vsa_associations": len(self.vsa_associations),
            "max_order": self.max_order,
        }
