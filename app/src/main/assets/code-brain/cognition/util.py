"""Small shared utilities: atomic JSON, hashing, normalization, budgets."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional


def stable_hash(*parts: Any, length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
    except Exception:
        pass
    return str(value)


def atomic_write_json(path: str, data: Any, indent: int = 2) -> None:
    """Atomic JSON write that tolerates concurrent writers.

    The temporary file lives in the same directory as the target so the
    final ``os.replace`` is atomic.  If several processes race, the last
    writer wins; no reader ever observes a truncated file.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json",
                              dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(data), handle, ensure_ascii=False,
                      indent=indent, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError:
        return []
    if limit is not None:
        rows = rows[-limit:]
    return rows


def append_jsonl(path: str, rows: Iterable[dict], max_rows: int = 0) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                exist_ok=True)
    rows = list(rows)
    if max_rows > 0:
        existing = read_jsonl(path, limit=max_rows)
        rows = existing + rows
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows[-max_rows:]:
                handle.write(json.dumps(json_safe(row),
                                        ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row),
                                    ensure_ascii=False) + "\n")


def cosine(a: Any, b: Any) -> float:
    try:
        import numpy as np
    except Exception:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass


@dataclass
class Budget:
    """A cooperative wall-clock / cycle budget with honest exhaustion."""
    max_seconds: float = 60.0
    max_cycles: int = 20
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.time()
        self.cycles = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, float(self.max_seconds) - self.elapsed)

    def remaining_cycles(self) -> int:
        return max(0, int(self.max_cycles) - self.cycles)

    def take_cycle(self) -> bool:
        if self.remaining_cycles() <= 0 or self.remaining_seconds <= 0:
            return False
        self.cycles += 1
        return True

    def exhausted_reason(self) -> str:
        if self.cycles >= self.max_cycles:
            return "cycle_budget"
        if self.elapsed >= self.max_seconds:
            return "time_budget"
        return ""
