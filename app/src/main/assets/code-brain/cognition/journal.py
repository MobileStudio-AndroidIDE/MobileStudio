"""Append-only growth journal with bounded size and honest limitations."""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional

from cognition.util import append_jsonl, read_jsonl
from cognition.util import stable_hash


class GrowthJournal:
    FILE = "growth_journal.jsonl"
    LIMIT = "limitations.json"

    def __init__(self, data_dir: str, max_rows: int = 4000):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, self.FILE)
        self.limits_path = os.path.join(data_dir, self.LIMIT)
        self.max_rows = max(100, int(max_rows))

    def log(self, event: str, cycle: int = 0, **data) -> dict:
        row = {
            "id": "j_" + stable_hash(event, cycle, time.time(), length=10),
            "at": time.time(),
            "event": event,
            "cycle": int(cycle),
            "data": data,
        }
        append_jsonl(self.path, [row], max_rows=self.max_rows)
        return row

    def recent(self, limit: int = 50,
               event: Optional[str] = None) -> list[dict]:
        rows = read_jsonl(self.path, limit=self.max_rows)
        if event:
            rows = [r for r in rows if r.get("event") == event]
        return rows[-max(1, limit):]

    def add_limitation(self, limitation: str, evidence: str = "") -> list:
        rows = read_jsonl(self.limits_path, limit=500)
        key = (limitation or "").strip().lower()
        if key and not any(str(r.get("limitation", "")).strip().lower() == key
                           for r in rows):
            rows.append({
                "limitation": limitation,
                "evidence": evidence,
                "at": time.time(),
            })
            tmp = self.limits_path + ".tmp"
            import json
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.limits_path)
        return rows

    def limitations(self) -> list[dict]:
        return read_jsonl(self.limits_path, limit=500)

    def summary(self) -> dict:
        rows = read_jsonl(self.path, limit=self.max_rows)
        counts: dict[str, int] = {}
        for row in rows:
            event = str(row.get("event") or "")
            counts[event] = counts.get(event, 0) + 1
        return {
            "events": len(rows),
            "last": rows[-1] if rows else None,
            "counts": counts,
            "limitations": self.limitations(),
        }
