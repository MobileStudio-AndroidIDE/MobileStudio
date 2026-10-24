"""Memory Store - stores learned cases, errors, and successful patterns."""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class MemoryType(Enum):
    CODE_EXAMPLE = "code_example"
    ERROR_CASE = "error_case"
    SUCCESS_CASE = "success_case"
    LEARNING_CASE = "learning_case"
    USER_PREFERENCE = "user_preference"
    REASONING_TRACE = "reasoning_trace"


@dataclass
class MemoryEntry:
    id: str
    memory_type: MemoryType
    content: str
    tags: list[str] = field(default_factory=list)
    language: str = ""
    confidence: float = 1.0
    use_count: int = 0
    last_used: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    def __init__(self, data_dir: str = None):
        self.memories: dict[str, MemoryEntry] = {}
        self.data_dir = data_dir
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._load()

    def store(self, entry: MemoryEntry) -> None:
        self.memories[entry.id] = entry
        self._save()

    def recall(self, memory_id: str) -> Optional[MemoryEntry]:
        entry = self.memories.get(memory_id)
        if entry:
            entry.use_count += 1
            entry.last_used = time.time()
            self._save()
        return entry

    def search(self, query: str, memory_type: MemoryType = None,
               language: str = None, tags: list[str] = None,
               limit: int = 10) -> list[MemoryEntry]:
        query_lower = query.lower()
        results = []
        for e in self.memories.values():
            if memory_type and e.memory_type != memory_type:
                continue
            if language and e.language != language:
                continue
            if tags and not any(t in e.tags for t in tags):
                continue
            if (query_lower in e.content.lower() or
                    any(query_lower in t.lower() for t in e.tags)):
                results.append(e)
        results.sort(key=lambda x: (-x.confidence, -x.use_count, -x.created_at))
        return results[:limit]

    def store_error(self, code: str, error: str, fix: str, language: str = "") -> MemoryEntry:
        entry = MemoryEntry(
            id=f"err_{len(self.memories)}",
            memory_type=MemoryType.ERROR_CASE,
            content=json.dumps({"code": code, "error": error, "fix": fix}),
            tags=["error", "fix"],
            language=language,
            metadata={"error_msg": error, "original_code": code, "fixed_code": fix},
        )
        self.store(entry)
        return entry

    def store_success(self, code: str, description: str, language: str = "") -> MemoryEntry:
        entry = MemoryEntry(
            id=f"suc_{len(self.memories)}",
            memory_type=MemoryType.SUCCESS_CASE,
            content=json.dumps({"code": code, "description": description}),
            tags=["success", "working"],
            language=language,
            metadata={"code": code, "description": description},
        )
        self.store(entry)
        return entry

    def store_learning(self, ir_data: dict, source_lang: str, target_lang: str,
                       pattern: str) -> MemoryEntry:
        entry = MemoryEntry(
            id=f"lrn_{len(self.memories)}",
            memory_type=MemoryType.LEARNING_CASE,
            content=json.dumps({"ir": ir_data, "source": source_lang, "target": target_lang, "pattern": pattern}),
            tags=["learning", source_lang, target_lang, pattern],
            language=target_lang,
            metadata={"ir_data": ir_data, "source_lang": source_lang,
                      "target_lang": target_lang, "pattern": pattern},
        )
        self.store(entry)
        return entry

    def get_most_used(self, limit: int = 10, memory_type: MemoryType = None) -> list[MemoryEntry]:
        entries = list(self.memories.values())
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        entries.sort(key=lambda x: -x.use_count)
        return entries[:limit]

    def get_recent(self, limit: int = 10, memory_type: MemoryType = None) -> list[MemoryEntry]:
        entries = list(self.memories.values())
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        entries.sort(key=lambda x: -x.created_at)
        return entries[:limit]

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save()
            return True
        return False

    def count(self, memory_type: MemoryType = None) -> int:
        if memory_type:
            return sum(1 for e in self.memories.values() if e.memory_type == memory_type)
        return len(self.memories)

    def _save(self):
        if not self.data_dir:
            return
        path = os.path.join(self.data_dir, "memory.json")
        data = []
        for e in self.memories.values():
            data.append({
                "id": e.id,
                "memory_type": e.memory_type.value,
                "content": e.content,
                "tags": e.tags,
                "language": e.language,
                "confidence": e.confidence,
                "use_count": e.use_count,
                "last_used": e.last_used,
                "created_at": e.created_at,
                "metadata": e.metadata,
            })
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _load(self):
        path = os.path.join(self.data_dir, "memory.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            try:
                os.replace(path, path + ".corrupt")
            except OSError:
                pass
            return
        if not isinstance(data, list):
            return
        for d in data:
            try:
                d["memory_type"] = MemoryType(d["memory_type"])
                e = MemoryEntry(**d)
            except (TypeError, KeyError, ValueError):
                continue
            self.memories[e.id] = e

    def export_json(self) -> str:
        data = []
        for e in self.memories.values():
            data.append({
                "id": e.id,
                "memory_type": e.memory_type.value,
                "content": e.content,
                "tags": e.tags,
                "language": e.language,
                "confidence": e.confidence,
                "use_count": e.use_count,
                "metadata": e.metadata,
            })
        return json.dumps(data, indent=2)

    def import_json(self, json_str: str) -> int:
        data = json.loads(json_str)
        count = 0
        for d in data:
            d["memory_type"] = MemoryType(d["memory_type"])
            e = MemoryEntry(**d)
            self.store(e)
            count += 1
        return count
