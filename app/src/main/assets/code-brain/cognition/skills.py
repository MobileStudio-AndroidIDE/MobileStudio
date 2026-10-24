"""Verification-gated skill registry.

A skill is a *candidate* until independent evidence says it works.  The
gate is deliberately boring and strict:

* at least ``gate_min_successes`` successful verified attempts,
* a success rate of at least ``gate_min_rate``,
* at least ``gate_min_holdout`` successes on held-out checks,
* a valid payload checksum (tampering retires the skill).

Only ``active`` skills are available for reuse.  Promotion, retirement,
transfer evidence and the full attempt ledger are persisted, so a later
run can audit why a capability is (or is not) registered.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.util import (atomic_write_json, clamp, read_json,
                            stable_hash)


@dataclass
class Skill:
    name: str
    domain: str
    kind: str
    payload: dict
    description: str = ""
    preconditions: dict = field(default_factory=dict)
    status: str = "candidate"
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    holdout_successes: int = 0
    holdout_failures: int = 0
    confidence: float = 0.5
    transfer: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    id: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_goal: str = ""
    capability_name: str = ""
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "s_" + stable_hash(self.name, self.domain, length=12)
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        return stable_hash(self.name, self.domain, self.kind,
                           sorted(self.payload.items(), key=lambda kv: kv[0]),
                           length=20)

    def integrity_ok(self) -> bool:
        return self.checksum == self.compute_checksum()

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def holdout_rate(self) -> float:
        total = self.holdout_successes + self.holdout_failures
        return self.holdout_successes / total if total else 0.0

    def record(self, success: bool, holdout: bool = False,
               note: str = "", transfer_domain: str = "") -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        if holdout:
            if success:
                self.holdout_successes += 1
            else:
                self.holdout_failures += 1
        if transfer_domain:
            row = self.transfer.setdefault(
                str(transfer_domain),
                {"trials": 0, "successes": 0})
            row["trials"] += 1
            if success:
                row["successes"] += 1
        # Laplace-smoothed confidence, not an overconfident raw ratio.
        self.confidence = clamp((self.successes + 1.0) /
                                (self.attempts + 2.0))
        self.updated_at = time.time()
        self.evidence.append({
            "at": self.updated_at,
            "success": bool(success),
            "holdout": bool(holdout),
            "transfer_domain": transfer_domain,
            "note": note[:400],
        })
        self.evidence = self.evidence[-200:]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "kind": self.kind,
            "payload": self.payload,
            "description": self.description,
            "preconditions": self.preconditions,
            "status": self.status,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "holdout_successes": self.holdout_successes,
            "holdout_failures": self.holdout_failures,
            "confidence": self.confidence,
            "transfer": self.transfer,
            "evidence": self.evidence,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_goal": self.source_goal,
            "capability_name": self.capability_name,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class VerificationGate:
    min_successes: int = 3
    min_rate: float = 0.75
    min_holdout: int = 1
    require_integrity: bool = True

    def evaluate(self, skill: Skill) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.require_integrity and not skill.integrity_ok():
            reasons.append("payload_checksum_mismatch")
        if skill.attempts < self.min_successes:
            reasons.append(
                f"attempts {skill.attempts} < {self.min_successes}")
        if skill.success_rate < self.min_rate:
            reasons.append(
                f"success_rate {skill.success_rate:.2f} < {self.min_rate}")
        if skill.holdout_successes < self.min_holdout:
            reasons.append(
                f"holdout_successes {skill.holdout_successes} < "
                f"{self.min_holdout}")
        if skill.failures > skill.successes:
            reasons.append("failures_exceed_successes")
        return (not reasons), reasons


class SkillRegistry:
    FILE = "skills.json"

    def __init__(self, data_dir: str, gate: Optional[VerificationGate] = None):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, self.FILE)
        self.gate = gate or VerificationGate()
        self.skills: dict[str, Skill] = {}
        self.promotions: list[dict] = []
        self.retirements: list[dict] = []
        self._load()

    def propose(self, name: str, domain: str, kind: str, payload: dict,
                description: str = "", preconditions: Optional[dict] = None,
                source_goal: str = "") -> Skill:
        existing = self.find(name, domain)
        if existing is not None:
            if existing.payload != payload:
                existing.payload = dict(payload)
                existing.checksum = existing.compute_checksum()
                existing.status = "candidate"
                existing.version += 1
                existing.updated_at = time.time()
            return existing
        skill = Skill(name=name, domain=domain, kind=kind,
                      payload=dict(payload), description=description,
                      preconditions=dict(preconditions or {}),
                      source_goal=source_goal)
        self.skills[skill.id] = skill
        return skill

    def find(self, name: str, domain: str = "") -> Optional[Skill]:
        for skill in self.skills.values():
            if skill.name == name and (not domain or skill.domain == domain):
                return skill
        return None

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def active(self, domain: str = "") -> list[Skill]:
        rows = [s for s in self.skills.values()
                if s.status == "active" and s.integrity_ok()
                and (not domain or s.domain == domain)]
        rows.sort(key=lambda s: (-s.confidence, -s.successes, s.name))
        return rows

    def candidates(self, domain: str = "") -> list[Skill]:
        return [s for s in self.skills.values()
                if s.status == "candidate"
                and (not domain or s.domain == domain)]

    def record_attempt(self, skill: Skill, success: bool,
                       holdout: bool = False, note: str = "",
                       transfer_domain: str = "",
                       auto_promote: bool = True) -> dict:
        skill.record(success, holdout=holdout, note=note,
                     transfer_domain=transfer_domain)
        promotion = None
        if auto_promote and skill.status == "candidate":
            promotion = self.try_promote(skill)
        return {
            "skill": skill.name,
            "success": success,
            "status": skill.status,
            "promotion": promotion,
        }

    def try_promote(self, skill: Skill,
                    metadata: Optional[dict] = None) -> Optional[dict]:
        ok, reasons = self.gate.evaluate(skill)
        if not ok:
            return None
        skill.status = "active"
        skill.updated_at = time.time()
        record = {
            "skill": skill.name,
            "domain": skill.domain,
            "at": skill.updated_at,
            "attempts": skill.attempts,
            "successes": skill.successes,
            "holdout_successes": skill.holdout_successes,
            "confidence": skill.confidence,
            "reasons": list(reasons),
            "metadata": dict(metadata or {}),
        }
        self.promotions.append(record)
        return record

    def retire(self, skill: Skill, reason: str) -> None:
        skill.status = "retired"
        skill.updated_at = time.time()
        self.retirements.append({
            "skill": skill.name,
            "domain": skill.domain,
            "reason": reason,
            "at": skill.updated_at,
        })

    def matching(self, domain: str, input_types: Optional[dict] = None,
                 limit: int = 5) -> list[Skill]:
        rows = self.active(domain)
        if input_types is not None:
            wanted = list(input_types.values())
            exact = [s for s in rows
                     if s.preconditions.get("param_types") == wanted]
            if exact:
                rows = exact
        return rows[:max(1, limit)]

    def integrity_check(self) -> list[str]:
        broken = []
        for skill in self.skills.values():
            if not skill.integrity_ok():
                if skill.status == "active":
                    self.retire(skill, "integrity_check_failed")
                broken.append(skill.name)
        return broken

    def sync_active_capabilities(self) -> list[str]:
        names = []
        for skill in self.active():
            name = skill.capability_name or (
                "cognition." + skill.domain + "." +
                skill.name.replace(" ", "_").replace("/", "_"))
            skill.capability_name = name
            names.append(name)
        return names

    def stats(self) -> dict:
        return {
            "skills": len(self.skills),
            "active": len(self.active()),
            "candidates": len(self.candidates()),
            "retired": sum(1 for s in self.skills.values()
                           if s.status == "retired"),
            "promotions": len(self.promotions),
            "domains": sorted({s.domain for s in self.skills.values()}),
        }

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "skills": [s.to_dict() for s in self.skills.values()],
            "promotions": self.promotions[-500:],
            "retirements": self.retirements[-500:],
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("skills") or []:
            try:
                skill = Skill.from_dict(row)
            except Exception:
                continue
            if skill.name:
                if not skill.integrity_ok():
                    skill.status = "retired"
                self.skills[skill.id] = skill
        self.promotions = list(data.get("promotions") or [])
        self.retirements = list(data.get("retirements") or [])
