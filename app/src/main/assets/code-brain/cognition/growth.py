"""Autonomous growth loop: failure -> goal -> practice -> verify -> skill."""
from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from cognition.abstraction import AbstractionEngine
from cognition.compositional import CompositionalReasoner
from cognition.config import GrowthConfig
from cognition.domains import (ConversationDomain, DataAnalysisDomain,
                               DomainAdapter, ExperimentContext,
                               FileDomain, GraphDomain, ImageDomain,
                               ListProgramDomain, LogicDomain, SequenceDomain,
                               Task)
from cognition.dsl import ProgramLibrary, Synthesizer, expr_from_dict
from cognition.journal import GrowthJournal
from cognition.memory import (ConceptMemory, Consolidator, EpisodicMemory,
                              WorkingMemory)
from cognition.predictive import PredictiveEngine
from cognition.sandbox import Sandbox, SelfModifier
from cognition.skills import Skill, SkillRegistry, VerificationGate
from cognition.util import (atomic_write_json, json_safe, read_json,
                            stable_hash)
from cognition.vsa import VSA


PERSISTENT_PAYLOAD = {
    "program": ("expression", "input_types", "code", "target_type",
                "macro_used"),
    "sequence_hypothesis": ("strategy",),
    "logic_result": ("strategy",),
    "graph_algorithm": ("strategy", "query", "code"),
    "image_classifier": ("mode", "centroids", "rules"),
    "image_renderer": ("renderer",),
    "file_plan": ("strategy", "operation"),
    "conversation_answer": ("strategy",),
    "data_algorithm": ("strategy", "query", "code"),
}

SKILL_PREFIX = {
    "program": "program",
    "sequence_hypothesis": "sequence",
    "logic_result": "logic",
    "graph_algorithm": "graph",
    "image_classifier": "image_classify",
    "image_renderer": "image_render",
    "file_plan": "file",
    "conversation_answer": "conversation",
    "data_algorithm": "data",
}


@dataclass
class GrowthReport:
    session_id: str
    started_at: float
    finished_at: float = 0.0
    duration_s: float = 0.0
    cycles: int = 0
    practice_cycles: int = 0
    tasks_attempted: int = 0
    tasks_verified: int = 0
    task_failures: int = 0
    goals_started: int = 0
    goals_completed: int = 0
    failure_goals: int = 0
    failure_recoveries: int = 0
    validation_trials: int = 0
    validation_successes: int = 0
    transfer_trials: int = 0
    transfer_successes: int = 0
    exploration_escalations: int = 0
    consolidations: int = 0
    abstractions_mined: int = 0
    skills_promoted: list[str] = field(default_factory=list)
    skills_active: list[str] = field(default_factory=list)
    skills_candidate: list[str] = field(default_factory=list)
    domains: dict = field(default_factory=dict)
    stopped_reason: str = ""
    limitations: list[str] = field(default_factory=list)
    self_modification_enabled: bool = False

    @property
    def transfers(self) -> int:
        return self.transfer_trials

    @property
    def success_rate(self) -> float:
        if not self.tasks_attempted:
            return 0.0
        return self.tasks_verified / self.tasks_attempted

    def to_dict(self) -> dict:
        data = asdict(self)
        data["transfers"] = self.transfer_trials
        data["success_rate"] = round(self.success_rate, 4)
        return data

    def save(self, path: str) -> None:
        atomic_write_json(path, self.to_dict())


class GrowthEngine:
    """One bounded autonomous-growth session over many experiment domains."""

    def __init__(self, config: Optional[GrowthConfig] = None,
                 domains: Optional[list[DomainAdapter]] = None):
        self.config = config or GrowthConfig()
        self.config.ensure_dirs()
        self.vsa = VSA(self.config.dimension, self.config.vector_seed)
        self.concepts = ConceptMemory(
            os.path.join(self.config.data_dir, "concepts"), self.vsa,
            self.config)
        self.working = WorkingMemory(self.vsa, self.config.working_capacity)
        self.episodes = EpisodicMemory(
            os.path.join(self.config.data_dir, "episodes"),
            self.config.episode_capacity)
        self.predictive = PredictiveEngine(
            os.path.join(self.config.data_dir, "predictive"), self.vsa)
        self.library = ProgramLibrary(
            os.path.join(self.config.data_dir, "library"))
        self.abstractions = AbstractionEngine(self.library, self.config)
        self.gate = VerificationGate(
            min_successes=self.config.gate_min_successes,
            min_rate=self.config.gate_min_rate,
            min_holdout=self.config.gate_min_holdout)
        self.skills = SkillRegistry(
            os.path.join(self.config.data_dir, "skills"), self.gate)
        self.reasoner = CompositionalReasoner(self.vsa)
        self.synthesizer = Synthesizer(
            max_candidates=self.config.max_synthesis_candidates,
            max_cost=6)
        self.sandbox = Sandbox(self.config)
        self.journal = GrowthJournal(
            os.path.join(self.config.data_dir, "journal"),
            self.config.journal_limit)
        self.consolidator = Consolidator(
            self.concepts, self.episodes, self.working, self.predictive,
            self.config)
        self.self_modifier = (SelfModifier(self.config)
                              if self.config.allow_self_modification
                              else None)
        self.rng = random.Random(self.config.seed)
        self.domains = self._build_domains(domains)
        self.goals_path = os.path.join(self.config.data_dir, "goals.json")
        self.reports_dir = os.path.join(self.config.data_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.goals = self._load_goals()
        self._session_goal_ids: set[str] = set()
        self._escalation: dict[str, int] = {}
        self._transferred: set[str] = set()
        self._promoted_ids: set[str] = {
            s.id for s in self.skills.active()}
        self._rehearsed: dict[str, int] = {}
        self._forced_domain = ""
        self._budget_deadline = 0.0
        self._cycle_deadline = 0.0
        self.report = GrowthReport(
            session_id="gr_" + stable_hash(time.time(), self.config.seed,
                                           length=12),
            started_at=time.time(),
            self_modification_enabled=self.self_modifier is not None)

    def _build_domains(self, domains: Optional[list[DomainAdapter]]
                       ) -> dict[str, DomainAdapter]:
        if domains is not None:
            return {d.name: d for d in domains}
        workspace = os.path.join(self.config.data_dir, "workspace")
        artifacts = os.path.join(self.config.data_dir, "artifacts")
        os.makedirs(artifacts, exist_ok=True)
        rows: list[DomainAdapter] = [
            ListProgramDomain("math", "lists"),
            ListProgramDomain("code", "lists"),
            SequenceDomain(),
            LogicDomain(),
            GraphDomain(),
            ImageDomain(artifacts),
            FileDomain(workspace),
            ConversationDomain(),
            DataAnalysisDomain(),
        ]
        return {d.name: d for d in rows}

    def _load_goals(self) -> list[dict]:
        data = read_json(self.goals_path, {"goals": []})
        rows = data.get("goals") if isinstance(data, dict) else []
        return [g for g in rows or [] if isinstance(g, dict)]

    def _save_goals(self) -> None:
        open_rows = [g for g in self.goals if g.get("status") == "open"]
        closed = [g for g in self.goals if g.get("status") != "open"][-200:]
        atomic_write_json(self.goals_path, {
            "version": 1,
            "goals": (open_rows + closed)[-260:],
        })

    def status(self) -> dict:
        domains = {}
        for name, adapter in self.domains.items():
            domains[name] = {
                "description": adapter.description,
                "active_skills": len(self.skills.active(name)),
                "candidate_skills": len(self.skills.candidates(name)),
            }
        goals = [g for g in self.goals if g.get("status") == "open"]
        return {
            "data_dir": self.config.data_dir,
            "skills": self.skills.stats(),
            "domains": domains,
            "concepts": self.concepts.stats(),
            "episodes": self.episodes.stats(),
            "predictive": self.predictive.stats(),
            "library": self.library.stats(),
            "open_goals": goals[-10:],
            "open_goal_count": len(goals),
            "journal": self.journal.summary(),
            "limitations": self.limitations(),
        }

    def limitations(self) -> list[str]:
        rows = [
            "This system does not use a pretrained model or an external "
            "LLM/API; its semantics and language coverage are limited to "
            "the generated task families in this package.",
            "Reported successes are on synthetic, independently checked "
            "tasks. They do not establish performance on natural data.",
            "The sandbox combines static checks, a separate process, file "
            "permissions and resource limits; it is not a hardened OS "
            "security boundary on Android.",
            "On Android/Termux the Bionic linker is incompatible with "
            "RLIMIT_AS, so the address-space cap is skipped there; CPU and "
            "file-descriptor limits remain active.",
            "Transfer is measured on fresh same-domain tasks, not on "
            "arbitrary out-of-distribution problems.",
            "Mined abstractions are typed sub-expressions of verified "
            "programs; they are reusable syntax, not causal world models.",
        ]
        if self.self_modifier is None:
            rows.append(
                "Self-modification is disabled by default; no source code "
                "was changed by this growth run.")
        else:
            rows.append(
                "Self-modification is enabled but every patch must pass "
                "tests and is rolled back on failure.")
        for adapter in self.domains.values():
            try:
                rows.extend(adapter.limitations())
            except Exception:
                continue
        return rows

    def save(self) -> None:
        self.concepts.save()
        self.episodes.save()
        self.predictive.save()
        self.library.save()
        self.skills.save()
        self._save_goals()

    def close(self) -> None:
        self.save()

    def _time_left(self) -> float:
        return self._budget_deadline - time.monotonic()

    def _cycle_left(self) -> float:
        return self._cycle_deadline - time.monotonic()

    def _ensure_domain_stat(self, domain: str) -> dict:
        return self.report.domains.setdefault(domain, {
            "attempts": 0,
            "verified": 0,
            "failed": 0,
            "skills_promoted": 0,
            "transfer_trials": 0,
            "transfer_successes": 0,
            "last_failure": "",
        })

    def _select_domain(self, cycle: int) -> Optional[DomainAdapter]:
        names = list(self.domains)
        if not names:
            return None
        if self._forced_domain in self.domains:
            return self.domains[self._forced_domain]
        open_failure = [
            g for g in self.goals
            if g.get("status") == "open"
            and g.get("source") == "failure"
            and int(g.get("attempts", 0)) < 3
            and g.get("domain") in self.domains
        ]
        if open_failure:
            open_failure.sort(key=lambda g: (
                -float(g.get("priority", 0.5)), g.get("updated_at", 0.0)))
            return self.domains[open_failure[0]["domain"]]
        unseen = [name for name in names
                  if self._ensure_domain_stat(name)["attempts"] == 0]
        if unseen:
            return self.domains[unseen[(cycle - 1) % len(unseen)]]
        if self.rng.random() < 0.2:
            return self.domains[self.rng.choice(names)]
        def weakness(name: str):
            row = self._ensure_domain_stat(name)
            rate = (row["verified"] / row["attempts"]
                    if row["attempts"] else 0.0)
            return (rate, -row["attempts"], name)
        return self.domains[min(names, key=weakness)]

    def _new_goal(self, adapter: DomainAdapter, source: str,
                  failure: str = "") -> dict:
        if failure:
            text = (f"diagnose {failure} and acquire a verified "
                    f"{adapter.name} strategy")
        else:
            text = (f"strengthen a verified capability in {adapter.name}: "
                    f"{adapter.description}")
        goal = {
            "id": "g_" + stable_hash(adapter.name, text, time.time(),
                                     self.rng.random(), length=12),
            "domain": adapter.name,
            "text": text,
            "source": source,
            "failure": failure,
            "status": "open",
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "priority": 0.8 if source == "failure" else 0.55,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self.goals.append(goal)
        self._session_goal_ids.add(goal["id"])
        self.report.goals_started += 1
        if source == "failure":
            self.report.failure_goals += 1
        self.journal.log("goal_created", cycle=self.report.cycles,
                         goal=goal["text"], domain=adapter.name,
                         source=source)
        return goal

    def _goal_for(self, adapter: DomainAdapter) -> dict:
        rows = [g for g in self.goals
                if g.get("domain") == adapter.name
                and g.get("status") == "open"
                and int(g.get("attempts", 0)) < 3]
        if rows:
            rows.sort(key=lambda g: (
                g.get("source") != "failure",
                -float(g.get("priority", 0.0)), g.get("created_at", 0.0)))
            goal = rows[0]
            self._session_goal_ids.add(goal["id"])
            return goal
        last_failure = self._ensure_domain_stat(
            adapter.name).get("last_failure") or ""
        return self._new_goal(adapter, "failure" if last_failure else "self",
                              last_failure)

    def grow(self, cycles: Optional[int] = None) -> GrowthReport:
        limit = self.config.max_cycles if cycles is None else int(cycles)
        limit = max(0, min(limit, self.config.max_cycles))
        started = time.time()
        self._budget_deadline = time.monotonic() + max(
            1.0, float(self.config.time_budget_seconds))
        self.report = GrowthReport(
            session_id=self.report.session_id,
            started_at=started,
            self_modification_enabled=self.self_modifier is not None)
        self.journal.log("session_start", cycle=0, cycles=limit,
                         seed=self.config.seed,
                         data_dir=self.config.data_dir)
        broken = self.skills.integrity_check()
        if broken:
            self.journal.log("integrity_retired", cycle=0, skills=broken)
        stopped = "cycle_budget_reached" if limit else "no_cycles"
        try:
            for cycle in range(1, limit + 1):
                if self._time_left() <= 0:
                    stopped = "time_budget_exhausted"
                    break
                self._cycle_deadline = (time.monotonic() +
                                        max(1.0, self.config.cycle_seconds))
                adapter = self._select_domain(cycle)
                if adapter is None:
                    stopped = "no_domains"
                    break
                goal = self._goal_for(adapter)
                wm_goal = self.working.push_goal(
                    goal["text"], priority=float(goal.get("priority", 0.5)),
                    metadata={"domain": adapter.name, "source":
                              goal.get("source", "")})
                self.report.cycles = cycle
                self.report.practice_cycles += 1
                solved = 0
                tasks = min(self.config.practice_per_cycle,
                            self.config.max_tasks_per_goal)
                for _ in range(max(1, tasks)):
                    if (self._time_left() <= 1.0
                            or self._cycle_left() <= 0.5):
                        break
                    if self._practice_task(adapter, goal):
                        solved += 1
                if solved:
                    goal["successes"] = int(goal.get("successes", 0)) + solved
                    goal["status"] = "done"
                    goal["updated_at"] = time.time()
                    self.report.goals_completed += 1
                    if goal.get("source") == "failure":
                        self.report.failure_recoveries += 1
                else:
                    goal["attempts"] = int(goal.get("attempts", 0)) + 1
                    goal["failures"] = int(goal.get("failures", 0)) + 1
                    goal["status"] = ("open" if goal["attempts"] < 3
                                      else "abandoned")
                    goal["updated_at"] = time.time()
                self.working.complete_goal(
                    wm_goal["id"], "done" if solved else "open")
                if (self.config.consolidation_interval > 0
                        and cycle % self.config.consolidation_interval == 0):
                    summary = self.consolidator.consolidate(seed=self.config.seed + cycle)
                    self.report.consolidations += 1
                    self.journal.log("consolidation", cycle=cycle,
                                     summary=summary)
                self.save()
                self.journal.log("cycle_end", cycle=cycle,
                                 domain=adapter.name,
                                 goal=goal["id"], solved=solved)
                if self._time_left() <= 0:
                    stopped = "time_budget_exhausted"
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
            self.journal.log("session_interrupted", cycle=self.report.cycles)
        except Exception as exc:
            stopped = "internal_error"
            self.journal.log("session_error", cycle=self.report.cycles,
                             error=f"{type(exc).__name__}: {exc}")
        finally:
            self.report.stopped_reason = stopped
            self.report.finished_at = time.time()
            self.report.duration_s = round(
                self.report.finished_at - started, 4)
            self.report.skills_active = [
                s.name for s in self.skills.active()]
            self.report.skills_candidate = [
                s.name for s in self.skills.candidates()]
            self.report.limitations = self.limitations()
            try:
                self.save()
            except Exception as exc:
                self.report.limitations.append(
                    f"Final persistence failed: {type(exc).__name__}: {exc}")
            self.journal.log("session_end", cycle=self.report.cycles,
                             stopped=stopped,
                             verified=self.report.tasks_verified,
                             promoted=self.report.skills_promoted)
            self.report.save(os.path.join(
                self.reports_dir, f"growth_{self.report.session_id}.json"))
            self.report.save(os.path.join(
                self.reports_dir, "last_growth_report.json"))
        return self.report

    def practice(self, domain: str = "", cycles: int = 1) -> GrowthReport:
        if domain and domain not in self.domains:
            raise ValueError(f"unknown domain: {domain}")
        self._forced_domain = domain
        try:
            return self.grow(cycles=cycles)
        finally:
            self._forced_domain = ""

    def _context(self, depth: int) -> ExperimentContext:
        return ExperimentContext(
            vsa=self.vsa,
            synthesizer=self.synthesizer,
            library=self.library,
            skills=self.skills,
            predictive=self.predictive,
            concepts=self.concepts,
            working=self.working,
            reasoner=self.reasoner,
            rng=self.rng,
            explore_depth=max(1, min(4, depth)),
            max_candidates=self.config.max_synthesis_candidates,
            sandbox=self.sandbox,
            task_budget_seconds=max(1.0, self._cycle_left()))

    @staticmethod
    def _feature_tag(task: Task) -> str:
        payload = task.payload or {}
        for key in ("query", "operation", "mode", "family", "target_type"):
            if payload.get(key):
                return f"{key}:{payload[key]}"
        tags = [tag for tag in task.tags if tag != task.domain]
        return tags[0] if tags else task.domain

    def _practice_task(self, adapter: DomainAdapter, goal: dict) -> bool:
        stat = self._ensure_domain_stat(adapter.name)
        stat["attempts"] += 1
        self.report.tasks_attempted += 1
        try:
            task = adapter.generate_task(
                self.rng,
                difficulty=min(0.9, 0.35 + 0.03 * stat["attempts"]),
                goal=goal["text"])
        except Exception as exc:
            stat["failed"] += 1
            self.report.task_failures += 1
            stat["last_failure"] = "task_generation"
            self.journal.log("task_generation_failed",
                             cycle=self.report.cycles,
                             domain=adapter.name,
                             error=f"{type(exc).__name__}: {exc}")
            return False
        self.concepts.add(
            task.domain, kind="domain",
            attributes={"description": adapter.description},
            relations=[("goal", goal["text"][:80])],
            source="growth", confidence=0.7)
        task_vector = adapter.task_vector(task, self.vsa)
        self.working.add(task.id, kind="task", vector=task_vector,
                         activation=0.8)
        failures: list[str] = []
        max_attempts = max(1, self.config.max_attempts_per_task)
        for attempt in range(1, max_attempts + 1):
            if self._time_left() <= 1.0 or self._cycle_left() <= 0.3:
                break
            depth = min(3, 2 + int(self._escalation.get(adapter.name, 0)))
            ctx = self._context(depth)
            try:
                outcome = adapter.experiment(task, ctx)
            except Exception as exc:
                failures.append("experiment_error")
                self.journal.log(
                    "experiment_error", cycle=self.report.cycles,
                    domain=adapter.name, task=task.id,
                    error=f"{type(exc).__name__}: {exc}")
                continue
            if outcome.candidate is None:
                kind = outcome.failure_kind or "no_candidate"
                failures.append(kind)
                self.journal.log(
                    "experiment_failed", cycle=self.report.cycles,
                    domain=adapter.name, task=task.id, kind=kind,
                    detail=outcome.detail[:300], tried=outcome.tried)
                continue
            try:
                verification = adapter.verify(task, outcome.candidate, ctx)
            except Exception as exc:
                failures.append("verification_error")
                verification = None
                self.journal.log(
                    "verification_error", cycle=self.report.cycles,
                    domain=adapter.name, task=task.id,
                    error=f"{type(exc).__name__}: {exc}")
            if verification is not None and verification.passed:
                skill, promoted = self._learn_success(
                    task, outcome.candidate, verification)
                stat["verified"] += 1
                self.report.tasks_verified += 1
                self._escalation[adapter.name] = 0
                self._record_episode(
                    task, True, outcome.candidate, verification, skill,
                    failure="")
                self.working.add(skill.name, kind="skill", activation=1.0)
                if skill.status in ("candidate", "active"):
                    self._rehearse_skill(skill, task)
                return True
            kind = "verification_failed"
            failures.append(kind)
            if verification is not None:
                self._learn_failure(task, outcome.candidate, verification)
                self.journal.log(
                    "verification_failed", cycle=self.report.cycles,
                    domain=adapter.name, task=task.id,
                    detail=verification.detail[:300],
                    checks=json_safe(verification.checks))
        kind = failures[-1] if failures else "budget_exhausted"
        stat["failed"] += 1
        stat["last_failure"] = kind
        self.report.task_failures += 1
        previous = int(self._escalation.get(adapter.name, 0))
        if previous < 1:
            self._escalation[adapter.name] = previous + 1
            self.report.exploration_escalations += 1
        self._record_episode(task, False, None, None, None,
                             failure=kind, attempts=attempt)
        self.journal.log("task_failed", cycle=self.report.cycles,
                         domain=adapter.name, task=task.id, kind=kind,
                         attempts=attempt)
        if goal.get("source") != "failure":
            goal["source"] = "failure"
            goal["failure"] = kind
            goal["priority"] = 0.8
            self.report.failure_goals += 1
        return False

    def _payload_for_skill(self, candidate) -> dict:
        keys = PERSISTENT_PAYLOAD.get(candidate.kind)
        payload = candidate.payload or {}
        if keys:
            return {key: json_safe(payload[key])
                    for key in keys if key in payload}
        return {key: json_safe(value) for key, value in payload.items()
                if key not in ("prediction", "prefix", "steps", "response")}

    def _skill_identity(self, task: Task, candidate) -> tuple[str, str]:
        payload = self._payload_for_skill(candidate)
        prefix = SKILL_PREFIX.get(candidate.kind, candidate.kind or "skill")
        strategy = candidate.strategy or candidate.kind
        if candidate.kind == "program":
            name = prefix + "_" + stable_hash(
                payload.get("expression"), payload.get("input_types"),
                length=10)
        elif candidate.kind in ("image_classifier", "image_renderer"):
            name = prefix + "_" + stable_hash(payload, length=10)
        else:
            name = f"{prefix}_{strategy}"
            if candidate.kind in ("graph_algorithm", "data_algorithm"):
                name += "_" + str(payload.get("query") or "")
            elif candidate.kind == "file_plan":
                name += "_" + str(payload.get("operation") or "")
        name = name.replace(" ", "_").replace("/", "_")[:120]
        capability = ("cognition." + task.domain + "." +
                      name.replace(".", "_"))
        return name, capability

    def _propose_for(self, task: Task, candidate) -> tuple[Skill, dict]:
        name, capability = self._skill_identity(task, candidate)
        payload = self._payload_for_skill(candidate)
        preconditions = {}
        if candidate.kind == "program" and payload.get("input_types"):
            preconditions["param_types"] = list(
                payload["input_types"].values())
        skill = self.skills.propose(
            name=name, domain=task.domain, kind=candidate.kind,
            payload=payload, description=candidate.description[:300],
            preconditions=preconditions, source_goal=task.goal[:200])
        if not skill.capability_name:
            skill.capability_name = capability
        return skill, payload

    def _learn_success(self, task: Task, candidate, verification
                       ) -> tuple[Skill, bool]:
        skill = None
        if candidate.skill_name:
            skill = self.skills.find(candidate.skill_name, task.domain)
        if skill is None:
            skill, _ = self._propose_for(task, candidate)
        result = self.skills.record_attempt(
            skill, True, holdout=True, note=verification.detail)
        promoted = bool(result.get("promotion"))
        if promoted:
            self._note_promotion(skill, task)
        self.concepts.add_relation(
            task.domain, "has_skill", skill.name,
            source_tag="growth", weight=skill.confidence)
        return skill, promoted

    def _note_promotion(self, skill: Skill, task: Task) -> None:
        if skill.id in self._promoted_ids:
            return
        self._promoted_ids.add(skill.id)
        self.report.skills_promoted.append(skill.name)
        self.report.domains[task.domain]["skills_promoted"] += 1
        self.journal.log(
            "skill_promoted", cycle=self.report.cycles,
            domain=task.domain, skill=skill.name, attempts=skill.attempts,
            confidence=round(skill.confidence, 3))
        if skill.kind != "program":
            return
        payload = skill.payload or {}
        input_types = payload.get("input_types") or {}
        try:
            self.library.add(
                list(input_types), list(input_types.values()),
                expr_from_dict(payload.get("expression") or {}),
                domain=task.domain, source_skill=skill.name)
        except Exception as exc:
            self.journal.log("macro_error", cycle=self.report.cycles,
                             skill=skill.name,
                             error=f"{type(exc).__name__}: {exc}")
        try:
            mined = self.abstractions.mine_skill(skill)
        except Exception as exc:
            mined = []
            self.journal.log("abstraction_error", cycle=self.report.cycles,
                             skill=skill.name,
                             error=f"{type(exc).__name__}: {exc}")
        if mined:
            self.report.abstractions_mined += len(mined)
            self.journal.log(
                "abstractions_mined", cycle=self.report.cycles,
                skill=skill.name, count=len(mined),
                names=[m.name for m in mined])

    def _learn_failure(self, task: Task, candidate, verification) -> None:
        skill = None
        if candidate.skill_name:
            skill = self.skills.find(candidate.skill_name, task.domain)
        if skill is None:
            try:
                skill, _ = self._propose_for(task, candidate)
            except Exception:
                return
        if skill.status == "retired":
            return
        self.skills.record_attempt(
            skill, False, holdout=False,
            note=(verification.detail or "verification failed")[:300])

    def _record_episode(self, task: Task, success: bool, candidate,
                        verification, skill: Optional[Skill],
                        failure: str = "", attempts: int = 1) -> None:
        domain = task.domain
        tag = self._feature_tag(task)
        if skill is not None:
            strategy = (candidate.strategy if candidate else "") or ""
            symbol_sequence = [domain, tag, "verified", skill.name]
            context = {
                "subject": domain,
                "relation": "has_skill",
                "skill": skill.name,
                "strategy": strategy,
                "failure": "",
                "symbol_sequence": symbol_sequence,
                "task": task.id,
            }
            action = f"verify:{skill.name}"
            reward = 1.0
        else:
            symbol_sequence = [domain, tag, "failed", failure or "unknown"]
            context = {
                "subject": domain,
                "relation": "weak_at",
                "skill": "",
                "strategy": "",
                "failure": failure or "unknown",
                "symbol_sequence": symbol_sequence,
                "task": task.id,
                "attempts": attempts,
            }
            action = f"attempt:{failure or 'unknown'}"
            reward = -0.5
        surprise = 0.0
        try:
            surprise = self.predictive.surprise(
                symbol_sequence[-1], symbol_sequence[:-1])
            self.predictive.observe_sequence(symbol_sequence)
        except Exception:
            pass
        outcome = {
            "task": task.id,
            "goal": task.goal[:200],
            "skill": skill.name if skill else "",
            "strategy": getattr(candidate, "strategy", "") if candidate else "",
            "failure": failure,
            "verification": verification.to_dict() if verification else {},
            "attempts": attempts,
        }
        self.episodes.record(
            domain=domain, action=action, outcome=outcome,
            success=success, reward=reward, surprise=surprise,
            context=context, tags=list(task.tags) + [domain],
            source="growth")
        if not success:
            self.concepts.add(
                "failure:" + str(failure or "unknown"),
                kind="failure_mode",
                attributes={"domain": domain},
                relations=[("seen_in", domain)],
                source="growth", confidence=0.55)

    def _candidate_uses_skill(self, candidate, skill: Skill) -> bool:
        if candidate is None:
            return False
        if candidate.skill_name:
            return candidate.skill_name == skill.name
        payload = candidate.payload or {}
        skill_payload = skill.payload or {}
        if candidate.kind == "program":
            return payload.get("expression") == skill_payload.get("expression")
        if candidate.kind in ("graph_algorithm", "data_algorithm"):
            return (payload.get("code") == skill_payload.get("code") and
                    payload.get("strategy") == skill_payload.get("strategy"))
        if candidate.kind == "image_classifier":
            return (payload.get("centroids") == skill_payload.get("centroids")
                    and payload.get("rules") == skill_payload.get("rules"))
        if candidate.kind == "image_renderer":
            return payload.get("renderer") == skill_payload.get("renderer")
        return (candidate.strategy and
                candidate.strategy == (skill_payload.get("strategy") or
                                       skill_payload.get("operation")))

    def _skill_candidate_for(self, task: Task, skill: Skill):
        try:
            from cognition.domains import Candidate
            payload = dict(skill.payload or {})
            if skill.kind == "program":
                return Candidate("program", payload, skill.description,
                                 strategy="skill_reuse",
                                 skill_name=skill.name)
            strategy = payload.get("strategy") or payload.get("operation") or ""
            return Candidate(skill.kind, payload, skill.description,
                             strategy=strategy, skill_name=skill.name)
        except Exception:
            return None

    def _rehearse_skill(self, skill: Skill, source_task: Task) -> None:
        adapter = self.domains.get(skill.domain)
        if adapter is None:
            return
        if skill.status == "active" and skill.id in self._transferred:
            return
        runs = int(self._rehearsed.get(skill.id, 0))
        if runs >= 2:
            return
        self._rehearsed[skill.id] = runs + 1
        if skill.status == "candidate":
            needed = max(0, self.gate.min_successes - skill.successes)
        else:
            needed = 0
        active_trials = 0 if skill.id in self._transferred else max(
            1, min(3, int(self.config.transfer_trials)))
        total = min(6, needed + active_trials + 1)
        for index in range(total):
            if self._time_left() <= 1.0 or self._cycle_left() <= 0.2:
                break
            try:
                task = adapter.transfer_task(source_task, self.rng)
            except Exception as exc:
                self.journal.log("rehearsal_task_error",
                                 cycle=self.report.cycles,
                                 skill=skill.name,
                                 error=f"{type(exc).__name__}: {exc}")
                break
            if task is None:
                break
            ctx = self._context(min(3, 2 + int(
                self._escalation.get(adapter.name, 0))))
            try:
                outcome = adapter.experiment(task, ctx)
                candidate = outcome.candidate
                if (candidate is None
                        or not self._candidate_uses_skill(candidate, skill)):
                    alternative = self._skill_candidate_for(task, skill)
                    if alternative is None:
                        self.journal.log(
                            "rehearsal_not_reused",
                            cycle=self.report.cycles, skill=skill.name,
                            domain=adapter.name, task=task.id)
                        break
                    candidate = alternative
                verification = adapter.verify(task, candidate, ctx)
            except Exception as exc:
                self.journal.log(
                    "rehearsal_error", cycle=self.report.cycles,
                    skill=skill.name, error=f"{type(exc).__name__}: {exc}")
                break
            was_active = skill.status == "active"
            transfer_domain = adapter.name if was_active else ""
            self.skills.record_attempt(
                skill, verification.passed, holdout=True,
                note=("transfer: " if was_active else "validation: ") +
                     verification.detail[:200],
                transfer_domain=transfer_domain)
            if not was_active:
                self.report.validation_trials += 1
                if verification.passed:
                    self.report.validation_successes += 1
            else:
                self.report.transfer_trials += 1
                self.report.domains[adapter.name]["transfer_trials"] += 1
            if not verification.passed:
                self.journal.log(
                    "rehearsal_failed", cycle=self.report.cycles,
                    skill=skill.name, domain=adapter.name, task=task.id,
                    detail=verification.detail[:300],
                    transfer=was_active)
                break
            if was_active:
                self.report.transfer_successes += 1
                self.report.domains[adapter.name]["transfer_successes"] += 1
                self.journal.log(
                    "transfer_success", cycle=self.report.cycles,
                    skill=skill.name, domain=adapter.name, task=task.id)
            else:
                self.journal.log(
                    "validation_success", cycle=self.report.cycles,
                    skill=skill.name, domain=adapter.name, task=task.id,
                    attempts=skill.attempts,
                    holdout_successes=skill.holdout_successes)
            self._record_episode(
                task, True, candidate, verification, skill, failure="")
            if skill.status == "active" and not was_active:
                self._note_promotion(skill, task)
            if was_active:
                active_trials -= 1
                if active_trials <= 0:
                    self._transferred.add(skill.id)
                    break
        if skill.status == "active" and active_trials <= 0:
            self._transferred.add(skill.id)
        self.save()
