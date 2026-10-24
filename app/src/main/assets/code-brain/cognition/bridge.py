"""Optional bridge between the cognitive growth subsystem and RENA-BR.

The existing CodeBrain pipeline is untouched.  A CognitionRuntime owns a
GrowthEngine and synchronizes only *verified* skills and tools into the
existing CapabilityRegistry.  Retired or unverified capabilities are
deactivated, so the registry never claims more than the evidence shows.
"""
from __future__ import annotations

import os
from typing import Optional

from cognition.config import GrowthConfig
from cognition.growth import GrowthEngine, GrowthReport
from cognition.skills import Skill
from cognition.tools import ToolCapability, ToolService, ToolVerifier
from cognition.util import read_json


class CognitionRuntime:
    def __init__(self, config: Optional[GrowthConfig] = None,
                 engine: Optional[GrowthEngine] = None,
                 brain=None, registry=None):
        self.engine = engine or GrowthEngine(config)
        self.config = self.engine.config
        self.brain = brain
        self.registry = (registry
                         or getattr(brain, "capability_registry", None))
        self.tool_verifier = ToolVerifier(self.config)
        self.tool_service = ToolService(self.config)

    def attach(self, brain, registry=None) -> "CognitionRuntime":
        self.brain = brain
        self.registry = (registry
                         or getattr(brain, "capability_registry", None))
        return self

    def grow(self, cycles: Optional[int] = None,
             domain: str = "") -> GrowthReport:
        report = self.engine.practice(domain, cycles) if domain else \
            self.engine.grow(cycles)
        self.sync_capabilities()
        return report

    def verify_tools(self, include_network: Optional[bool] = None) -> dict:
        summary = self.tool_verifier.run(include_network=include_network)
        self.sync_capabilities()
        return summary

    def _active_skill_rows(self) -> list[Skill]:
        return self.engine.skills.active()

    def _verified_tool_rows(self) -> list[ToolCapability]:
        return self.tool_verifier.verified()

    def sync_capabilities(self) -> list[str]:
        active_skills = self._active_skill_rows()
        verified_tools = self._verified_tool_rows()
        names: list[str] = []
        rows = []
        for skill in active_skills:
            name = skill.capability_name or (
                "cognition." + skill.domain + "." +
                skill.name.replace(" ", "_").replace("/", "_"))
            skill.capability_name = name
            rows.append((name, skill.name, skill.description, "cognition"))
        for tool in verified_tools:
            rows.append((tool.name, tool.label, tool.description,
                         "cognition_tool"))
        for name, label, description, category in rows:
            names.append(name)
            if self.registry is None:
                continue
            cap = self.registry.get(name)
            if cap is None:
                self.registry._add(
                    name, "Cognition: " + label, description, category)
            else:
                cap.description = "Cognition: " + label + ": " + description
                cap.category = category
                cap.is_implemented = True
                cap.is_active = True
        if self.registry is not None:
            keep = set(names)
            for cap in self.registry.capabilities.values():
                if cap.name.startswith("cognition.") and cap.name not in keep:
                    cap.is_active = False
        return sorted(names)

    def status(self) -> dict:
        data = self.engine.status()
        data["tools"] = self.tool_verifier.summarize()
        data["active_capabilities"] = sorted(
            [s.capability_name or s.name for s in self._active_skill_rows()] +
            [t.name for t in self._verified_tool_rows()])
        data["attached_registry"] = self.registry is not None
        return data

    def capabilities(self) -> list[dict]:
        rows = []
        for skill in self._active_skill_rows():
            rows.append({
                "name": skill.capability_name or skill.name,
                "kind": "skill",
                "domain": skill.domain,
                "description": skill.description,
                "confidence": round(skill.confidence, 4),
                "attempts": skill.attempts,
                "successes": skill.successes,
                "holdout_successes": skill.holdout_successes,
                "transfer": skill.transfer,
            })
        for tool in self._verified_tool_rows():
            rows.append({
                "name": tool.name,
                "kind": "tool",
                "domain": tool.category,
                "description": tool.description,
                "checks": tool.checks,
                "limitations": tool.limitations,
            })
        return rows

    def limitations(self) -> list[str]:
        rows = self.engine.limitations()
        for tool in self.tool_verifier.tools.values():
            if tool.status == "failed":
                rows.append(
                    f"Tool capability {tool.name} failed its current "
                    "verification and is not registered.")
            elif tool.status == "skipped":
                rows.append(
                    f"Tool capability {tool.name} was skipped "
                    "(network disabled by configuration).")
        return rows

    def last_report(self) -> dict:
        path = os.path.join(self.config.data_dir, "reports",
                            "last_growth_report.json")
        data = read_json(path, {})
        return data if isinstance(data, dict) else {}

    def close(self) -> None:
        self.engine.close()
