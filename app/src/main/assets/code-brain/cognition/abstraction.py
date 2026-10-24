"""Mine reusable sub-programs from independently verified programs.

Whole verified programs already enter the macro library.  This module
adds a second, finer layer: any typed sub-expression with at least one
free variable can become a macro of its own.  Later synthesis can then
compose these learned components instead of rediscovering them.
"""
from __future__ import annotations

from typing import Optional

from cognition.config import GrowthConfig
from cognition.dsl import (Macro, ProgramLibrary, expr_from_dict)
from cognition.skills import Skill


_SUPPORTED_TYPES = {"num", "list", "bool", "str"}


class AbstractionEngine:
    """Extract and register reusable sub-expressions from learned code."""

    def __init__(self, library: ProgramLibrary,
                 config: Optional[GrowthConfig] = None):
        self.library = library
        self.config = config or GrowthConfig()
        self.mined_total = 0

    def mine_skill(self, skill: Skill) -> list[Macro]:
        if getattr(skill, "kind", "") != "program":
            return []
        payload = skill.payload or {}
        data = payload.get("expression")
        input_types = payload.get("input_types") or {}
        if not isinstance(data, dict) or not data:
            return []
        try:
            expression = expr_from_dict(data)
        except Exception:
            return []
        return self.mine(expression, input_types=input_types,
                         domain=skill.domain,
                         source_skill=skill.name)

    def mine(self, expression, input_types: Optional[dict] = None,
             domain: str = "general", source_skill: str = "",
             max_new: Optional[int] = None) -> list[Macro]:
        if not getattr(self.config, "abstraction_enabled", True):
            return []
        declared = dict(input_types or {})
        limit = (int(max_new) if max_new is not None
                 else int(self.config.abstraction_max_per_skill))
        limit = max(0, limit)
        if limit == 0:
            return []
        minimum_depth = max(1, int(self.config.abstraction_min_depth))
        candidates = []
        seen = set()

        def walk(node):
            yield node
            for child in node.args:
                yield from walk(child)

        for node in walk(expression):
            if node is expression:
                continue
            if node.op in ("var", "const"):
                continue
            if _contains_macro(node):
                continue
            variables = node.variables()
            if not variables or len(variables) > 3:
                continue
            variable_types = _variable_types(node)
            if declared and any(name not in declared for name in variables):
                continue
            param_types = [declared.get(name) or variable_types.get(name)
                           for name in variables]
            if any(kind not in _SUPPORTED_TYPES for kind in param_types):
                continue
            depth = _expr_depth(node)
            size = _expr_size(node)
            if depth < minimum_depth or size < 3:
                continue
            signature = repr(node.to_dict())
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((depth, size, variables, param_types, node))
        candidates.sort(key=lambda row: (-row[0], -row[1]))
        before = set(self.library.macros)
        added: list[Macro] = []
        for _depth, _size, params, param_types, body in candidates:
            if len(added) >= limit:
                break
            macro = self.library.add(
                list(params), list(param_types), body, domain=domain,
                source_skill=source_skill)
            if macro.name not in before:
                before.add(macro.name)
                added.append(macro)
        self.mined_total += len(added)
        return added

    def stats(self) -> dict:
        return {
            "mined_total": self.mined_total,
            "library_macros": len(self.library.macros),
        }


def _contains_macro(node) -> bool:
    if node.op == "macro":
        return True
    return any(_contains_macro(child) for child in node.args)


def _expr_depth(node) -> int:
    if not node.args:
        return 1
    return 1 + max(_expr_depth(child) for child in node.args)


def _expr_size(node) -> int:
    return 1 + sum(_expr_size(child) for child in node.args)


def _variable_types(node) -> dict:
    rows: dict[str, str] = {}

    def walk(item) -> None:
        if item.op == "var":
            rows[str(item.value)] = item.out_type
        for child in item.args:
            walk(child)

    walk(node)
    return rows
