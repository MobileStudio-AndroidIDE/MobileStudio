"""Shared data structures for explanation engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from code.ir.ir_types import IRNode, IRNodeType


@dataclass
class LinePurpose:
    line_type: str
    node: IRNode
    context: dict
    detail: str = ""

@dataclass
class ClassPurpose:
    name: str
    properties: list[str]
    reason: str = ""
    alternatives: list = field(default_factory=list)

@dataclass
class FunctionPurpose:
    name: str
    params: list[tuple[str, str]]
    return_type: str
    operations: list[str]
    reason: str = ""
    alternatives: list = field(default_factory=list)

@dataclass
class PipelinePurpose:
    stage_name: str
    operation: Optional[object] = None
    operation_order: list[str] = field(default_factory=list)