"""Symbol Table - manages names, types, scopes, declarations, and references."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Symbol:
    name: str
    symbol_type: str  # "function", "class", "variable", "parameter", "property", "enum", "interface"
    scope: str  # "global", "class", "function", "block"
    type_ref: str = ""
    declaration_line: int = 0
    file_path: str = ""
    references: list[tuple[str, int]] = field(default_factory=list)  # (file, line)
    metadata: dict = field(default_factory=dict)


class SymbolTable:
    def __init__(self):
        self.symbols: dict[str, Symbol] = {}
        self._scope_stack: list[str] = ["global"]
        self._current_scope = "global"

    def enter_scope(self, scope_name: str) -> None:
        self._current_scope = f"{self._current_scope}.{scope_name}"
        self._scope_stack.append(self._current_scope)

    def exit_scope(self) -> None:
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()
            self._current_scope = self._scope_stack[-1]

    def declare(self, name: str, symbol_type: str, type_ref: str = "",
                file_path: str = "", line: int = 0, metadata: dict = None) -> Symbol:
        qualified_name = f"{self._current_scope}.{name}"
        symbol = Symbol(
            name=name,
            symbol_type=symbol_type,
            scope=self._current_scope,
            type_ref=type_ref,
            file_path=file_path,
            declaration_line=line,
            metadata=metadata or {},
        )
        self.symbols[qualified_name] = symbol
        return symbol

    def lookup(self, name: str, current_scope_only: bool = False) -> Optional[Symbol]:
        if current_scope_only:
            qualified = f"{self._current_scope}.{name}"
            return self.symbols.get(qualified)
        for i in range(len(self._scope_stack) - 1, -1, -1):
            scope = self._scope_stack[i]
            qualified = f"{scope}.{name}"
            if qualified in self.symbols:
                return self.symbols[qualified]
        return None

    def add_reference(self, name: str, file_path: str = "", line: int = 0) -> bool:
        symbol = self.lookup(name)
        if symbol:
            symbol.references.append((file_path, line))
            return True
        return False

    def get_all_in_scope(self, scope: str = None) -> list[Symbol]:
        scope = scope or self._current_scope
        return [s for s in self.symbols.values() if s.scope.startswith(scope)]

    def get_by_type(self, symbol_type: str) -> list[Symbol]:
        return [s for s in self.symbols.values() if s.symbol_type == symbol_type]

    def clear(self) -> None:
        self.symbols.clear()
        self._scope_stack = ["global"]
        self._current_scope = "global"

    def to_dict(self) -> dict:
        return {
            qualified: {
                "name": s.name,
                "symbol_type": s.symbol_type,
                "scope": s.scope,
                "type_ref": s.type_ref,
                "file_path": s.file_path,
                "declaration_line": s.declaration_line,
                "references": s.references,
                "metadata": s.metadata,
            }
            for qualified, s in self.symbols.items()
        }

    def from_dict(self, data: dict) -> None:
        self.symbols.clear()
        for qualified, d in data.items():
            s = Symbol(
                name=d["name"],
                symbol_type=d["symbol_type"],
                scope=d["scope"],
                type_ref=d.get("type_ref", ""),
                file_path=d.get("file_path", ""),
                declaration_line=d.get("declaration_line", 0),
                references=d.get("references", []),
                metadata=d.get("metadata", {}),
            )
            self.symbols[qualified] = s
