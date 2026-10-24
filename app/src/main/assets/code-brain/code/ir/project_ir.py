"""Project IR - Structural intermediate representation for multi-file projects."""
from __future__ import annotations
import ast
import re
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class FileType(Enum):
    SOURCE = "source"
    BUILD = "build"
    MANIFEST = "manifest"
    RESOURCE = "resource"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    TEST = "test"


class Platform(Enum):
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    DESKTOP = "desktop"
    CLI = "cli"
    LIBRARY = "library"
    BACKEND = "backend"


class DependencyType(Enum):
    COMPILE = "compile"
    RUNTIME = "runtime"
    TEST = "test"
    DEVELOPMENT = "development"
    PERMISSION = "permission"
    PLATFORM = "platform"


@dataclass
class FileDependency:
    source_file: str
    target_file: str
    dependency_type: DependencyType = DependencyType.COMPILE
    reason: str = ""


@dataclass
class ModuleDependency:
    source_module: str
    target_module: str
    dependency_type: DependencyType = DependencyType.COMPILE
    reason: str = ""


@dataclass
class FileIR:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    path: str = ""
    file_type: FileType = FileType.SOURCE
    language: str = ""
    content_type: str = ""
    imports: list[str] = field(default_factory=list)
    declarations: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    dependencies: list[FileDependency] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    code_ir: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "file_type": self.file_type.value,
            "language": self.language,
            "content_type": self.content_type,
            "imports": self.imports,
            "declarations": self.declarations,
            "permissions": self.permissions,
            "dependencies": [
                {"source": d.source_file, "target": d.target_file,
                 "type": d.dependency_type.value, "reason": d.reason}
                for d in self.dependencies
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FileIR:
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            path=d.get("path", ""),
            file_type=FileType(d.get("file_type", "source")),
            language=d.get("language", ""),
            content_type=d.get("content_type", ""),
            imports=d.get("imports", []),
            declarations=d.get("declarations", []),
            permissions=d.get("permissions", []),
            dependencies=[
                FileDependency(
                    source_file=dep.get("source", ""),
                    target_file=dep.get("target", ""),
                    dependency_type=DependencyType(dep.get("type", "compile")),
                    reason=dep.get("reason", ""),
                )
                for dep in d.get("dependencies", [])
            ],
            metadata=d.get("metadata", {}),
        )


@dataclass
class ModuleIR:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    path: str = ""
    files: list[FileIR] = field(default_factory=list)
    dependencies: list[ModuleDependency] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_file(self, file: FileIR) -> FileIR:
        self.files.append(file)
        return file

    def find_file(self, path: str) -> Optional[FileIR]:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def get_source_files(self) -> list[FileIR]:
        return [f for f in self.files if f.file_type == FileType.SOURCE]

    def get_build_files(self) -> list[FileIR]:
        return [f for f in self.files if f.file_type == FileType.BUILD]

    def get_manifest_files(self) -> list[FileIR]:
        return [f for f in self.files if f.file_type == FileType.MANIFEST]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "files": [f.to_dict() for f in self.files],
            "dependencies": [
                {"source": d.source_module, "target": d.target_module,
                 "type": d.dependency_type.value, "reason": d.reason}
                for d in self.dependencies
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModuleIR:
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            name=d.get("name", ""),
            path=d.get("path", ""),
            files=[FileIR.from_dict(f) for f in d.get("files", [])],
            dependencies=[
                ModuleDependency(
                    source_module=dep.get("source", ""),
                    target_module=dep.get("target", ""),
                    dependency_type=DependencyType(dep.get("type", "compile")),
                    reason=dep.get("reason", ""),
                )
                for dep in d.get("dependencies", [])
            ],
            metadata=d.get("metadata", {}),
        )


@dataclass
class ProjectIR:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    language: str = ""
    platform: Platform = Platform.CLI
    modules: list[ModuleIR] = field(default_factory=list)
    root_dependencies: list[ModuleDependency] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_module(self, module: ModuleIR) -> ModuleIR:
        self.modules.append(module)
        return module

    def find_module(self, name: str) -> Optional[ModuleIR]:
        for m in self.modules:
            if m.name == name:
                return m
        return None

    def find_file(self, path: str) -> Optional[FileIR]:
        for m in self.modules:
            f = m.find_file(path)
            if f:
                return f
        return None

    def get_all_files(self) -> list[FileIR]:
        files = []
        for m in self.modules:
            files.extend(m.files)
        return files

    def get_all_source_files(self) -> list[FileIR]:
        files = []
        for m in self.modules:
            files.extend(m.get_source_files())
        return files

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "platform": self.platform.value,
            "modules": [m.to_dict() for m in self.modules],
            "dependencies": [
                {"source": d.source_module, "target": d.target_module,
                 "type": d.dependency_type.value, "reason": d.reason}
                for d in self.root_dependencies
            ],
            "metadata": self.metadata,
        }
        # CONSUME POINT (a): every serialization runs the inferred
        # dependency validation; issues are surfaced in the dict so the
        # project verifier (or any consumer of to_dict) sees them.  The
        # key appears ONLY when issues exist, so projects without issues
        # serialize exactly as before (backward compatible).
        validation = self.validate_dependencies()
        if validation["issues"]:
            d["dependency_issues"] = list(validation["issues"])
            d["dependency_validation"] = validation
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProjectIR:
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            name=d.get("name", ""),
            language=d.get("language", ""),
            platform=Platform(d.get("platform", "cli")),
            modules=[ModuleIR.from_dict(m) for m in d.get("modules", [])],
            root_dependencies=[
                ModuleDependency(
                    source_module=dep.get("source", ""),
                    target_module=dep.get("target", ""),
                    dependency_type=DependencyType(dep.get("type", "compile")),
                    reason=dep.get("reason", ""),
                )
                for dep in d.get("dependencies", [])
            ],
            metadata=d.get("metadata", {}),
        )

    def pretty_print(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}project name={self.name} language={self.language} platform={self.platform.value}"]
        for m in self.modules:
            lines.append(f"{prefix}  module name={m.name} path={m.path}")
            for f in m.files:
                lines.append(f"{prefix}    file path={f.path} type={f.file_type.value} lang={f.language}")
                if f.imports:
                    lines.append(f"{prefix}      imports={f.imports}")
                if f.declarations:
                    lines.append(f"{prefix}      declarations={f.declarations}")
                if f.permissions:
                    lines.append(f"{prefix}      permissions={f.permissions}")
        for d in self.root_dependencies:
            lines.append(f"{prefix}  dependency {d.source_module} -> {d.target_module} ({d.dependency_type.value})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Inferred project dependency graph
    # ------------------------------------------------------------------ #
    # The graph is INFERRED by parsing the files' import statements,
    # never by trusting the declared root_dependencies list.  Only
    # imports that resolve to a file that exists in THIS project become
    # nodes/edges; stdlib and third-party imports are ignored (their
    # validity is not knowable from the IR).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _module_candidates(path: str) -> list[str]:
        """Dotted-module candidates for a project file path."""
        if not path:
            return []
        p = path.replace("\\", "/")
        if p.endswith(".py"):
            p = p[:-3]
            if p.endswith("/__init__"):
                p = p[:-len("/__init__")]
            elif p == "__init__":
                p = ""
            return [p.replace("/", ".")] if p else []
        return []

    @classmethod
    def _build_module_registry(cls, files: list[FileIR]) -> dict:
        """Map dotted module names (full and leaf) to file paths.

        Deterministic: keys keep the first registered path; files are
        registered in path order.
        """
        registry: dict[str, str] = {}
        for f in sorted(files, key=lambda x: x.path):
            for cand in cls._module_candidates(f.path):
                for name in (cand, cand.rsplit(".", 1)[-1]):
                    registry.setdefault(name, f.path)
        return registry

    @staticmethod
    def _ast_imports(source: str) -> list[tuple[int, str, list[str]]]:
        """(level, module, names) import entries from Python source."""
        entries: list[tuple[int, str, list[str]]] = []
        try:
            tree = ast.parse(source or "")
        except SyntaxError:
            return entries
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    entries.append((0, alias.name, []))
            elif isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names
                         if a.name and a.name != "*"]
                entries.append((node.level, node.module or "", names))
        return entries

    @staticmethod
    def _regex_imports(source: str, language: str) -> list[tuple[int, str, list[str]]]:
        """(level, module, names) extraction for non-Python languages."""
        entries: list[tuple[int, str, list[str]]] = []
        lang = (language or "").lower()

        def clean(raw: str) -> str:
            raw = raw.strip().strip(";").strip()
            raw = raw.split(" as ")[0].strip()
            return raw

        if lang in ("kotlin", "java", "scala"):
            pattern = r'^\s*import\s+([A-Za-z_][\w.]*)\s*(?:as\s+\w+)?;?'
            for m in re.finditer(pattern, source, re.MULTILINE):
                entries.append((0, clean(m.group(1)), []))
        elif lang in ("javascript", "typescript", "tsx", "jsx"):
            for m in re.finditer(
                    r'(?:from\s+["\']([^"\']+)["\'])|'
                    r'(?:require\s*\(\s*["\']([^"\']+)["\']\s*\))', source):
                raw = m.group(1) or m.group(2) or ""
                if raw:
                    raw = clean(raw)
                    entries.append((1, raw[1:], []) if raw.startswith(".")
                                   else (0, raw, []))
        elif lang in ("c", "cpp", "c++", "objc"):
            for m in re.finditer(r'^\s*#include\s+["<]([^">]+)[">]', source, re.MULTILINE):
                raw = m.group(1).strip()
                if raw:
                    entries.append((0, clean(raw), []))
        elif lang == "go":
            for m in re.finditer(r'"([A-Za-z_][\w./-]*)"', source):
                entries.append((0, clean(m.group(1)), []))
        return [(lv, mod, names) for lv, mod, names in entries if mod or names]

    def _extract_file_imports(self, file_ir: FileIR,
                              sources: dict) -> list[tuple[int, str, list[str]]]:
        """(level, module, names) import entries for one file.

        Prefers real source (ast.parse for .py, regex for other
        languages), falls back to the FileIR.imports list the planner
        already extracted.
        """
        source = ""
        if sources and file_ir.path in sources:
            source = str(sources.get(file_ir.path) or "")
        if not source:
            meta = file_ir.metadata or {}
            source = str(meta.get("content") or meta.get("source") or "")
        if source:
            lang = (file_ir.language or "").lower()
            if file_ir.path.endswith(".py") or lang == "python":
                entries = self._ast_imports(source)
            else:
                entries = self._regex_imports(source, lang)
            if entries:
                return entries
        result: list[tuple[int, str, list[str]]] = []
        for imp in (file_ir.imports or []):
            raw = str(imp)
            if raw.startswith("."):
                result.append((len(raw) - len(raw.lstrip(".")),
                               raw.lstrip("."), []))
            else:
                result.append((0, raw, []))
        return result

    def dependency_graph(self, sources: Optional[dict] = None) -> dict:
        """Infer the project's import graph.

        Args:
            sources: optional {file_path: source_text}.  When given (or
                when a FileIR carries embedded source), imports are
                parsed from the text; otherwise the FileIR.imports list
                is used.

        Returns a deterministic dict:
            {"nodes": [paths sorted],
             "edges": [{"from", "to", "kind": "import"} sorted],
             "unresolved": [{"file", "import"} sorted],
             "cycles": [[paths...] sorted]}
        """
        files = [f for f in self.get_all_files()]
        registry = self._build_module_registry(files)
        nodes = sorted(f.path for f in files)

        unresolved: list[dict] = []
        raw_edges: set[tuple[str, str]] = set()
        for file_ir in sorted(files, key=lambda x: x.path):
            entries = self._extract_file_imports(file_ir, sources)
            for level, module, names in entries:
                target = self._resolve_leveled(level, module, names,
                                               file_ir, registry)
                if target is None:
                    if level > 0:
                        label = "." * level + module
                        unresolved.append({"file": file_ir.path,
                                           "import": label or "."})
                    continue
                if target == file_ir.path:
                    continue
                raw_edges.add((file_ir.path, target))

        edges = [{"from": src, "to": dst, "kind": "import"}
                 for src, dst in sorted(raw_edges)]
        return {
            "nodes": nodes,
            "edges": edges,
            "unresolved": sorted(unresolved, key=lambda u: (u["file"],
                                                            u["import"])),
            "cycles": self._find_cycles(nodes, edges),
        }

    @staticmethod
    def _is_project_local_import(imp: str) -> bool:
        """Only unambiguous project-relative imports are reported as
        unresolved; an unmatched absolute import is assumed stdlib or
        third-party (never a project error)."""
        return str(imp).startswith(".")

    @staticmethod
    def _dotted_candidates(dotted: str) -> list[str]:
        parts = [p for p in str(dotted).split(".") if p]
        cands = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        if parts and parts[-1] not in cands:
            cands.append(parts[-1])
        return [c for c in cands if c]

    def _resolve_leveled(self, level: int, module: str,
                         names: list[str], file_ir: FileIR,
                         registry: dict) -> Optional[str]:
        """Resolve one (level, module, names) import entry to a file."""
        if level <= 0:
            for cand in self._dotted_candidates(module):
                if cand in registry:
                    return registry[cand]
            for name in names:
                for cand in self._dotted_candidates(
                        f"{module}.{name}" if module else name):
                    if cand in registry:
                        return registry[cand]
            return None
        # relative: build the package base from the importing file
        parts = [p for p in file_ir.path.replace("\\", "/").split("/")[:-1]
                 if p]
        base = parts[:len(parts) - (level - 1)]
        base_dotted = ".".join(base)
        if module:
            for cand in self._dotted_candidates(
                    f"{base_dotted}.{module}" if base_dotted else module):
                if cand in registry:
                    return registry[cand]
        for name in names:
            for cand in self._dotted_candidates(
                    f"{base_dotted}.{name}" if base_dotted else name):
                if cand in registry:
                    return registry[cand]
        if base_dotted and base_dotted in registry:
            return registry[base_dotted]
        return None

    @staticmethod
    def _find_cycles(nodes: list[str], edges: list[dict]) -> list[list[str]]:
        """All elementary cycles (each canonicalized from its strongest
        node), deterministic order.  Iterative DFS with colors."""
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        for e in edges:
            adj.setdefault(e["from"], []).append(e["to"])
        for n in adj:
            adj[n] = sorted(set(adj[n]))

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in adj}
        stack: list[str] = []
        cycles: set[tuple[str, ...]] = set()

        def dfs(start: str) -> None:
            color[start] = GRAY
            stack.append(start)
            for nxt in adj.get(start, []):
                if color.get(nxt, WHITE) == WHITE:
                    dfs(nxt)
                elif color.get(nxt) == GRAY:
                    idx = stack.index(nxt)
                    cycle = tuple(stack[idx:])
                    pivot = min(range(len(cycle)),
                                key=lambda i: (cycle[i], i))
                    canon = cycle[pivot:] + cycle[:pivot]
                    cycles.add(canon)
            stack.pop()
            color[start] = BLACK

        for n in sorted(adj):
            if color[n] == WHITE:
                dfs(n)
        return [list(c) for c in sorted(cycles)]

    @classmethod
    def _declared_dependency_covered(cls, dep: ModuleDependency,
                                     edges: list[dict]) -> bool:
        src_stem = dep.source_module.replace("\\", "/").rstrip("/")
        dst_stem = dep.target_module.replace("\\", "/").rstrip("/")
        for e in edges:
            src_head = e["from"].replace("\\", "/").split("/")[0]
            src_ok = (src_head == src_stem
                      or src_head.startswith(src_stem + ".")
                      or src_head.startswith(src_stem + "_"))
            if not src_ok:
                continue
            dst_parts = [p for p in
                         e["to"].replace("\\", "/").split("/") if p]
            if not dst_parts:
                continue
            dst_module = "/".join(dst_parts)
            if dst_module.endswith(".py"):
                dst_module = dst_module[:-3]
            dst_module = dst_module.replace("/", ".")
            if (dst_module == dst_stem
                    or dst_module.startswith(dst_stem + ".")
                    or dst_module.endswith("." + dst_stem)):
                return True
            if len(dst_parts) > 1 and (
                    dst_parts[0] == dst_stem
                    or dst_parts[0].startswith(dst_stem + ".")):
                return True
        return False

    def validate_dependencies(self) -> dict:
        """Validate declared dependencies against the INFERRED graph.

        Returns {"issues": [str], "declarations_checked": N,
                 "declarations_used": M}.  Issues:
          - import cycles;
          - unresolved project-relative imports (module not in project);
          - declared ModuleDependency entries no inferred edge realizes.
        """
        graph = self.dependency_graph()
        issues: list[str] = []

        for cycle in graph["cycles"]:
            issues.append("import cycle: " + " -> ".join(cycle))
        for item in graph["unresolved"]:
            issues.append(
                f"unresolved import: {item['file']} -> {item['import']} "
                f"(module not found in project)")

        declarations_checked = 0
        declarations_used = 0
        for dep in self.root_dependencies:
            declarations_checked += 1
            if self._declared_dependency_covered(dep, graph["edges"]):
                declarations_used += 1
            else:
                issues.append(
                    f"unused declared dependency: "
                    f"{dep.source_module} -> {dep.target_module} "
                    f"(no inferred import edge realizes it)")

        return {
            "issues": sorted(issues, key=lambda s: (s.split(":", 1)[0], s)),
            "declarations_checked": declarations_checked,
            "declarations_used": declarations_used,
        }
