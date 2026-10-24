"""Multi-language code generator - generates source code from IR for all 8 languages."""
from __future__ import annotations
from typing import Optional
from code.ir.ir_types import IRNode, IRNodeType
from languages.python.python_gen import PythonGenerator
from languages.kotlin.kotlin_gen import KotlinGenerator
from languages.java.java_gen import JavaGenerator
from languages.c.c_gen import CGenerator
from languages.cpp.cpp_gen import CppGenerator
from languages.javascript.js_gen import JavaScriptGenerator
from languages.go.go_gen import GoGenerator
from languages.rust.rust_gen import RustGenerator


GENERATORS = {
    "python": PythonGenerator,
    "kotlin": KotlinGenerator,
    "java": JavaGenerator,
    "c": CGenerator,
    "cpp": CppGenerator,
    "javascript": JavaScriptGenerator,
    "go": GoGenerator,
    "rust": RustGenerator,
}


# The IR's standard helper vocabulary: builders reference these
# (a copy of a list, an empty list) — the python rendering defines
# them so the generated program is executable, not just readable.
_PY_HELPER_DEFS = (
    ("copy_list", "def copy_list(items):\n    return list(items)"),
    ("new_list", "def new_list():\n    return []"),
)


class CodeGenerator:
    def __init__(self):
        self._generators = {k: v() for k, v in GENERATORS.items()}

    def _ensure_python_helpers(self, code: str) -> str:
        """Define any standard helper the code calls but does not define
        (the helper vocabulary comes from the IR, not from the request)."""
        for name, definition in _PY_HELPER_DEFS:
            if f"{name}(" in code and f"def {name}" not in code:
                code = definition + "\n\n" + code
        return code

    def generate(self, node: IRNode, language: str) -> str:
        gen = self._generators.get(language)
        if not gen:
            raise ValueError(f"Unsupported language: {language}")
        code = gen.generate(node, indent=0)
        if language == "python":
            code = self._ensure_python_helpers(code)
        return code

    def generate_all(self, node: IRNode) -> dict[str, str]:
        results = {}
        for lang, gen in self._generators.items():
            try:
                code = gen.generate(node, indent=0)
                if lang == "python":
                    code = self._ensure_python_helpers(code)
                results[lang] = code
            except Exception as e:
                results[lang] = f"// Error generating {lang}: {e}"
        return results

    def get_supported_languages(self) -> list[str]:
        return list(self._generators.keys())
