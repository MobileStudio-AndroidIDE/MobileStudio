"""C++ backend stub.

The JavaScript backend is implemented; the C++ emitter is not yet, so
it fails honestly with TranspileError and callers keep the python-only
result.
"""

from __future__ import annotations

import ast

from code.transpiler.py_ast_transpiler import TranspileError


def emit_cpp(tree: ast.Module) -> str:
    """Emit C++ for a python module tree (not yet implemented)."""
    raise TranspileError("C++ backend not yet implemented")
