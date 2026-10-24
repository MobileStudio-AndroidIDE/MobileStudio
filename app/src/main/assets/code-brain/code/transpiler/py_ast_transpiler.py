"""AST-based transpiler: Python source -> dynamically typed targets.

Design rules (RENA-BR conventions):
- Source is parsed with ``ast`` (never regex/line munging).
- A type pass annotates every expression node with a value type so the
  backend can pick container-aware operations (Map for dict, Set for
  set, negative index handling, python ``//``/``%``/truthiness rules).
- Python has no block scope, JS does: every function-scoped assigned
  name is hoisted to the top of the emitted function.
- Any construct that cannot be translated faithfully raises
  ``TranspileError``; callers must then keep the python-only output
  instead of emitting wrong code.
- Emitted builtin names match python's (``int``, ``len``, ``print``...)
  because the runtime prelude defines them with the same names -- that
  keeps shadowing semantics identical to python.  Attribute-method
  helpers live in the ``$`` namespace (``$`` cannot appear in python
  identifiers, so no user name can ever collide with them).
"""
from __future__ import annotations

import ast


class TranspileError(Exception):
    """Unsupported construct -- caller falls back to python-only."""


# ---------------------------------------------------------------------------
# value types
# ---------------------------------------------------------------------------

class VT:
    """Static approximation of a python value's runtime shape."""

    __slots__ = ("kind", "elem", "kval", "vval", "mod")

    def __init__(self, kind: str, elem: "VT | None" = None,
                 kval: "VT | None" = None, vval: "VT | None" = None,
                 mod: str | None = None):
        self.kind = kind          # int float num str bool none list dict set
                                  # tuple deque counter range func module
                                  # unknown
        self.elem = elem          # list/tuple/set/deque element type
        self.kval = kval          # dict/counter key type
        self.vval = vval          # dict/counter value type
        self.mod = mod            # module symbol path for kind == module

    def __repr__(self) -> str:    # pragma: no cover - debugging aid
        parts = [self.kind]
        if self.elem is not None:
            parts.append("elem=" + self.elem.kind)
        if self.kval is not None:
            parts.append("key=" + self.kval.kind)
        if self.vval is not None:
            parts.append("val=" + self.vval.kind)
        if self.mod:
            parts.append("mod=" + self.mod)
        return "VT(" + ",".join(parts) + ")"

    @property
    def is_num(self) -> bool:
        return self.kind in ("int", "float", "num", "bool")

    @property
    def is_container(self) -> bool:
        return self.kind in ("list", "dict", "set", "tuple", "deque",
                             "counter")

    def is_int_family(self) -> bool:
        return self.kind in ("int", "bool")


NONE = VT("none")
UNKNOWN = VT("unknown")
INT = VT("int")
FLOAT = VT("float")
NUM = VT("num")
BOOL = VT("bool")
STR = VT("str")
RANGE = VT("range")
LIST_ANY = VT("list")
DICT_ANY = VT("dict")
SET_ANY = VT("set")
FUNC_ANY = VT("func")

_NUMERIC_JOIN = {
    ("int", "int"): "int",
    ("int", "num"): "num",
    ("int", "float"): "float",
    ("int", "bool"): "int",
    ("bool", "int"): "int",
    ("bool", "bool"): "bool",
    ("bool", "num"): "num",
    ("bool", "float"): "float",
    ("num", "int"): "num",
    ("num", "num"): "num",
    ("num", "float"): "float",
    ("float", "int"): "float",
    ("float", "num"): "float",
    ("float", "float"): "float",
}


def join_vt(a: VT | None, b: VT | None) -> VT:
    """Least upper bound of two value types (branch merges, + of mixed)."""
    if a is None:
        return b if b is not None else UNKNOWN
    if b is None:
        return a
    if a is b:
        return a
    if a.kind == "none":
        return b
    if b.kind == "none":
        return a
    if a.kind == "unknown" or b.kind == "unknown":
        return UNKNOWN
    if a.kind == b.kind:
        if a.kind in ("list", "tuple", "set", "deque"):
            return VT(a.kind, elem=join_vt(a.elem, b.elem))
        if a.kind in ("dict", "counter"):
            return VT(a.kind, kval=join_vt(a.kval, b.kval),
                      vval=join_vt(a.vval, b.vval))
        if a.kind == "module":
            return a if a.mod == b.mod else UNKNOWN
        return a
    if a.kind == "func" or b.kind == "func":
        if a.kind == b.kind:
            return VT("func", elem=join_vt(a.elem, b.elem))
        return UNKNOWN
    key = (a.kind, b.kind)
    if key in _NUMERIC_JOIN:
        return VT(_NUMERIC_JOIN[key])
    rkey = (b.kind, a.kind)
    if rkey in _NUMERIC_JOIN:
        return VT(_NUMERIC_JOIN[rkey])
    return UNKNOWN


# ---------------------------------------------------------------------------
# scope / environment
# ---------------------------------------------------------------------------

class FuncScope:
    """Names a function body binds, plus its nested function defs."""

    __slots__ = ("hoist", "_seen", "defs", "nonlocals")

    def __init__(self) -> None:
        self.hoist: list[str] = []
        self._seen: set[str] = set()
        self.defs: set[str] = set()
        self.nonlocals: set[str] = set()

    def add_hoist(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.hoist.append(name)

    def want_hoist(self, name: str) -> bool:
        """True when a binding of ``name`` belongs in the hoist list."""
        return (name not in self._seen
                and name not in self.nonlocals
                and name not in self.defs)


class Env:
    """Lexical environment: variable types + import bindings, chained."""

    __slots__ = ("vars", "imports", "parent", "scope")

    def __init__(self, parent: "Env | None" = None,
                 scope: "FuncScope | None" = None):
        self.vars: dict[str, VT] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}
        self.parent = parent
        self.scope = scope

    def lookup(self, name: str) -> VT | None:
        e: Env | None = self
        while e is not None:
            v = e.vars.get(name)
            if v is not None:
                return v
            e = e.parent
        return None

    def import_lookup(self, name: str) -> tuple[str, str | None] | None:
        e: Env | None = self
        while e is not None:
            v = e.imports.get(name)
            if v is not None:
                return v
            e = e.parent
        return None

    def bind(self, name: str, vt: VT) -> None:
        self.vars[name] = vt

    def mark_hoist(self, name: str) -> None:
        if self.scope is not None and self.scope.want_hoist(name):
            self.scope.add_hoist(name)


# ---------------------------------------------------------------------------
# builtin / module knowledge (language API, not problem specific)
# ---------------------------------------------------------------------------

# Builtins that have no faithful translation in the runtime prelude.
UNSAFE_BUILTINS = {
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "dir", "getattr", "setattr", "delattr", "hasattr", "callable", "id",
    "type", "super", "object", "property", "staticmethod", "classmethod",
    "memoryview", "bytearray", "bytes", "frozenset", "slice", "iter",
    "next", "help", "input",  # input is allowed below (prelude provides it)
    "type",  # allowed below (prelude provides it for type(None) etc)
}
UNSAFE_BUILTINS.discard("type")
UNSAFE_BUILTINS.discard("input")

# Every builtin the prelude defines with python's own name.
KNOWN_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "format", "hex", "int", "isinstance",
    "len", "list", "map", "max", "min", "oct", "ord", "pow", "print",
    "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
    "zip", "input", "reversed", "type", "repr", "hash",
}

# Attribute paths on imported modules -> static type of the result.
# {module: {attr path: VT or "call:<kind>"}}
_MODULE_ATTRS: dict[str, dict[str, VT]] = {
    "sys": {"stdin.read": STR, "stdin.readline": STR,
            "stdout.write": NONE, "stdout.flush": NONE, "exit": NONE,
            "setrecursionlimit": NONE, "getrecursionlimit": INT,
            "argv": VT("list", elem=STR), "maxsize": INT},
    "math": {"factorial": INT, "isqrt": INT, "sqrt": FLOAT, "floor": INT,
             "ceil": INT, "log": FLOAT, "log2": FLOAT, "log10": FLOAT,
             "sin": FLOAT, "cos": FLOAT, "tan": FLOAT, "pi": FLOAT,
             "e": FLOAT, "inf": FLOAT, "gcd": INT, "comb": INT, "perm": INT,
             "lcm": INT, "trunc": INT, "hypot": FLOAT, "exp": FLOAT,
             "asin": FLOAT, "acos": FLOAT, "atan": FLOAT, "atan2": FLOAT,
             "fabs": FLOAT, "degrees": FLOAT, "radians": FLOAT,
             "log1p": FLOAT, "asin": FLOAT, "isclose": BOOL},
    "re": {"findall": VT("list", elem=STR), "finditer": LIST_ANY,
           "match": VT("dict"), "search": VT("dict"), "sub": STR,
           "split": VT("list", elem=STR), "fullmatch": VT("dict"),
           "escape": STR, "I": INT, "M": INT, "S": INT, "X": INT, "A": INT,
           "IGNORECASE": INT, "MULTILINE": INT, "DOTALL": INT,
           "VERBOSE": INT, "ASCII": INT},
    "heapq": {"heappush": NONE, "heappop": UNKNOWN, "heapify": NONE,
              "nlargest": LIST_ANY, "nsmallest": LIST_ANY},
    "itertools": {"accumulate": LIST_ANY, "permutations": LIST_ANY,
                  "combinations": LIST_ANY, "product": LIST_ANY,
                  "chain": LIST_ANY, "groupby": LIST_ANY},
    "bisect": {"bisect": INT, "bisect_left": INT, "bisect_right": INT,
               "insort": NONE},
    "collections": {"defaultdict": DICT_ANY, "OrderedDict": DICT_ANY,
                    "namedtuple": UNKNOWN},
}

# ``from X import Y`` symbols whose *call* result type is special.
_FROM_IMPORT_CALL: dict[tuple[str, str], VT] = {
    ("collections", "Counter"): VT("counter", kval=UNKNOWN, vval=INT),
    ("collections", "deque"): VT("deque"),
    ("collections", "defaultdict"): VT("dict", kval=UNKNOWN, vval=UNKNOWN),
    ("itertools", "accumulate"): LIST_ANY,
    ("itertools", "permutations"): VT("list", elem=VT("tuple")),
    ("itertools", "combinations"): VT("list", elem=VT("tuple")),
    ("itertools", "product"): VT("list", elem=VT("tuple")),
    ("itertools", "chain"): LIST_ANY,
    ("heapq", "heapify"): NONE,
}


# ---------------------------------------------------------------------------
# type pass
# ---------------------------------------------------------------------------

class Typer:
    """Annotates every expression node with ``_vt`` and collects scopes."""

    def __init__(self) -> None:
        self.defs: dict[str, ast.FunctionDef] = {}

    # -- module ------------------------------------------------------------
    def module(self, tree: ast.Module) -> None:
        env = Env()
        for st in tree.body:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                self.stmt_import(st, env)
            elif isinstance(st, ast.FunctionDef):
                self.function(st, env)
            else:
                self.stmt(st, env)

    def function(self, fd: ast.FunctionDef, parent: Env) -> None:
        if fd.decorator_list:
            raise TranspileError("decorated function")
        args = fd.args
        if args.vararg or args.kwarg or args.kwonlyargs:
            raise TranspileError("variadic/kw-only function")
        if args.defaults or args.kw_defaults:
            raise TranspileError("function default arguments")
        self.defs[fd.name] = fd
        scope = FuncScope()
        env = Env(parent, scope)
        for a in args.args:
            if a.arg in ("self", "cls"):
                continue
            env.bind(a.arg, UNKNOWN)
        rets: list[VT] = []
        fd._scope = scope           # type: ignore[attr-defined]
        fd._env = env               # type: ignore[attr-defined]
        for st in fd.body:
            self.stmt(st, env, rets)
        ret: VT = NONE
        for r in rets:
            ret = join_vt(ret, r)
        fd._retvt = ret             # type: ignore[attr-defined]

    # -- statements --------------------------------------------------------
    def stmt(self, node: ast.stmt, env: Env,
             rets: list[VT] | None = None) -> None:
        if isinstance(node, ast.Assign):
            self.expr(node.value, env)
            for t in node.targets:
                self.assign_target(t, node.value, env)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.expr(node.value, env)
                self.assign_target(node.target, node.value, env)
        elif isinstance(node, ast.AugAssign):
            self.expr(node.value, env)
            self.assign_target(node.target, node.value, env, aug=True)
        elif isinstance(node, ast.Expr):
            self.expr(node.value, env)
        elif isinstance(node, ast.If):
            self.expr(node.test, env)
            pre = dict(env.vars)
            for st in node.body:
                self.stmt(st, env, rets)
            mid = dict(env.vars)
            env.vars = dict(pre)
            for st in node.orelse:
                self.stmt(st, env, rets)
            post = dict(env.vars)
            merged: dict[str, VT] = {}
            for k in set(mid) | set(post) | set(pre):
                a = mid.get(k, pre.get(k))
                b = post.get(k, pre.get(k))
                if a is None and b is None:
                    continue
                merged[k] = join_vt(a, b)
            env.vars = merged
        elif isinstance(node, (ast.For, ast.While)):
            if node.orelse:
                raise TranspileError("for/while-else")
            if isinstance(node, ast.For):
                it_vt = self.expr(node.iter, env)
                self.bind_target(node.target, _elem_of(it_vt), env)
                self.for_targets(node.target, env)
            else:
                self.expr(node.test, env)
            for st in node.body:
                self.stmt(st, env, rets)
        elif isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            pass
        elif isinstance(node, ast.Return):
            if rets is None:
                raise TranspileError("return outside function")
            if node.value is None:
                rets.append(NONE)
            else:
                rets.append(self.expr(node.value, env))
        elif isinstance(node, ast.Raise):
            if node.exc is not None:
                self.expr(node.exc, env)
            if node.cause is not None:
                self.expr(node.cause, env)
        elif isinstance(node, ast.Assert):
            self.expr(node.test, env)
            if node.msg is not None:
                self.expr(node.msg, env)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self.stmt_import(node, env)
        elif isinstance(node, ast.FunctionDef):
            self.function(node, env)
            env.bind(node.name, VT("func",
                                   elem=getattr(node, "_retvt", NONE)))
            if env.scope is not None:
                env.scope.defs.add(node.name)
        elif isinstance(node, ast.Try):
            if node.finalbody:
                for st in node.finalbody:
                    self.stmt(st, env, rets)
            if node.orelse:
                raise TranspileError("try-else")
            if len(node.handlers) > 1:
                raise TranspileError("multiple except handlers")
            for h in node.handlers:
                if h.type is not None and h.name:
                    pass
                if getattr(h, "test", None):
                    raise TranspileError("conditional except")
            pre = dict(env.vars)
            for st in node.body:
                self.stmt(st, env, rets)
            body_vars = dict(env.vars)
            all_vars = dict(body_vars)
            for h in node.handlers:
                env.vars = dict(pre)
                if h.type is not None:
                    self.expr(h.type, env)
                if h.name:
                    if env.scope is not None:
                        env.scope.add_hoist(h.name)
                    env.bind(h.name, UNKNOWN)
                for st in h.body:
                    self.stmt(st, env, rets)
                for k, v in env.vars.items():
                    prev = all_vars.get(k)
                    all_vars[k] = v if prev is None else join_vt(prev, v)
            if not node.handlers:
                all_vars = body_vars
            env.vars = all_vars
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript):
                    self.expr(t.value, env)
                    self.visit_slice(t.slice, env)
                elif isinstance(t, ast.Name):
                    if env.scope is not None and t.id in env.vars:
                        env.vars[t.id] = NONE
                else:
                    raise TranspileError("unsupported delete target")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            if env.scope is not None:
                env.scope.nonlocals.update(node.names)
                for n in node.names:
                    env.scope._seen.discard(n)   # noqa: SLF001
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            raise TranspileError("with statement")
        elif isinstance(node, (ast.AsyncFor, ast.AsyncFunctionDef)):
            raise TranspileError("async function")
        elif isinstance(node, (ast.ClassDef, ast.Match)):
            raise TranspileError("class/match statement")
        elif isinstance(node, ast.While):
            raise TranspileError("while")
        else:
            raise TranspileError(f"unsupported statement {type(node).__name__}")

    def stmt_import(self, node: ast.stmt, env: Env) -> None:
        allowed = set(_MODULE_ATTRS) | {"numpy"}
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in allowed:
                    raise TranspileError(f"module {a.name} not transpilable")
                env.imports[a.asname or root] = (root, None)
                name = a.asname or root
                env.bind(name, VT("module", mod=root))
                env.mark_hoist(name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise TranspileError("relative import")
            if node.module.split(".")[0] not in allowed:
                raise TranspileError(f"module {node.module} not transpilable")
            for a in node.names:
                if a.name == "*":
                    raise TranspileError("star import")
                env.imports[a.asname or a.name] = (node.module, a.name)
                name = a.asname or a.name
                env.bind(name, VT("module",
                                  mod=f"{node.module}.{a.name}"))
                env.mark_hoist(name)

    # -- assignment targets ------------------------------------------------
    def assign_target(self, t: ast.expr, value: ast.expr, env: Env,
                      aug: bool = False) -> None:
        if isinstance(t, ast.Name):
            if aug:
                cur = env.lookup(t.id)
                vt = join_vt(cur, self.expr(value, env))
                if cur is None or cur.kind == "unknown":
                    vt = self.expr(value, env)
            else:
                vt = self.expr(value, env)
            env.bind(t.id, vt)
            env.mark_hoist(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            vt = self.expr(value, env)
            self.bind_target(t, vt, env)
        elif isinstance(t, ast.Starred):
            raise TranspileError("starred assignment target")
        elif isinstance(t, ast.Subscript):
            self.expr(t.value, env)
            self.visit_slice(t.slice, env)
            rhs = self.expr(value, env)
            if not aug:
                self.record_store(t, rhs, env)
        elif isinstance(t, ast.Attribute):
            raise TranspileError("attribute assignment target")
        else:
            raise TranspileError("unsupported assignment target")

    def bind_target(self, t: ast.expr, vt: VT, env: Env) -> None:
        """Bind names of a (possibly nested) target from the value type."""
        if isinstance(t, ast.Name):
            env.bind(t.id, vt)
            env.mark_hoist(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            inner = _elem_of(vt)
            for elt in t.elts:
                if isinstance(elt, ast.Starred):
                    raise TranspileError("starred target")
                self.bind_target(elt, inner, env)
        elif isinstance(t, ast.Subscript):
            self.expr(t.value, env)
            self.visit_slice(t.slice, env)
        elif isinstance(t, ast.Starred):
            raise TranspileError("starred target")
        elif isinstance(t, ast.Attribute):
            raise TranspileError("attribute target")

    def for_targets(self, t: ast.expr, env: Env) -> None:
        """Hoist every name bound by a for-target (python loop vars leak)."""
        if isinstance(t, ast.Name):
            env.mark_hoist(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                self.for_targets(elt, env)
        elif isinstance(t, ast.Subscript):
            self.expr(t.value, env)
            self.visit_slice(t.slice, env)

    def visit_slice(self, node: ast.expr, env: Env) -> None:
        if isinstance(node, ast.Slice):
            for part in (_slice_part(node, "lower"),
                         _slice_part(node, "upper"), node.step):
                if part is not None:
                    self.expr(part, env)
        else:
            self.expr(node, env)

    def record_store(self, sub: ast.Subscript, rhs: VT, env: Env) -> None:
        """Learn the element/value type from ``container[k] = v``."""
        base = sub.value
        if isinstance(base, ast.Name):
            cont = env.lookup(base.id)
            if cont is None:
                return
            if cont.kind in ("list", "tuple"):
                if cont.elem is None or cont.elem.kind == "unknown":
                    cont.elem = rhs
                else:
                    cont.elem = join_vt(cont.elem, rhs)
            elif cont.kind in ("dict", "counter"):
                if not isinstance(sub.slice, ast.Slice):
                    svt = self.expr(sub.slice, env)
                    if svt.kind == "list":
                        raise TranspileError("unhashable dict key")
                if cont.vval is None or cont.vval.kind == "unknown":
                    cont.vval = rhs
                else:
                    cont.vval = join_vt(cont.vval, rhs)
        elif isinstance(base, ast.Subscript):
            pass  # nested store: container kind of the base result unknown

    # -- expressions -------------------------------------------------------
    def expr(self, node: ast.expr, env: Env) -> VT:
        vt = self._expr(node, env)
        node._vt = vt              # type: ignore[attr-defined]
        return vt

    def _expr(self, node: ast.expr, env: Env) -> VT:
        if isinstance(node, ast.Constant):
            v = node.value
            if v is None:
                return NONE
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                return INT
            if isinstance(v, float):
                return FLOAT
            if isinstance(v, str):
                return STR
            if isinstance(v, (bytes, complex, tuple, frozenset)):
                raise TranspileError(f"constant {type(v).__name__}")
            return UNKNOWN
        if isinstance(node, ast.Name):
            return self.name_vt(node, env)
        if isinstance(node, ast.BinOp):
            return self.binop_vt(node, env)
        if isinstance(node, ast.UnaryOp):
            opnd = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return BOOL
            if isinstance(node.op, ast.Invert):
                return INT
            if opnd.kind == "int":
                return INT
            if opnd.kind == "bool":
                return INT
            if opnd.kind == "float":
                return FLOAT
            if opnd.kind == "num":
                return NUM
            return NUM
        if isinstance(node, ast.BoolOp):
            out: VT | None = None
            for v in node.values:
                out = join_vt(out, self.expr(v, env))
            return out or UNKNOWN
        if isinstance(node, ast.Compare):
            for n in [node.left] + list(node.comparators):
                self.expr(n, env)
            return BOOL
        if isinstance(node, ast.IfExp):
            self.expr(node.test, env)
            a = self.expr(node.body, env)
            b = self.expr(node.orelse, env)
            return join_vt(a, b)
        if isinstance(node, ast.Call):
            return self.call_vt(node, env)
        if isinstance(node, ast.Subscript):
            return self.subscript_vt(node, env)
        if isinstance(node, ast.Attribute):
            return self.attribute_vt(node, env)
        if isinstance(node, ast.List):
            elem: VT | None = None
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    raise TranspileError("starred list element")
                elem = join_vt(elem, self.expr(e, env))
            return VT("list", elem=elem)
        if isinstance(node, ast.Tuple):
            elem = None
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    raise TranspileError("starred tuple element")
                elem = join_vt(elem, self.expr(e, env))
            return VT("tuple", elem=elem)
        if isinstance(node, ast.Set):
            elem = None
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    raise TranspileError("starred set element")
                ev = self.expr(e, env)
                if ev.kind == "list":
                    raise TranspileError("unhashable set element")
                elem = join_vt(elem, ev)
            return VT("set", elem=elem)
        if isinstance(node, ast.Dict):
            kval: VT | None = None
            vval: VT | None = None
            for k, v in zip(node.keys, node.values):
                if k is None:
                    raise TranspileError("dict unpacking")
                kvt = self.expr(k, env)
                if kvt.kind == "list":
                    raise TranspileError("unhashable dict key")
                kval = join_vt(kval, kvt)
                vval = join_vt(vval, self.expr(v, env))
            return VT("dict", kval=kval, vval=vval)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            vt = self.comp_vt(node, env)
            if isinstance(node, ast.GeneratorExp):
                return VT("list", elem=vt.elem) if vt.kind != "unknown" else LIST_ANY
            return vt
        if isinstance(node, ast.DictComp):
            return self.dict_comp_vt(node, env)
        if isinstance(node, ast.SetComp):
            vt = self.comp_vt(node, env)
            if vt.elem and vt.elem.kind == "list":
                raise TranspileError("set of unhashable elements")
            return VT("set", elem=vt.elem)
        if isinstance(node, ast.Lambda):
            if node.args.defaults or node.args.kwonlyargs or node.args.vararg:
                raise TranspileError("lambda defaults/variadic")
            lenv = Env(env, None)
            for a in node.args.args:
                lenv.bind(a.arg, UNKNOWN)
            body = self.expr(node.body, lenv)
            return VT("func", elem=body)
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                self.expr(v, env)
            return STR
        if isinstance(node, ast.FormattedValue):
            self.expr(node.value, env)
            if node.format_spec is not None:
                self.expr(node.format_spec, env)
            return STR
        if isinstance(node, ast.Starred):
            raise TranspileError("starred expression")
        if isinstance(node, ast.NamedExpr):
            raise TranspileError("walrus operator")
        raise TranspileError(f"unsupported expression {type(node).__name__}")

    def name_vt(self, node: ast.Name, env: Env) -> VT:
        v = env.lookup(node.id)
        if v is not None:
            return v
        imp = env.import_lookup(node.id)
        if imp is not None:
            mod, sym = imp
            return VT("module", mod=f"{mod}.{sym}" if sym else mod)
        if node.id in UNSAFE_BUILTINS:
            raise TranspileError(f"builtin {node.id} not transpilable")
        if node.id in KNOWN_BUILTINS:
            return VT("module", mod="builtin." + node.id)
        return UNKNOWN

    def binop_vt(self, node: ast.BinOp, env: Env) -> VT:
        a = self.expr(node.left, env)
        b = self.expr(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            if a.kind in ("list", "tuple") and b.kind in ("list", "tuple"):
                return VT("list", elem=join_vt(a.elem, b.elem))
            if a.kind == "list":
                return VT("list", elem=a.elem)
            if a.kind == "str" and b.kind == "str":
                return STR
            if a.kind == "str" or b.kind == "str":
                return STR
            return _arith_vt(a, b)
        if isinstance(op, ast.Sub):
            return _arith_vt(a, b)
        if isinstance(op, ast.Mult):
            if a.kind == "list" and (b.is_num or b.kind == "unknown"):
                return VT("list", elem=a.elem)
            if b.kind == "list" and (a.is_num or a.kind == "unknown"):
                return VT("list", elem=b.elem)
            if a.kind == "str" and b.is_num:
                return STR
            if b.kind == "str" and a.is_num:
                return STR
            return _arith_vt(a, b)
        if isinstance(op, ast.Div):
            return FLOAT
        if isinstance(op, ast.FloorDiv):
            if a.kind == "int" and b.kind == "int":
                return INT
            return NUM if a.is_num or b.is_num or a.kind == "unknown" else UNKNOWN
        if isinstance(op, ast.Mod):
            if a.kind == "str":
                return STR
            if a.kind == "int" and b.kind == "int":
                return INT
            return NUM if a.is_num or b.is_num or a.kind == "unknown" else UNKNOWN
        if isinstance(op, ast.Pow):
            if a.kind == "int" and b.kind == "int":
                return INT
            return NUM if a.is_num or b.is_num or a.kind == "unknown" else UNKNOWN
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor,
                           ast.LShift, ast.RShift)):
            return INT
        if isinstance(op, ast.MatMult):
            raise TranspileError("matrix multiply")
        return UNKNOWN

    def call_vt(self, node: ast.Call, env: Env) -> VT:
        arg_vts: list[VT] = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise TranspileError("starred call argument")
            arg_vts.append(self.expr(a, env))
        for kw in node.keywords:
            if kw.arg is None:
                raise TranspileError("**kwargs call")
            self.expr(kw.value, env)

        f = node.func
        if isinstance(f, ast.Name):
            fvt = self.name_vt(f, env)
            f._vt = fvt                   # type: ignore[attr-defined]
            if node.keywords and not (
                    fvt.kind == "module" and fvt.mod
                    and fvt.mod.startswith("builtin.")
                    and f.id in ("print", "sorted", "max", "min")):
                raise TranspileError(f"keyword arguments to {f.id}")
            if fvt.kind == "func":
                return fvt.elem if fvt.elem is not None else NONE
            if fvt.kind == "module" and fvt.mod:
                if fvt.mod.startswith("builtin."):
                    return self.builtin_call_vt(f.id, arg_vts, node)
                parts = fvt.mod.split(".")
                if len(parts) == 2:
                    special = _FROM_IMPORT_CALL.get((parts[0], parts[1]))
                    if special is not None:
                        return _fresh(special)
                return UNKNOWN
            return UNKNOWN              # local shadow / forward function

        if isinstance(f, ast.Attribute):
            obj = self.expr(f.value, env)
            if node.keywords and obj.kind != "module" \
                    and f.attr not in ("format", "sort"):
                raise TranspileError(f"keyword arguments to .{f.attr}")
            if obj.kind == "module":
                return self.module_call_vt(obj, f.attr, arg_vts)
            return self.method_vt(obj, f.attr, arg_vts)

        self.expr(f, env)
        return UNKNOWN

    def builtin_call_vt(self, name: str, argvts: list[VT],
                        node: ast.Call) -> VT:
        if name in ("int", "len", "ord"):
            return INT
        if name in ("str", "chr", "format"):
            return STR
        if name == "float":
            return FLOAT
        if name == "bool":
            return BOOL
        if name == "range":
            return RANGE
        if name == "input":
            return STR
        if name == "print":
            return NONE
        if name == "abs":
            return argvts[0] if argvts and argvts[0].is_num else NUM
        if name == "pow":
            if len(argvts) >= 2 and argvts[0].kind == "int" \
                    and argvts[1].kind == "int":
                return INT
            return NUM
        if name == "round":
            return INT if len(argvts) < 2 else FLOAT
        if name == "divmod":
            return VT("tuple", elem=NUM)
        if name in ("bin", "hex", "oct"):
            return STR
        if name in ("any", "all", "isinstance", "issubclass"):
            return BOOL
        if name == "repr":
            return STR
        if name == "hash":
            return INT
        if name in ("eval", "exec", "compile", "globals", "locals", "vars",
                    "dir", "getattr", "setattr", "delattr", "hasattr",
                    "callable", "id", "super", "object", "property",
                    "staticmethod", "classmethod", "memoryview", "bytearray",
                    "bytes", "frozenset", "slice", "iter", "next", "help"):
            raise TranspileError(f"builtin {name} not transpilable")
        if name == "type":
            return UNKNOWN
        a0 = argvts[0] if argvts else None
        if name == "list":
            return _list_vt(a0)
        if name == "tuple":
            if a0 is None:
                return VT("tuple")
            if a0.kind in ("list", "tuple", "deque"):
                return VT("tuple", elem=a0.elem)
            if a0.kind in ("dict", "counter"):
                return VT("tuple", elem=a0.kval)
            if a0.kind == "set":
                return VT("tuple", elem=a0.elem)
            if a0.kind == "str":
                return VT("tuple", elem=STR)
            if a0.kind == "range":
                return VT("tuple", elem=INT)
            return VT("tuple")
        if name == "set":
            elem = None
            if a0 is not None:
                if a0.kind in ("list", "tuple", "set", "deque"):
                    elem = a0.elem
                elif a0.kind in ("dict", "counter"):
                    elem = a0.kval
                elif a0.kind == "range":
                    elem = INT
                elif a0.kind == "str":
                    elem = STR
                elif a0.kind == "unknown":
                    elem = None
            if elem is not None and elem.kind == "list":
                raise TranspileError("set of unhashable elements")
            return VT("set", elem=elem)
        if name == "dict":
            if a0 is not None and a0.kind in ("dict", "counter"):
                return VT("dict", kval=a0.kval, vval=a0.vval)
            return VT("dict", kval=UNKNOWN, vval=UNKNOWN)
        if name in ("sorted", "reversed"):
            return _list_vt(a0)
        if name == "filter":
            return _list_vt(argvts[1] if len(argvts) > 1 else None)
        if name == "map":
            elem: VT | None = None
            if node.args and isinstance(node.args[0], ast.Name):
                fn = node.args[0].id
                if fn == "int":
                    elem = INT
                elif fn == "str":
                    elem = STR
                elif fn == "float":
                    elem = FLOAT
            if elem is None and a0 is not None:
                if a0.kind == "func":
                    elem = a0.elem
                elif a0.kind == "module" and a0.mod == "builtin.int":
                    elem = INT
                elif a0.kind == "module" and a0.mod == "builtin.str":
                    elem = STR
            return VT("list", elem=elem)
        if name == "enumerate":
            it = argvts[0] if argvts else None
            inner = it.elem if it is not None and it.kind in (
                "list", "tuple", "deque", "set", "str", "range",
                "dict", "counter") else None
            if it is not None and it.kind == "str":
                inner = STR
            if it is not None and it.kind == "range":
                inner = INT
            if it is not None and it.kind in ("dict", "counter"):
                inner = it.kval
            return VT("list", elem=VT("tuple", elem=join_vt(INT, inner)))
        if name == "zip":
            elem = None
            for z in argvts:
                if z.kind in ("list", "tuple", "deque", "set"):
                    elem = join_vt(elem, z.elem)
                elif z.kind == "str":
                    elem = join_vt(elem, STR)
                elif z.kind == "range":
                    elem = join_vt(elem, INT)
                elif z.kind in ("dict", "counter"):
                    elem = join_vt(elem, z.kval)
            return VT("list", elem=VT("tuple", elem=elem))
        if name == "sum":
            e = a0.elem if a0 is not None and a0.kind in (
                "list", "tuple", "deque", "range") else None
            if e is not None and e.kind == "int":
                return INT
            if e is not None and e.kind == "float":
                return FLOAT
            if e is not None and e.kind == "bool":
                return INT
            if e is not None and e.kind in ("num", "unknown"):
                return NUM if e.kind == "num" else NUM
            return NUM
        if name in ("max", "min"):
            if a0 is not None and a0.kind in ("list", "tuple", "deque",
                                               "set", "range"):
                return a0.elem or NUM
            if a0 is not None and a0.kind == "str":
                return STR
            if a0 is not None and a0.kind in ("dict", "counter"):
                return a0.kval or NUM
            if len(argvts) > 1:
                out: VT | None = None
                for x in argvts:
                    out = join_vt(out, x)
                return out or NUM
            return a0 if a0 is not None and a0.is_num else NUM
        if name == "reversed":
            return _list_vt(a0)
        if name == "input":
            return STR
        if name == "print":
            return NONE
        if a0 is not None and a0.kind == "func":
            return a0.elem or UNKNOWN
        return UNKNOWN

    def module_call_vt(self, obj: VT, attr: str | None,
                       argvts: list[VT]) -> VT:
        path = obj.mod or ""
        full = f"{path}.{attr}" if attr else path
        root = path.split(".")[0]
        attrs = _MODULE_ATTRS.get(root, {})
        if full in attrs:
            return _fresh(attrs[full])
        if full.startswith(root + ".") and full[len(root) + 1:] in attrs:
            return _fresh(attrs[full[len(root) + 1:]])
        parts = full.split(".")
        for i in range(1, len(parts)):
            special = _FROM_IMPORT_CALL.get((".".join(parts[:i]), parts[i]))
            if special is not None:
                return _fresh(special)
        return UNKNOWN

    def method_vt(self, obj: VT, attr: str, argvts: list[VT]) -> VT:
        k = obj.kind
        if k == "str":
            if attr in ("split", "splitlines"):
                return VT("list", elem=STR)
            if attr in ("strip", "lstrip", "rstrip", "replace", "format",
                        "translate", "lower", "upper", "capitalize", "title",
                        "zfill", "ljust", "rjust", "center", "encode",
                        "decode", "expandtabs", "rstrip", "casefold",
                        "removeprefix", "removesuffix", "swapcase"):
                return STR
            if attr in ("find", "index", "rfind", "rindex", "count"):
                return INT
            if attr in ("join",):
                return STR
            if attr in ("startswith", "endswith", "isdigit", "isalpha",
                        "isalnum", "isspace", "islower", "isupper",
                        "istitle", "isdecimal", "isnumeric"):
                return BOOL
            if attr in ("partition", "rpartition"):
                return VT("tuple", elem=STR)
            if attr in ("casefold",):
                return STR
            return UNKNOWN
        if k in ("list", "tuple"):
            if attr == "pop":
                return obj.elem or UNKNOWN
            if attr in ("index", "count"):
                return INT
            if attr == "copy":
                return VT("list", elem=obj.elem)
            if attr == "join":
                return STR
            if attr in ("append", "extend", "insert", "remove", "sort",
                        "reverse", "clear"):
                return NONE
            return UNKNOWN
        if k in ("dict", "counter"):
            if attr == "items":
                return VT("list", elem=VT(
                    "tuple", elem=join_vt(obj.kval or UNKNOWN,
                                          obj.vval or UNKNOWN)))
            if attr == "keys":
                return VT("list", elem=obj.kval)
            if attr == "values":
                return VT("list", elem=obj.vval)
            if attr == "get":
                dflt = argvts[1] if len(argvts) > 1 else NONE
                return join_vt(obj.vval, dflt)
            if attr == "pop":
                dflt = argvts[1] if len(argvts) > 1 else NONE
                return join_vt(obj.vval, dflt)
            if attr == "setdefault":
                return join_vt(obj.vval, NONE)
            if attr in ("update", "clear", "setdefault"):
                return NONE
            if attr == "copy":
                return VT("dict", kval=obj.kval, vval=obj.vval)
            if attr in ("values", "keys"):
                return VT("list")
            return UNKNOWN
        if k == "set":
            if attr == "pop":
                return obj.elem or UNKNOWN
            if attr in ("add", "discard", "remove", "clear", "update"):
                return NONE
            if attr == "copy":
                return VT("set", elem=obj.elem)
            if attr in ("union", "intersection", "difference",
                        "symmetric_difference"):
                return VT("set", elem=obj.elem)
            if attr == "isdisjoint":
                return BOOL
            return UNKNOWN
        if k == "deque":
            if attr in ("popleft", "pop"):
                return obj.elem or UNKNOWN
            if attr in ("append", "appendleft", "extend", "extendleft",
                        "rotate", "clear"):
                return NONE
            if attr == "copy":
                return VT("deque", elem=obj.elem)
            if attr in ("count", "index"):
                return INT
            return UNKNOWN
        if k == "module":
            return UNKNOWN
        return UNKNOWN

    def attribute_vt(self, node: ast.Attribute, env: Env) -> VT:
        obj = self.expr(node.value, env)
        if obj.kind == "module":
            path = f"{obj.mod}.{node.attr}"
            root = (obj.mod or "").split(".")[0]
            attrs = _MODULE_ATTRS.get(root, {})
            if path in attrs:
                return _fresh(attrs[path])
            if path.startswith(root + ".") and path[len(root) + 1:] in attrs:
                return _fresh(attrs[path[len(root) + 1:]])
            return VT("module", mod=path)
        return self.method_vt(obj, node.attr, [])

    def subscript_vt(self, node: ast.Subscript, env: Env) -> VT:
        cont = self.expr(node.value, env)
        if isinstance(node.slice, ast.Slice):
            for part in (_slice_part(node.slice, "lower"),
                         _slice_part(node.slice, "upper"),
                         node.slice.step):
                if part is not None:
                    self.expr(part, env)
            if cont.kind == "str":
                return STR
            if cont.kind in ("list", "tuple", "deque"):
                return VT(cont.kind, elem=cont.elem)
            return UNKNOWN
        idx = self.expr(node.slice, env)
        k = cont.kind
        if k in ("dict", "counter"):
            if idx.kind == "list":
                raise TranspileError("unhashable dict key")
            return cont.vval if cont.vval is not None else UNKNOWN
        if k in ("list", "tuple", "deque"):
            return cont.elem or UNKNOWN
        if k == "str":
            return STR
        if k == "range":
            return INT
        if k == "set":
            raise TranspileError("set subscript")
        return UNKNOWN

    def comp_vt(self, node, env: Env) -> VT:
        cenv = Env(env, None)
        for i, gen in enumerate(node.generators):
            iter_env = env if i == 0 else cenv
            it_vt = self.expr(gen.iter, iter_env)
            self.bind_target(gen.target, _elem_of(it_vt), cenv)
            for cond in gen.ifs:
                self.expr(cond, cenv)
        elt = self.expr(node.elt, cenv)
        return VT("list", elem=elt)

    def dict_comp_vt(self, node: ast.DictComp, env: Env) -> VT:
        cenv = Env(env, None)
        for i, gen in enumerate(node.generators):
            iter_env = env if i == 0 else cenv
            it_vt = self.expr(gen.iter, iter_env)
            self.bind_target(gen.target, _elem_of(it_vt), cenv)
            for cond in gen.ifs:
                self.expr(cond, cenv)
        if node.key is None:
            raise TranspileError("dict unpacking")
        kvt = self.expr(node.key, cenv)
        if kvt.kind == "list":
            raise TranspileError("unhashable dict key")
        vvt = self.expr(node.value, cenv)
        return VT("dict", kval=kvt, vval=vvt)


def _fresh(vt: VT) -> VT:
    return VT(vt.kind, elem=vt.elem, kval=vt.kval, vval=vt.vval, mod=vt.mod)


def _elem_of(vt: VT) -> VT:
    """Type of one element produced when iterating ``vt``."""
    if vt.kind in ("list", "tuple", "deque", "set", "range"):
        return vt.elem or UNKNOWN
    if vt.kind in ("dict", "counter"):
        return vt.kval or UNKNOWN
    if vt.kind == "str":
        return STR
    return UNKNOWN


def _slice_part(slc: ast.Slice, name: str) -> ast.expr | None:
    """``start``/``stop`` were renamed ``lower``/``upper`` in py3.9+."""
    if name == "lower":
        v = getattr(slc, "lower", None)
        if v is None:
            v = getattr(slc, "start", None)
        return v
    v = getattr(slc, "upper", None)
    if v is None:
        v = getattr(slc, "stop", None)
    return v


def _arith_vt(a: VT, b: VT) -> VT:
    if a.kind in ("unknown",) or b.kind in ("unknown",):
        return NUM if (a.is_num or b.is_num) else UNKNOWN
    joined = join_vt(a, b)
    if joined.kind in ("int", "float", "num"):
        return joined
    if joined.kind == "bool":
        return INT
    return NUM


def _list_vt(a: VT | None) -> VT:
    if a is None:
        return VT("list")
    if a.kind in ("list", "tuple", "deque", "set"):
        return VT("list", elem=a.elem)
    if a.kind in ("dict", "counter"):
        return VT("list", elem=a.kval)
    if a.kind == "range":
        return VT("list", elem=INT)
    if a.kind == "str":
        return VT("list", elem=STR)
    if a.kind == "unknown":
        return VT("list")
    return VT("list")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def transpile(py_code: str, target: str) -> str:
    """Transpile ``py_code`` into ``target`` (``"javascript"`` / ``"cpp"``).

    Raises TranspileError for anything that cannot be translated
    faithfully -- callers then keep the python-only result.
    """
    try:
        tree = ast.parse(py_code)
    except SyntaxError as exc:
        raise TranspileError(f"syntax error: {exc}") from exc
    Typer().module(tree)
    tgt = (target or "").strip().lower()
    if tgt == "javascript":
        from code.transpiler.js_backend import emit_js
        return emit_js(tree)
    if tgt == "cpp":
        from code.transpiler.cpp_backend import emit_cpp
        return emit_cpp(tree)
    raise TranspileError(f"unsupported transpile target: {target}")

