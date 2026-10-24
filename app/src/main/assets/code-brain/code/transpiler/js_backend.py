"""JavaScript backend for the AST transpiler.

Emits JS whose builtin names match python's own (``int``, ``len``,
``print``...) -- the runtime prelude defines them with identical
shadowing semantics -- while attribute-method helpers live in the
collision-proof ``$`` namespace.  Every function-scoped assigned name is
hoisted to the top of the emitted function because python has no block
scope but JS does.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from code.transpiler.py_ast_transpiler import (
    KNOWN_BUILTINS,
    TranspileError,
    VT,
    _FROM_IMPORT_CALL,
    _MODULE_ATTRS,
    _slice_part,
    UNKNOWN,
)


def _read_prelude() -> str:
    try:
        return Path(__file__).with_name("js_prelude.js").read_text(
            encoding="utf-8")
    except OSError:
        return ""

# JS reserved words (plus read-only globals) a python identifier must not
# collide with.  Mangling appends "$" which python can never contain.
JS_RESERVED = {
    "arguments", "await", "break", "case", "catch", "class", "const",
    "continue", "debugger", "default", "delete", "do", "else", "enum",
    "eval", "export", "extends", "false", "finally", "for", "function",
    "if", "implements", "import", "in", "Infinity", "instanceof", "interface",
    "let", "NaN", "new", "null", "package", "private", "protected", "public",
    "return", "static", "super", "switch", "this", "throw", "true", "try",
    "typeof", "var", "void", "while", "with", "yield", "undefined",
}

# Modules the prelude provides real objects for.
KNOWN_MODULES = {"sys", "math", "re", "heapq", "itertools", "bisect",
                 "collections"}

CATCH_NAME = "__py$err"

SIMPLE_EXPR = (ast.Name, ast.Constant)


def js_name(name: str) -> str:
    return name + "$" if name in JS_RESERVED else name


class JSBackend:
    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.out: list[str] = []
        self.temps: set[str] = set()   # hoisted temps (current function)
        self.tmpn = 0

    # -- helpers -----------------------------------------------------------
    def pad(self, level: int) -> str:
        return "    " * level

    def hoisttmp(self) -> str:
        self.tmpn += 1
        t = f"__py$t{self.tmpn}"
        self.temps.add(t)
        return t

    def blocktmp(self, prefix: str = "t") -> str:
        self.tmpn += 1
        return f"__py${prefix}{self.tmpn}"

    def kwobj(self, kws: list[ast.keyword]) -> str | None:
        if not kws:
            return None
        parts = [f"{k.arg}: {self.expr(k.value)}" if k.arg else
                 f"...{self.expr(k.value)}" for k in kws]
        return "$.kw({" + ", ".join(parts) + "})"

    # -- module ------------------------------------------------------------
    def emit(self) -> str:
        entry: str | None = None
        defs: list[ast.FunctionDef] = []
        for st in self.tree.body:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                self.import_stmt(st, 0, self.out)
            elif isinstance(st, ast.FunctionDef):
                defs.append(st)
                self.function(st, 0, self.out)
            else:
                self.stmt(st, 0, self.out)
        if not self._has_main_guard():
            names = [d.name for d in defs]
            if "solution" in names:
                entry = "solution"
            elif len(defs) == 1:
                entry = defs[0].name
            if entry and not self._has_module_call(defs):
                self.out.append(f"{js_name(entry)}();")
        head = ""
        if self.temps:
            head = "let " + ", ".join(sorted(self.temps)) + ";\n"
        return _read_prelude() + "\n" + head + "\n".join(self.out) + "\n"

    def _has_main_guard(self) -> bool:
        for st in self.tree.body:
            if isinstance(st, ast.If) and isinstance(st.test, ast.Compare):
                left = st.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return True
        return False

    def _has_module_call(self, defs: list[ast.FunctionDef]) -> bool:
        """True when the module body already invokes one of its functions.

        The program manages its own entry then; appending another call
        would execute it twice.
        """
        names = {d.name for d in defs}
        for st in self.tree.body:
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                f = st.value.func
                if isinstance(f, ast.Name) and f.id in names:
                    return True
        return False

    def function(self, fd: ast.FunctionDef, level: int,
                 out: list[str]) -> None:
        saved_temps, saved_n = self.temps, self.tmpn
        self.temps, self.tmpn = set(), 0
        body: list[str] = []
        for st in fd.body:
            self.stmt(st, level + 1, body)
        scope = getattr(fd, "_scope", None)
        hoist: list[str] = [js_name(n) for n in scope.hoist] if scope else []
        hoist.extend(sorted(self.temps))
        hoist.append(CATCH_NAME)
        pad = self.pad(level)
        params = ", ".join(js_name(a.arg) for a in fd.args.args)
        seen: set[str] = set()
        pset = {js_name(a.arg) for a in fd.args.args}
        uniq = [h for h in hoist if h not in pset
                and not (h in seen or seen.add(h))]
        out.append(f"{pad}function {js_name(fd.name)}({params}) {{")
        if uniq:
            out.append(f"{pad}    let " + ", ".join(uniq) + ";")
        out.extend(body)
        out.append(f"{pad}}}")
        self.temps, self.tmpn = saved_temps, saved_n

    # -- imports -----------------------------------------------------------
    def import_stmt(self, node: ast.stmt, level: int,
                    out: list[str]) -> None:
        pad = self.pad(level)
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                name = js_name(a.asname or root)
                rhs = (f"$.{root}" if root in KNOWN_MODULES
                       else f"$.mod({json.dumps(root)})")
                if level == 0:
                    out.append(f"{pad}var {name} = {rhs};")
                else:
                    out.append(f"{pad}{name} = {rhs};")
        else:
            assert isinstance(node, ast.ImportFrom)
            root = (node.module or "").split(".")[0]
            for a in node.names:
                if not self._ok_from(root, a.name):
                    raise TranspileError(
                        f"symbol {root}.{a.name} not transpilable")
                name = js_name(a.asname or a.name)
                if root in KNOWN_MODULES:
                    rhs = f"$.{root}.{a.name}"
                else:
                    rhs = f"$.mod({json.dumps(root)}).{a.name}"
                if level == 0:
                    out.append(f"{pad}var {name} = {rhs};")
                else:
                    out.append(f"{pad}{name} = {rhs};")

    # -- statements --------------------------------------------------------
    def stmt(self, node: ast.stmt, level: int,
             out: list[str] | None = None) -> None:
        if out is None:
            out = self.out
        pad = self.pad(level)

        if isinstance(node, ast.Assign):
            self.assign(node, level, out)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.assign_from(node.target, node.value, out, pad, level)
        elif isinstance(node, ast.AugAssign):
            self.aug_assign(node, out, pad)
        elif isinstance(node, ast.Expr):
            out.append(f"{pad}({self.expr(node.value)});")
        elif isinstance(node, ast.If):
            self.if_stmt(node, out, level)
        elif isinstance(node, ast.For):
            self.for_stmt(node, out, level)
        elif isinstance(node, ast.While):
            if node.orelse:
                raise TranspileError("while-else unsupported")
            out.append(f"{pad}while ({self.cond(node.test)}) {{")
            for st in node.body:
                self.stmt(st, level + 1, out)
            out.append(f"{pad}}}")
        elif isinstance(node, ast.Break):
            out.append(f"{pad}break;")
        elif isinstance(node, ast.Continue):
            out.append(f"{pad}continue;")
        elif isinstance(node, ast.Pass):
            out.append(f"{pad};")
        elif isinstance(node, ast.Return):
            v = self.expr(node.value) if node.value is not None else "null"
            out.append(f"{pad}return {v};")
        elif isinstance(node, ast.Raise):
            self.raise_stmt(node, out, pad)
        elif isinstance(node, ast.Assert):
            msg = (self.expr(node.msg) if node.msg is not None
                   else "'assertion failed'")
            out.append(f"{pad}if (!({self.cond(node.test)})) "
                       f"throw new Error($.str({msg}));")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self.import_stmt(node, level, out)
        elif isinstance(node, ast.FunctionDef):
            self.function(node, level, out)
        elif isinstance(node, ast.Try):
            self.try_stmt(node, out, level)
        elif isinstance(node, ast.Delete):
            self.delete_stmt(node, out, pad)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            pass
        else:
            raise TranspileError(f"unsupported statement "
                                 f"{type(node).__name__}")

    def assign(self, node: ast.Assign, level: int, out: list[str]) -> None:
        pad = self.pad(level)
        rhs_node = node.value
        rhs = self.expr(rhs_node)
        targets = node.targets
        if len(targets) == 1:
            self.assign_from(targets[0], rhs_node, out, pad, level,
                             rhs_str=rhs)
            return
        # multiple targets: python evaluates the value once
        if all(isinstance(t, ast.Name) for t in targets):
            chain = " = ".join([js_name(t.id) for t in targets] + [rhs])
            if level == 0:
                out.append(f"{pad}var {chain};")
            else:
                out.append(f"{pad}{chain};")
            return
        t = self.hoisttmp()
        out.append(f"{pad}{t} = {rhs};")
        for tgt in targets:
            self.assign_from(tgt, rhs_node, out, pad, level, rhs_str=t)

    def assign_from(self, target: ast.expr, rhs_node: ast.expr,
                    out: list[str], pad: str, level: int,
                    rhs_str: str | None = None) -> None:
        if rhs_str is None:
            rhs_str = self.expr(rhs_node)
        if isinstance(target, ast.Name):
            if level == 0:
                out.append(f"{pad}var {js_name(target.id)} = {rhs_str};")
            else:
                out.append(f"{pad}{js_name(target.id)} = {rhs_str};")
        elif isinstance(target, (ast.Tuple, ast.List)):
            pat = self.pattern(target)
            out.append(f"{pad}({pat} = $.unpack({rhs_str}));")
        elif isinstance(target, ast.Subscript):
            self.subscript_store(target, rhs_str, out, pad)
        else:
            raise TranspileError("unsupported assignment target")

    def subscript_store(self, target: ast.Subscript, rhs: str,
                        out: list[str], pad: str) -> None:
        cont = self.expr(target.value)
        cvt = getattr(target.value, "_vt", UNKNOWN)
        if isinstance(target.slice, ast.Slice):
            lo = (self.expr(_slice_part(target.slice, "lower"))
                  if _slice_part(target.slice, "lower") else "null")
            hi = (self.expr(_slice_part(target.slice, "upper"))
                  if _slice_part(target.slice, "upper") else "null")
            st = (self.expr(target.slice.step)
                  if target.slice.step else "null")
            out.append(f"{pad}$.sliceAssign({cont}, {lo}, {hi}, {st}, "
                       f"{rhs});")
            return
        idx = self.expr(target.slice)
        if cvt.kind in ("dict", "counter"):
            out.append(f"{pad}({cont}).set({idx}, {rhs});")
        elif (isinstance(target.slice, ast.Constant)
              and isinstance(target.slice.value, int)
              and target.slice.value >= 0):
            out.append(f"{pad}({cont})[{target.slice.value}] = {rhs};")
        else:
            out.append(f"{pad}$.aset({cont}, {idx}, {rhs});")

    def _ok_from(self, root: str, sym: str) -> bool:
        attrs = _MODULE_ATTRS.get(root, {})
        if sym in attrs:
            return True
        if any(k.startswith(sym + ".") for k in attrs):
            return True
        return (root, sym) in _FROM_IMPORT_CALL

    def aug_assign(self, node: ast.AugAssign, out: list[str],
                   pad: str) -> None:
        opsym = _AUG_OPS.get(type(node.op))
        if opsym is None:
            raise TranspileError("unsupported augmented assignment")
        if isinstance(node.target, ast.Name):
            n = js_name(node.target.id)
            rhs = self.expr(node.value)
            lvt = getattr(node.target, "_vt", UNKNOWN)
            rvt = getattr(node.value, "_vt", UNKNOWN)
            if (isinstance(node.op, ast.Add)
                    and (self._is_seq_vt(lvt) or lvt.kind == "counter")):
                val = f"$.iadd({n}, {rhs})"
            else:
                val = self.binop_str(node.op, n, rhs, lvt, rvt)
            out.append(f"{pad}{n} = {val};")
        elif isinstance(node.target, ast.Subscript):
            cont = self.expr(node.target.value)
            idx = self.expr(node.target.slice)
            rhs = self.expr(node.value)
            out.append(f"{pad}$.aug({cont}, {idx}, {json.dumps(opsym)}, "
                       f"{rhs});")
        else:
            raise TranspileError("unsupported augmented target")

    def if_stmt(self, node: ast.If, out: list[str], level: int) -> None:
        pad = self.pad(level)
        out.append(f"{pad}if ({self.cond(node.test)}) {{")
        for st in node.body:
            self.stmt(st, level + 1, out)
        if node.orelse:
            if (len(node.orelse) == 1
                    and isinstance(node.orelse[0], ast.If)):
                # convert to "} else if" chain
                out.append(f"{pad}}} else if "
                           f"({self.cond(node.orelse[0].test)}) {{")
                inner = node.orelse[0]
                # flatten: emit body, then recurse for its orelse
                for st in inner.body:
                    self.stmt(st, level + 1, out)
                rest = inner.orelse
                while True:
                    if not rest:
                        out.append(f"{pad}}}")
                        break
                    if len(rest) == 1 and isinstance(rest[0], ast.If):
                        out.append(f"{pad}}} else if "
                                   f"({self.cond(rest[0].test)}) {{")
                        for st in rest[0].body:
                            self.stmt(st, level + 1, out)
                        rest = rest[0].orelse
                        continue
                    out.append(f"{pad}}} else {{")
                    for st in rest:
                        self.stmt(st, level + 1, out)
                    out.append(f"{pad}}}")
                    break
                return
            out.append(f"{pad}}} else {{")
            for st in node.orelse:
                self.stmt(st, level + 1, out)
            out.append(f"{pad}}}")
        else:
            out.append(f"{pad}}}")

    def for_stmt(self, node: ast.For, out: list[str], level: int) -> None:
        if node.orelse:
            raise TranspileError("for-else unsupported")
        pad = self.pad(level)
        it = node.iter
        tgt = node.target
        if isinstance(tgt, ast.Subscript):
            raise TranspileError("subscript for-target")
        if self._is_range(it) and isinstance(tgt, ast.Name):
            self._for_range(node, out, level)
            return
        itexpr = self.iter_expr(it)
        if isinstance(tgt, ast.Name):
            head = f"{js_name(tgt.id)} of {itexpr}"
        else:
            head = f"{self.pattern(tgt)} of {itexpr}"
        out.append(f"{pad}for ({head}) {{")
        for st in node.body:
            self.stmt(st, level + 1, out)
        out.append(f"{pad}}}")

    def _is_range(self, node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range")

    def _for_range(self, node: ast.For, out: list[str],
                   level: int) -> None:
        pad = self.pad(level)
        call = node.iter
        args = call.args
        n = len(args)
        if n == 1:
            start_n, stop_n, step_n = None, args[0], None
        elif n == 2:
            start_n, stop_n, step_n = args[0], args[1], None
        elif n == 3:
            start_n, stop_n, step_n = args[0], args[1], args[2]
        else:
            raise TranspileError("range arity")
        tgt = js_name(node.target.id)

        def part(n_: ast.expr | None, default: str) -> str:
            return self.expr(n_) if n_ is not None else default

        def step_const() -> int | None:
            if step_n is None:
                return 1
            if isinstance(step_n, ast.Constant) and isinstance(
                    step_n.value, int):
                return step_n.value
            return None

        def emit_body(prefix_lines: list[str], s: str, e: str,
                      t: str) -> None:
            sc = step_const()
            if sc is not None:
                cond = (f"{tgt} < {e}" if sc > 0 else f"{tgt} > {e}")
            else:
                cond = f"{t} > 0 ? {tgt} < {e} : {tgt} > {e}"
            out.extend(prefix_lines)
            out.append(f"{pad}for ({tgt} = {s}; {cond}; "
                       f"{tgt} += {t}) {{")
            for st in node.body:
                self.stmt(st, level + 1, out)
            out.append(f"{pad}}}")
            if prefix_lines:
                out.append(f"{pad}}}")

        # python evaluates the range() arguments once at loop entry; block
        # temps snapshot start/stop/step so a body mutation cannot change them
        ts = self.blocktmp("s")
        te = self.blocktmp("e")
        tt = self.blocktmp("k")
        s_val = part(start_n, "0")
        e_val = part(stop_n, "0")
        t_val = part(step_n, "1")
        emit_body([f"{pad}{{ let {ts} = {s_val}, {te} = {e_val}, "
                   f"{tt} = {t_val};"],
                  ts, te, tt)

    def iter_expr(self, node: ast.expr) -> str:
        vt = getattr(node, "_vt", UNKNOWN)
        e = self.expr(node)
        if vt.kind in ("dict", "counter"):
            return f"({e}).keys()"
        if vt.kind in ("list", "tuple", "deque", "set", "str", "range"):
            return e
        return f"$.iter({e})"

    def try_stmt(self, node: ast.Try, out: list[str], level: int) -> None:
        if node.orelse:
            raise TranspileError("try-else unsupported")
        pad = self.pad(level)
        out.append(f"{pad}try {{")
        for st in node.body:
            self.stmt(st, level + 1, out)
        for h in node.handlers:
            out.append(f"{pad}}} catch ({CATCH_NAME}) {{")
            cond = self._handler_cond(h)
            ind = pad + "    "
            if cond is not None:
                out.append(f"{ind}if ({cond}) {{")
                ind += "    "
            if h.name:
                out.append(f"{ind}{js_name(h.name)} = {CATCH_NAME};")
            for st in h.body:
                self.stmt(st, level + 1, out)
            if cond is not None:
                out.append(f"{ind[:-4]}}} else {{")
                out.append(f"{ind[:-4]}    throw {CATCH_NAME};")
                out.append(f"{ind[:-4]}}}")
        if node.finalbody:
            out.append(f"{pad}}} finally {{")
            for st in node.finalbody:
                self.stmt(st, level + 1, out)
        out.append(f"{pad}}}")

    def _handler_cond(self, h: ast.ExceptHandler) -> str | None:
        """JS condition selecting this handler; None means catch-all."""
        t = h.type
        if t is None:
            return None
        if isinstance(t, ast.Name):
            if t.id in ("Exception", "BaseException"):
                return None
            return f"{CATCH_NAME}.pytype === {json.dumps(t.id)}"
        if isinstance(t, ast.Tuple):
            parts = [f"{CATCH_NAME}.pytype === {json.dumps(e.id)}"
                     for e in t.elts if isinstance(e, ast.Name)]
            if not parts:
                return None
            return "(" + " || ".join(parts) + ")"
        return None

    def raise_stmt(self, node: ast.Raise, out: list[str],
                   pad: str) -> None:
        if node.exc is None:
            out.append(f"{pad}throw {CATCH_NAME};")
            return
        exc = node.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            fname = exc.func.id
            if fname == "SystemExit":
                code = self.expr(exc.args[0]) if exc.args else "0"
                out.append(f"{pad}process.exit({code});")
                return
            msg = (self.expr(exc.args[0]) if exc.args else
                   json.dumps(fname))
            out.append(f"{pad}throw $.pyerr({json.dumps(fname)}, "
                       f"$.str({msg}));")
            return
        if isinstance(exc, ast.Name):
            if exc.id == "SystemExit":
                out.append(f"{pad}process.exit(0);")
            else:
                out.append(f"{pad}throw $.pyerr({json.dumps(exc.id)}, "
                           f"{json.dumps(exc.id)});")
            return
        out.append(f"{pad}throw $.pyerr(\"Exception\", "
                   f"$.str({self.expr(exc)}));")

    def delete_stmt(self, node: ast.Delete, out: list[str],
                    pad: str) -> None:
        for t in node.targets:
            if isinstance(t, ast.Name):
                out.append(f"{pad}{js_name(t.id)} = null;")
            elif isinstance(t, ast.Subscript):
                cont = self.expr(t.value)
                cvt = getattr(t.value, "_vt", UNKNOWN)
                if isinstance(t.slice, ast.Slice):
                    raise TranspileError("slice delete")
                idx = self.expr(t.slice)
                if cvt.kind in ("dict", "counter"):
                    out.append(f"{pad}({cont}).delete({idx});")
                else:
                    out.append(f"{pad}$.delitem({cont}, {idx});")
            else:
                raise TranspileError("unsupported delete target")

    def pattern(self, t: ast.expr) -> str:
        """JS destructuring pattern (no declaration) for a tuple target."""
        if isinstance(t, ast.Name):
            return js_name(t.id)
        if isinstance(t, (ast.Tuple, ast.List)):
            return "[" + ", ".join(self.pattern(e) for e in t.elts) + "]"
        if isinstance(t, ast.Subscript):
            cont = self.expr(t.value)
            if isinstance(t.slice, ast.Slice):
                raise TranspileError("slice in pattern")
            idx = self.expr(t.slice)
            return f"({cont})[{idx}]"
        if isinstance(t, ast.Starred):
            raise TranspileError("starred pattern")
        raise TranspileError("unsupported pattern")

    # -- expressions -------------------------------------------------------
    def expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            return self.const(node)
        if isinstance(node, ast.Name):
            return js_name(node.id)
        if isinstance(node, ast.BinOp):
            return self.binop(node)
        if isinstance(node, ast.UnaryOp):
            return self.unary(node)
        if isinstance(node, ast.BoolOp):
            return self.boolop(node)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.IfExp):
            return (f"({self.cond(node.test)} ? {self.expr(node.body)} : "
                    f"{self.expr(node.orelse)})")
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.List):
            return "[" + ", ".join(self.expr(e) for e in node.elts) + "]"
        if isinstance(node, ast.Tuple):
            inner = ", ".join(self.expr(e) for e in node.elts)
            return f"$.tup({inner})" if node.elts else "$.tup()"
        if isinstance(node, ast.Set):
            inner = ", ".join(self.expr(e) for e in node.elts)
            return f"$.fset({inner})" if node.elts else "$.fset()"
        if isinstance(node, ast.Dict):
            pairs = [f"[{self.expr(k)}, {self.expr(v)}]"
                     for k, v in zip(node.keys, node.values)]
            return (f"$.dict([{', '.join(pairs)}])" if pairs
                    else "$.dict()")
        if isinstance(node, (ast.ListComp, ast.SetComp,
                             ast.GeneratorExp, ast.DictComp)):
            return self.comp(node)
        if isinstance(node, ast.Lambda):
            params = ", ".join(js_name(a.arg) for a in node.args.args)
            return f"(({params}) => ({self.expr(node.body)}))"
        if isinstance(node, ast.JoinedStr):
            return self.joined_str(node)
        if isinstance(node, ast.FormattedValue):
            return self.fmt_part(node)
        raise TranspileError(f"unsupported expression "
                             f"{type(node).__name__}")

    def const(self, node: ast.Constant) -> str:
        v = node.value
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            return (str(v) + "n"
                    if v > 9007199254740991 or v < -9007199254740991
                    else str(v))
        if isinstance(v, float):
            if v != v:
                return "NaN"
            if v == float("inf"):
                return "Infinity"
            if v == float("-inf"):
                return "-Infinity"
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=True)
        raise TranspileError(f"constant {type(v).__name__}")

    # -- binary / unary ----------------------------------------------------
    @staticmethod
    def _is_num_vt(vt: VT) -> bool:
        return vt.kind in ("int", "float", "num", "bool")

    @staticmethod
    def _is_seq_vt(vt: VT) -> bool:
        return vt.kind in ("list", "tuple")

    def binop(self, node: ast.BinOp) -> str:
        return self.binop_str(node.op, self.expr(node.left),
                              self.expr(node.right),
                              getattr(node.left, "_vt", UNKNOWN),
                              getattr(node.right, "_vt", UNKNOWN))

    def binop_str(self, op: ast.operator, ls: str, rs: str,
                  lvt: VT, rvt: VT) -> str:
        ln, rn = lvt.kind, rvt.kind
        if isinstance(op, ast.Add):
            if self._is_seq_vt(lvt) or self._is_seq_vt(rvt):
                return f"$.concat({ls}, {rs})"
            if ln == "str" and rn == "str":
                return f"({ls} + {rs})"
            if ln == "str" or rn == "str":
                return f"$.strcat({ls}, {rs})"
            return f"$.pyadd({ls}, {rs})"
        if isinstance(op, ast.Sub):
            return f"$.pysub({ls}, {rs})"
        if isinstance(op, ast.Mult):
            if self._is_seq_vt(lvt) and (
                    self._is_num_vt(rvt) or rn == "unknown"):
                return f"$.repeat({ls}, {rs})"
            if self._is_seq_vt(rvt) and (
                    self._is_num_vt(lvt) or ln == "unknown"):
                return f"$.repeat({rs}, {ls})"
            if ln == "str" and (self._is_num_vt(rvt) or rn == "unknown"):
                return f"$.strmul({ls}, {rs})"
            if rn == "str" and (self._is_num_vt(lvt) or ln == "unknown"):
                return f"$.strmul({rs}, {ls})"
            return f"$.pymul({ls}, {rs})"
        if isinstance(op, ast.Div):
            return f"$.truediv({ls}, {rs})"
        if isinstance(op, ast.FloorDiv):
            return f"$.floordiv({ls}, {rs})"
        if isinstance(op, ast.Mod):
            if ln == "str":
                raise TranspileError("string % formatting")
            return f"$.pymod({ls}, {rs})"
        if isinstance(op, ast.Pow):
            return f"$.pow({ls}, {rs})"
        if isinstance(op, ast.BitAnd):
            return f"$.and({ls}, {rs})"
        if isinstance(op, ast.BitOr):
            return f"$.or({ls}, {rs})"
        if isinstance(op, ast.BitXor):
            return f"$.xor({ls}, {rs})"
        if isinstance(op, ast.LShift):
            return f"$.shl({ls}, {rs})"
        if isinstance(op, ast.RShift):
            return f"$.shr({ls}, {rs})"
        raise TranspileError("unsupported binary operator")

    def unary(self, node: ast.UnaryOp) -> str:
        if isinstance(node.op, ast.Not):
            opnd = node.operand
            inner = self._truthy(self.expr(opnd),
                                 getattr(opnd, "_vt", UNKNOWN))
            return f"(!{inner})"
        if isinstance(node.op, ast.USub):
            if (isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)):
                return f"(-({self.const(node.operand)}))"
            return f"(-{self.expr(node.operand)})"
        if isinstance(node.op, ast.UAdd):
            if (isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, int)):
                return self.const(node.operand)
            return f"(+{self.expr(node.operand)})"
        if isinstance(node.op, ast.Invert):
            return f"$.invert({self.expr(node.operand)})"
        raise TranspileError("unsupported unary operator")

    _TRUTHY_WRAP = ("list", "dict", "set", "tuple", "deque", "counter",
                    "range", "unknown")

    def _truthy(self, js: str, vt: VT) -> str:
        if vt.kind in self._TRUTHY_WRAP:
            return f"$.truthy({js})"
        return js

    def cond(self, node: ast.expr) -> str:
        if isinstance(node, ast.Compare):
            return self.compare(node)
        return self._truthy(self.expr(node),
                            getattr(node, "_vt", UNKNOWN))

    def boolop(self, node: ast.BoolOp) -> str:
        vals = [self.expr(v) for v in node.values]
        vts = [getattr(v, "_vt", UNKNOWN) for v in node.values]
        is_and = isinstance(node.op, ast.And)
        cur = self.hoisttmp()
        parts: list[str] = [f"{cur} = {vals[0]}"]
        for i in range(1, len(vals)):
            test = self._truthy(cur, vts[i - 1])
            branch = (f"({test} ? {vals[i]} : {cur})" if is_and
                      else f"(!({test}) ? {vals[i]} : {cur})")
            if i == len(vals) - 1:
                parts.append(branch)
            else:
                nxt = self.hoisttmp()
                parts.append(f"{nxt} = {branch}")
                cur = nxt
        return "(" + ", ".join(parts) + ")"

    # -- comparisons -------------------------------------------------------
    _CMP_OK_NUM = ("int", "float", "num", "bool")

    @staticmethod
    def _is_range_call(node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range")

    def compare(self, node: ast.Compare) -> str:
        ops = node.ops
        if len(ops) == 1 and isinstance(ops[0], (ast.In, ast.NotIn)):
            rn = node.comparators[0]
            if self._is_range_call(rn) and len(rn.args) in (1, 2, 3):
                args = [self.expr(a) for a in rn.args]
                if len(args) == 1:
                    args = ["0", args[0]]
                ljs = self.expr(node.left)
                res = f"$.inRange({ljs}, {', '.join(args)})"
                if isinstance(ops[0], ast.NotIn):
                    return f"(!{res})"
                return res
        lvt0 = getattr(node.left, "_vt", UNKNOWN)
        if len(ops) == 1:
            r0 = node.comparators[0]
            return self._cmp1(self.expr(node.left), ops[0],
                              self.expr(r0), lvt0,
                              getattr(r0, "_vt", UNKNOWN))
        # chained: evaluate each operand exactly once, python order
        lt = self.hoisttmp()
        rt = self.hoisttmp()
        r0 = node.comparators[0]
        s = (f"{lt} = {self.expr(node.left)}, {rt} = {self.expr(r0)}, "
             f"{self._cmp1(lt, ops[0], rt, lvt0, getattr(r0, '_vt', UNKNOWN))}")
        expr_s = f"({s})"
        for i in range(1, len(ops)):
            lvt = getattr(node.comparators[i - 1], "_vt", UNKNOWN)
            rvt = getattr(node.comparators[i], "_vt", UNKNOWN)
            c = self._cmp1(lt, ops[i], rt, lvt, rvt)
            body = (f"{lt} = {rt}, "
                    f"{rt} = {self.expr(node.comparators[i])}, {c}")
            expr_s = f"({expr_s} ? ({body}) : false)"
        return expr_s

    def _cmp1(self, ls: str, op: ast.cmpop, rs: str,
              lvt: VT, rvt: VT) -> str:
        if isinstance(op, ast.Eq):
            return f"$.eq({ls}, {rs})"
        if isinstance(op, ast.NotEq):
            return f"!$.eq({ls}, {rs})"
        if isinstance(op, ast.Is):
            return f"({ls} === {rs})"
        if isinstance(op, ast.IsNot):
            return f"({ls} !== {rs})"
        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                   ast.GtE: ">="}[type(op)]
            if (lvt.kind in self._CMP_OK_NUM
                    and rvt.kind in self._CMP_OK_NUM
                    and not ls.rstrip(")").endswith("n")
                    and not rs.rstrip(")").endswith("n")):
                return f"({ls} {sym} {rs})"
            if lvt.kind == "str" and rvt.kind == "str":
                return f"({ls} {sym} {rs})"
            return f"$.cmp({ls}, {rs}) {sym} 0"
        if isinstance(op, (ast.In, ast.NotIn)):
            res = f"$.contains({rs}, {ls})"
            if isinstance(op, ast.NotIn):
                return f"(!{res})"
            return res
        raise TranspileError("unsupported comparison")

    # -- calls -------------------------------------------------------------
    def call(self, node: ast.Call) -> str:
        args = [self.expr(a) for a in node.args]
        kwmark = self.kwobj(node.keywords) if node.keywords else None
        f = node.func
        if isinstance(f, ast.Name):
            fvt = getattr(f, "_vt", UNKNOWN)
            if (fvt.kind == "module" and fvt.mod
                    and fvt.mod.startswith("builtin.")):
                return self.builtin_call(node, args, kwmark)
            if kwmark is not None:
                raise TranspileError(f"keyword arguments to {f.id}")
            if fvt.kind == "module" and fvt.mod:
                parts = fvt.mod.split(".")
                if len(parts) == 2 and not self._ok_from(parts[0], parts[1]):
                    raise TranspileError(
                        f"symbol {fvt.mod} not transpilable")
            return f"{js_name(f.id)}({', '.join(args)})"
        if isinstance(f, ast.Attribute):
            ovt = getattr(f.value, "_vt", UNKNOWN)
            if ovt.kind == "module":
                self._check_module_attr(ovt, f.attr)
                recv = self.expr(f.value)
                arglist = ", ".join(args)
                if kwmark is not None:
                    if (ovt.mod or "").split(".")[0] != "re":
                        raise TranspileError(
                            f"keyword arguments to .{f.attr}")
                    arglist = (arglist + ", " + kwmark
                               if arglist else kwmark)
                return f"({recv}).{f.attr}({arglist})"
            if not self._method_ok(ovt, f.attr):
                raise TranspileError(
                    f"method .{f.attr} not transpilable for {ovt.kind}")
            recv = self.expr(f.value)
            arglist = ", ".join(args)
            if kwmark is not None:
                arglist = arglist + ", " + kwmark if arglist else kwmark
            return f"$.callm({recv}, {json.dumps(f.attr)}, [{arglist}])"
        return f"({self.expr(f)})({', '.join(args)})"

    def builtin_call(self, node: ast.Call, args: list[str],
                     kwmark: str | None) -> str:
        raw = node.func.id
        name = js_name(raw)
        if raw == "print":
            return self.print_call(node, args)
        if raw == "isinstance":
            if len(args) != 2:
                raise TranspileError("isinstance arity")
            hint = (", 1"
                    if getattr(node.args[0], "_vt", UNKNOWN).kind == "float"
                    else "")
            return (f"$.isinstance({args[0]}, "
                    f"{self._type_desc(node.args[1])}{hint})")
        if raw == "str" and node.args:
            if getattr(node.args[0], "_vt", UNKNOWN).kind == "float":
                return f"$.strf({args[0]})"
        if kwmark is not None:
            if raw not in ("sorted", "max", "min"):
                raise TranspileError(f"keyword arguments to {raw}")
            return f"{name}({', '.join(args)}, {kwmark})"
        return f"{name}({', '.join(args)})"

    def print_call(self, node: ast.Call, args: list[str]) -> str:
        mask = 0
        for i, a in enumerate(node.args):
            if i < 30 and getattr(a, "_vt", UNKNOWN).kind == "float":
                mask |= (1 << i)
        sep = "undefined"
        end = "undefined"
        for kw in node.keywords:
            if kw.arg == "sep":
                sep = self.expr(kw.value)
            elif kw.arg == "end":
                end = self.expr(kw.value)
            else:
                raise TranspileError(f"print keyword {kw.arg}")
        return f"$.pyprint([{', '.join(args)}], {mask}, {sep}, {end})"

    def _type_desc(self, t: ast.expr) -> str:
        if isinstance(t, ast.Tuple):
            return "[" + ", ".join(self._type_desc(e) for e in t.elts) + "]"
        if isinstance(t, ast.Name):
            return json.dumps(t.id)
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            return json.dumps(t.value.id)
        if (isinstance(t, ast.Call) and isinstance(t.func, ast.Name)
                and t.func.id == "type" and len(t.args) == 1
                and isinstance(t.args[0], ast.Constant)
                and t.args[0].value is None):
            return '"NoneType"'
        raise TranspileError("isinstance type operand")

    def _check_module_attr(self, ovt: VT, attr: str) -> None:
        path = ovt.mod or ""
        root = path.split(".")[0]
        if root == "builtin":
            rel = path[len("builtin."):]
            if attr not in _BUILTIN_STATIC.get(rel, ()):
                raise TranspileError(
                    f"builtin.{rel}.{attr} not transpilable")
            return
        if root == "numpy":
            return   # unreachable: the import itself raises ImportError
        if root not in KNOWN_MODULES:
            raise TranspileError(f"module {root} not transpilable")
        attrs = _MODULE_ATTRS.get(root, {})
        rel = attr if path == root else path[len(root) + 1:] + "." + attr
        if rel in attrs:
            return
        if any(k.startswith(rel + ".") for k in attrs):
            return
        if (root, attr) in _FROM_IMPORT_CALL:
            return
        raise TranspileError(f"{path}.{attr} not transpilable")

    def _method_ok(self, vt: VT, attr: str) -> bool:
        k = vt.kind
        if k == "str":
            return attr in _STR_METHODS
        if k in ("list", "tuple", "range"):
            return attr in _LIST_METHODS
        if k in ("dict", "counter"):
            return attr in _DICT_METHODS or attr in _MATCH_METHODS
        if k == "set":
            return attr in _SET_METHODS
        if k == "deque":
            return attr in _DEQUE_METHODS
        if k in ("int", "float", "num", "bool"):
            return attr in _NUM_METHODS
        return True

    # -- subscripts / attributes ------------------------------------------
    def subscript(self, node: ast.Subscript) -> str:
        cont = self.expr(node.value)
        cvt = getattr(node.value, "_vt", UNKNOWN)
        if isinstance(node.slice, ast.Slice):
            lo = _slice_part(node.slice, "lower")
            hi = _slice_part(node.slice, "upper")
            lo_js = self.expr(lo) if lo is not None else "undefined"
            hi_js = self.expr(hi) if hi is not None else "undefined"
            st = node.slice.step
            st_js = self.expr(st) if st is not None else "undefined"
            return f"$.slice({cont}, {lo_js}, {hi_js}, {st_js})"
        if (isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
                and node.slice.value >= 0
                and cvt.kind in ("list", "tuple", "str")):
            return f"({cont})[{node.slice.value}]"
        idx = self.expr(node.slice)
        if cvt.kind in ("dict", "counter"):
            return f"$.dget({cont}, {idx})"
        return f"$.at({cont}, {idx})"

    def attribute(self, node: ast.Attribute) -> str:
        ovt = getattr(node.value, "_vt", UNKNOWN)
        recv = self.expr(node.value)
        if ovt.kind == "module":
            self._check_module_attr(ovt, node.attr)
            return f"({recv}).{node.attr}"
        if ovt.kind in ("int", "float", "num", "bool"):
            if node.attr == "real":
                return f"({recv})"
            if node.attr == "imag":
                return "0"
            raise TranspileError(f"attribute .{node.attr}")
        return f"$.bound({recv}, {json.dumps(node.attr)})"

    # -- comprehensions ----------------------------------------------------
    def _comp_pattern(self, t: ast.expr) -> str:
        if isinstance(t, ast.Subscript):
            raise TranspileError("subscript comprehension target")
        return self.pattern(t)

    def comp(self, node) -> str:
        if isinstance(node, ast.DictComp):
            first = "const __py$r = $.dict();"
        elif isinstance(node, ast.SetComp):
            first = "const __py$r = $.fset();"
        else:
            first = "const __py$r = [];"
        lines = ["(() => {", "  " + first]
        ind = "  "
        for gen in node.generators:
            it = self.iter_expr(gen.iter)
            lines.append(f"{ind}for (let {self._comp_pattern(gen.target)} "
                         f"of {it}) {{")
            ind += "  "
            for cond_n in gen.ifs:
                lines.append(f"{ind}if ({self.cond(cond_n)}) {{")
                ind += "  "
        if isinstance(node, ast.DictComp):
            lines.append(f"{ind}__py$r.set({self.expr(node.key)}, "
                         f"{self.expr(node.value)});")
        elif isinstance(node, ast.SetComp):
            lines.append(f"{ind}__py$r.add({self.expr(node.elt)});")
        else:
            lines.append(f"{ind}__py$r.push({self.expr(node.elt)});")
        for gen in node.generators:
            for _ in gen.ifs:
                ind = ind[:-2]
                lines.append(ind + "}")
            ind = ind[:-2]
            lines.append(ind + "}")
        lines.append("  return __py$r;")
        lines.append("})()")
        return "\n".join(lines)

    # -- f-strings ---------------------------------------------------------
    def joined_str(self, node: ast.JoinedStr) -> str:
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(json.dumps(str(v.value)))
            elif isinstance(v, ast.FormattedValue):
                parts.append(self.fmt_part(v))
            else:
                raise TranspileError("f-string part")
        if not parts:
            return '""'
        return "(" + " + ".join(parts) + ")"

    def fmt_part(self, node: ast.FormattedValue) -> str:
        v = self.expr(node.value)
        is_float = (getattr(node.value, "_vt", UNKNOWN).kind == "float")
        conv = node.conversion
        if node.format_spec is not None:
            spec_js = self.expr(node.format_spec)
            c_js = "" if conv in (-1, None) else f", {conv}"
            f_js = ", 1" if is_float else ""
            return f"$.fmt({v}, {spec_js}{c_js}{f_js})"
        if conv in (114, 97):
            return f"$.repr({v})"
        if is_float:
            return f"$.strf({v})"
        return f"$.str({v})"


_AUG_OPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
}

# Method names each runtime receiver type supports (mirrors the prelude
# dispatch tables; used to reject untranspilable calls honestly).
_STR_METHODS = {
    "split", "splitlines", "strip", "lstrip", "rstrip", "replace",
    "format", "translate", "lower", "upper", "capitalize", "title",
    "zfill", "ljust", "rjust", "center", "encode", "decode",
    "expandtabs", "casefold", "removeprefix", "removesuffix", "swapcase",
    "find", "index", "rfind", "rindex", "count", "join", "startswith",
    "endswith", "isdigit", "isalpha", "isalnum", "isspace", "islower",
    "isupper", "istitle", "isdecimal", "isnumeric", "partition",
    "rpartition",
}
_LIST_METHODS = {
    "pop", "index", "count", "copy", "append", "extend", "insert",
    "remove", "sort", "reverse", "clear",
}
_DICT_METHODS = {
    "items", "keys", "values", "get", "pop", "setdefault", "update",
    "clear", "copy", "popitem", "most_common", "total", "elements",
}
_MATCH_METHODS = {
    "group", "groups", "groupdict", "start", "end", "span", "expand",
}
_SET_METHODS = {
    "pop", "add", "discard", "remove", "clear", "update", "copy",
    "union", "intersection", "difference", "symmetric_difference",
    "isdisjoint", "issubset", "issuperset",
}
_DEQUE_METHODS = {
    "popleft", "pop", "append", "appendleft", "extend", "extendleft",
    "rotate", "clear", "copy", "count", "index",
}
_NUM_METHODS = {
    "bit_length", "to_bytes", "is_integer", "as_integer_ratio",
}
_BUILTIN_STATIC = {
    "str": ("maketrans",),
    "int": ("from_bytes", "bit_length", "to_bytes"),
}


def emit_js(tree: ast.Module) -> str:
    return JSBackend(tree).emit()
