"""Exact math core: rationals, safe eval, symbolic lite, number theory, stats, units."""

from __future__ import annotations

import ast
import calendar
import datetime as _dt
import math
import operator
import random
import re
from fractions import Fraction

try:
    import numpy as _np
except Exception:
    _np = None

F = Fraction
ZERO = F(0)
ONE = F(1)


def parse_rational(x):
    if isinstance(x, bool):
        return F(int(x))
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return F(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise ValueError("non-finite number")
        return F(str(x))
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("empty number")
        try:
            return F(s)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"bad rational: {x!r}") from exc
    raise ValueError(f"bad rational: {x!r}")


def format_exact(x):
    v = parse_rational(x)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


_EXPR_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "log2": math.log2, "log10": math.log10,
    "sqrt": math.sqrt, "cbrt": math.cbrt, "abs": abs, "floor": math.floor,
    "ceil": math.ceil, "round": round, "min": min, "max": max, "pow": pow,
    "hypot": math.hypot, "atan2": math.atan2, "deg": math.degrees, "rad": math.radians,
}
_EXPR_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_EXPR_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_EXPR_NODE_TYPES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
    ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.UAdd, ast.USub,
)


def _validate_expr_tree(tree, scope=None):
    scope = scope or {}
    for node in ast.walk(tree):
        if not isinstance(node, _EXPR_NODE_TYPES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ValueError("private names not allowed")
            if node.id in scope:
                continue
            if node.id not in _EXPR_NAMES and node.id not in _EXPR_FUNCS:
                raise ValueError(f"unknown name: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _EXPR_FUNCS:
                raise ValueError("only whitelisted function calls allowed")
            if node.keywords:
                raise ValueError("keyword arguments not allowed")


def _eval_node(node, vars):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in vars:
            return vars[node.id]
        return _EXPR_NAMES[node.id]
    if isinstance(node, ast.BinOp):
        op = _EXPR_BINOPS[type(node.op)]
        left = _eval_node(node.left, vars)
        right = _eval_node(node.right, vars)
        if isinstance(node.op, ast.Pow) and isinstance(right, int) and abs(right) > 10000:
            raise ValueError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, vars)
        if isinstance(node.op, ast.USub):
            return -val
        return +val
    if isinstance(node, ast.Call):
        func = _EXPR_FUNCS[node.func.id]
        args = [_eval_node(a, vars) for a in node.args]
        if node.func.id == "pow" and len(args) == 2 and isinstance(args[1], int) and abs(args[1]) > 10000:
            raise ValueError("exponent too large")
        return func(*args)
    raise ValueError("unsupported expression")


def eval_expr(expr, vars=None):
    if not isinstance(expr, str):
        raise ValueError("expression must be a string")
    if len(expr) > 100000:
        raise ValueError("expression too long")
    scope = {} if vars is None else {str(k): v for k, v in dict(vars).items() if not str(k).startswith("_")}
    cleaned = expr.replace("^", "**")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ValueError("invalid expression") from exc
    _validate_expr_tree(tree, scope)
    try:
        result = _eval_node(tree.body, scope)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("non-finite result")
    return result


_SYM_FUNCS = {"sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs"}
_SYM_BINOPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.BitXor: "pow",
    ast.Pow: "pow",
}


def _parse_sym(node, src):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants allowed")
        seg = ast.get_source_segment(src, node)
        return ("num", parse_rational(seg if seg is not None else repr(node.value)))
    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise ValueError("private names not allowed")
        return ("var", node.id)
    if isinstance(node, ast.BinOp):
        tag = _SYM_BINOPS.get(type(node.op))
        if tag is None:
            raise ValueError("unsupported operator")
        return (tag, _parse_sym(node.left, src), _parse_sym(node.right, src))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return ("neg", _parse_sym(node.operand, src))
        if isinstance(node.op, ast.UAdd):
            return _parse_sym(node.operand, src)
        raise ValueError("unsupported unary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SYM_FUNCS:
            raise ValueError("unsupported function")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("functions take exactly one argument")
        name = "log" if node.func.id == "ln" else node.func.id
        return (name, _parse_sym(node.args[0], src))
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def parse_expr(expr):
    if isinstance(expr, tuple):
        return expr
    if not isinstance(expr, str):
        raise ValueError("expression must be a string")
    if len(expr) > 100000:
        raise ValueError("expression too long")
    cleaned = expr.replace("^", "**")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ValueError("invalid expression") from exc
    return _parse_sym(tree.body, cleaned)


_NUM0 = ("num", ZERO)
_NUM1 = ("num", ONE)


def _num(x):
    return ("num", F(x))


def _simplify_pow(a, b):
    if b[0] == "num":
        exp = b[1]
        if exp == 0:
            return _NUM1
        if exp == 1:
            return a
        if a[0] == "num":
            if exp.denominator == 1:
                e = int(exp)
                if abs(e) <= 100000:
                    if a[1] == 0 and e < 0:
                        raise ValueError("division by zero in expression")
                    return _num(a[1] ** e)
            if a[1] == 0:
                return _NUM0
    if a[0] == "num":
        if a[1] == 1:
            return _NUM1
        if a[1] == 0:
            return _NUM0
    return ("pow", a, b)


def _simplify_func(tag, a):
    if a[0] == "num":
        v = a[1]
        if tag == "sin" and v == 0:
            return _NUM0
        if tag == "cos" and v == 0:
            return _NUM1
        if tag == "tan" and v == 0:
            return _NUM0
        if tag == "exp" and v == 0:
            return _NUM1
        if tag == "log" and v == 1:
            return _NUM0
        if tag == "sqrt" and v >= 0:
            n = math.isqrt(v.numerator)
            d = math.isqrt(v.denominator)
            if n * n == v.numerator and d * d == v.denominator:
                return _num(F(n, d))
        if tag == "abs":
            return _num(abs(v))
    return (tag, a)


def _collect_mul(e):
    tag = e[0]
    if tag == "num":
        return e[1], {}
    if tag == "mul":
        c1, f1 = _collect_mul(e[1])
        c2, f2 = _collect_mul(e[2])
        for k, v in f2.items():
            f1[k] = f1.get(k, ZERO) + v
        return c1 * c2, f1
    if tag == "neg":
        c, f = _collect_mul(e[1])
        return -c, f
    if tag == "div":
        c1, f1 = _collect_mul(e[1])
        if e[2][0] == "num":
            if e[2][1] == 0:
                raise ValueError("division by zero in expression")
            return c1 / e[2][1], f1
        c2, f2 = _collect_mul(e[2])
        if c2 == 0:
            raise ValueError("division by zero in expression")
        for k, v in f2.items():
            f1[k] = f1.get(k, ZERO) - v
        return c1 / c2, f1
    if tag == "pow" and e[2][0] == "num":
        base, exp = e[1], e[2][1]
        if base[0] == "num" and exp.denominator == 1 and abs(int(exp)) <= 100000:
            if base[1] == 0 and exp < 0:
                raise ValueError("division by zero in expression")
            return base[1] ** int(exp), {}
        return ONE, {base: exp}
    return ONE, {e: ONE}


def _factor_sort_key(k):
    if k[0] == "var":
        return (0, k[1])
    if k[0] == "pow" and k[1][0] == "var":
        return (1, k[1][1], str(k[2]))
    return (2, str(k))


def _build_mul(coeff, facs):
    factors = []
    for k, exp in facs.items():
        if exp == 0:
            continue
        factors.append((k, exp))
    factors.sort(key=lambda kv: _factor_sort_key(kv[0]))
    if coeff == 0:
        return _NUM0
    if not factors:
        return _num(coeff)
    if all(exp < 0 for _, exp in factors):
        den_factors = []
        for k, exp in factors:
            den_factors.append(k if exp == -1 else ("pow", k, _num(-exp)))
        den = den_factors[0]
        for f in den_factors[1:]:
            den = ("mul", den, f)
        if coeff == 1:
            return ("div", _NUM1, den)
        if coeff == -1:
            return ("neg", ("div", _NUM1, den))
        return ("div", _num(coeff), den)
    nodes = [k if exp == 1 else ("pow", k, _num(exp)) for k, exp in factors]
    if coeff == -1:
        node = nodes[0]
        for f in nodes[1:]:
            node = ("mul", node, f)
        return ("neg", node)
    if coeff != 1:
        nodes.insert(0, _num(coeff))
    node = nodes[0]
    for f in nodes[1:]:
        node = ("mul", node, f)
    return node


def _collect_add(e):
    tag = e[0]
    if tag == "add":
        return _collect_add(e[1]) + _collect_add(e[2])
    if tag == "sub":
        return _collect_add(e[1]) + [(-c, k) for c, k in _collect_add(e[2])]
    if tag == "neg":
        return [(-c, k) for c, k in _collect_add(e[1])]
    if tag == "num":
        return [(e[1], None)]
    if tag == "mul":
        coeff, facs = _collect_mul(e)
        if not facs:
            return [(coeff, None)]
        return [(coeff, _build_mul(ONE, facs))]
    return [(ONE, e)]


def _scale(coeff, key):
    if key is None:
        return _num(coeff)
    if coeff == 1:
        return key
    if coeff == -1:
        return ("neg", key)
    return ("mul", _num(coeff), key)


def simplify(expr):
    e = parse_expr(expr)
    tag = e[0]
    if tag in ("num", "var"):
        return e
    if tag == "neg":
        a = simplify(e[1])
        if a[0] == "num":
            return _num(-a[1])
        if a[0] == "neg":
            return a[1]
        return ("neg", a)
    if tag in ("add", "sub"):
        left = simplify(e[1])
        right = simplify(e[2])
        terms = _collect_add(("sub", left, right) if tag == "sub" else ("add", left, right))
        const = ZERO
        groups = {}
        order = []
        for c, k in terms:
            if k is None:
                const += c
            else:
                if k not in groups:
                    groups[k] = ZERO
                    order.append(k)
                groups[k] += c
        parts = []
        for k in order:
            if groups[k] != 0:
                parts.append(_scale(groups[k], k))
        node = None
        for p in parts:
            node = p if node is None else ("add", node, p)
        if const != 0 or node is None:
            cnode = _num(const)
            node = cnode if node is None else ("add", node, cnode)
        return node
    if tag == "mul":
        a = simplify(e[1])
        b = simplify(e[2])
        coeff, facs = _collect_mul(("mul", a, b))
        return _build_mul(coeff, facs)
    if tag == "div":
        a = simplify(e[1])
        b = simplify(e[2])
        if b[0] == "num":
            if b[1] == 0:
                raise ValueError("division by zero in expression")
            if a[0] == "num":
                return _num(a[1] / b[1])
            if b[1] == 1:
                return a
        if a[0] == "num" and a[1] == 0:
            return _NUM0
        if a[0] != "num" and b[0] != "num":
            ca, fa = _collect_mul(a)
            cb, fb = _collect_mul(b)
            if cb == 0:
                raise ValueError("division by zero in expression")
            shared = [k for k in fb if k in fa]
            for k in shared:
                fa[k] = fa[k] - fb[k]
                if fa[k] == 0:
                    del fa[k]
                del fb[k]
            if not fb:
                return _build_mul(ca / cb, fa)
        return ("div", a, b)
    if tag == "pow":
        return _simplify_pow(simplify(e[1]), simplify(e[2]))
    if tag in _SYM_FUNCS:
        return _simplify_func(tag, simplify(e[1]))
    raise ValueError("unsupported expression")


def _str_node(e):
    tag = e[0]
    if tag == "num":
        return format_exact(e[1]), 4
    if tag == "var":
        return e[1], 4
    if tag == "neg":
        t, p = _str_node(e[1])
        if p < 2:
            t = f"({t})"
        return "-" + t, 2
    if tag == "add":
        ta, pa = _str_node(e[1])
        if pa < 1:
            ta = f"({ta})"
        b = e[2]
        if (b[0] == "num" and b[1] < 0) or b[0] == "neg":
            inner = b[1] if b[0] == "neg" else _num(-b[1])
            tb, pb = _str_node(inner)
            if pb < 2:
                tb = f"({tb})"
            return f"{ta} - {tb}", 1
        tb, pb = _str_node(b)
        if pb < 2:
            tb = f"({tb})"
        return f"{ta} + {tb}", 1
    if tag == "sub":
        ta, pa = _str_node(e[1])
        if pa < 1:
            ta = f"({ta})"
        tb, pb = _str_node(e[2])
        if pb < 2:
            tb = f"({tb})"
        return f"{ta} - {tb}", 1
    if tag == "mul":
        ta, pa = _str_node(e[1])
        if pa < 2:
            ta = f"({ta})"
        tb, pb = _str_node(e[2])
        if pb < 2:
            tb = f"({tb})"
        return f"{ta}*{tb}", 2
    if tag == "div":
        ta, pa = _str_node(e[1])
        if pa < 2:
            ta = f"({ta})"
        tb, pb = _str_node(e[2])
        if pb < 3:
            tb = f"({tb})"
        return f"{ta}/{tb}", 2
    if tag == "pow":
        ta, pa = _str_node(e[1])
        if pa < 3 or (e[1][0] == "num" and e[1][1] < 0):
            ta = f"({ta})"
        tb, pb = _str_node(e[2])
        if pb < 4:
            tb = f"({tb})"
        return f"{ta}^{tb}", 3
    if tag in _SYM_FUNCS:
        return f"{tag}({_str_node(e[1])[0]})", 4
    raise ValueError("unsupported expression")


def to_string(expr):
    return _str_node(parse_expr(expr) if not isinstance(expr, tuple) else expr)[0]


def _diff(e, var):
    tag = e[0]
    if tag == "num":
        return _NUM0
    if tag == "var":
        return _NUM1 if e[1] == var else _NUM0
    if tag == "add":
        return ("add", _diff(e[1], var), _diff(e[2], var))
    if tag == "sub":
        return ("sub", _diff(e[1], var), _diff(e[2], var))
    if tag == "neg":
        return ("neg", _diff(e[1], var))
    if tag == "mul":
        return ("add", ("mul", _diff(e[1], var), e[2]), ("mul", e[1], _diff(e[2], var)))
    if tag == "div":
        a, b = e[1], e[2]
        return ("div",
                ("sub", ("mul", _diff(a, var), b), ("mul", a, _diff(b, var))),
                ("pow", b, _num(2)))
    if tag == "pow":
        a, b = e[1], e[2]
        if b[0] == "num":
            n = b[1]
            return ("mul", ("mul", _num(n), ("pow", a, _num(n - 1))), _diff(a, var))
        da = _diff(a, var)
        db = _diff(b, var)
        inner = ("add", ("mul", db, ("log", a)), ("div", ("mul", b, da), a))
        return ("mul", ("pow", a, b), inner)
    if tag == "sin":
        return ("mul", ("cos", e[1]), _diff(e[1], var))
    if tag == "cos":
        return ("mul", ("neg", ("sin", e[1])), _diff(e[1], var))
    if tag == "tan":
        return ("mul", ("div", _NUM1, ("pow", ("cos", e[1]), _num(2))), _diff(e[1], var))
    if tag == "exp":
        return ("mul", ("exp", e[1]), _diff(e[1], var))
    if tag == "log":
        return ("mul", ("div", _NUM1, e[1]), _diff(e[1], var))
    if tag == "sqrt":
        return ("mul", ("div", _NUM1, ("mul", _num(2), ("sqrt", e[1]))), _diff(e[1], var))
    raise ValueError("cannot differentiate")


def differentiate(expr, var="x"):
    return simplify(_diff(parse_expr(expr), var))


def _poly_add(p, q, sign=1):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, ZERO) + sign * v
    return out


def _poly_mul(p, q):
    out = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            out[k1 + k2] = out.get(k1 + k2, ZERO) + v1 * v2
    return out


def _expand(e, var):
    tag = e[0]
    if tag == "num":
        return {0: e[1]}
    if tag == "var":
        return {1: ONE} if e[1] == var else None
    if tag == "neg":
        p = _expand(e[1], var)
        return None if p is None else {k: -v for k, v in p.items()}
    if tag == "add":
        p = _expand(e[1], var)
        q = _expand(e[2], var)
        return None if p is None or q is None else _poly_add(p, q)
    if tag == "sub":
        p = _expand(e[1], var)
        q = _expand(e[2], var)
        return None if p is None or q is None else _poly_add(p, q, -1)
    if tag == "mul":
        p = _expand(e[1], var)
        q = _expand(e[2], var)
        return None if p is None or q is None else _poly_mul(p, q)
    if tag == "div":
        p = _expand(e[1], var)
        q = _expand(e[2], var)
        if p is None or q is None or set(q) - {0} or q.get(0, ZERO) == 0:
            return None
        d = q[0]
        return {k: v / d for k, v in p.items()}
    if tag == "pow":
        p = _expand(e[1], var)
        b = e[2]
        if p is None or b[0] != "num" or b[1].denominator != 1:
            return None
        n = int(b[1])
        if n < 0 or n > 1000:
            return None
        out = {0: ONE}
        for _ in range(n):
            out = _poly_mul(out, p)
        return out
    return None


def poly_coefficients(expr, var="x"):
    p = _expand(parse_expr(expr), var)
    if p is None:
        raise ValueError("not a polynomial")
    deg = max(p)
    return [p.get(i, ZERO) for i in range(deg + 1)]


def poly_degree(expr, var="x"):
    return len(poly_coefficients(expr, var)) - 1


def solve_linear(a, b):
    a = parse_rational(a)
    b = parse_rational(b)
    if a == 0:
        raise ValueError("not a linear equation")
    return -b / a


def _exact_sqrt(q):
    n = math.isqrt(q.numerator)
    d = math.isqrt(q.denominator)
    if n * n == q.numerator and d * d == q.denominator:
        return F(n, d)
    raise ValueError("irrational root")


def solve_quadratic(a, b, c):
    a = parse_rational(a)
    b = parse_rational(b)
    c = parse_rational(c)
    if a == 0:
        return (solve_linear(b, c),)
    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("no real roots")
    s = _exact_sqrt(disc)
    r1 = (-b - s) / (2 * a)
    r2 = (-b + s) / (2 * a)
    return (r1, r2) if r1 <= r2 else (r2, r1)


def cramer_2x2(a, b, c, d, e, f):
    a, b, c, d, e, f = (parse_rational(x) for x in (a, b, c, d, e, f))
    det = a * d - b * c
    if det == 0:
        raise ValueError("singular system")
    return ((e * d - b * f) / det, (a * f - e * c) / det)


_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("is_prime expects an integer")
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("next_prime expects an integer")
    cand = n + 1
    if cand <= 2:
        return 2
    if cand % 2 == 0:
        cand += 1
    while not is_prime(cand):
        cand += 2
    return cand


def prime_sieve(limit):
    if not isinstance(limit, int) or limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i:limit + 1:i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i, flag in enumerate(sieve) if flag]


def _pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    rng = random.Random(0xC0FFEE)
    while True:
        c = rng.randrange(1, n)
        f = lambda x: (x * x + c) % n
        x = y = rng.randrange(2, n)
        d = 1
        while d == 1:
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factorize(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("factorize expects an integer")
    if n == 0:
        raise ValueError("cannot factorize zero")
    n = abs(n)
    factors = {}
    for p in prime_sieve(10000):
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors[m] = factors.get(m, 0) + 1
            continue
        d = _pollard_rho(m)
        stack.append(d)
        stack.append(m // d)
    return dict(sorted(factors.items()))


def gcd(*args):
    if not args:
        return 0
    result = 0
    for a in args:
        result = math.gcd(result, int(a))
    return result


def lcm(*args):
    if not args:
        return 1
    result = 1
    for a in args:
        a = int(a)
        if a == 0:
            return 0
        result = abs(result * a) // math.gcd(result, a) if result else abs(a)
    return abs(result)


def modpow(a, e, m):
    if m == 0:
        raise ValueError("modulus must be non-zero")
    return pow(int(a), int(e), int(m))


def euler_phi(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("euler_phi expects a positive integer")
    result = n
    for p in factorize(n):
        result -= result // p
    return result


def divisors(n):
    if not isinstance(n, int) or n == 0:
        raise ValueError("divisors expects a non-zero integer")
    n = abs(n)
    out = set()
    for i in range(1, math.isqrt(n) + 1):
        if n % i == 0:
            out.add(i)
            out.add(n // i)
    return sorted(out)


def digit_sum(n):
    if isinstance(n, bool) or not isinstance(n, (int, str)):
        raise ValueError("digit_sum expects an integer or digit string")
    return sum(int(ch) for ch in str(abs(int(n))))


def factorial(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("factorial expects a non-negative integer")
    return math.factorial(n)


def nCr(n, r):
    n = int(n)
    r = int(r)
    if r < 0 or n < 0 or r > n:
        return 0
    return math.comb(n, r)


def nPr(n, r):
    n = int(n)
    r = int(r)
    if r < 0 or n < 0 or r > n:
        return 0
    return math.perm(n, r)


def catalan(n):
    n = int(n)
    if n < 0:
        raise ValueError("catalan expects a non-negative integer")
    return math.comb(2 * n, n) // (n + 1)


def fib(n):
    n = int(n)
    neg = n < 0
    n = abs(n)

    def fd(k):
        if k == 0:
            return (0, 1)
        a, b = fd(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)

    value = fd(n)[0]
    if neg and n % 2 == 0:
        value = -value
    return value


def _exact_values(xs):
    return all(isinstance(x, (int, Fraction)) and not isinstance(x, bool) for x in xs)


def _sum_exact(xs):
    total = ZERO
    for x in xs:
        total += F(x) if not isinstance(x, float) else x
    return total


def mean(xs):
    xs = list(xs)
    if not xs:
        raise ValueError("empty data")
    if _exact_values(xs):
        return _sum_exact(xs) / len(xs)
    return sum(xs) / len(xs)


def median(xs):
    xs = sorted(xs)
    if not xs:
        raise ValueError("empty data")
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    if _exact_values(xs):
        return (F(xs[n // 2 - 1]) + F(xs[n // 2])) / 2
    return (xs[n // 2 - 1] + xs[n // 2]) / 2


def mode(xs):
    xs = list(xs)
    if not xs:
        raise ValueError("empty data")
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)


def variance(xs, sample=False):
    xs = list(xs)
    n = len(xs)
    if n == 0 or (sample and n == 1):
        raise ValueError("not enough data")
    m = mean(xs)
    total = ZERO
    for x in xs:
        total += (F(x) - m) ** 2 if not isinstance(x, float) and not isinstance(m, float) else (x - m) ** 2
    denom = n - 1 if sample else n
    return total / denom


def std(xs, sample=False):
    return math.sqrt(float(variance(xs, sample=sample)))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        raise ValueError("empty data")
    if q < 0 or q > 1:
        raise ValueError("q must be in [0, 1]")
    pos = F(q) * (len(xs) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(xs):
        return xs[-1]
    a = xs[lo]
    b = xs[lo + 1]
    if frac == 0:
        return a
    if _exact_values(xs):
        return F(a) + (F(b) - F(a)) * frac
    return a + (b - a) * float(frac)


def covariance(xs, ys, sample=False):
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys) or not xs:
        raise ValueError("data length mismatch")
    if sample and len(xs) < 2:
        raise ValueError("not enough data")
    mx = mean(xs)
    my = mean(ys)
    total = ZERO
    exact = _exact_values(xs) and _exact_values(ys)
    for x, y in zip(xs, ys):
        if exact:
            total += (F(x) - mx) * (F(y) - my)
        else:
            total += (x - mx) * (y - my)
    return total / (len(xs) - 1 if sample else len(xs))


def pearson(xs, ys):
    cv = float(covariance(xs, ys))
    sx = float(variance(xs))
    sy = float(variance(ys))
    if sx == 0 or sy == 0:
        raise ValueError("zero variance")
    return cv / math.sqrt(sx * sy)


def _ranks(xs):
    xs = list(xs)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def linear_regression(xs, ys):
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two paired points")
    vx = variance(xs)
    if vx == 0:
        raise ValueError("zero variance in x")
    slope = covariance(xs, ys) / vx
    intercept = mean(ys) - slope * mean(xs)
    return slope, intercept


def _mul_f(*xs):
    out = ONE
    for x in xs:
        out *= x
    return out


_LB = F("0.45359237")
_FT = F("0.3048")
_G0 = F("9.80665")
_LBF = _LB * _G0
_PSI = _LBF / (_FT ** 2)
_GAL_US = F("0.003785411784")

_UNITS = {
    "length": {
        "m": ONE, "km": F(1000), "cm": F(1, 100), "mm": F(1, 1000),
        "um": F(1, 10 ** 6), "nm": F(1, 10 ** 9), "mile": F("1609.344"),
        "yd": F("0.9144"), "ft": _FT, "in": F("0.0254"), "nmi": F(1852),
    },
    "mass": {
        "kg": ONE, "g": F(1, 1000), "mg": F(1, 10 ** 6), "t": F(1000),
        "lb": _LB, "oz": _LB / 16, "stone": _LB * 14,
    },
    "time": {
        "s": ONE, "ms": F(1, 1000), "us": F(1, 10 ** 6), "ns": F(1, 10 ** 9),
        "min": F(60), "h": F(3600), "day": F(86400), "week": F(604800),
        "fortnight": F(1209600), "year": F(31557600),
    },
    "area": {
        "m2": ONE, "km2": F(10 ** 6), "cm2": F(1, 10 ** 4), "mm2": F(1, 10 ** 6),
        "ha": F(10000), "acre": F("4046.8564224"), "ft2": _FT ** 2,
        "in2": F("0.0254") ** 2, "yd2": F("0.9144") ** 2, "mi2": F("1609.344") ** 2,
    },
    "volume": {
        "m3": ONE, "l": F(1, 1000), "ml": F(1, 10 ** 6), "cm3": F(1, 10 ** 6),
        "gal_us": _GAL_US, "gal_uk": F("0.00454609"),
        "floz_us": _GAL_US / 128, "cup_us": _GAL_US / 16, "pint_us": _GAL_US / 8,
        "quart_us": _GAL_US / 4, "tsp_us": _GAL_US / 768, "tbsp_us": _GAL_US / 256,
        "acre_ft": F("1233.48183754752"),
    },
    "speed": {
        "mps": ONE, "kmph": F(5, 18), "mph": F("0.44704"),
        "knot": F(463, 900), "fps": _FT,
    },
    "energy": {
        "J": ONE, "kJ": F(1000), "cal": F("4.184"), "kcal": F(4184),
        "Wh": F(3600), "kWh": F(3600000), "eV": F("1.602176634e-19"),
        "Btu": F("1055.05585262"), "ftlb": _mul_f(_FT, _LBF),
    },
    "power": {
        "W": ONE, "kW": F(1000), "MW": F(10 ** 6), "hp": _mul_f(F(550), _FT, _LBF),
    },
    "pressure": {
        "Pa": ONE, "kPa": F(1000), "MPa": F(10 ** 6), "bar": F(10 ** 5),
        "mbar": F(100), "atm": F(101325), "psi": _PSI, "torr": F(101325, 760),
        "mmHg": F("133.322387415"),
    },
    "force": {
        "N": ONE, "kN": F(1000), "dyn": F(1, 10 ** 5), "lbf": _LBF, "kgf": _G0,
    },
    "data": {
        "bit": F(1, 8), "byte": ONE, "KB": F(1000), "MB": F(10 ** 6),
        "GB": F(10 ** 9), "TB": F(10 ** 12), "PB": F(10 ** 15),
        "KiB": F(1024), "MiB": F(1024 ** 2), "GiB": F(1024 ** 3), "TiB": F(1024 ** 4),
        "kbit": F(125), "Mbit": F(125000), "Gbit": F(125 * 10 ** 6),
    },
}

_TEMP_UNITS = {"c", "f", "k", "r"}

_UNIT_ALIASES = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "um": "um", "micrometer": "um", "micron": "um", "microns": "um",
    "nm": "nm", "nanometer": "nm", "nanometers": "nm",
    "mile": "mile", "miles": "mile", "mi": "mile",
    "yd": "yd", "yard": "yd", "yards": "yd",
    "ft": "ft", "foot": "ft", "feet": "ft",
    "in": "in", "inch": "in", "inches": "in",
    "nmi": "nmi", "nauticalmile": "nmi",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "g": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "t": "t", "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "stone": "stone", "stones": "stone",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "us": "us", "microsecond": "us", "microseconds": "us",
    "ns": "ns", "nanosecond": "ns", "nanoseconds": "ns",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "day": "day", "days": "day", "week": "week", "weeks": "week",
    "fortnight": "fortnight", "fortnights": "fortnight",
    "year": "year", "years": "year", "yr": "year", "yrs": "year",
    "m2": "m2", "km2": "km2", "cm2": "cm2", "mm2": "mm2", "sqm": "m2",
    "ha": "ha", "hectare": "ha", "hectares": "ha",
    "acre": "acre", "acres": "acre",
    "ft2": "ft2", "in2": "in2", "yd2": "yd2", "mi2": "mi2",
    "m3": "m3", "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "cm3": "cm3", "cc": "cm3",
    "gal": "gal_us", "gallon": "gal_us", "gallons": "gal_us",
    "galus": "gal_us", "galuk": "gal_uk", "impgallon": "gal_uk",
    "floz": "floz_us", "flozus": "floz_us", "cup": "cup_us", "cups": "cup_us",
    "pint": "pint_us", "pints": "pint_us", "quart": "quart_us", "quarts": "quart_us",
    "tsp": "tsp_us", "tbsp": "tbsp_us", "acreft": "acre_ft",
    "mps": "mps", "kmph": "kmph", "kph": "kmph", "mph": "mph",
    "knot": "knot", "knots": "knot", "fps": "fps",
    "j": "J", "joule": "J", "joules": "J",
    "kj": "kJ", "kilojoule": "kJ", "kilojoules": "kJ",
    "cal": "cal", "calorie": "cal", "calories": "cal",
    "kcal": "kcal", "kilocalorie": "kcal", "kilocalories": "kcal",
    "wh": "Wh", "kilowatthour": "kWh", "kwh": "kWh",
    "ev": "eV", "btu": "Btu", "ftlb": "ftlb",
    "w": "W", "watt": "W", "watts": "W", "kw": "kW", "kilowatt": "kW", "kilowatts": "kW",
    "mw": "MW", "megawatt": "MW", "megawatts": "MW",
    "hp": "hp", "horsepower": "hp",
    "pa": "Pa", "pascal": "Pa", "pascals": "Pa",
    "kpa": "kPa", "mpa": "MPa", "bar": "bar", "bars": "bar", "mbar": "mbar",
    "atm": "atm", "atmosphere": "atm", "atmospheres": "atm",
    "psi": "psi", "torr": "torr", "mmhg": "mmHg",
    "n": "N", "newton": "N", "newtons": "N", "kn": "kN", "kilonewton": "kN",
    "dyn": "dyn", "dyne": "dyn", "dynes": "dyn",
    "lbf": "lbf", "kgf": "kgf", "kilogramforce": "kgf",
    "B": "byte", "byte": "byte", "bytes": "byte",
    "b": "bit", "bit": "bit", "bits": "bit",
    "KB": "KB", "kb": "KB", "kilobyte": "KB", "kilobytes": "KB",
    "MB": "MB", "mb": "MB", "megabyte": "MB", "megabytes": "MB",
    "GB": "GB", "gb": "GB", "gigabyte": "GB", "gigabytes": "GB",
    "TB": "TB", "tb": "TB", "terabyte": "TB", "terabytes": "TB",
    "PB": "PB", "pb": "PB", "petabyte": "PB", "petabytes": "PB",
    "KiB": "KiB", "kib": "KiB", "kibibyte": "KiB", "kibibytes": "KiB",
    "MiB": "MiB", "mib": "MiB", "mebibyte": "MiB", "mebibytes": "MiB",
    "GiB": "GiB", "gib": "GiB", "gibibyte": "GiB", "gibibytes": "GiB",
    "TiB": "TiB", "tib": "TiB", "tebibyte": "TiB", "tebibytes": "TiB",
    "kbit": "kbit", "kilobit": "kbit", "kilobits": "kbit",
    "Mbit": "Mbit", "mbit": "Mbit", "megabit": "Mbit", "megabits": "Mbit",
    "Gbit": "Gbit", "gbit": "Gbit", "gigabit": "Gbit", "gigabits": "Gbit",
    "c": "c", "degc": "c", "celsius": "c", "centigrade": "c", "ccelsius": "c",
    "f": "f", "degf": "f", "fahrenheit": "f",
    "k": "k", "kelvin": "k", "kelvins": "k",
    "r": "r", "rankine": "r",
    "deg": "deg", "degree": "deg", "degrees": "deg",
    "rad": "rad", "radian": "rad", "radians": "rad",
    "grad": "grad", "gradian": "grad", "gradians": "grad", "gon": "gon",
    "turn": "turn", "turns": "turn", "rev": "turn", "revolution": "turn",
    "arcmin": "arcmin", "arcminute": "arcmin", "arcminates": "arcmin",
    "arcsec": "arcsec", "arcsecond": "arcsec", "arcseconds": "arcsec",
}

for _cat in _UNITS.values():
    for _k in _cat:
        _UNIT_ALIASES.setdefault(_k, _k)


def _norm_unit(unit):
    if not isinstance(unit, str):
        raise ValueError("unit must be a string")
    s = unit.strip()
    if not s:
        raise ValueError("empty unit")
    if s in _UNIT_ALIASES:
        return _UNIT_ALIASES[s]
    low = s.lower()
    if low in _UNIT_ALIASES:
        return _UNIT_ALIASES[low]
    if low.endswith("s") and low[:-1] in _UNIT_ALIASES:
        return _UNIT_ALIASES[low[:-1]]
    low2 = low.replace("°", "").strip()
    if low2 in _UNIT_ALIASES:
        return _UNIT_ALIASES[low2]
    raise ValueError(f"unknown unit: {unit}")


def _unit_category(u):
    if u in _TEMP_UNITS:
        return "temperature"
    for cat, table in _UNITS.items():
        if u in table:
            return cat
    if u in ("deg", "rad", "grad", "gon", "turn", "arcmin", "arcsec"):
        return "angle"
    raise ValueError(f"unknown unit: {u}")


_ANGLE_DEG = {
    "deg": ONE, "grad": F(9, 10), "gon": F(9, 10), "turn": F(360),
    "arcmin": F(1, 60), "arcsec": F(1, 3600),
}


def _convert_temp(value, f, t):
    exact = not isinstance(value, float)
    v = F(value) if exact else value
    if f == "c":
        c = v
    elif f == "f":
        c = (v - 32) * F(5, 9)
    elif f == "k":
        c = v - F(27315, 100)
    else:
        c = (v - F(49167, 100)) * F(5, 9)
    if t == "c":
        out = c
    elif t == "f":
        out = c * F(9, 5) + 32
    elif t == "k":
        out = c + F(27315, 100)
    else:
        out = c * F(9, 5) + F(49167, 100)
    return out if exact else float(out)


def _convert_angle(value, f, t):
    if f == t:
        return value
    v = float(value)
    if f == "rad":
        deg = math.degrees(v)
    else:
        deg = v * float(_ANGLE_DEG[f])
    if t == "rad":
        return math.radians(deg)
    return deg / float(_ANGLE_DEG[t])


def convert(value, from_unit, to_unit):
    f = _norm_unit(from_unit)
    t = _norm_unit(to_unit)
    exact = not isinstance(value, float)
    if isinstance(value, str):
        value = parse_rational(value)
    cf = _unit_category(f)
    ct = _unit_category(t)
    if cf != ct:
        raise ValueError(f"incompatible units: {from_unit} -> {to_unit}")
    if cf == "temperature":
        return _convert_temp(value, f, t)
    if cf == "angle":
        return _convert_angle(value, f, t)
    ratio = _UNITS[cf][f] / _UNITS[cf][t]
    out = F(value) * ratio if exact else value * float(ratio)
    return out


_CONVERT_RE = re.compile(
    r"^\s*(?:convert\s+)?([-+]?(?:\d+/\d+|\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"([A-Za-z\u00b0][A-Za-z0-9\u00b0/^\-]*)\s*(?:to|in|->|\u2192)\s*"
    r"([A-Za-z\u00b0][A-Za-z0-9\u00b0/^\-]*)\s*$",
    re.IGNORECASE,
)


def parse_convert(text):
    if not isinstance(text, str):
        raise ValueError("conversion must be a string")
    m = _CONVERT_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse conversion: {text!r}")
    value = parse_rational(m.group(1))
    return convert(value, m.group(2), m.group(3)), _norm_unit(m.group(3))


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def int_to_base(n, base):
    if not 2 <= int(base) <= 36 or isinstance(base, bool):
        raise ValueError("base must be in 2..36")
    base = int(base)
    n = int(n)
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(_DIGITS[r])
    return sign + "".join(reversed(out))


def from_base(text, base):
    if not 2 <= int(base) <= 36 or isinstance(base, bool):
        raise ValueError("base must be in 2..36")
    base = int(base)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("invalid digit string")
    s = text.strip().lower()
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    if not s:
        raise ValueError("invalid digit string")
    value = 0
    for ch in s:
        d = _DIGITS.find(ch)
        if d < 0 or d >= base:
            raise ValueError(f"invalid digit for base {base}: {ch!r}")
        value = value * base + d
    return sign * value


_ROMAN_TABLE = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
    (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
    (5, "V"), (4, "IV"), (1, "I"),
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def to_roman(n):
    n = int(n)
    if not 1 <= n <= 3999:
        raise ValueError("roman numerals support 1..3999")
    out = []
    for value, sym in _ROMAN_TABLE:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def from_roman(text):
    if not isinstance(text, str):
        raise ValueError("roman numeral must be a string")
    s = text.strip().upper()
    if not s:
        raise ValueError("empty roman numeral")
    total = 0
    prev = 0
    for ch in reversed(s):
        if ch not in _ROMAN_VALUES:
            raise ValueError(f"invalid roman digit: {ch!r}")
        v = _ROMAN_VALUES[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    if to_roman(total) != s:
        raise ValueError(f"not a canonical roman numeral: {text!r}")
    return total


def percent_of(a, b=None):
    if b is None:
        if not isinstance(a, str):
            raise ValueError("percent_of needs 'p% of x' or two numbers")
        m = re.match(
            r"^\s*([-+]?(?:\d+/\d+|\d+\.?\d*|\.\d+))\s*%\s*of\s*"
            r"([-+]?(?:\d+/\d+|\d+\.?\d*|\.\d+))\s*$",
            a, re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"cannot parse percent phrase: {a!r}")
        a, b = m.group(1), m.group(2)
    pa, pb = parse_rational(a), parse_rational(b)
    return pa / 100 * pb


def what_percent(x, y=None):
    if y is None:
        if not isinstance(x, str):
            raise ValueError("what_percent needs 'x is what percent of y' or two numbers")
        m = re.match(
            r"^\s*([-+]?(?:\d+/\d+|\d+\.?\d*|\.\d+))\s+is\s+what\s+percent\s+of\s+"
            r"([-+]?(?:\d+/\d+|\d+\.?\d*|\.\d+))\s*\??\s*$",
            x, re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"cannot parse percent phrase: {x!r}")
        x, y = m.group(1), m.group(2)
    px, py = parse_rational(x), parse_rational(y)
    if py == 0:
        raise ValueError("division by zero")
    return px / py * 100


def percent_change(old, new):
    po, pn = parse_rational(old), parse_rational(new)
    if po == 0:
        raise ValueError("division by zero")
    return (pn - po) / po * 100


def simple_interest(principal, rate, years):
    p, r, t = parse_rational(principal), parse_rational(rate), parse_rational(years)
    exact = not any(isinstance(v, float) for v in (principal, rate, years))
    out = p * r * t
    return out if exact else float(out)


def compound_interest(principal, rate, years, n=1):
    p, r, t = parse_rational(principal), parse_rational(rate), parse_rational(years)
    ni = int(n)
    if ni <= 0:
        raise ValueError("n must be positive")
    periods = t * ni
    exact = not any(isinstance(v, float) for v in (principal, rate, years))
    if exact and periods.denominator == 1:
        return p * (1 + r / ni) ** int(periods)
    return float(p) * (1 + float(r) / ni) ** (float(t) * ni)


def _as_date(d):
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, _dt.date):
        return d
    if isinstance(d, str):
        try:
            return _dt.datetime.fromisoformat(d).date()
        except ValueError as exc:
            raise ValueError(f"bad date: {d!r}") from exc
    raise ValueError("expected date, datetime or ISO string")


def days_between(a, b):
    da, db = _as_date(a), _as_date(b)
    return (db - da).days


def add_months(d, n):
    if isinstance(d, str):
        d = _dt.datetime.fromisoformat(d)
        if d.hour == 0 and d.minute == 0 and d.second == 0 and "T" not in d.isoformat():
            pass
    if not isinstance(d, (_dt.date, _dt.datetime)):
        raise ValueError("expected date or datetime")
    total = d.year * 12 + (d.month - 1) + int(n)
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def _as_matrix(m):
    if _np is None:
        raise ValueError("numpy is not available")
    try:
        return _np.asarray(m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bad matrix") from exc


def matrix_det(m):
    a = _as_matrix(m)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    return float(_np.linalg.det(a))


def matrix_inverse(m):
    a = _as_matrix(m)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    try:
        return _np.linalg.inv(a)
    except _np.linalg.LinAlgError as exc:
        raise ValueError("singular matrix") from exc


def matrix_solve(a, b):
    am = _as_matrix(a)
    bv = _as_matrix(b)
    if am.ndim != 2 or am.shape[0] != am.shape[1]:
        raise ValueError("matrix must be square")
    try:
        return _np.linalg.solve(am, bv)
    except _np.linalg.LinAlgError as exc:
        raise ValueError("singular matrix") from exc


def to_polar(z):
    z = complex(z)
    return abs(z), math.atan2(z.imag, z.real)


def from_polar(r, theta):
    r = float(r)
    theta = float(theta)
    return complex(r * math.cos(theta), r * math.sin(theta))


def roots_of_unity(n):
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    return [complex(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]







