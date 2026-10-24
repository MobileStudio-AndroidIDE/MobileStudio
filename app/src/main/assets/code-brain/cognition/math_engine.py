"""Small symbolic math engine: parse, evaluate, differentiate, solve.

This is deliberately a small algebra system, not a computer algebra
package.  It supports numbers, variables, + - * / ^, parentheses and a
short list of unary functions.  Differentiation and simplification are
symbolic; every answer can be checked by evaluating it on sample points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_TOKEN_RE = __import__("re").compile(
    r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|"
    r"(\*\*|[+\-*/^(),]))")
_FUNCTIONS = {"sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs"}


@dataclass(frozen=True)
class MathNode:
    op: str
    args: tuple = ()
    value: object = None

    def variables(self) -> list[str]:
        if self.op == "var":
            return [str(self.value)]
        out: list[str] = []
        for arg in self.args:
            for name in arg.variables():
                if name not in out:
                    out.append(name)
        return out

    def to_string(self, parent_precedence: int = 0) -> str:
        if self.op == "num":
            return _number_text(self.value)
        if self.op == "var":
            return str(self.value)
        if self.op == "neg":
            text = "-" + self.args[0].to_string(3)
            return "(" + text + ")" if parent_precedence > 3 else text
        if self.op in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs"):
            name = "log" if self.op == "ln" else self.op
            return f"{name}({self.args[0].to_string()})"
        precedence = {"add": 1, "sub": 1, "mul": 2, "div": 2,
                      "pow": 4}[self.op]
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/",
                  "pow": "^"}[self.op]
        left = self.args[0].to_string(precedence)
        right = self.args[1].to_string(precedence + 1)
        text = f"{left}{symbol}{right}"
        return "(" + text + ")" if precedence < parent_precedence else text


class MathParseError(ValueError):
    pass


def parse_expression(text: str) -> MathNode:
    parser = _Parser(str(text or ""))
    node = parser.parse_expression()
    parser.expect_end()
    return node


class _Parser:
    def __init__(self, text: str):
        self.tokens = []
        position = 0
        while position < len(text):
            match = _TOKEN_RE.match(text, position)
            if not match:
                if text[position:].strip():
                    raise MathParseError(
                        "unexpected input at %d" % position)
                break
            position = match.end()
            number, name, symbol = match.groups()
            if number is not None:
                value = float(number) if "." in number else int(number)
                self.tokens.append(("num", value))
            elif name is not None:
                self.tokens.append(("name", name))
            elif symbol is not None:
                self.tokens.append(("sym", symbol))
        self.index = 0

    def peek(self):
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def take(self, kind: str = "", value=None):
        token = self.peek()
        if token is None:
            raise MathParseError("unexpected end of expression")
        if kind and token[0] != kind:
            raise MathParseError("expected %s, got %r" % (kind, token))
        if value is not None and token[1] != value:
            raise MathParseError("expected %r, got %r" % (value, token))
        self.index += 1
        return token

    def expect_end(self) -> None:
        if self.peek() is not None:
            raise MathParseError("unexpected token %r" % (self.peek(),))

    def parse_expression(self) -> MathNode:
        node = self.parse_term()
        while self.peek() in (("sym", "+"), ("sym", "-")):
            symbol = self.take()[1]
            right = self.parse_term()
            node = MathNode("add" if symbol == "+" else "sub",
                            (node, right))
        return node

    def parse_term(self) -> MathNode:
        node = self.parse_power()
        while self.peek() in (("sym", "*"), ("sym", "/")):
            symbol = self.take()[1]
            right = self.parse_power()
            node = MathNode("mul" if symbol == "*" else "div",
                            (node, right))
        return node

    def parse_power(self) -> MathNode:
        node = self.parse_unary()
        if self.peek() == ("sym", "^") or self.peek() == ("sym", "**"):
            self.take()
            exponent = self.parse_power()
            node = MathNode("pow", (node, exponent))
        return node

    def parse_unary(self) -> MathNode:
        if self.peek() == ("sym", "+"):
            self.take()
            return self.parse_unary()
        if self.peek() == ("sym", "-"):
            self.take()
            return MathNode("neg", (self.parse_unary(),))
        return self.parse_primary()

    def parse_primary(self) -> MathNode:
        token = self.peek()
        if token is None:
            raise MathParseError("unexpected end of expression")
        if token[0] == "num":
            self.take()
            return MathNode("num", (), token[1])
        if token[0] == "name":
            self.take()
            name = token[1].lower()
            if self.peek() == ("sym", "("):
                if name not in _FUNCTIONS:
                    raise MathParseError("unknown function " + name)
                self.take()
                argument = self.parse_expression()
                self.take("sym", ")")
                return MathNode("log" if name == "ln" else name,
                                (argument,))
            return MathNode("var", (), token[1])
        if token == ("sym", "("):
            self.take()
            node = self.parse_expression()
            self.take("sym", ")")
            return node
        raise MathParseError("unexpected token %r" % (token,))


def evaluate(node: MathNode, env: Optional[dict] = None) -> float:
    env = env or {}
    if node.op == "num":
        return float(node.value)
    if node.op == "var":
        if str(node.value) not in env:
            raise ValueError("unbound variable " + str(node.value))
        return float(env[str(node.value)])
    if node.op == "neg":
        return -evaluate(node.args[0], env)
    if node.op == "add":
        return evaluate(node.args[0], env) + evaluate(node.args[1], env)
    if node.op == "sub":
        return evaluate(node.args[0], env) - evaluate(node.args[1], env)
    if node.op == "mul":
        return evaluate(node.args[0], env) * evaluate(node.args[1], env)
    if node.op == "div":
        return evaluate(node.args[0], env) / evaluate(node.args[1], env)
    if node.op == "pow":
        return evaluate(node.args[0], env) ** evaluate(node.args[1], env)
    value = evaluate(node.args[0], env)
    if node.op == "sin":
        return math.sin(value)
    if node.op == "cos":
        return math.cos(value)
    if node.op == "tan":
        return math.tan(value)
    if node.op == "exp":
        return math.exp(value)
    if node.op == "log":
        return math.log(value)
    if node.op == "sqrt":
        return math.sqrt(value)
    if node.op == "abs":
        return abs(value)
    raise ValueError("unknown op " + node.op)


def derivative(node: MathNode, variable: str = "x") -> MathNode:
    variable = str(variable)
    if node.op == "num":
        return _num(0)
    if node.op == "var":
        return _num(1) if node.value == variable else _num(0)
    if node.op == "neg":
        return MathNode("neg", (derivative(node.args[0], variable),))
    if node.op in ("add", "sub"):
        return MathNode(node.op, (
            derivative(node.args[0], variable),
            derivative(node.args[1], variable)))
    if node.op == "mul":
        left, right = node.args
        return MathNode("add", (
            MathNode("mul", (derivative(left, variable), right)),
            MathNode("mul", (left, derivative(right, variable)))))
    if node.op == "div":
        left, right = node.args
        numerator = MathNode("sub", (
            MathNode("mul", (derivative(left, variable), right)),
            MathNode("mul", (left, derivative(right, variable)))))
        return MathNode("div", (
            numerator, MathNode("pow", (right, _num(2)))))
    if node.op == "pow":
        base, exponent = node.args
        if not exponent.variables() and exponent.op == "num":
            return MathNode("mul", (
                MathNode("mul", (exponent, MathNode(
                    "pow", (base, _num(float(exponent.value) - 1))))),
                derivative(base, variable)))
        if not base.variables() and base.op == "num":
            return MathNode("mul", (
                MathNode("mul", (node, _num(math.log(float(base.value))))),
                derivative(exponent, variable)))
        return MathNode("mul", (
            node, MathNode("add", (
                MathNode("mul", (derivative(exponent, variable),
                                 MathNode("log", (base,)))),
                MathNode("mul", (exponent, MathNode(
                    "div", (derivative(base, variable), base))))))))
    if node.op == "sin":
        return MathNode("mul", (MathNode("cos", (node.args[0],)),
                                derivative(node.args[0], variable)))
    if node.op == "cos":
        return MathNode("mul", (
            MathNode("neg", (MathNode("sin", (node.args[0],)),)),
            derivative(node.args[0], variable)))
    if node.op == "tan":
        cosine = MathNode("cos", (node.args[0],))
        return MathNode("div", (
            derivative(node.args[0], variable),
            MathNode("pow", (cosine, _num(2)))))
    if node.op == "exp":
        return MathNode("mul", (
            MathNode("exp", (node.args[0],)),
            derivative(node.args[0], variable)))
    if node.op == "log":
        return MathNode("div", (
            derivative(node.args[0], variable), node.args[0]))
    if node.op == "sqrt":
        return MathNode("div", (
            derivative(node.args[0], variable),
            MathNode("mul", (_num(2), MathNode("sqrt", (node.args[0],))))))
    if node.op == "abs":
        return derivative(node.args[0], variable)
    raise ValueError("unknown op " + node.op)


def simplify(node: MathNode) -> MathNode:
    if node.op in ("num", "var"):
        return node
    args = tuple(simplify(arg) for arg in node.args)
    node = MathNode(node.op, args, node.value)
    if node.op == "neg":
        child = node.args[0]
        if child.op == "num":
            return _num(-float(child.value))
        if child.op == "neg":
            return child.args[0]
        return node
    if node.op in ("add", "sub", "mul", "div", "pow"):
        left, right = node.args
        if left.op == "num" and right.op == "num":
            try:
                value = evaluate(node)
            except Exception:
                return node
            if not math.isfinite(value):
                return node
            if float(value).is_integer():
                value = int(value)
            return _num(value)
        if node.op == "add":
            terms = _flatten(node, "add")
            constant = 0.0
            rest = []
            for term in terms:
                if term.op == "num":
                    constant += float(term.value)
                else:
                    rest.append(term)
            if abs(constant) > 0.0:
                rest.append(_num(constant))
            if not rest:
                return _num(0)
            if len(rest) == 1:
                return rest[0]
            return _rebuild("add", rest)
        if node.op == "sub":
            if right.op == "num" and float(right.value) == 0:
                return left
            if left.op == "num" and float(left.value) == 0:
                return MathNode("neg", (right,))
            if left.to_string() == right.to_string():
                return _num(0)
        if node.op == "mul":
            factors = _flatten(node, "mul")
            constant = 1.0
            rest = []
            for factor in factors:
                if factor.op == "num":
                    constant *= float(factor.value)
                else:
                    rest.append(factor)
            if abs(constant) < 1e-15:
                return _num(0)
            if not rest:
                return _num(constant)
            if abs(constant - 1.0) > 1e-15 or constant < 0:
                rest.insert(0, _num(constant))
            if len(rest) == 1:
                return rest[0]
            return _rebuild("mul", rest)
        if node.op == "div":
            if right.op == "num" and float(right.value) == 1:
                return left
            if left.op == "num" and float(left.value) == 0:
                return _num(0)
        if node.op == "pow":
            if right.op == "num" and float(right.value) == 1:
                return left
            if right.op == "num" and float(right.value) == 0:
                return _num(1)
            if left.op == "num" and float(left.value) == 1:
                return _num(1)
        return node
    return MathNode(node.op, args, node.value)


def differentiate(text: str, variable: str = "x") -> str:
    tree = simplify(derivative(parse_expression(text), variable))
    return tree.to_string()


def solve_linear(text: str, variable: str = "x") -> Optional[float]:
    text = str(text or "")
    if "=" in text:
        left, right = text.split("=", 1)
        equation = MathNode("sub", (parse_expression(left),
                                    parse_expression(right)))
    else:
        equation = parse_expression(text)
    coefficients = _linear_coefficients(equation, variable)
    if coefficients is None:
        return None
    slope, intercept = coefficients
    if abs(slope) < 1e-12:
        return None
    return -intercept / slope


def _linear_coefficients(node: MathNode, variable: str
                         ) -> Optional[tuple[float, float]]:
    if node.op == "num":
        return (0.0, float(node.value))
    if node.op == "var":
        return (1.0, 0.0) if node.value == variable else (0.0, 0.0)
    if node.op == "neg":
        row = _linear_coefficients(node.args[0], variable)
        return None if row is None else (-row[0], -row[1])
    if node.op in ("add", "sub"):
        left = _linear_coefficients(node.args[0], variable)
        right = _linear_coefficients(node.args[1], variable)
        if left is None or right is None:
            return None
        sign = 1.0 if node.op == "add" else -1.0
        return (left[0] + sign * right[0], left[1] + sign * right[1])
    if node.op == "mul":
        left = _linear_coefficients(node.args[0], variable)
        right = _linear_coefficients(node.args[1], variable)
        if left is None or right is None:
            return None
        if abs(left[0]) > 1e-12 and abs(right[0]) > 1e-12:
            return None
        return (left[0] * right[1] + right[0] * left[1],
                left[1] * right[1])
    if node.op == "div":
        left = _linear_coefficients(node.args[0], variable)
        right = _linear_coefficients(node.args[1], variable)
        if left is None or right is None or abs(right[0]) > 1e-12:
            return None
        if abs(right[1]) < 1e-12:
            return None
        return (left[0] / right[1], left[1] / right[1])
    if node.op == "pow":
        base = _linear_coefficients(node.args[0], variable)
        exponent = _linear_coefficients(node.args[1], variable)
        if base is None or exponent is None:
            return None
        if abs(base[0]) > 1e-12:
            return None
        if abs(exponent[0]) < 1e-12:
            value = base[1] ** exponent[1]
            return (0.0, float(value))
        if base[1] == 1.0:
            return (0.0, 1.0)
        return None
    return None


def _num(value) -> MathNode:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return MathNode("num", (), value)


def _number_text(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return ("%.10g" % value)


def _flatten(node: MathNode, op: str) -> list[MathNode]:
    if node.op != op:
        return [node]
    out: list[MathNode] = []
    for argument in node.args:
        out.extend(_flatten(argument, op))
    return out


def _rebuild(op: str, terms: list[MathNode]) -> MathNode:
    node = terms[0]
    for term in terms[1:]:
        node = MathNode(op, (node, term))
    return node
