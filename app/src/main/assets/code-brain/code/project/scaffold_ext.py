"""Scale project scaffold generator (additive extension).

Generates complete, runnable multi-module projects from a declarative
spec.  The generator is deterministic (same spec -> byte-identical
output), dependency-free (stdlib only), and refuses to emit broken code:
unknown body kinds and invalid functions are skipped and reported in the
returned manifest instead of being rendered.

    spec = {
        "name": "demo",
        "language": "python",
        "modules": [
            {"name": "math_basics", "doc": "arithmetic helpers",
             "functions": [
                 {"name": "add", "body": {"kind": "add"}},
             ],
             "classes": [
                 {"name": "Stats", "doc": "list statistics",
                  "methods": [{"name": "mean", "target": "mean"}],
                  "sample": "[1, 2, 3]"},
             ]},
        ],
        "entry": "main.py",
        "readme": True,
        "tests": True,
    }
    manifest = generate_project(spec, out_dir)
"""
from __future__ import annotations

import hashlib
import json
import os
import re

SCAFFOLD_VERSION = "1.0"


class ScaffoldError(ValueError):
    """Raised when the spec is structurally unusable."""


def _kind(name, params, py_return, ts_return, summary, py_body, ts_body,
          sample, expected, imports=()):
    return {
        "name": name,
        "params": tuple(params),
        "py_return": py_return,
        "ts_return": ts_return,
        "summary": summary,
        "py_body": tuple(py_body),
        "ts_body": tuple(ts_body),
        "sample_args": tuple(sample),
        "sample_expected": expected,
        "imports": tuple(imports),
    }


KINDS = {}


def _register(entries):
    for entry in entries:
        KINDS[entry["name"]] = entry


_register([
    _kind(
        "add",
        (("a", "float", "number"), ("b", "float", "number")),
        "float", "number", "Add two numbers",
        ("return @0@ + @1@",),
        ("return @0@ + @1@;",),
        ("2", "3"), "5",
    ),
    _kind(
        "subtract",
        (("a", "float", "number"), ("b", "float", "number")),
        "float", "number", "Subtract the second number from the first",
        ("return @0@ - @1@",),
        ("return @0@ - @1@;",),
        ("5", "2"), "3",
    ),
    _kind(
        "mul",
        (("a", "float", "number"), ("b", "float", "number")),
        "float", "number", "Multiply two numbers",
        ("return @0@ * @1@",),
        ("return @0@ * @1@;",),
        ("4", "3"), "12",
    ),
    _kind(
        "div_safe",
        (("a", "float", "number"), ("b", "float", "number")),
        "float", "number", "Divide two numbers, returning 0.0 on zero divisor",
        ("if @1@ == 0:",
         "    return 0.0",
         "return @0@ / @1@"),
        ("if (@1@ === 0) {",
         "  return 0.0;",
         "}",
         "return @0@ / @1@;"),
        ("6", "2"), "3.0",
    ),
    _kind(
        "mean",
        (("values", "list[float]", "number[]"),),
        "float", "number", "Compute the arithmetic mean of a list",
        ("if not @0@:",
         "    return 0.0",
         "return sum(@0@) / len(@0@)"),
        ("if (@0@.length === 0) {",
         "  return 0.0;",
         "}",
         "const total = @0@.reduce((acc, value) => acc + value, 0);",
         "return total / @0@.length;"),
        ("[1, 2, 3]",), "2.0",
    ),
    _kind(
        "median",
        (("values", "list[float]", "number[]"),),
        "float", "number", "Compute the median of a list",
        ("if not @0@:",
         "    return 0.0",
         "ordered = sorted(@0@)",
         "count = len(ordered)",
         "middle = count // 2",
         "if count % 2 == 1:",
         "    return float(ordered[middle])",
         "return (ordered[middle - 1] + ordered[middle]) / 2.0"),
        ("if (@0@.length === 0) {",
         "  return 0.0;",
         "}",
         "const ordered = [...@0@].sort((left, right) => left - right);",
         "const middle = Math.floor(ordered.length / 2);",
         "if (ordered.length % 2 === 1) {",
         "  return ordered[middle];",
         "}",
         "return (ordered[middle - 1] + ordered[middle]) / 2.0;"),
        ("[3, 1, 2]",), "2.0",
    ),
    _kind(
        "filter_even",
        (("values", "list[int]", "number[]"),),
        "list[int]", "number[]", "Keep only the even numbers",
        ("return [value for value in @0@ if value % 2 == 0]",),
        ("return @0@.filter((value) => value % 2 === 0);",),
        ("[1, 2, 3, 4]",), "[2, 4]",
    ),
    _kind(
        "filter_gt",
        (("values", "list[float]", "number[]"),
         ("threshold", "float", "number")),
        "list[float]", "number[]", "Keep values strictly greater than a threshold",
        ("return [value for value in @0@ if value > @1@]",),
        ("return @0@.filter((value) => value > @1@);",),
        ("[1, 5, 3]", "2"), "[5, 3]",
    ),
    _kind(
        "sort_desc",
        (("values", "list[float]", "number[]"),),
        "list[float]", "number[]", "Sort values in descending order",
        ("return sorted(@0@, reverse=True)",),
        ("return [...@0@].sort((left, right) => right - left);",),
        ("[1, 3, 2]",), "[3, 2, 1]",
    ),
    _kind(
        "unique_sorted",
        (("values", "list[float]", "number[]"),),
        "list[float]", "number[]", "Remove duplicates and sort ascending",
        ("return sorted(set(@0@))",),
        ("return Array.from(new Set(@0@)).sort((left, right) => left - right);",),
        ("[3, 1, 3, 2]",), "[1, 2, 3]",
    ),
    _kind(
        "word_count",
        (("text", "str", "string"),),
        "dict[str, int]", "Record<string, number>",
        "Count word frequencies in a text",
        ("counts = {}",
         "for word in @0@.split():",
         "    key = word.lower()",
         "    counts[key] = counts.get(key, 0) + 1",
         "return counts"),
        ("const counts: Record<string, number> = {};",
         "for (const word of @0@.split(/\\s+/)) {",
         "  if (!word) {",
         "    continue;",
         "  }",
         "  const key = word.toLowerCase();",
         "  counts[key] = (counts[key] ?? 0) + 1;",
         "}",
         "return counts;"),
        ('"a b a"',), '{"a": 2, "b": 1}',
    ),
    _kind(
        "char_freq",
        (("text", "str", "string"),),
        "dict[str, int]", "Record<string, number>",
        "Count character frequencies in a text",
        ("counts = {}",
         "for char in @0@:",
         "    if char.strip():",
         "        counts[char] = counts.get(char, 0) + 1",
         "return counts"),
        ("const counts: Record<string, number> = {};",
         "for (const char of @0@) {",
         "  if (!char.trim()) {",
         "    continue;",
         "  }",
         "  counts[char] = (counts[char] ?? 0) + 1;",
         "}",
         "return counts;"),
        ('"aba"',), '{"a": 2, "b": 1}',
    ),
    _kind(
        "running_max",
        (("values", "list[float]", "number[]"),),
        "list[float]", "number[]", "Track the running maximum of a list",
        ("if not @0@:",
         "    return []",
         "out = [@0@[0]]",
         "for value in @0@[1:]:",
         "    out.append(value if value > out[-1] else out[-1])",
         "return out"),
        ("if (@0@.length === 0) {",
         "  return [];",
         "}",
         "const out: number[] = [@0@[0]];",
         "for (const value of @0@.slice(1)) {",
         "  out.push(value > out[out.length - 1] ? value "
         ": out[out.length - 1]);",
         "}",
         "return out;"),
        ("[1, 3, 2]",), "[1, 3, 3]",
    ),
    _kind(
        "running_sum",
        (("values", "list[float]", "number[]"),),
        "list[float]", "number[]", "Track the running sum of a list",
        ("out = []",
         "total = 0",
         "for value in @0@:",
         "    total += value",
         "    out.append(total)",
         "return out"),
        ("const out: number[] = [];",
         "let total = 0;",
         "for (const value of @0@) {",
         "  total += value;",
         "  out.push(total);",
         "}",
         "return out;"),
        ("[1, 2, 3]",), "[1, 3, 6]",
    ),
    _kind(
        "min_max",
        (("values", "list[float]", "number[]"),),
        "tuple[float, float]", "[number, number]",
        "Return the minimum and maximum of a list",
        ("if not @0@:",
         "    return (0.0, 0.0)",
         "return (float(min(@0@)), float(max(@0@)))"),
        ("if (@0@.length === 0) {",
         "  return [0.0, 0.0];",
         "}",
         "return [Math.min(...@0@), Math.max(...@0@)];"),
        ("[2, 9, -1]",), "(-1.0, 9.0)",
    ),
    _kind(
        "clamp",
        (("value", "float", "number"), ("low", "float", "number"),
         ("high", "float", "number")),
        "float", "number", "Clamp a value into an inclusive range",
        ("lower, upper = @1@, @2@",
         "if lower > upper:",
         "    lower, upper = upper, lower",
         "return max(lower, min(@0@, upper))"),
        ("const lower = Math.min(@1@, @2@);",
         "const upper = Math.max(@1@, @2@);",
         "return Math.max(lower, Math.min(@0@, upper));"),
        ("5", "0", "3"), "3",
    ),
    _kind(
        "lerp",
        (("a", "float", "number"), ("b", "float", "number"),
         ("t", "float", "number")),
        "float", "number", "Linearly interpolate between two values",
        ("return @0@ + (@1@ - @0@) * @2@",),
        ("return @0@ + (@1@ - @0@) * @2@;",),
        ("0", "10", "0.5"), "5.0",
    ),
    _kind(
        "gradient",
        (("values", "list[float]", "number[]"),),
        "list[float]", "number[]", "Compute consecutive differences",
        ("if len(@0@) < 2:",
         "    return []",
         "return [@0@[index + 1] - @0@[index] "
         "for index in range(len(@0@) - 1)]"),
        ("if (@0@.length < 2) {",
         "  return [];",
         "}",
         "const out: number[] = [];",
         "for (let index = 0; index + 1 < @0@.length; index += 1) {",
         "  out.push(@0@[index + 1] - @0@[index]);",
         "}",
         "return out;"),
        ("[1, 3, 6]",), "[2, 3]",
    ),
    _kind(
        "dot",
        (("left", "list[float]", "number[]"),
         ("right", "list[float]", "number[]")),
        "float", "number", "Compute the dot product of two vectors",
        ("if len(@0@) != len(@1@):",
         '    raise ValueError("dot: vectors must have equal length")',
         "return sum(x * y for x, y in zip(@0@, @1@))"),
        ("if (@0@.length !== @1@.length) {",
         '  throw new Error("dot: vectors must have equal length");',
         "}",
         "let total = 0;",
         "for (let index = 0; index < @0@.length; index += 1) {",
         "  total += @0@[index] * @1@[index];",
         "}",
         "return total;"),
        ("[1, 2]", "[3, 4]"), "11",
    ),
    _kind(
        "norm",
        (("values", "list[float]", "number[]"),),
        "float", "number", "Compute the Euclidean norm of a vector",
        ("return math.sqrt(sum(value * value for value in @0@))",),
        ("return Math.sqrt(@0@.reduce((acc, value) => "
         "acc + value * value, 0));",),
        ("[3, 4]",), "5.0",
        imports=("math",),
    ),
    _kind(
        "is_prime",
        (("number", "int", "number"),),
        "bool", "boolean", "Check whether an integer is prime",
        ("if @0@ < 2:",
         "    return False",
         "if @0@ < 4:",
         "    return True",
         "if @0@ % 2 == 0:",
         "    return False",
         "divisor = 3",
         "while divisor * divisor <= @0@:",
         "    if @0@ % divisor == 0:",
         "        return False",
         "    divisor += 2",
         "return True"),
        ("if (@0@ < 2) {",
         "  return false;",
         "}",
         "if (@0@ < 4) {",
         "  return true;",
         "}",
         "if (@0@ % 2 === 0) {",
         "  return false;",
         "}",
         "for (let divisor = 3; divisor * divisor <= @0@; "
         "divisor += 2) {",
         "  if (@0@ % divisor === 0) {",
         "    return false;",
         "  }",
         "}",
         "return true;"),
        ("7",), "True",
    ),
    _kind(
        "fib",
        (("number", "int", "number"),),
        "int", "number", "Return the n-th Fibonacci number",
        ("if @0@ <= 0:",
         "    return 0",
         "previous, current = 0, 1",
         "for _ in range(@0@):",
         "    previous, current = current, previous + current",
         "return previous"),
        ("if (@0@ <= 0) {",
         "  return 0;",
         "}",
         "let previous = 0;",
         "let current = 1;",
         "for (let index = 0; index < @0@; index += 1) {",
         "  const next = previous + current;",
         "  previous = current;",
         "  current = next;",
         "}",
         "return previous;"),
        ("10",), "55",
    ),
    _kind(
        "fact",
        (("number", "int", "number"),),
        "int", "number", "Return the factorial of a non-negative integer",
        ("if @0@ < 0:",
         '    raise ValueError("fact: input must be non-negative")',
         "result = 1",
         "for factor in range(2, @0@ + 1):",
         "    result *= factor",
         "return result"),
        ("if (@0@ < 0) {",
         '  throw new Error("fact: input must be non-negative");',
         "}",
         "let result = 1;",
         "for (let factor = 2; factor <= @0@; factor += 1) {",
         "  result *= factor;",
         "}",
         "return result;"),
        ("5",), "120",
    ),
    _kind(
        "reverse_words",
        (("text", "str", "string"),),
        "str", "string", "Reverse the order of words in a text",
        ('return " ".join(reversed(@0@.split()))',),
        ('return @0@.trim().split(/\\s+/).reverse().join(" ");',),
        ('"one two three"',), '"three two one"',
    ),
    _kind(
        "capitalize_words",
        (("text", "str", "string"),),
        "str", "string", "Capitalize the first letter of every word",
        ('return " ".join(word.capitalize() for word in @0@.split())',),
        ('return @0@.trim().split(/\\s+/).map((word) => '
         'word.charAt(0).toUpperCase() + '
         'word.slice(1).toLowerCase()).join(" ");',),
        ('"hello world"',), '"Hello World"',
    ),
    _kind(
        "count_vowels",
        (("text", "str", "string"),),
        "int", "number", "Count the vowels in a text",
        ('return sum(1 for char in @0@.lower() if char in "aeiou")',),
        ("let total = 0;",
         'for (const char of @0@.toLowerCase()) {',
         '  if ("aeiou".includes(char)) {',
         "    total += 1;",
         "  }",
         "}",
         "return total;"),
        ('"hello"',), "2",
    ),
    _kind(
        "gcd",
        (("left", "int", "number"), ("right", "int", "number")),
        "int", "number", "Return the greatest common divisor",
        ("while @1@:",
         "    @0@, @1@ = @1@, @0@ % @1@",
         "return abs(@0@)"),
        ("let a = Math.abs(@0@);",
         "let b = Math.abs(@1@);",
         "while (b !== 0) {",
         "  const remainder = a % b;",
         "  a = b;",
         "  b = remainder;",
         "}",
         "return a;"),
        ("12", "18"), "6",
    ),
    _kind(
        "lcm",
        (("left", "int", "number"), ("right", "int", "number")),
        "int", "number", "Return the least common multiple",
        ("if @0@ == 0 or @1@ == 0:",
         "    return 0",
         "a, b = abs(@0@), abs(@1@)",
         "while b:",
         "    a, b = b, a % b",
         "return abs(@0@ * @1@) // a"),
        ("if (@0@ === 0 || @1@ === 0) {",
         "  return 0;",
         "}",
         "let a = Math.abs(@0@);",
         "let b = Math.abs(@1@);",
         "while (b !== 0) {",
         "  const remainder = a % b;",
         "  a = b;",
         "  b = remainder;",
         "}",
         "return Math.abs(@0@ * @1@) / a;"),
        ("4", "6"), "12",
    ),
    _kind(
        "binary_search",
        (("values", "list[int]", "number[]"), ("target", "int", "number")),
        "int", "number", "Return the index of a target in a sorted list",
        ("low, high = 0, len(@0@) - 1",
         "while low <= high:",
         "    middle = (low + high) // 2",
         "    if @0@[middle] == @1@:",
         "        return middle",
         "    if @0@[middle] < @1@:",
         "        low = middle + 1",
         "    else:",
         "        high = middle - 1",
         "return -1"),
        ("let low = 0;",
         "let high = @0@.length - 1;",
         "while (low <= high) {",
         "  const middle = Math.floor((low + high) / 2);",
         "  if (@0@[middle] === @1@) {",
         "    return middle;",
         "  }",
         "  if (@0@[middle] < @1@) {",
         "    low = middle + 1;",
         "  } else {",
         "    high = middle - 1;",
         "  }",
         "}",
         "return -1;"),
        ("[1, 2, 3, 4]", "3"), "2",
    ),
    _kind(
        "merge_sorted",
        (("left", "list[float]", "number[]"),
         ("right", "list[float]", "number[]")),
        "list[float]", "number[]", "Merge two sorted lists into one",
        ("left_index, right_index = 0, 0",
         "merged = []",
         "while left_index < len(@0@) and right_index < len(@1@):",
         "    if @0@[left_index] <= @1@[right_index]:",
         "        merged.append(@0@[left_index])",
         "        left_index += 1",
         "    else:",
         "        merged.append(@1@[right_index])",
         "        right_index += 1",
         "merged.extend(@0@[left_index:])",
         "merged.extend(@1@[right_index:])",
         "return merged"),
        ("const merged: number[] = [];",
         "let leftIndex = 0;",
         "let rightIndex = 0;",
         "while (leftIndex < @0@.length && rightIndex < @1@.length) {",
         "  if (@0@[leftIndex] <= @1@[rightIndex]) {",
         "    merged.push(@0@[leftIndex]);",
         "    leftIndex += 1;",
         "  } else {",
         "    merged.push(@1@[rightIndex]);",
         "    rightIndex += 1;",
         "  }",
         "}",
         "return merged.concat(@0@.slice(leftIndex), "
         "@1@.slice(rightIndex));"),
        ("[1, 3]", "[2, 4]"), "[1, 2, 3, 4]",
    ),
    _kind(
        "chunk",
        (("values", "list[float]", "number[]"), ("size", "int", "number")),
        "list[list[float]]", "number[][]",
        "Split a list into fixed-size chunks",
        ("if @1@ <= 0:",
         '    raise ValueError("chunk: size must be positive")',
         "return [@0@[index:index + @1@] "
         "for index in range(0, len(@0@), @1@)]"),
        ("if (@1@ <= 0) {",
         '  throw new Error("chunk: size must be positive");',
         "}",
         "const out: number[][] = [];",
         "for (let index = 0; index < @0@.length; index += @1@) {",
         "  out.push(@0@.slice(index, index + @1@));",
         "}",
         "return out;"),
        ("[1, 2, 3, 4, 5]", "2"), "[[1, 2], [3, 4], [5]]",
    ),
    _kind(
        "flatten",
        (("values", "list[object]", "unknown[]"),),
        "list[object]", "unknown[]",
        "Flatten arbitrarily nested lists into one",
        ("out = []",
         "for item in @0@:",
         "    if isinstance(item, list):",
         "        out.extend(flatten(item))",
         "    else:",
         "        out.append(item)",
         "return out"),
        ("const out: unknown[] = [];",
         "for (const item of @0@) {",
         "  if (Array.isArray(item)) {",
         "    out.push(...flatten(item));",
         "  } else {",
         "    out.push(item);",
         "  }",
         "}",
         "return out;"),
        ("[[1, [2]], 3]",), "[1, 2, 3]",
    ),
    _kind(
        "zip_agg",
        (("keys", "list[str]", "string[]"),
         ("values", "list[float]", "number[]")),
        "dict[str, float]", "Record<string, number>",
        "Zip keys and values into a mapping",
        ("if len(@0@) != len(@1@):",
         '    raise ValueError("zip_agg: keys and values must have '
         'equal length")',
         "return {str(key): value for key, value in zip(@0@, @1@)}"),
        ("if (@0@.length !== @1@.length) {",
         '  throw new Error("zip_agg: keys and values must have '
         'equal length");',
         "}",
         "const out: Record<string, number> = {};",
         "for (let index = 0; index < @0@.length; index += 1) {",
         "  out[String(@0@[index])] = @1@[index];",
         "}",
         "return out;"),
        ('["a", "b"]', "[1, 2]"), '{"a": 1, "b": 2}',
    ),
    _kind(
        "to_json",
        (("value", "object", "unknown"),),
        "str", "string", "Serialize a value to deterministic JSON",
        ("return json.dumps(@0@, sort_keys=True, ensure_ascii=False)",),
        ("return JSON.stringify(@0@);",),
        ('{"b": 2, "a": 1}',), '\'{"a": 1, "b": 2}\'',
        imports=("json",),
    ),
    _kind(
        "from_json",
        (("text", "str", "string"),),
        "object", "unknown", "Parse a JSON document",
        ("return json.loads(@0@)",),
        ("return JSON.parse(@0@);",),
        ('\'{"a": 1}\'',), '{"a": 1}',
        imports=("json",),
    ),
    _kind(
        "slugify",
        (("text", "str", "string"),),
        "str", "string", "Convert text into a URL-friendly slug",
        ("value = @0@.strip().lower()",
         'value = re.sub(r"[^a-z0-9]+", "-", value)',
         'return value.strip("-")'),
        ("return @0@.trim().toLowerCase()",
         '  .replace(/[^a-z0-9]+/g, "-")',
         '  .replace(/^-+|-+$/g, "");'),
        ('"Hello World!"',), '"hello-world"',
        imports=("re",),
    ),
    _kind(
        "title_case",
        (("text", "str", "string"),),
        "str", "string", "Title-case a text",
        ('return " ".join(word[:1].upper() + word[1:].lower() '
         "for word in @0@.split())",),
        ('return @0@.trim().split(/\\s+/).map((word) => '
         'word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())'
         '.join(" ");',),
        ('"hELLO wORLD"',), '"Hello World"',
    ),
    _kind(
        "trim_punct",
        (("text", "str", "string"),),
        "str", "string", "Strip leading and trailing punctuation",
        ("return @0@.strip(string.punctuation)",),
        ('return @0@.replace(/^\\W+|\\W+$/g, "");',),
        ('"!!hello!!"',), '"hello"',
        imports=("string",),
    ),
    _kind(
        "safe_get",
        (("data", "dict", "Record<string, unknown>"),
         ("key", "str", "string"), ("default", "object", "unknown")),
        "object", "unknown",
        "Read a mapping key with a fallback default",
        ("if isinstance(@0@, dict):",
         "    return @0@.get(@1@, @2@)",
         "return @2@"),
        ("if (typeof @0@ === \"object\" && @0@ !== null) {",
         "  const record = @0@ as Record<string, unknown>;",
         "  if (@1@ in record) {",
         "    return record[@1@];",
         "  }",
         "}",
         "return @2@;"),
        ('{"a": 1}', '"a"', "0"), "1",
    ),
    _kind(
        "moving_average",
        (("values", "list[float]", "number[]"), ("window", "int", "number")),
        "list[float]", "number[]",
        "Compute a trailing moving average",
        ("if @1@ <= 0:",
         '    raise ValueError("moving_average: window must be positive")',
         "out = []",
         "for index in range(len(@0@)):",
         "    start = max(0, index - @1@ + 1)",
         "    window_values = @0@[start:index + 1]",
         "    out.append(sum(window_values) / len(window_values))",
         "return out"),
        ("if (@1@ <= 0) {",
         '  throw new Error("moving_average: window must be positive");',
         "}",
         "const out: number[] = [];",
         "for (let index = 0; index < @0@.length; index += 1) {",
         "  const start = Math.max(0, index - @1@ + 1);",
         "  const windowValues = @0@.slice(start, index + 1);",
         "  const total = windowValues.reduce((acc, value) => "
         "acc + value, 0);",
         "  out.push(total / windowValues.length);",
         "}",
         "return out;"),
        ("[1, 2, 3, 4]", "2"), "[1.0, 1.5, 2.5, 3.5]",
    ),
])


# ---------------------------------------------------------------------------
# spec normalization and validation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "float": ("float", "number"),
    "int": ("int", "number"),
    "str": ("str", "string"),
    "string": ("str", "string"),
    "bool": ("bool", "boolean"),
    "list": ("list", "unknown[]"),
    "list[float]": ("list[float]", "number[]"),
    "list[int]": ("list[int]", "number[]"),
    "dict": ("dict", "Record<string, unknown>"),
    "dict[str, int]": ("dict[str, int]", "Record<string, number>"),
    "tuple[float, float]": ("tuple[float, float]", "[number, number]"),
}


def _slug(text, fallback="app"):
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(text or "")).strip("_").lower()
    if not value:
        value = fallback
    if value[0].isdigit():
        value = "_" + value
    return value


def _ts_name(text):
    return re.sub(r"[^0-9a-z]+", "-", str(text or "").lower()).strip("-") or "app"


def _normalize_params(raw, kind):
    if not raw:
        return [tuple(p) for p in kind["params"]]
    if len(raw) != len(kind["params"]):
        raise ScaffoldError(
            f"arity mismatch: expects {len(kind['params'])} params")
    params = []
    for position, item in enumerate(raw):
        if isinstance(item, str):
            name = item
            declared = None
        elif isinstance(item, dict):
            name = str(item.get("name") or "")
            declared = item.get("type")
        else:
            raise ScaffoldError(f"invalid param at position {position}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            raise ScaffoldError(f"invalid param name: {name!r}")
        if declared:
            py_type, ts_type = _TYPE_MAP.get(
                declared, (kind["params"][position][1],
                           kind["params"][position][2]))
        else:
            py_type, ts_type = kind["params"][position][1], \
                kind["params"][position][2]
        params.append((name, py_type, ts_type))
    return params


def _normalize_returns(raw, kind):
    if not raw:
        return kind["py_return"], kind["ts_return"]
    if isinstance(raw, dict):
        raw = raw.get("type") or raw.get("name")
    if not raw:
        return kind["py_return"], kind["ts_return"]
    if raw in _TYPE_MAP:
        return _TYPE_MAP[raw]
    return str(raw), "unknown"


def _plan(spec):
    if not isinstance(spec, dict):
        raise ScaffoldError("spec must be a dict")
    name = str(spec.get("name") or "").strip()
    if not name:
        raise ScaffoldError("spec.name is required")
    language = str(spec.get("language") or "python").strip().lower()
    if language not in ("python", "typescript"):
        raise ScaffoldError(f"unsupported language: {language!r}")
    raw_modules = spec.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ScaffoldError("spec.modules must be a non-empty list")

    issues = []
    skipped = []
    modules = []
    seen_modules = set()
    for index, raw in enumerate(raw_modules):
        if not isinstance(raw, dict):
            raise ScaffoldError(f"module #{index} must be a dict")
        module_name = _slug(raw.get("name"), fallback=f"module_{index + 1}")
        if module_name in seen_modules:
            raise ScaffoldError(f"duplicate module name: {module_name}")
        seen_modules.add(module_name)
        functions = []
        function_names = set()
        for item in raw.get("functions") or []:
            if not isinstance(item, dict):
                skipped.append({"module": module_name, "function": "?",
                                "reason": "function entry must be a dict"})
                continue
            function_name = str(item.get("name") or "").strip()
            body = item.get("body")
            kind_name = None
            if isinstance(body, dict):
                kind_name = body.get("kind")
            elif isinstance(body, str):
                kind_name = body
            kind_name = kind_name or item.get("kind")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name):
                skipped.append({"module": module_name,
                                "function": function_name or "?",
                                "reason": "invalid function name"})
                continue
            if kind_name not in KINDS:
                skipped.append({"module": module_name,
                                "function": function_name,
                                "reason": f"unknown body kind: {kind_name!r}"})
                continue
            kind = KINDS[kind_name]
            try:
                params = _normalize_params(item.get("params"), kind)
            except ScaffoldError as error:
                skipped.append({"module": module_name,
                                "function": function_name,
                                "reason": str(error)})
                continue
            if function_name in function_names:
                skipped.append({"module": module_name,
                                "function": function_name,
                                "reason": "duplicate top-level symbol"})
                continue
            function_names.add(function_name)
            py_return, ts_return = _normalize_returns(
                item.get("returns"), kind)
            functions.append({
                "name": function_name,
                "kind": kind,
                "params": params,
                "py_return": py_return,
                "ts_return": ts_return,
                "doc": str(item.get("doc") or kind["summary"]),
            })
        classes = []
        class_names = set()
        for item in raw.get("classes") or []:
            if not isinstance(item, dict):
                skipped.append({"module": module_name, "class": "?",
                                "reason": "class entry must be a dict"})
                continue
            class_name = str(item.get("name") or "").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", class_name):
                skipped.append({"module": module_name,
                                "class": class_name or "?",
                                "reason": "invalid class name"})
                continue
            if class_name in class_names or class_name in function_names:
                skipped.append({"module": module_name, "class": class_name,
                                "reason": "duplicate top-level symbol"})
                continue
            methods = []
            for method in item.get("methods") or []:
                if not isinstance(method, dict):
                    skipped.append({"module": module_name, "class": class_name,
                                    "reason": "method entry must be a dict"})
                    continue
                method_name = str(method.get("name") or "").strip()
                target = str(method.get("target") or method_name).strip()
                matched = next((f for f in functions
                                if f["name"] == target), None)
                if matched is None:
                    skipped.append({
                        "module": module_name, "class": class_name,
                        "reason": f"method target not generated: {target}"})
                    continue
                if len(matched["params"]) != 1:
                    skipped.append({
                        "module": module_name, "class": class_name,
                        "reason": f"method target {target} needs one param"})
                    continue
                methods.append({"name": method_name, "target": target})
            if not methods:
                skipped.append({"module": module_name, "class": class_name,
                                "reason": "class has no usable methods"})
                continue
            class_names.add(class_name)
            classes.append({
                "name": class_name,
                "doc": str(item.get("doc") or f"Aggregate over values"),
                "methods": methods,
                "sample": str(item.get("sample") or "[1, 2, 3]"),
            })
        modules.append({
            "name": module_name,
            "doc": str(raw.get("doc") or f"{module_name} helpers"),
            "functions": functions,
            "classes": classes,
        })
        for skipped_item in skipped:
            if "reason" in skipped_item:
                issues.append(
                    f"{skipped_item.get('module', '?')}: "
                    f"{skipped_item.get('function') or skipped_item.get('class') or '?'}: "
                    f"{skipped_item['reason']}")
    return {
        "name": name,
        "package": _slug(name),
        "ts_name": _ts_name(name),
        "language": language,
        "modules": modules,
        "issues": issues,
        "skipped": skipped,
        "entry": str(spec.get("entry") or "main.py"),
        "readme": spec.get("readme", True),
        "tests": spec.get("tests", True),
    }


def _generated_function_count(plan):
    return sum(len(m["functions"]) for m in plan["modules"])


def _fill(text, params):
    for index, (name, _py, _ts) in enumerate(params):
        text = text.replace(f"@{index}@", name)
    return text


def _py_signature(function):
    params = ", ".join(f"{name}: {py_type}"
                       for name, py_type, _ts in function["params"])
    return f"{function['name']}({params}) -> {function['py_return']}"


def _ts_signature(function):
    params = ", ".join(f"{name}: {ts_type}"
                       for name, _py, ts_type in function["params"])
    return (f"export function {function['name']}"
            f"({params}): {function['ts_return']}")


def _py_function(function):
    kind = function["kind"]
    lines = [f"def {_py_signature(function)}:", f'    """{function["doc"]}.']
    if function["params"]:
        lines.append("")
        lines.append("    Args:")
        for name, _py, _ts in function["params"]:
            lines.append(f"        {name}: Input value for the operation.")
    lines.append("")
    lines.append("    Returns:")
    lines.append(f"        Result of {function['name'].replace('_', ' ')}.")
    lines.append('    """')
    for body_line in kind["py_body"]:
        lines.append("    " + _fill(body_line, function["params"]))
    lines.append("")
    return lines


def _ts_function(function):
    kind = function["kind"]
    lines = ["/**", f" * {function['doc']}."]
    if function["params"]:
        lines.append(" *")
        for name, _py, _ts in function["params"]:
            lines.append(f" * @param {name} - Input value for the operation.")
    lines.append(" *")
    lines.append(f" * @returns Result of {function['name'].replace('_', ' ')}.")
    lines.append(" */")
    lines.append(f"{_ts_signature(function)} {{")
    for body_line in kind["ts_body"]:
        lines.append("  " + _fill(body_line, function["params"]))
    lines.append("}")
    lines.append("")
    return lines


def _py_class(cls, functions_by_name):
    lines = [f"class {cls['name']}:",
             f'    """{cls["doc"]}."""',
             "",
             "    def __init__(self, values):",
             "        self.values = list(values)",
             ""]
    for method in cls["methods"]:
        target = functions_by_name[method["target"]]
        lines.append(f"    def {method['name']}(self) -> "
                     f"{target['py_return']}:")
        lines.append(f'        """Delegate to {method["target"]}."""')
        lines.append(f"        return {method['target']}(self.values)")
        lines.append("")
    return lines


def _ts_class(cls, functions_by_name):
    lines = ["export class " + cls["name"] + " {",
             "  private readonly values: number[];",
             "",
             "  constructor(values: number[]) {",
             "    this.values = [...values];",
             "  }",
             ""]
    for method in cls["methods"]:
        target = functions_by_name[method["target"]]
        lines.append(f"  {method['name']}(): {target['ts_return']} {{")
        lines.append(f"    return {method['target']}(this.values);")
        lines.append("  }")
        lines.append("")
    lines.append("}")
    lines.append("")
    return lines


def _py_module(plan, module):
    lines = [f'"""{module["doc"]}."""',
             "",
             "from __future__ import annotations",
             ""]
    imports = sorted({imp for function in module["functions"]
                      for imp in function["kind"]["imports"]})
    for imp in imports:
        lines.append(f"import {imp}")
    if imports:
        lines.append("")
    exports = [f["name"] for f in module["functions"]] + \
        [c["name"] for c in module["classes"]]
    lines.append("__all__ = [")
    for name in exports:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")
    lines.append("")
    functions_by_name = {f["name"]: f for f in module["functions"]}
    for function in module["functions"]:
        lines.extend(_py_function(function))
        lines.append("")
    for cls in module["classes"]:
        lines.extend(_py_class(cls, functions_by_name))
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _ts_module(module):
    lines = ["/**",
             f" * {module['doc']}.",
             " *",
             " * Generated by the RENA-BR scaffold generator: deterministic and",
             " * dependency-free.  Every export has a strict signature.",
             " */",
             ""]
    for function in module["functions"]:
        lines.extend(_ts_function(function))
    functions_by_name = {f["name"]: f for f in module["functions"]}
    for cls in module["classes"]:
        lines.extend(_ts_class(cls, functions_by_name))
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _py_init(plan):
    lines = [f'"""{plan["name"]}: generated package."""',
             "",
             "from __future__ import annotations",
             ""]
    exports = []
    for module in plan["modules"]:
        names = [f["name"] for f in module["functions"]] + \
            [c["name"] for c in module["classes"]]
        if not names:
            continue
        for name in names:
            exports.append(name)
        lines.append(f"from .{module['name']} import (")
        for name in names:
            lines.append(f"    {name},")
        lines.append(")")
        lines.append("")
    lines.append("__all__ = [")
    for name in exports:
        lines.append(f'    "{name}",')
    lines.append("]")
    return "\n".join(lines) + "\n"


def _py_main(plan):
    lines = [f'"""Entry point for {plan["name"]}: exercises every generated '
             'function."""',
             "",
             "from __future__ import annotations",
             ""]
    for module in plan["modules"]:
        names = [f["name"] for f in module["functions"]] + \
            [c["name"] for c in module["classes"]]
        if not names:
            continue
        lines.append(f"from {plan['package']}.{module['name']} import (")
        for name in names:
            lines.append(f"    {name},")
        lines.append(")")
    lines.append("")
    lines.append("")
    lines.append("def _exercise() -> int:")
    lines.append("    failures = 0")
    for module in plan["modules"]:
        for function in module["functions"]:
            call = ", ".join(function["kind"]["sample_args"])
            lines.append("    try:")
            lines.append(
                f'        print("{function["name"]}:", '
                f'{function["name"]}({call}))')
            lines.append("    except Exception as error:  # noqa: BLE001")
            lines.append("        failures += 1")
            lines.append(
                f'        print("{function["name"]}: ERROR", error)')
        for cls in module["classes"]:
            lines.append("    try:")
            lines.append(f"        instance = {cls['name']}({cls['sample']})")
            for method in cls["methods"]:
                lines.append(
                    f'        print("{cls["name"]}.{method["name"]}:", '
                    f"instance.{method['name']}())")
            lines.append("    except Exception as error:  # noqa: BLE001")
            lines.append("        failures += 1")
            lines.append(
                f'        print("{cls["name"]}: ERROR", error)')
    lines.append("    if failures:")
    lines.append('        print(f"FAILED: {failures} call(s)")')
    lines.append("        return 1")
    lines.append('    print("OK")')
    lines.append("    return 0")
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    raise SystemExit(_exercise())")
    return "\n".join(lines) + "\n"


def _py_tests(plan):
    lines = [f'"""Generated unittest coverage for {plan["name"]}."""',
             "",
             "from __future__ import annotations",
             "",
             "import unittest",
             ""]
    for module in plan["modules"]:
        names = [f["name"] for f in module["functions"]] + \
            [c["name"] for c in module["classes"]]
        if not names:
            continue
        lines.append(f"from {plan['package']}.{module['name']} import (")
        for name in names:
            lines.append(f"    {name},")
        lines.append(")")
    lines.append("")
    lines.append("")
    for module in plan["modules"]:
        class_name = "".join(part.capitalize()
                             for part in module["name"].split("_")) + "Test"
        lines.append(f"class {class_name}(unittest.TestCase):")
        wrote = False
        for function in module["functions"]:
            call = ", ".join(function["kind"]["sample_args"])
            lines.append(f"    def test_{function['name']}(self):")
            lines.append(
                f"        self.assertEqual({function['name']}({call}), "
                f"{function['kind']['sample_expected']})")
            wrote = True
        for cls in module["classes"]:
            lines.append(f"    def test_{cls['name'].lower()}_class(self):")
            lines.append(f"        instance = {cls['name']}({cls['sample']})")
            for method in cls["methods"]:
                lines.append(
                    f"        self.assertIsNotNone("
                    f"instance.{method['name']}())")
            wrote = True
        if not wrote:
            lines.append("    def test_module_importable(self):")
            lines.append("        self.assertTrue(True)")
        lines.append("")
    return "\n".join(lines) + "\n"


def _py_project_files(plan):
    files = {}
    package = plan["package"]
    files[f"{package}/__init__.py"] = _py_init(plan)
    for module in plan["modules"]:
        files[f"{package}/{module['name']}.py"] = _py_module(plan, module)
    files["main.py"] = _py_main(plan)
    files["tests/__init__.py"] = ""
    if plan["tests"]:
        files["tests/test_generated.py"] = _py_tests(plan)
    files["pyproject.toml"] = (
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[project]\n"
        f'name = "{_ts_name(plan["name"])}"\n'
        'version = "0.1.0"\n'
        f'description = "{plan["name"]} generated scaffold"\n'
        'requires-python = ">=3.10"\n'
        "\n"
        "[tool.setuptools.packages.find]\n"
        'where = ["."]\n'
        f'include = ["{package}*"]\n'
    )
    files["README.md"] = _readme(plan, "python")
    return files


def _ts_project_files(plan):
    files = {}
    for module in plan["modules"]:
        files[f"src/{module['name']}.ts"] = _ts_module(module)
    export_lines = ["/**",
                    f" * {plan['name']}: generated barrel export.",
                    " *",
                    " * Every module is pure and dependency-free; the strict",
                    " * compiler settings in tsconfig.json apply to all exports.",
                    " */",
                    ""]
    for module in plan["modules"]:
        export_lines.append(f'export * from "./{module["name"]}";')
    export_lines.append("")
    export_lines.append(f'export const PROJECT_NAME = "{plan["ts_name"]}";')
    export_lines.append("")
    export_lines.append("export const GENERATED_MODULES: string[] = [")
    for module in plan["modules"]:
        export_lines.append(f'  "{module["name"]}",')
    export_lines.append("];")
    files["src/index.ts"] = "\n".join(export_lines) + "\n"
    files["tsconfig.json"] = (
        "{\n"
        '  "compilerOptions": {\n'
        '    "target": "ES2022",\n'
        '    "module": "ESNext",\n'
        '    "moduleResolution": "Bundler",\n'
        '    "strict": true,\n'
        '    "declaration": true,\n'
        '    "outDir": "dist",\n'
        '    "rootDir": "src",\n'
        '    "skipLibCheck": true,\n'
        '    "forceConsistentCasingInFileNames": true\n'
        "  },\n"
        '  "include": ["src/**/*.ts"]\n'
        "}\n"
    )
    files["package.json"] = json.dumps({
        "name": plan["ts_name"],
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "main": "dist/index.js",
        "types": "dist/index.d.ts",
        "scripts": {
            "build": "tsc -p tsconfig.json",
            "typecheck": "tsc -p tsconfig.json --noEmit",
        },
    }, indent=2) + "\n"
    files["README.md"] = _readme(plan, "typescript")
    return files


def _readme(plan, language):
    if isinstance(plan["readme"], str) and plan["readme"].strip():
        return plan["readme"].rstrip() + "\n"
    lines = [f"# {plan['name']}", "",
             f"Generated by the RENA-BR scaffold generator "
             f"v{SCAFFOLD_VERSION} ({language}).", "",
             "## Modules", ""]
    for module in plan["modules"]:
        lines.append(f"### {module['name']}")
        lines.append("")
        lines.append(module["doc"] + ".")
        lines.append("")
        if module["functions"]:
            lines.append("| function | kind |")
            lines.append("| --- | --- |")
            for function in module["functions"]:
                lines.append(
                    f"| `{function['name']}` | `{function['kind']['name']}` |")
            lines.append("")
    if language == "python":
        lines.append("## Usage")
        lines.append("")
        lines.append("```bash")
        lines.append("python3 main.py")
        lines.append("python3 -m unittest")
        lines.append("```")
    else:
        lines.append("## Usage")
        lines.append("")
        lines.append("```bash")
        lines.append("npm install")
        lines.append("npm run build")
        lines.append("```")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append("The same spec always produces byte-identical output.")
    return "\n".join(lines) + "\n"


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_project(spec, out_dir):
    """Generate a project tree from ``spec`` into ``out_dir``.

    Returns a manifest describing generated files, skipped symbols and
    validation issues.  Raises ``ScaffoldError`` for structurally
    unusable specs; individual bad functions are skipped and reported
    instead of emitting broken code.
    """
    plan = _plan(spec)
    if plan["language"] == "python":
        files = _py_project_files(plan)
    else:
        files = _ts_project_files(plan)
    os.makedirs(out_dir, exist_ok=True)
    file_entries = []
    for path in sorted(files):
        content = files[path]
        full_path = os.path.join(out_dir, path)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        file_entries.append({
            "path": path,
            "lines": content.count("\n"),
            "bytes": len(content.encode("utf-8")),
            "sha256": _sha256(content),
        })
    total_lines = sum(entry["lines"] for entry in file_entries)
    python_lines = sum(entry["lines"] for entry in file_entries
                       if entry["path"].endswith(".py"))
    typescript_lines = sum(entry["lines"] for entry in file_entries
                           if entry["path"].endswith(".ts"))
    return {
        "scaffold_version": SCAFFOLD_VERSION,
        "project": plan["name"],
        "package": plan["package"],
        "language": plan["language"],
        "out_dir": os.path.abspath(out_dir),
        "files": file_entries,
        "modules": [m["name"] for m in plan["modules"]],
        "functions_generated": _generated_function_count(plan),
        "functions_skipped": list(plan["skipped"]),
        "issues": list(plan["issues"]),
        "totals": {
            "files": len(file_entries),
            "lines": total_lines,
            "python_lines": python_lines,
            "typescript_lines": typescript_lines,
        },
    }


_PROMPT_KEYWORDS = [
    ("moving average", "moving_average"),
    ("이동 평균", "moving_average"),
    ("fibonacci", "fib"),
    ("피보나치", "fib"),
    ("factorial", "fact"),
    ("팩토리얼", "fact"),
    ("binary search", "binary_search"),
    ("이진 탐색", "binary_search"),
    ("prime", "is_prime"),
    ("소수", "is_prime"),
    ("gcd", "gcd"),
    ("최대공약수", "gcd"),
    ("lcm", "lcm"),
    ("최소공배수", "lcm"),
    ("slug", "slugify"),
    ("슬러그", "slugify"),
    ("median", "median"),
    ("중앙값", "median"),
    ("average", "mean"),
    ("mean", "mean"),
    ("평균", "mean"),
    ("clamp", "clamp"),
    ("클램프", "clamp"),
    ("norm", "norm"),
    ("정규화", "norm"),
    ("dot", "dot"),
    ("내적", "dot"),
    ("vowel", "count_vowels"),
    ("모음", "count_vowels"),
    ("capitalize", "capitalize_words"),
    ("대문자", "capitalize_words"),
    ("reverse", "reverse_words"),
    ("뒤집", "reverse_words"),
    ("word", "word_count"),
    ("단어", "word_count"),
    ("char", "char_freq"),
    ("문자 빈도", "char_freq"),
    ("unique", "unique_sorted"),
    ("중복", "unique_sorted"),
    ("even", "filter_even"),
    ("짝수", "filter_even"),
    ("sort", "sort_desc"),
    ("정렬", "sort_desc"),
    ("merge", "merge_sorted"),
    ("병합", "merge_sorted"),
    ("chunk", "chunk"),
    ("청크", "chunk"),
    ("flatten", "flatten"),
    ("평탄화", "flatten"),
    ("json", "to_json"),
    ("add", "add"),
    ("더하", "add"),
    ("subtract", "subtract"),
    ("빼", "subtract"),
    ("multiply", "mul"),
    ("곱", "mul"),
    ("divide", "div_safe"),
    ("나누", "div_safe"),
]

_PROMPT_MODULE = {
    "add": "math_basics", "subtract": "math_basics", "mul": "math_basics",
    "div_safe": "math_basics", "mean": "math_basics", "median": "math_basics",
    "clamp": "math_basics", "norm": "math_basics", "dot": "math_basics",
    "is_prime": "math_basics", "fib": "math_basics", "fact": "math_basics",
    "gcd": "math_basics", "lcm": "math_basics", "moving_average": "math_basics",
    "filter_even": "lists", "filter_gt": "lists", "sort_desc": "lists",
    "unique_sorted": "lists", "merge_sorted": "lists",
    "binary_search": "lists", "chunk": "lists", "flatten": "lists",
    "word_count": "text", "char_freq": "text", "reverse_words": "text",
    "capitalize_words": "text", "count_vowels": "text", "slugify": "text",
    "to_json": "data",
}


def generate_from_prompt(text):
    """Best-effort heuristic: scan a Korean/English request and build a
    minimal spec.  Documented as best-effort — unknown requests fall back
    to a small analytics scaffold."""
    prompt = str(text or "").lower()
    matched = []
    for keyword, kind in _PROMPT_KEYWORDS:
        if keyword in prompt and kind not in matched:
            matched.append(kind)
    if not matched:
        matched = ["mean", "sort_desc", "to_json"]
    language = "python"
    if any(token in prompt for token in
           ("typescript", "타입스크립트", "ts ", "node")):
        language = "typescript"
    grouped = {}
    for kind in matched:
        grouped.setdefault(_PROMPT_MODULE.get(kind, "utils"), []).append(kind)
    modules = []
    for module_name in sorted(grouped):
        kinds = grouped[module_name]
        modules.append({
            "name": module_name,
            "doc": f"{module_name} helpers detected from the request",
            "functions": [
                {"name": kind, "body": {"kind": kind}} for kind in kinds
            ],
            "classes": [],
        })
    words = re.findall(r"[0-9a-zA-Z가-힣]+", str(text or ""))
    base = "_".join(words[:3]) if words else "generated_project"
    return {
        "name": base or "generated_project",
        "language": language,
        "modules": modules,
        "entry": "main.py",
        "readme": True,
        "tests": True,
    }
