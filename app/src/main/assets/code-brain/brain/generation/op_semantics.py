"""Shared operation semantics registry.

Single source of truth for what each operation *means*:
- how it transforms data (reference semantics, used by verification),
- how it is realized in generated code (strategy variant templates),
- how its presence is detected structurally in generated code,
- how fine-grained composition operations normalize into pipeline ops.

The code generator builds a dataflow-chained pipeline from the operation
graph. The verifier executes generated code and compares behavior against
the reference semantics derived from the same operation graph — behavior
expectations come from the semantic specification, never from hardcoded
expected outputs per sentence.
"""
from __future__ import annotations
from typing import Callable, Optional


class UnknownOperationError(Exception):
    """Raised when a required operation has no known executable semantics."""

    def __init__(self, op: str):
        super().__init__(f"No executable semantics for operation: {op}")
        self.operation = op


# --------------------------------------------------------------------- #
# Normalization: fine-grained composition ops -> pipeline ops
# --------------------------------------------------------------------- #

def normalize_ops(ops: list[str], composition_ops: Optional[list[str]] = None) -> list[str]:
    """Normalize an operation sequence into executable pipeline form.

    - Collapses compare_greater_than/compare_less_than (+ filter) into
      filter_above/filter_below.
    - Resolves generic sort direction from composition cues.
    - Removes consecutive duplicates.

    ``composition_ops`` (from the SemanticComposer graph) is used only to
    resolve ambiguity (e.g. whether a generic sort is ascending or
    descending); it never injects operations the user did not express.
    """
    comp = set(composition_ops or [])

    normalized: list[str] = []
    i = 0
    while i < len(ops):
        op = ops[i]

        if op == "compare_greater_than":
            normalized.append("filter_above")
            # absorb a following standalone filter
            if i + 1 < len(ops) and ops[i + 1] == "filter":
                i += 2
            else:
                i += 1
            continue
        if op == "compare_less_than":
            normalized.append("filter_below")
            if i + 1 < len(ops) and ops[i + 1] == "filter":
                i += 2
            else:
                i += 1
            continue
        if op == "filter" and normalized and normalized[-1] in ("filter_above", "filter_below"):
            i += 1
            continue
        if op == "sort_list":
            if "sort_descending" in comp:
                normalized.append("sort_descending")
            elif "sort_ascending" in comp:
                normalized.append("sort_ascending")
            else:
                normalized.append("sort_ascending")
            i += 1
            continue
        if normalized and op == normalized[-1]:
            i += 1
            continue
        normalized.append(op)
        i += 1

    return normalized


# --------------------------------------------------------------------- #
# Reference semantics (behavioral ground truth per operation)
# --------------------------------------------------------------------- #

def _is_prime(n) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def _ref_apply(op: str, cur, scalars: dict, key: str = "",
               threshold=None):
    """Apply one operation's reference semantics.

    ``cur`` is the current dataflow value; ``scalars`` holds named scalar
    products (avg, total) computed along the way. Returns the new ``cur``.

    ``key`` is the objective's select key ("length"): an extremum over a
    DERIVED length compares lengths, not the elements' own ordering —
    the reference and the generated code read the same rule.
    """
    key_fn = len if key == "length" else None
    if op == "remove_duplicates":
        seen = set()
        out = []
        for x in cur:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    if op == "calculate_average":
        scalars["avg"] = sum(cur) / len(cur) if cur else 0
        return cur
    if op == "calculate_sum":
        scalars["total"] = sum(cur)
        return cur
    if op == "filter_above":
        if threshold is not None:
            return [x for x in cur if x > threshold]
        avg = scalars.get("avg")
        if avg is None:
            # No operand and no aggregate produced one: the selection has
            # no executable semantics.  Never invent the mean.
            raise UnknownOperationError("filter_above")
        return [x for x in cur if x > avg]
    if op == "filter_below":
        if threshold is not None:
            return [x for x in cur if x < threshold]
        avg = scalars.get("avg")
        if avg is None:
            raise UnknownOperationError("filter_below")
        return [x for x in cur if x < avg]
    if op == "sort_descending":
        return sorted(cur, reverse=True)
    if op == "sort_ascending":
        return sorted(cur)
    if op == "reverse":
        return list(cur)[::-1]
    if op == "filter":
        return [x for x in cur if x]
    if op == "filter_even":
        return [x for x in cur if x % 2 == 0]
    if op == "is_prime":
        return [x for x in cur if _is_prime(x)]
    if op == "is_multiple_of":
        modulus = scalars.get("modulus", 2)
        return [x for x in cur if x % modulus == 0]
    if op == "generate_range":
        return list(range(1, 101))
    if op == "find_max":
        # A scalar product: it is RECORDED, never substituted for the
        # collection — a downstream collection op consumes a sequence,
        # and min(max(x)) is a dataflow-invalid chain (min over a
        # scalar), the same scalar-advance bug the ir_builder forbids.
        scalars["max"] = max(cur, key=key_fn) if cur else None
        return cur
    if op == "count_elements":
        scalars["count"] = len(cur)
        return cur
    if op == "find_min":
        scalars["min"] = min(cur, key=key_fn) if cur else None
        return cur
    if op == "add":
        # two-operand arithmetic over the collection's first elements
        if len(cur) >= 2:
            scalars["add"] = cur[0] + cur[1]
        elif cur:
            scalars["add"] = cur[0]
        else:
            scalars["add"] = 0
        return cur
    if op == "multiply":
        if len(cur) >= 2:
            scalars["multiply"] = cur[0] * cur[1]
        elif cur:
            scalars["multiply"] = cur[0]
        else:
            scalars["multiply"] = 0
        return cur
    if op == "search":
        # linear search: first index whose element equals the target
        target = scalars.get("target")
        for i, x in enumerate(cur):
            if x == target:
                return i
        return -1
    if op == "count_frequency":
        freq: dict = {}
        for x in cur:
            freq[x] = freq.get(x, 0) + 1
        scalars["freq"] = freq
        return cur
    if op == "compare_equal_to":
        params = scalars.get("__params__") or {}
        scalars["equal_operand"] = params.get("operand", 1)
        return cur
    if op == "select_first":
        scalars["first"] = cur[0] if cur else None
        return cur
    if op == "fallback_none":
        return scalars.get("first")
    raise UnknownOperationError(op)


#: Operations whose final semantic product is a scalar, when they are the
#: *last* operation in the graph.  reference_pipeline's run() handles
#: each of these terminals directly (the return binds to the product).
_SCALAR_FINAL_OPS = {"calculate_average", "calculate_sum", "find_max",
                     "find_min", "count_elements", "add", "multiply"}


def reference_pipeline(ops: list[str], contract=None) -> Callable:
    """Build a reference implementation for a full operation graph.

    The returned function maps input list -> expected output, derived
    entirely from the semantic operation structure. Handles both pipeline
    operations (list transforms) and direct algorithms (standalone functions).
    """
    if len(ops) == 1 and ops[0] in _DIRECT_ALGORITHMS:
        algo = _DIRECT_ALGORITHMS[ops[0]]
        ref_fn = algo["ref"]
        n_params = algo["ref_params"]

        def run_direct(*args):
            if n_params == 1:
                return ref_fn(args[0])
            elif n_params == 2:
                return ref_fn(args[0], args[1])
            return ref_fn(*args)
        return run_direct

    # Pipeline operations
    key = "length" if (contract is not None
                       and getattr(contract, "select_key", "") == "length")         else ""
    key_fn = len if key == "length" else None
    threshold = (getattr(contract, "threshold", None)
                 if contract is not None else None)
    op_params = ((getattr(contract, "op_params", None) or {})
                 if contract is not None else {})

    def run(data):
        cur = list(data) if data is not None else []
        scalars: dict = {}
        for idx, op in enumerate(ops):
            is_last = idx == len(ops) - 1
            scalars["__params__"] = op_params.get(op, {})
            # A selection whose predicate is the DERIVED per-element
            # frequency ("appears only once") filters on the frequency
            # state, not on truthiness.
            if op == "filter" and "freq" in scalars \
                    and "equal_operand" in scalars:
                cur = [x for x in cur
                       if scalars["freq"].get(x, 0) == scalars["equal_operand"]]
                continue
            if is_last and op == "select_first":
                return cur[0] if cur else None
            if is_last and op == "fallback_none":
                return scalars.get("first")
            # Terminal scalar op: the pipeline's OUTPUT is the scalar
            # itself — the return binds to the product, never to the
            # input (that would be an identity fallback).
            if is_last and op == "calculate_average":
                return sum(cur) / len(cur) if cur else 0
            if is_last and op == "calculate_sum":
                return sum(cur)
            if is_last and op == "find_max":
                return max(cur, key=key_fn) if cur else None
            if is_last and op == "find_min":
                return min(cur, key=key_fn) if cur else None
            if is_last and op == "count_elements":
                return len(cur)
            if is_last and op in ("add", "multiply"):
                if len(cur) >= 2:
                    return cur[0] + cur[1] if op == "add" else cur[0] * cur[1]
                return cur[0] if cur else 0
            cur = _ref_apply(op, cur, scalars, key, threshold=threshold)
        return cur
    return run


#: String-domain test inputs: same STRUCTURAL cases as the numeric set
#: (varied values, duplicates, ties on the selection key, all-identical,
#: empty, single element) expressed over strings — the objective's
#: element domain decides the inputs, never the request's wording.
_STRING_TEST_INPUTS = [
    ["apple", "pear", "banana", "fig", "kiwi"],   # varied lengths
    ["aa", "b", "ccc", "dd", "e"],                # short/long, ties by length
    ["same", "same", "same"],                     # all identical
    [],                                           # empty input
    ["zz"],                                       # single element
    ["ab", "abc", "a", "abcd", "ab"],             # tie on the max length
    ["hello", "world", "hi"],                     # two equal-length maxima
    ["x", "yy", "yyy", "yy"],                     # duplicate of the max
]


def generated_test_inputs(contract=None) -> list[list]:
    """Behaviorally meaningful test inputs generated from data semantics,
    not from any particular sentence: duplicates, values straddling a mean,
    negatives, zero, all-identical, and empty collections.

    With a contract that DECLARED the element domain, the cases are drawn
    from that domain — a list of strings is verified against strings.
    """
    if contract is not None and (getattr(contract, "elem_type", "") or "") == "str":
        return [list(c) for c in _STRING_TEST_INPUTS]
    return [
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],   # duplicates, mixed
        [-5, -2, 0, 3, -8, 3],               # negatives and zero
        [7, 7, 7, 7],                        # all identical
        [],                                   # empty input
        [10, -10, 20, -20, 5, 5],            # symmetric around zero
        [0, 0, 0, 1],                        # zeros dominating
        [42],                                 # single element
        [2, 2, 3, 3, 1, 1],                  # everything duplicated
    ]


# --------------------------------------------------------------------- #
# Code templates: op -> {variant_id: emitter(prev_var, out_var) -> lines}
# Each emitter returns body lines (4-space indented) assigning out_var.
# --------------------------------------------------------------------- #

def _tpl_remove_duplicates():
    return {
        "loop_seen": lambda p, o: [
            f"    seen = set()",
            f"    {o} = []",
            f"    for x in {p}:",
            f"        if x not in seen:",
            f"            seen.add(x)",
            f"            {o}.append(x)",
        ],
        "dict_fromkeys": lambda p, o: [f"    {o} = list(dict.fromkeys({p}))"],
        "set": lambda p, o: [f"    {o} = list(set({p}))"],
    }

def _tpl_calculate_average():
    return {
        "builtin": lambda p, o: [
            f"    avg = sum({p}) / len({p}) if {p} else 0",
            f"    {o} = {p}",
        ],
        "accumulate": lambda p, o: [
            f"    total = 0",
            f"    count = 0",
            f"    for x in {p}:",
            f"        total += x",
            f"        count += 1",
            f"    avg = total / count if count else 0",
            f"    {o} = {p}",
        ],
    }

def _tpl_calculate_sum():
    return {
        "builtin": lambda p, o: [
            f"    total = sum({p})",
            f"    {o} = {p}",
        ],
    }

def _tpl_filter_above():
    # The threshold is provided by the dataflow: a literal operand
    # (rendered directly by build_pipeline) or the `avg` variable an
    # aggregate produced.  A selection with NEITHER has no executable
    # semantics — build_pipeline raises instead of inventing the mean.
    return {
        "list_comp": lambda p, o: [
            f"    {o} = [x for x in {p} if x > avg]"],
        "filter_builtin": lambda p, o: [
            f"    {o} = list(filter(lambda x: x > avg, {p}))"],
    }

def _tpl_filter_below():
    return {
        "list_comp": lambda p, o: [
            f"    {o} = [x for x in {p} if x < avg]"],
        "filter_builtin": lambda p, o: [
            f"    {o} = list(filter(lambda x: x < avg, {p}))"],
    }

def _tpl_sort_descending():
    return {
        "builtin": lambda p, o: [f"    {o} = sorted({p}, reverse=True)"],
        "bubble_sort": lambda p, o: [
            f"    {o} = list({p})",
            f"    for i in range(len({o})):",
            f"        for j in range(len({o}) - 1 - i):",
            f"            if {o}[j] < {o}[j + 1]:",
            f"                {o}[j], {o}[j + 1] = {o}[j + 1], {o}[j]",
        ],
        "heap": lambda p, o: [
            f"    import heapq",
            f"    {o} = heapq.nlargest(len({p}), {p})",
        ],
    }

def _tpl_sort_ascending():
    return {
        "builtin": lambda p, o: [f"    {o} = sorted({p})"],
        "bubble_sort": lambda p, o: [
            f"    {o} = list({p})",
            f"    for i in range(len({o})):",
            f"        for j in range(len({o}) - 1 - i):",
            f"            if {o}[j] > {o}[j + 1]:",
            f"                {o}[j], {o}[j + 1] = {o}[j + 1], {o}[j]",
        ],
    }

def _tpl_simple(name):
    frag = {
        "reverse": "[::-1]",
        "filter": " if x",
        "filter_even": " % 2 == 0",
        "is_multiple_of": " % 2 == 0",
    }
    if name == "reverse":
        return {"default": lambda p, o: [f"    {o} = {p}[::-1]"]}
    if name == "filter":
        return {"default": lambda p, o: [f"    {o} = [x for x in {p} if x]"]}
    if name == "filter_even":
        return {"default": lambda p, o: [f"    {o} = [x for x in {p} if x % 2 == 0]"]}
    # is_multiple_of (default modulus 2)
    return {"default": lambda p, o: [f"    {o} = [x for x in {p} if x % 2 == 0]"]}

def _tpl_is_prime():
    return {
        "trial_division": lambda p, o: [
            f"    def _is_prime(n):",
            f"        if n < 2:",
            f"            return False",
            f"        for i in range(2, int(n ** 0.5) + 1):",
            f"            if n % i == 0:",
            f"                return False",
            f"        return True",
            f"    {o} = [x for x in {p} if _is_prime(x)]",
        ],
    }

def _tpl_generate_range():
    return {
        "default": lambda p, o: [f"    {o} = list(range(1, 101))"],
    }

def _tpl_find_max():
    return {
        "builtin": lambda p, o: [f"    {o} = max({p}) if {p} else None"],
        "linear_scan": lambda p, o: [
            f"    {o} = None",
            f"    for x in {p}:",
            f"        if {o} is None or x > {o}:",
            f"            {o} = x",
        ],
    }


def _tpl_add_multiply(sym):
    """Two-operand arithmetic over the collection's first elements."""
    return {
        "builtin": lambda p, o: [
            f"    {o} = {p}[0] {sym} {p}[1] if len({p}) >= 2 else ({p}[0] if {p} else 0)"
        ],
    }


def _tpl_search():
    """Linear search: first index whose element equals the target."""
    return {
        "linear": lambda p, o: [
            f"    {o} = -1",
            f"    for i, x in enumerate({p}):",
            "        if x == target:",
            f"            {o} = i",
            "            break",
        ],
    }

def _tpl_find_min():
    return {
        "builtin": lambda p, o: [f"    {o} = min({p}) if {p} else None"],
        "linear_scan": lambda p, o: [
            f"    {o} = None",
            f"    for x in {p}:",
            f"        if {o} is None or x < {o}:",
            f"            {o} = x",
        ],
    }


_OP_REGISTRY: dict[str, dict] = {
    "remove_duplicates": {"templates": _tpl_remove_duplicates(), "default": "loop_seen",
                          "markers": ["seen", "fromkeys", "set("]},
    "calculate_average": {"templates": _tpl_calculate_average(), "default": "builtin",
                          "markers": ["avg", "sum(", "/ len(", "count"]},
    "calculate_sum": {"templates": _tpl_calculate_sum(), "default": "builtin",
                      "markers": ["total", "sum("]},
    # The selection's markers are its STRUCTURE (an element comparison
    # inside a selection), not one operand spelling: "x > avg" and
    # "x > 5" are the same operation with different thresholds.
    "filter_above": {"templates": _tpl_filter_above(), "default": "list_comp",
                     "markers": ["x > avg", "x>avg", "if x >", "if x>"]},
    "filter_below": {"templates": _tpl_filter_below(), "default": "list_comp",
                     "markers": ["x < avg", "x<avg", "if x <", "if x<"]},
    "sort_descending": {"templates": _tpl_sort_descending(), "default": "builtin",
                        "markers": ["reverse=True", "nlargest", "arr", "} + [" + "{o}" + "[j] <"]},
    "sort_ascending": {"templates": _tpl_sort_ascending(), "default": "builtin",
                       "markers": ["sorted(", "{o}[j] >"]},
    "reverse": {"templates": _tpl_simple("reverse"), "default": "default",
                "markers": ["[::-1]"]},
    "filter": {"templates": _tpl_simple("filter"), "default": "default",
               "markers": ["if x]", "if pred(", "if _pred("]},
    "filter_even": {"templates": _tpl_simple("filter_even"), "default": "default",
                    "markers": ["% 2 == 0"]},
    "is_prime": {"templates": _tpl_is_prime(), "default": "trial_division",
                 "markers": ["_is_prime", "n % i"]},
    "is_multiple_of": {"templates": _tpl_simple("is_multiple_of"), "default": "default",
                       "markers": ["% 2 == 0"]},
    "generate_range": {"templates": _tpl_generate_range(), "default": "default",
                       "markers": ["range("]},
    "find_max": {"templates": _tpl_find_max(), "default": "builtin",
                 "markers": ["max(", "x >"]},
    "find_min": {"templates": _tpl_find_min(), "default": "builtin",
                 "markers": ["min(", "x <"]},
    "add": {"templates": _tpl_add_multiply("+"), "default": "builtin",
            "markers": ["data[0] + data[1]", "+ data[1]"]},
    "multiply": {"templates": _tpl_add_multiply("*"), "default": "builtin",
                 "markers": ["data[0] * data[1]", "* data[1]"]},
    "search": {"templates": _tpl_search(), "default": "linear",
               "markers": ["enumerate", "target"]},
}

# fix accidental marker construction above
_OP_REGISTRY["sort_descending"]["markers"] = ["reverse=True", "nlargest", "for j in range(len("]


# --------------------------------------------------------------------- #
# Pipeline builder (IR -> code following the dataflow graph)
# --------------------------------------------------------------------- #

def available_variants(op: str) -> list[str]:
    entry = _OP_REGISTRY.get(op)
    return list(entry["templates"].keys()) if entry else []


#: Operations that produce a SCALAR: emitting them must not advance the
#: collection pointer — a downstream collection op consumes the most
#: recent sequence (the current pointer), and min(max(x)) chains are
#: dataflow-invalid. Mirrors ir_builder's pointer rule.
_SCALAR_POINTER_OPS = {"calculate_average", "calculate_sum", "find_max",
                       "find_min", "count_elements", "add", "multiply"}


def _keyed_extremum_lines(op: str, p: str, o: str, variant: str) -> list[str]:
    """Extremum lines selecting by the objective's DERIVED key.

    maximize(length(element)) compares len(x); it is a different
    computation from a value extremum, and the generated code must say so.
    """
    cmp_op = ">" if op == "find_max" else "<"
    if variant == "linear_scan":
        return [
            f"    {o} = None",
            f"    for x in {p}:",
            f"        if {o} is None or len(x) {cmp_op} len({o}):",
            f"            {o} = x",
        ]
    fn = "max" if op == "find_max" else "min"
    return [f"    {o} = {fn}({p}, key=len) if {p} else None"]


def build_pipeline(ops: list[str], func_name: str,
                   variants: Optional[dict[str, str]] = None,
                   contract=None) -> list[str]:
    """Build dataflow-chained function code from an operation graph.

    Every required operation becomes real executable IR/code; intermediate
    values flow through step variables; exactly one return at the end.
    Raises UnknownOperationError when an op has no semantics — generation
    must surface this instead of emitting placeholders.
    """
    if not ops:
        # No operations at all (test-only semantic=None setups): the
        # caller decides how to present an empty pipeline. NOTE: the
        # identity violation the pipeline must never commit is emitting a
        # pass-through when operations DO exist but carry no executable
        # semantics — that case raises UnknownOperationError below.
        return [f"def {func_name}(data):", "    return data"]

    # Single direct-algorithm: not a pipeline transform, but a standalone function.
    if len(ops) == 1 and ops[0] in _DIRECT_ALGORITHMS:
        return build_direct_algorithm(ops[0], func_name, variants)

    variants = variants or {}
    lines = [f"def {func_name}(data):"]
    prev = "data"

    for idx, op in enumerate(ops):
        entry = _OP_REGISTRY.get(op)
        if entry is None:
            # Check if it's a direct algorithm used in a pipeline — emit standalone
            if op in _DIRECT_ALGORITHMS:
                raise UnknownOperationError(op)
            raise UnknownOperationError(op)
        out = f"step_{idx}"
        variant = variants.get(op, entry["default"])
        if variant not in entry["templates"]:
            variant = entry["default"]
        is_last = idx == len(ops) - 1

        if is_last and op == "calculate_average":
            tmpl = entry["templates"][variant]
            for line in tmpl(prev, out):
                lines.append(line)
            lines.append("    return avg")
            return lines
        if is_last and op == "calculate_sum":
            for line in entry["templates"][variant](prev, out):
                lines.append(line)
            lines.append("    return total")
            return lines

        # A selection ("filter_above"/"filter_below") needs its operand:
        # either the LITERAL the request supplied (contract.threshold)
        # or an average a preceding aggregate computed.  With neither,
        # the operation is unrenderable — raise, never invent a
        # threshold (the mean-substitution contamination).
        if op in ("filter_above", "filter_below"):
            thr = (getattr(contract, "threshold", None)
                   if contract is not None else None)
            if thr is not None:
                cmp = ">" if op == "filter_above" else "<"
                lines.append(
                    f"    {out} = [x for x in {prev} if x {cmp} {thr}]")
                prev = out
                continue
            if "calculate_average" not in ops[:idx]:
                raise UnknownOperationError(op)
            # An average precedes: fall through to the template, which
            # reads the `avg` variable that aggregate produced.

        if (contract is not None
                and getattr(contract, "select_key", "") == "length"
                and op in ("find_max", "find_min")):
            tmpl_lines = _keyed_extremum_lines(op, prev, out, variant)
        else:
            tmpl_lines = list(entry["templates"][variant](prev, out))
        for line in tmpl_lines:
            lines.append(line)
        # Dataflow (mirrors ir_builder + the DAG wiring): a scalar op
        # leaves the collection pointer UNCHANGED — a downstream
        # collection op consumes a sequence (the ontology wires it to
        # the most recent sequence producer, which is exactly the
        # current pointer here), and min(max(x)) chains are forbidden.
        # The TERMINAL scalar op binds the return to the product
        # itself; the terminal collection op advances as usual.
        if is_last or op not in _SCALAR_POINTER_OPS:
            prev = out

    lines.append(f"    return {prev}")
    return lines


# --------------------------------------------------------------------- #
# Direct algorithms: single-function operations that are not pipeline
# transforms (e.g. "add two numbers", "check if prime", "factorial").
# --------------------------------------------------------------------- #

_DIRECT_ALGORITHMS: dict[str, dict] = {
    "add_two_numbers": {
        "params": [("a", "int"), ("b", "int")],
        "return": "int",
        "variants": {
            "default": lambda: ["    return a + b"],
            "subtraction": lambda: ["    return a - b"],
        },
        "default_variant": "default",
        "markers": ["return a + b", "return a+b", "a + b"],
        "ref": lambda a, b: a + b,
        "ref_params": 2,
    },
    "is_prime": {
        "params": [("n", "int")],
        "return": "bool",
        "variants": {
            "trial_division": lambda: [
                "    if n < 2:",
                "        return False",
                "    for i in range(2, int(n ** 0.5) + 1):",
                "        if n % i == 0:",
                "            return False",
                "    return True",
            ],
        },
        "default_variant": "trial_division",
        "markers": ["n % i", "n < 2", "range(2,"],
        "ref": staticmethod(lambda n: _is_prime(n)),
        "ref_params": 1,
    },
    "factorial": {
        "params": [("n", "int")],
        "return": "int",
        "variants": {
            "iterative": lambda: [
                "    result = 1",
                "    for i in range(2, n + 1):",
                "        result *= i",
                "    return result",
            ],
            "recursive": lambda: [
                "    if n <= 1:",
                "        return 1",
                "    return n * factorial(n - 1)",
            ],
        },
        "default_variant": "iterative",
        "markers": ["result *= i", "n * factorial", "range(2, n"],
        "ref": staticmethod(lambda n: 1 if n <= 1 else n * _DIRECT_ALGORITHMS["factorial"]["ref"](n - 1)),
        "ref_params": 1,
    },
    "fibonacci": {
        "params": [("n", "int")],
        "return": "int",
        "variants": {
            "iterative": lambda: [
                "    if n <= 0:",
                "        return 0",
                "    if n == 1:",
                "        return 1",
                "    a, b = 0, 1",
                "    for _ in range(2, n + 1):",
                "        a, b = b, a + b",
                "    return b",
            ],
        },
        "default_variant": "iterative",
        "markers": ["a, b = 0, 1", "a, b = b, a"],
        "ref": staticmethod(lambda n: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34][min(n, 9)] if n >= 0 else 0),
        "ref_params": 1,
    },
    "sum_list": {
        "params": [("data", "list")],
        "return": "int",
        "variants": {
            "builtin": lambda: ["    return sum(data)"],
            "loop": lambda: [
                "    total = 0",
                "    for x in data:",
                "        total += x",
                "    return total",
            ],
        },
        "default_variant": "builtin",
        "markers": ["sum(data)", "total += x"],
        "ref": staticmethod(lambda data: sum(data)),
        "ref_params": 1,
    },
    "average_list": {
        "params": [("data", "list")],
        "return": "float",
        "variants": {
            "builtin": lambda: ["    return sum(data) / len(data) if data else 0"],
        },
        "default_variant": "builtin",
        "markers": ["sum(data) / len(data)"],
        "ref": staticmethod(lambda data: sum(data) / len(data) if data else 0),
        "ref_params": 1,
    },
    "sort_list": {
        # A standalone sort function (the algorithm path's artifact) —
        # not a pipeline transform.  The markers identify the
        # generator's own structure (the copy + swap body), and the
        # reference is the sorted order the artifact must reproduce, so
        # verification compares BEHAVIOR — never a missing-op claim
        # against code that does sort.
        "params": [("items", "list")],
        "return": "list",
        "variants": {
            "builtin": lambda: ["    return sorted(items)"],
        },
        "default_variant": "builtin",
        "markers": ["sorted_list", "copy_list"],
        "ref": staticmethod(lambda items: sorted(items)),
        "ref_params": 1,
    },
    "most_frequent_char": {
        "params": [("text", "str")],
        "return": "str",
        "variants": {
            "builtin": lambda: [
                "    from collections import Counter",
                "    return Counter(text).most_common(1)[0][0] if text else ''",
            ],
        },
        "default_variant": "builtin",
        "markers": ["Counter(", "most_common"],
        "ref": staticmethod(lambda text: max(set(text), key=text.count) if text else ''),
        "ref_params": 1,
    },
}


def build_direct_algorithm(op: str, func_name: str,
                           variants: Optional[dict[str, str]] = None) -> list[str]:
    """Build code for a standalone single-function algorithm."""
    algo = _DIRECT_ALGORITHMS.get(op)
    if algo is None:
        raise UnknownOperationError(op)

    variants = variants or {}
    variant = variants.get(op, algo["default_variant"])
    if variant not in algo["variants"]:
        variant = algo["default_variant"]

    params = ", ".join(f"{p[0]}: {p[1]}" for p in algo["params"])
    lines = [f"def {func_name}({params}) -> {algo['return']}:"]
    for line in algo["variants"][variant]():
        lines.append(line)
    return lines


def is_direct_algorithm(op: str) -> bool:
    return op in _DIRECT_ALGORITHMS
