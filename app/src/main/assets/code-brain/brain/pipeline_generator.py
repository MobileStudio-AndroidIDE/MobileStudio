"""Multi-language pipeline code generator.

Generates equivalent code for all 10 target languages from an operation
pipeline. Each operation is mapped to language-specific implementations
using proper data structures and syntax for that language.

Dataflow contract (the invariant the renderer must preserve):

    Semantic Output Type == IR Output Type
        == Generated Function Return Type == Verified Behavioral Type

Every operation CONSUMES a value and PRODUCES a value.  The function's
return binds to the TERMINAL operation's output — never to the input.
An identity assignment (`step = data; return step`) is not an
implementation of an operation: it is forbidden, and an operation the
renderer has no template for is an honest failure (ValueError), never a
silent passthrough that returns the input unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Type mappings per language
TYPE_MAP = {
    "python": {"list": "List[float]", "float": "float", "int": "int", "bool": "bool", "str": "str"},
    "kotlin": {"list": "List<Double>", "float": "Double", "int": "Int", "bool": "Boolean", "str": "String"},
    "java": {"list": "List<Double>", "float": "double", "int": "int", "bool": "boolean", "str": "String"},
    "c": {"list": "double*", "float": "double", "int": "int", "bool": "int", "str": "char*"},
    "cpp": {"list": "std::vector<double>", "float": "double", "int": "int", "bool": "bool", "str": "std::string"},
    "javascript": {"list": "number[]", "float": "number", "int": "number", "bool": "boolean", "str": "string"},
    "typescript": {"list": "number[]", "float": "number", "int": "number", "bool": "boolean", "str": "string"},
    "go": {"list": "[]float64", "float": "float64", "int": "int", "bool": "bool", "str": "string"},
    "rust": {"list": "Vec<f64>", "float": "f64", "int": "i32", "bool": "bool", "str": "String"},
    "dart": {"list": "List<double>", "float": "double", "int": "int", "bool": "bool", "str": "String"},
}


def _get_type(lang: str, base_type: str) -> str:
    """Get language-specific type."""
    return TYPE_MAP.get(lang, TYPE_MAP["python"]).get(base_type, base_type)


# The operation dataflow contract: the TYPE each SCALAR-producing
# operation's output carries.  Every operation not listed here produces
# a collection (the transformed sequence).
_SCALAR_OUTPUT_TYPES = {
    "calculate_average": "float",
    "calculate_sum": "float",
    "find_max": "float",
    "find_min": "float",
    "add": "float",
    "multiply": "float",
    "count_elements": "int",
    "fibonacci": "int",
    "search": "int",
}


def _pipeline_return_type(ops: list) -> str:
    """The pipeline's OUTPUT type: the terminal operation's contract.

    `calculate_average` as the last operation means the function returns
    a Number, not the input sequence — the return TYPE follows the
    dataflow, never a blanket "list" annotation.
    """
    if not ops:
        return "list"
    return _SCALAR_OUTPUT_TYPES.get(ops[-1], "list")


def _pipeline_elem_type(ops: list) -> str:
    """The element domain the pipeline operates over — from the OPERATION
    VOCABULARY, never from text sniffing.

    Parity predicates (even/odd) are defined only over integers, so a
    pipeline containing filter_even / is_multiple_of consumes and
    yields ints; every other pipeline stays with the generic Number
    domain (float). Mirrors ir_builder.build_ir's element rule so the
    IR's type_ref and the rendered signature agree.
    """
    if any(op in ("filter_even", "is_multiple_of") for op in ops):
        return "int"
    return "float"


class PipelineCodeGenerator:
    """Generates code for all 8 target languages from an operation pipeline."""

    def __init__(self, contract=None):
        """`contract` (brain.generation.contract.PipelineContract) carries the
        declared element domain and the objective's select key.  With no
        contract the generator derives types from the operation vocabulary
        exactly as before — the contract only REPLACES a derivation when
        the semantics actually declared something."""
        self.contract = contract

    def _elem(self, ops: list) -> str:
        """The element domain: the DECLARED element when known."""
        declared = (getattr(self.contract, "elem_type", "") or "") \
            if self.contract is not None else ""
        if declared:
            return declared
        return _pipeline_elem_type(ops)

    def _keyed(self) -> bool:
        """Does the extremum select by a derived key (e.g. string length)?"""
        return bool(self.contract is not None
                    and getattr(self.contract, "keyed", False))

    def _declared_elem(self) -> str:
        """The declared element domain ("" when the semantics declared none)."""
        if self.contract is None:
            return ""
        return (getattr(self.contract, "elem_type", "") or "")

    # Languages with a template for a string element AND for an extremum
    # that selects by a derived key.  Every other language renders the
    # numeric domain only.
    _STR_DOMAIN_LANGS = {"python", "kotlin", "javascript",
                         "typescript", "dart"}

    def _require_support(self, language: str) -> None:
        """Refuse to render a domain this language has no template for.

        A language that cannot express the declared element domain or the
        select key is ABSENT from the result (the module's honest-absence
        contract) — it is never rendered as a value comparison, which
        would be wrong code rather than no code.
        """
        if self.contract is None:
            return
        declared = (getattr(self.contract, "elem_type", "") or "")
        if not self._keyed() and declared != "str":
            return  # numeric domain: every language renders, as before
        if language not in self._STR_DOMAIN_LANGS:
            raise ValueError(
                f"element domain '{declared or 'keyed'}' "
                f"has no template in {language}")

    def generate_all(self, func_name: str, ops: list[str],
                     language: str = "python") -> dict[str, dict]:
        """Generate code for all languages, EACH INDEPENDENTLY.

        A language with no template for an operation is ABSENT from the
        result — an honest absence.  It must not render as an identity
        passthrough, and it must not drag the languages that CAN render
        the pipeline down with it (the renderer's per-language failure
        is never a reason to fall back to a different pipeline's code).
        """
        renderers = (
            ("python", self._gen_python),
            ("kotlin", self._gen_kotlin),
            ("java", self._gen_java),
            ("c", self._gen_c),
            ("cpp", self._gen_cpp),
            ("javascript", self._gen_javascript),
            ("typescript", self._gen_typescript),
            ("dart", self._gen_dart),
            ("go", self._gen_go),
            ("rust", self._gen_rust),
        )
        out: dict[str, dict] = {}
        for lang, render in renderers:
            try:
                out[lang] = {"code": render(func_name, ops)}
            except ValueError:
                # No honest template for this operation in this
                # language: the language is absent, not wrong.
                continue
        return out

    # ------------------------------------------------------------------ #
    # IR-driven generation: the COMPLETE structural IR is the input
    # ------------------------------------------------------------------ #
    def generate_from_ir(self, ir) -> dict[str, dict]:
        """Generate code for all languages from the structural IR.

        The IR (an IRNode FUNCTION tree) is the source of truth: the
        operation sequence is read from the IR's op-marked nodes, then
        rendered with each language's operator semantics. Deterministic —
        no parser results, no keyword re-matching.
        """
        ops, func_name = self._ops_from_ir(ir)
        return self.generate_all(func_name, ops)

    def _ops_from_ir(self, ir) -> tuple[list[str], str]:
        """Extract the operation sequence and function name from the IR.

        The ir_builder marks each operation's top-level node with an
        "op" property; the op sequence is the order of those nodes.
        """
        from code.ir.ir_types import IRNodeType
        func_name = "process_data"
        ops: list[str] = []

        if getattr(ir, "node_type", None) == IRNodeType.FUNCTION:
            func_name = ir.name or func_name

        def _walk(node):
            op = node.properties.get("op") if node.properties else None
            if op:
                ops.append(op)
            for child in node.children:
                _walk(child)

        _walk(ir)
        return ops, func_name

    # ------------------------------------------------------------------ #
    # Python
    # ------------------------------------------------------------------ #
    def _gen_python(self, name: str, ops: list[str]) -> str:
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # sum / extremum over the integer domain yields an int — the
        # signature, the IR type_ref and the behavioral output agree.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        lines = ["from typing import List", "", ""]
        lines.append(f"def {name}(data: List[{elem}]) -> "
                     f"{_get_type('python', ret_type)}:")
        lines.append('    """')
        lines.append("    Process data through operation pipeline:")
        for i, op in enumerate(ops):
            lines.append(f"        {i+1}. {op.replace('_', ' ')}")
        lines.append('    """')

        var = "data"       # the collection the next op consumes
        ret = "data"       # the TERMINAL value: the last op's output
        for i, op in enumerate(ops):
            nxt = f"step_{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"    seen = set()",
                    f"    {nxt} = []",
                    f"    for x in {var}:",
                    f"        if x not in seen:",
                    f"            seen.add(x)",
                    f"            {nxt}.append(x)",
                ])
                ret = nxt
            elif op == "calculate_average":
                # A SCALAR op: it consumes the collection and produces
                # the average.  The collection pointer stays (a
                # downstream filter may consume both); the terminal
                # value is the scalar itself — never the input.
                lines.append(f"    avg_{i+1} = sum({var}) / len({var}) if {var} else 0")
                ret = f"avg_{i+1}"
            elif op == "filter_above":
                avg = self._find_avg_var(ops, i)
                lines.append(f"    {nxt} = [x for x in {var} if x > {avg}]")
                ret = nxt
            elif op == "filter_below":
                avg = self._find_avg_var(ops, i)
                lines.append(f"    {nxt} = [x for x in {var} if x < {avg}]")
                ret = nxt
            elif op == "sort_descending":
                lines.append(f"    {nxt} = sorted({var}, reverse=True)")
                ret = nxt
            elif op == "sort_ascending":
                lines.append(f"    {nxt} = sorted({var})")
                ret = nxt
            elif op == "filter_even":
                lines.append(f"    {nxt} = [x for x in {var} if x % 2 == 0]")
                ret = nxt
            elif op == "calculate_sum":
                lines.append(f"    {nxt} = sum({var})")
                ret = nxt
            elif op in ("add", "multiply"):
                # The operation contract (the reference semantics
                # decide, never the renderer): two operands → their
                # combination; one → that value; none → 0.
                sym = "+" if op == "add" else "*"
                lines.append(f"    {nxt} = {var}[0] {sym} {var}[1] "
                             f"if len({var}) >= 2 "
                             f"else ({var}[0] if {var} else 0)")
                ret = nxt
            elif op == "find_max":
                if self._keyed():
                    # Selecting by a DERIVED length: the extremum is over
                    # len(element), not over the element's own ordering.
                    lines.append(f"    {nxt} = max({var}, key=len) if {var} else None")
                else:
                    lines.append(f"    {nxt} = max({var})")
                ret = nxt
            elif op == "find_min":
                if self._keyed():
                    lines.append(f"    {nxt} = min({var}, key=len) if {var} else None")
                else:
                    lines.append(f"    {nxt} = min({var})")
                ret = nxt
            elif op == "count_elements":
                lines.append(f"    {nxt} = len({var})")
                ret = nxt
            elif op in ("reverse_sequence", "reverse"):
                lines.append(f"    {nxt} = {var}[::-1]")
                ret = nxt
            elif op == "fibonacci":
                # iterative fibonacci: two rolling variables
                lines.extend([
                    "    a, b = 0, 1",
                    f"    for _ in range({var}):",
                    "        a, b = b, a + b",
                ])
                ret = "a"
            elif op == "search":
                # linear search: first index whose element equals the target
                lines.extend([
                    f"    {nxt} = -1",
                    f"    for i, x in enumerate({var}):",
                    "        if x == target:",
                    "            " + f"{nxt} = i",
                    "            break",
                ])
                ret = nxt
            else:
                raise ValueError(f"No pipeline template for operation '{op}'")
            # A collection op advances the collection pointer; a scalar
            # op leaves it for the next consumer.
            if op not in ("calculate_average", "fibonacci"):
                var = nxt

        lines.append(f"    return {ret}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Kotlin
    # ------------------------------------------------------------------ #
    def _gen_kotlin(self, name: str, ops: list[str]) -> str:
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python: a terminal sum/extremum over the element
        # domain yields THAT element type — the signature follows the
        # dataflow, never a blanket Double.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        ek = _get_type("kotlin", elem)          # Int / Double / String
        zero = "0" if elem == "int" else ('""' if elem == "str" else "0.0")
        if ret_type == "list":
            ret_k = f"List<{ek}>"
        else:
            ret_k = _get_type("kotlin", ret_type)
        lines = [f"fun {name}(data: List<{ek}>): {ret_k} {{"]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"    val seen = mutableSetOf<{ek}>()",
                    f"    val {nxt} = mutableListOf<{ek}>()",
                    f"    for (x in {var}) {{",
                    f"        if (x !in seen) {{",
                    f"            seen.add(x)",
                    f"            {nxt}.add(x)",
                    f"        }}",
                    f"    }}",
                ])
            elif op == "calculate_average":
                lines.append(f"    val avg{i+1} = if ({var}.isNotEmpty()) {var}.average() else 0.0")
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.append(f"    val {nxt} = {var}.sum()")
            elif op == "find_max":
                if self._keyed():
                    lines.append(f"    val {nxt} = {var}.maxByOrNull {{ it.length }} ?: \"\"")
                else:
                    lines.append(f"    val {nxt} = {var}.maxOrNull() ?: {zero}")
            elif op == "find_min":
                if self._keyed():
                    lines.append(f"    val {nxt} = {var}.minByOrNull {{ it.length }} ?: \"\"")
                else:
                    lines.append(f"    val {nxt} = {var}.minOrNull() ?: {zero}")
            elif op == "count_elements":
                lines.append(f"    val {nxt} = {var}.size")
            elif op == "filter_above":
                avg = self._find_avg_var_kotlin(ops, i)
                lines.append(f"    val {nxt} = {var}.filter {{ it > {avg} }}")
            elif op == "filter_below":
                avg = self._find_avg_var_kotlin(ops, i)
                lines.append(f"    val {nxt} = {var}.filter {{ it < {avg} }}")
            elif op == "sort_descending":
                lines.append(f"    val {nxt} = {var}.sortedDescending()")
            elif op == "sort_ascending":
                lines.append(f"    val {nxt} = {var}.sorted()")
            elif op == "filter_even":
                if elem == "int":
                    lines.append(f"    val {nxt} = {var}.filter {{ it % 2 == 0 }}")
                else:
                    lines.append(f"    val {nxt} = {var}.filter {{ it % 2.0 == 0.0 }}")
            elif op in ("reverse_sequence", "reverse"):
                lines.append(f"    val {nxt} = {var}.reversed()")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (kotlin)")
            ret = nxt
            var = nxt

        lines.append(f"    return {ret}")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Java
    # ------------------------------------------------------------------ #
    def _gen_java(self, name: str, ops: list[str]) -> str:
        self._require_support("java")
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python: a terminal sum/extremum over the element
        # domain yields THAT element type — the signature follows the
        # dataflow, never a blanket Double.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        jek = "Integer" if elem == "int" else "Double"   # boxed element
        zero = "0" if elem == "int" else "0.0"
        class_name = name.capitalize() if name.islower() else name
        if ret_type == "list":
            rt_ann = f"List<{jek}>"
        else:
            rt_ann = _get_type('java', ret_type)
        lines = [
            "import java.util.*;",
            "",
            f"public class {class_name} {{",
            f"    public static {rt_ann} {name}(List<{jek}> data) {{",
        ]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"        LinkedHashSet<{jek}> seen = new LinkedHashSet<>({var});",
                    f"        List<{jek}> {nxt} = new ArrayList<>(seen);",
                ])
            elif op == "calculate_average":
                lines.append(f"        double avg{i+1} = {var}.stream().mapToDouble(x -> x.doubleValue()).average().orElse(0.0);")
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                if elem == "int":
                    lines.append(f"        int {nxt} = {var}.stream().mapToInt(x -> x.intValue()).sum();")
                else:
                    lines.append(f"        double {nxt} = {var}.stream().mapToDouble(x -> x.doubleValue()).sum();")
            elif op == "find_max":
                if elem == "int":
                    lines.append(f"        int {nxt} = {var}.stream().mapToInt(x -> x.intValue()).max().orElse({zero});")
                else:
                    lines.append(f"        double {nxt} = {var}.stream().mapToDouble(x -> x.doubleValue()).max().orElse({zero});")
            elif op == "find_min":
                if elem == "int":
                    lines.append(f"        int {nxt} = {var}.stream().mapToInt(x -> x.intValue()).min().orElse({zero});")
                else:
                    lines.append(f"        double {nxt} = {var}.stream().mapToDouble(x -> x.doubleValue()).min().orElse({zero});")
            elif op == "count_elements":
                lines.append(f"        int {nxt} = {var}.size();")
            elif op == "filter_above":
                avg = self._find_avg_var_java(ops, i)
                lines.extend([
                    f"        List<{jek}> {nxt} = new ArrayList<>();",
                    f"        for ({jek} x : {var}) {{",
                    f"            if (x.doubleValue() > {avg}) {nxt}.add(x);",
                    f"        }}",
                ])
            elif op == "filter_below":
                avg = self._find_avg_var_java(ops, i)
                lines.extend([
                    f"        List<{jek}> {nxt} = new ArrayList<>();",
                    f"        for ({jek} x : {var}) {{",
                    f"            if (x.doubleValue() < {avg}) {nxt}.add(x);",
                    f"        }}",
                ])
            elif op == "sort_descending":
                lines.extend([
                    f"        List<{jek}> {nxt} = new ArrayList<>({var});",
                    f"        {nxt}.sort(Collections.reverseOrder());",
                ])
            elif op == "sort_ascending":
                lines.extend([
                    f"        List<{jek}> {nxt} = new ArrayList<>({var});",
                    f"        Collections.sort({nxt});",
                ])
            elif op == "filter_even":
                if elem == "int":
                    lines.extend([
                        f"        List<{jek}> {nxt} = new ArrayList<>();",
                        f"        for ({jek} x : {var}) {{",
                        f"            if (x % 2 == 0) {nxt}.add(x);",
                        f"        }}",
                    ])
                else:
                    lines.extend([
                        f"        List<{jek}> {nxt} = new ArrayList<>();",
                        f"        for ({jek} x : {var}) {{",
                        f"            if (x % 2.0 == 0.0) {nxt}.add(x);",
                        f"        }}",
                    ])
            elif op in ("reverse_sequence", "reverse"):
                lines.extend([
                    f"        List<{jek}> {nxt} = new ArrayList<>({var});",
                    f"        Collections.reverse({nxt});",
                ])
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (java)")
            ret = nxt
            var = nxt

        lines.append(f"        return {ret};")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # C
    # ------------------------------------------------------------------ #
    def _gen_c(self, name: str, ops: list[str]) -> str:
        self._require_support("c")
        ret_type = _pipeline_return_type(ops)
        if ret_type == "list":
            lines = [
                "#include <stdlib.h>",
                "#include <math.h>",
                "",
                f"double* {name}(double* data, int data_len, int* out_len) {{",
                f"    *out_len = data_len;",
                f"    double* result = malloc(data_len * sizeof(double));",
                f"    for (int i = 0; i < data_len; i++) result[i] = data[i];",
            ]
        else:
            lines = [
                "#include <stdlib.h>",
                "#include <math.h>",
                "",
                f"{_get_type('c', ret_type)} {name}(double* data, int data_len) {{",
                f"    double* result = malloc(data_len * sizeof(double));",
                f"    for (int i = 0; i < data_len; i++) result[i] = data[i];",
                f"    int out_len = data_len;",
            ]

        ret_expr = "result"
        for i, op in enumerate(ops):
            if op == "remove_duplicates":
                lines.extend([
                    f"    // Remove duplicates",
                    f"    int new_len = 0;",
                    f"    for (int i = 0; i < out_len; i++) {{",
                    f"        int found = 0;",
                    f"        for (int j = 0; j < new_len; j++) {{",
                    f"            if (fabs(result[i] - result[j]) < 1e-9) {{ found = 1; break; }}",
                    f"        }}",
                    f"        if (!found) result[new_len++] = result[i];",
                    f"    }}",
                    f"    out_len = new_len;",
                ])
                ret_expr = "result"
            elif op == "calculate_average":
                lines.extend([
                    f"    // Calculate average",
                    f"    double avg{i+1} = 0;",
                    f"    for (int i = 0; i < out_len; i++) avg{i+1} += result[i];",
                    f"    avg{i+1} /= (out_len > 0) ? out_len : 1;",
                ])
                ret_expr = f"avg{i+1}"
            elif op == "calculate_sum":
                lines.extend([
                    f"    double sumv{i+1} = 0;",
                    f"    for (int i = 0; i < out_len; i++) sumv{i+1} += result[i];",
                ])
                ret_expr = f"sumv{i+1}"
            elif op == "find_max":
                lines.extend([
                    f"    double mx{i+1} = (out_len > 0) ? result[0] : 0;",
                    f"    for (int i = 1; i < out_len; i++) if (result[i] > mx{i+1}) mx{i+1} = result[i];",
                ])
                ret_expr = f"mx{i+1}"
            elif op == "find_min":
                lines.extend([
                    f"    double mn{i+1} = (out_len > 0) ? result[0] : 0;",
                    f"    for (int i = 1; i < out_len; i++) if (result[i] < mn{i+1}) mn{i+1} = result[i];",
                ])
                ret_expr = f"mn{i+1}"
            elif op == "count_elements":
                ret_expr = "out_len"
            elif op == "filter_above":
                avg = f"avg{self._find_avg_idx(ops, i)+1}"
                lines.extend([
                    f"    // Filter above average",
                    f"    int new_len = 0;",
                    f"    for (int i = 0; i < out_len; i++) {{",
                    f"        if (result[i] > {avg}) result[new_len++] = result[i];",
                    f"    }}",
                    f"    out_len = new_len;",
                ])
                ret_expr = "result"
            elif op == "filter_below":
                avg = f"avg{self._find_avg_idx(ops, i)+1}"
                lines.extend([
                    f"    // Filter below average",
                    f"    int new_len = 0;",
                    f"    for (int i = 0; i < out_len; i++) {{",
                    f"        if (result[i] < {avg}) result[new_len++] = result[i];",
                    f"    }}",
                    f"    out_len = new_len;",
                ])
                ret_expr = "result"
            elif op == "sort_descending":
                lines.extend([
                    f"    // Sort descending",
                    f"    for (int i = 0; i < out_len - 1; i++) {{",
                    f"        for (int j = i + 1; j < out_len; j++) {{",
                    f"            if (result[i] < result[j]) {{",
                    f"                double tmp = result[i]; result[i] = result[j]; result[j] = tmp;",
                    f"            }}",
                    f"        }}",
                    f"    }}",
                ])
                ret_expr = "result"
            elif op == "sort_ascending":
                lines.extend([
                    f"    // Sort ascending",
                    f"    for (int i = 0; i < out_len - 1; i++) {{",
                    f"        for (int j = i + 1; j < out_len; j++) {{",
                    f"            if (result[i] > result[j]) {{",
                    f"                double tmp = result[i]; result[i] = result[j]; result[j] = tmp;",
                    f"            }}",
                    f"        }}",
                    f"    }}",
                ])
                ret_expr = "result"
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (c)")

        lines.append(f"    return {ret_expr};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # C++
    # ------------------------------------------------------------------ #
    def _gen_cpp(self, name: str, ops: list[str]) -> str:
        self._require_support("cpp")
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python: terminal sum/extremum yields the element
        # type; every container, literal and lambda follows the same
        # element domain — never a blanket double.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        ct = "int" if elem == "int" else "double"
        vc = f"std::vector<{ct}>"
        if ret_type == "list":
            ret_c = vc
        else:
            ret_c = _get_type("cpp", ret_type)
        lines = [
            "#include <vector>",
            "#include <algorithm>",
            "#include <numeric>",
            "#include <cmath>",
            "",
            f"{ret_c} {name}({vc} data) {{",
        ]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                if ct == "int":
                    pred = f"if (x == y)"
                else:
                    pred = f"if (std::abs(x - y) < 1e-9)"
                lines.extend([
                    f"    {vc} {nxt};",
                    f"    for (const auto& x : {var}) {{",
                    f"        bool found = false;",
                    f"        for (const auto& y : {nxt}) {{",
                    f"            {pred} {{ found = true; break; }}",
                    f"        }}",
                    f"        if (!found) {nxt}.push_back(x);",
                    f"    }}",
                ])
            elif op == "calculate_average":
                lines.extend([
                    f"    double avg{i+1} = {var}.empty() ? 0.0 :",
                    f"        std::accumulate({var}.begin(), {var}.end(), 0.0) / {var}.size();",
                ])
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                if ct == "int":
                    lines.append(f"    int {nxt} = std::accumulate({var}.begin(), {var}.end(), 0);")
                else:
                    lines.append(f"    double {nxt} = std::accumulate({var}.begin(), {var}.end(), 0.0);")
            elif op == "find_max":
                lines.append(f"    {ct} {nxt} = *std::max_element({var}.begin(), {var}.end());")
            elif op == "find_min":
                lines.append(f"    {ct} {nxt} = *std::min_element({var}.begin(), {var}.end());")
            elif op == "count_elements":
                lines.append(f"    int {nxt} = static_cast<int>({var}.size());")
            elif op == "filter_above":
                avg = self._find_avg_var_cpp(ops, i)
                lines.extend([
                    f"    {vc} {nxt};",
                    f"    std::copy_if({var}.begin(), {var}.end(), std::back_inserter({nxt}),",
                    f"        [&]({ct} x) {{ return x > {avg}; }});",
                ])
            elif op == "filter_below":
                avg = self._find_avg_var_cpp(ops, i)
                lines.extend([
                    f"    {vc} {nxt};",
                    f"    std::copy_if({var}.begin(), {var}.end(), std::back_inserter({nxt}),",
                    f"        [&]({ct} x) {{ return x < {avg}; }});",
                ])
            elif op == "sort_descending":
                lines.extend([
                    f"    {vc} {nxt} = {var};",
                    f"    std::sort({nxt}.begin(), {nxt}.end(), std::greater<{ct}>());",
                ])
            elif op == "sort_ascending":
                lines.extend([
                    f"    {vc} {nxt} = {var};",
                    f"    std::sort({nxt}.begin(), {nxt}.end());",
                ])
            elif op == "filter_even":
                lines.append(f"    {vc} {nxt};")
                lines.append(f"    std::copy_if({var}.begin(), {var}.end(), std::back_inserter({nxt}),")
                if ct == "int":
                    lines.append(f"        [&]({ct} x) {{ return x % 2 == 0; }});")
                else:
                    lines.append(f"        [&]({ct} x) {{ return std::fmod(x, 2.0) == 0.0; }});")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (cpp)")
            ret = nxt
            var = nxt

        lines.append(f"    return {ret};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # JavaScript
    # ------------------------------------------------------------------ #
    def _gen_javascript(self, name: str, ops: list[str]) -> str:
        lines = [f"function {name}(data) {{"]
        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.append(f"    const {nxt} = [...new Set({var})];")
            elif op == "calculate_average":
                lines.append(f"    const avg{i+1} = {var}.length > 0 ? {var}.reduce((a, b) => a + b, 0) / {var}.length : 0;")
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.append(f"    const {nxt} = {var}.reduce((a, b) => a + b, 0);")
            elif op == "find_max":
                if self._keyed():
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a.length > b.length ? a : b) : \"\";")
                elif self._declared_elem() == "str":
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a > b ? a : b) : \"\";")
                else:
                    lines.append(f"    const {nxt} = Math.max(...{var});")
            elif op == "find_min":
                if self._keyed():
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a.length < b.length ? a : b) : \"\";")
                elif self._declared_elem() == "str":
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a < b ? a : b) : \"\";")
                else:
                    lines.append(f"    const {nxt} = Math.min(...{var});")
            elif op == "count_elements":
                lines.append(f"    const {nxt} = {var}.length;")
            elif op == "filter_above":
                avg = self._find_avg_var_js(ops, i)
                lines.append(f"    const {nxt} = {var}.filter(x => x > {avg});")
            elif op == "filter_below":
                avg = self._find_avg_var_js(ops, i)
                lines.append(f"    const {nxt} = {var}.filter(x => x < {avg});")
            elif op == "sort_descending":
                lines.append(f"    const {nxt} = [...{var}].sort((a, b) => b - a);")
            elif op == "sort_ascending":
                lines.append(f"    const {nxt} = [...{var}].sort((a, b) => a - b);")
            elif op == "filter_even":
                lines.append(f"    const {nxt} = {var}.filter(x => x % 2 === 0);")
            elif op in ("reverse_sequence", "reverse"):
                lines.append(f"    const {nxt} = [...{var}].reverse();")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (javascript)")
            ret = nxt
            var = nxt

        lines.append(f"    return {ret};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # TypeScript
    # ------------------------------------------------------------------ #
    def _gen_typescript(self, name: str, ops: list[str]) -> str:
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python/kotlin: terminal sum/extremum yields the
        # element domain's scalar — the signature follows the dataflow.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        arr_t = f"{_get_type('typescript', elem)}[]"
        if ret_type == "list":
            ret_t = arr_t
        else:
            ret_t = _get_type("typescript", ret_type)
        lines = [f"function {name}(data: {arr_t}): {ret_t} {{"]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.append(f"    const {nxt} = [...new Set({var})];")
            elif op == "calculate_average":
                lines.append(
                    f"    const avg{i+1} = {var}.length > 0 "
                    f"? {var}.reduce((a, b) => a + b, 0) / {var}.length : 0;"
                )
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.append(f"    const {nxt} = {var}.reduce((a, b) => a + b, 0);")
            elif op == "find_max":
                if self._keyed():
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a.length > b.length ? a : b) : \"\";")
                elif self._declared_elem() == "str":
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a > b ? a : b) : \"\";")
                else:
                    lines.append(f"    const {nxt} = Math.max(...{var});")
            elif op == "find_min":
                if self._keyed():
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a.length < b.length ? a : b) : \"\";")
                elif self._declared_elem() == "str":
                    lines.append(f"    const {nxt} = {var}.length ? "
                                 f"{var}.reduce((a, b) => a < b ? a : b) : \"\";")
                else:
                    lines.append(f"    const {nxt} = Math.min(...{var});")
            elif op == "count_elements":
                lines.append(f"    const {nxt} = {var}.length;")
            elif op == "filter_above":
                avg = self._find_avg_var_js(ops, i)
                lines.append(f"    const {nxt} = {var}.filter(x => x > {avg});")
            elif op == "filter_below":
                avg = self._find_avg_var_js(ops, i)
                lines.append(f"    const {nxt} = {var}.filter(x => x < {avg});")
            elif op == "sort_descending":
                lines.append(f"    const {nxt} = [...{var}].sort((a, b) => b - a);")
            elif op == "sort_ascending":
                lines.append(f"    const {nxt} = [...{var}].sort((a, b) => a - b);")
            elif op == "filter_even":
                lines.append(f"    const {nxt} = {var}.filter(x => x % 2 === 0);")
            elif op in ("reverse_sequence", "reverse"):
                lines.append(f"    const {nxt} = [...{var}].reverse();")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (typescript)")
            ret = nxt
            var = nxt

        lines.append(f"    return {ret};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Dart
    # ------------------------------------------------------------------ #
    def _gen_dart(self, name: str, ops: list[str]) -> str:
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python/kotlin/cpp: terminal sum/extremum yields
        # the element type; containers and lambdas follow the same
        # element domain — never a blanket double.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        de = _get_type("dart", elem)              # int / double / String
        zero = "0" if elem == "int" else ('""' if elem == "str" else "0.0")
        if ret_type == "list":
            ret_d = f"List<{de}>"
        else:
            ret_d = _get_type("dart", ret_type)
        lines = [f"{ret_d} {name}(List<{de}> data) {{"]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"  final seen = <{de}>{{}};",
                    f"  final {nxt} = <{de}>[];",
                    f"  for (final x in {var}) {{",
                    f"    if (seen.add(x)) {{",
                    f"      {nxt}.add(x);",
                    f"    }}",
                    f"  }}",
                ])
            elif op == "calculate_average":
                lines.append(
                    f"  final avg{i+1} = {var}.isNotEmpty "
                    f"? {var}.reduce((a, b) => a + b) / {var}.length : 0.0;"
                )
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.append(f"  final {nxt} = {var}.fold<{de}>({zero}, (a, b) => a + b);")
            elif op == "find_max":
                if self._keyed():
                    lines.append(
                        f"  final {nxt} = {var}.isEmpty ? \"\" "
                        f": {var}.reduce((a, b) => a.length > b.length ? a : b);"
                    )
                else:
                    lines.append(
                        f"  final {nxt} = {var}.isEmpty ? {zero} "
                        f": {var}.reduce((a, b) => a > b ? a : b);"
                    )
            elif op == "find_min":
                if self._keyed():
                    lines.append(
                        f"  final {nxt} = {var}.isEmpty ? \"\" "
                        f": {var}.reduce((a, b) => a.length < b.length ? a : b);"
                    )
                else:
                    lines.append(
                        f"  final {nxt} = {var}.isEmpty ? {zero} "
                        f": {var}.reduce((a, b) => a < b ? a : b);"
                    )
            elif op == "count_elements":
                lines.append(f"  final {nxt} = {var}.length;")
            elif op == "filter_above":
                avg = self._find_avg_var_dart(ops, i)
                lines.append(f"  final {nxt} = {var}.where((x) => x > {avg}).toList();")
            elif op == "filter_below":
                avg = self._find_avg_var_dart(ops, i)
                lines.append(f"  final {nxt} = {var}.where((x) => x < {avg}).toList();")
            elif op == "sort_descending":
                lines.append(f"  final {nxt} = [...{var}]..sort((a, b) => b.compareTo(a));")
            elif op == "sort_ascending":
                lines.append(f"  final {nxt} = [...{var}]..sort((a, b) => a.compareTo(b));")
            elif op == "filter_even":
                if elem == "int":
                    lines.append(f"  final {nxt} = {var}.where((x) => x % 2 == 0).toList();")
                else:
                    lines.append(f"  final {nxt} = {var}.where((x) => x % 2.0 == 0.0).toList();")
            elif op in ("reverse_sequence", "reverse"):
                lines.append(f"  final {nxt} = {var}.reversed.toList();")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (dart)")
            ret = nxt
            var = nxt

        lines.append(f"  return {ret};")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Go
    # ------------------------------------------------------------------ #
    def _gen_go(self, name: str, ops: list[str]) -> str:
        self._require_support("go")
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python: a terminal sum/extremum over the element
        # domain yields THAT element type — the signature follows the
        # dataflow, never a blanket float64.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        gk = "int" if elem == "int" else "float64"
        zero = "0" if elem == "int" else "0.0"
        if ret_type == "list":
            sig_ret = f"[]{gk}"
        else:
            sig_ret = _get_type('go', ret_type)
        lines = [
            "package main",
            "",
            "import (",
            '    "sort"',
            ")",
            "",
            f"func {name}(data []{gk}) {sig_ret} {{",
        ]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"    seen := make(map[{gk}]bool)",
                    f"    {nxt} := make([]{gk}, 0)",
                    f"    for _, x := range {var} {{",
                    f"        if !seen[x] {{",
                    f"            seen[x] = true",
                    f"            {nxt} = append({nxt}, x)",
                    f"        }}",
                    f"    }}",
                ])
            elif op == "calculate_average":
                if elem == "int":
                    lines.extend([
                        f"    sum := 0.0",
                        f"    for _, x := range {var} {{ sum += float64(x) }}",
                        f"    avg{i+1} := 0.0",
                        f"    if len({var}) > 0 {{ avg{i+1} = sum / float64(len({var})) }}",
                    ])
                else:
                    lines.extend([
                        f"    sum := 0.0",
                        f"    for _, x := range {var} {{ sum += x }}",
                        f"    avg{i+1} := 0.0",
                        f"    if len({var}) > 0 {{ avg{i+1} = sum / float64(len({var})) }}",
                    ])
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.extend([
                    f"    {nxt} := {zero}",
                    f"    for _, x := range {var} {{ {nxt} += x }}",
                ])
            elif op == "find_max":
                lines.extend([
                    f"    {nxt} := {zero}",
                    f"    if len({var}) > 0 {{",
                    f"        {nxt} = {var}[0]",
                    f"        for _, x := range {var} {{",
                    f"            if x > {nxt} {{ {nxt} = x }}",
                    f"        }}",
                    f"    }}",
                ])
            elif op == "find_min":
                lines.extend([
                    f"    {nxt} := {zero}",
                    f"    if len({var}) > 0 {{",
                    f"        {nxt} = {var}[0]",
                    f"        for _, x := range {var} {{",
                    f"            if x < {nxt} {{ {nxt} = x }}",
                    f"        }}",
                    f"    }}",
                ])
            elif op == "count_elements":
                lines.append(f"    {nxt} := len({var})")
            elif op == "filter_above":
                avg = self._find_avg_var_go(ops, i)
                cmp = f"float64(x) > {avg}" if elem == "int" else f"x > {avg}"
                lines.extend([
                    f"    {nxt} := make([]{gk}, 0)",
                    f"    for _, x := range {var} {{",
                    f"        if {cmp} {{ {nxt} = append({nxt}, x) }}",
                    f"    }}",
                ])
            elif op == "filter_below":
                avg = self._find_avg_var_go(ops, i)
                cmp = f"float64(x) < {avg}" if elem == "int" else f"x < {avg}"
                lines.extend([
                    f"    {nxt} := make([]{gk}, 0)",
                    f"    for _, x := range {var} {{",
                    f"        if {cmp} {{ {nxt} = append({nxt}, x) }}",
                    f"    }}",
                ])
            elif op == "sort_descending":
                lines.extend([
                    f"    {nxt} := make([]{gk}, len({var}))",
                    f"    copy({nxt}, {var})",
                    f"    sort.Slice({nxt}, func(i, j int) bool {{ return {nxt}[i] > {nxt}[j] }})",
                ])
            elif op == "sort_ascending":
                lines.extend([
                    f"    {nxt} := make([]{gk}, len({var}))",
                    f"    copy({nxt}, {var})",
                    f"    {'sort.Ints' if elem == 'int' else 'sort.Float64s'}({nxt})",
                ])
            elif op == "filter_even":
                if elem == "int":
                    lines.extend([
                        f"    {nxt} := make([]{gk}, 0)",
                        f"    for _, x := range {var} {{",
                        f"        if x % 2 == 0 {{ {nxt} = append({nxt}, x) }}",
                        f"    }}",
                    ])
                else:
                    lines.extend([
                        f"    {nxt} := make([]{gk}, 0)",
                        f"    for _, x := range {var} {{",
                        f"        if x == float64(int(x/2)*2) {{ {nxt} = append({nxt}, x) }}",
                        f"    }}",
                    ])
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (go)")
            ret = nxt
            var = nxt

        lines.extend([f"    return {ret}", "}"])
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Rust
    # ------------------------------------------------------------------ #
    def _gen_rust(self, name: str, ops: list[str]) -> str:
        self._require_support("rust")
        ret_type = _pipeline_return_type(ops)
        elem = self._elem(ops)
        # Parity with python: a terminal sum/extremum over the element
        # domain yields THAT element type — the signature follows the
        # dataflow, never a blanket f64.
        if ops and ops[-1] in ("calculate_sum", "find_max", "find_min"):
            ret_type = elem
        rk = "i32" if elem == "int" else "f64"
        zero = "0" if elem == "int" else "0.0"
        if ret_type == "list":
            sig_ret = f"Vec<{rk}>"
        else:
            sig_ret = _get_type('rust', ret_type)
        lines = [
            f"fn {name}(data: Vec<{rk}>) -> {sig_ret} {{",
        ]

        var = "data"
        ret = "data"
        for i, op in enumerate(ops):
            nxt = f"step{i+1}"
            if op == "remove_duplicates":
                lines.extend([
                    f"    let mut seen = std::collections::HashSet::new();",
                    f"    let mut {nxt}: Vec<{rk}> = Vec::new();",
                    f"    for &x in &{var} {{",
                    f"        if seen.insert(x) {{ {nxt}.push(x); }}",
                    f"    }}",
                ])
            elif op == "calculate_average":
                lines.extend([
                    f"    let avg{i+1}: f64 = if {var}.is_empty() {{ 0.0 }}",
                    f"        else {{ {var}.iter().map(|&x| x as f64).sum::<f64>() / {var}.len() as f64 }};",
                ])
                ret = f"avg{i+1}"
                continue
            elif op == "calculate_sum":
                lines.append(f"    let {nxt} = {var}.iter().sum::<{rk}>();")
            elif op == "find_max":
                if elem == "int":
                    lines.append(f"    let {nxt} = {var}.iter().copied().fold(i32::MIN, |a, x| a.max(x));")
                else:
                    lines.append(f"    let {nxt} = {var}.iter().cloned().fold(f64::MIN, f64::max);")
            elif op == "find_min":
                if elem == "int":
                    lines.append(f"    let {nxt} = {var}.iter().copied().fold(i32::MAX, |a, x| a.min(x));")
                else:
                    lines.append(f"    let {nxt} = {var}.iter().cloned().fold(f64::MAX, f64::min);")
            elif op == "count_elements":
                lines.append(f"    let {nxt} = {var}.len() as i32;")
            elif op == "filter_above":
                avg = self._find_avg_var_rust(ops, i)
                pred = f"*x as f64 > {avg}" if elem == "int" else f"*x > {avg}"
                lines.append(f"    let {nxt}: Vec<{rk}> = {var}.iter().cloned().filter(|x| {pred}).collect();")
            elif op == "filter_below":
                avg = self._find_avg_var_rust(ops, i)
                pred = f"*x as f64 < {avg}" if elem == "int" else f"*x < {avg}"
                lines.append(f"    let {nxt}: Vec<{rk}> = {var}.iter().cloned().filter(|x| {pred}).collect();")
            elif op == "sort_descending":
                lines.extend([
                    f"    let mut {nxt} = {var}.clone();",
                    f"    {nxt}.sort_by(|a, b| b.partial_cmp(a).unwrap());",
                ])
            elif op == "sort_ascending":
                lines.extend([
                    f"    let mut {nxt} = {var}.clone();",
                    f"    {nxt}.sort_by(|a, b| a.partial_cmp(b).unwrap());",
                ])
            elif op == "filter_even":
                if elem == "int":
                    lines.append(f"    let {nxt}: Vec<{rk}> = {var}.iter().cloned().filter(|x| *x % 2 == 0).collect();")
                else:
                    lines.append(f"    let {nxt}: Vec<{rk}> = {var}.iter().cloned().filter(|x| x % 2.0 == 0.0).collect();")
            else:
                raise ValueError(f"No pipeline template for operation '{op}' (rust)")
            ret = nxt
            var = nxt

        lines.append(f"    {ret}")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helper methods for finding average variable names
    # ------------------------------------------------------------------ #
    def _contract_threshold(self) -> str | None:
        """The LITERAL selection operand the semantics declared.

        When the request states a comparison operand ("greater than 5"),
        the selection thresholds THAT constant — the aggregate's average
        is only the operand when the request says "the average".  A
        declared literal is rendered here; None means the threshold is
        the average variable an aggregate produced.
        """
        if self.contract is None:
            return None
        thr = getattr(self.contract, "threshold", None)
        if thr is None:
            return None
        if float(thr).is_integer():
            return str(int(thr))
        return str(thr)

    def _find_avg_var(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg_{j+1}"
        return "(sum(data) / len(data) if data else 0)"

    def _find_avg_var_kotlin(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "data.average()"

    def _find_avg_var_dart(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "data.fold<num>(0, (a, b) => a + b) / data.length"

    def _find_avg_var_java(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "data.stream().mapToDouble(x -> x.doubleValue()).average().orElse(0.0)"

    def _find_avg_var_cpp(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "std::accumulate(data.begin(), data.end(), 0.0) / data.size()"

    def _find_avg_var_js(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "data.reduce((a, b) => a + b, 0) / data.length"

    def _find_avg_var_rust(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        if self._elem(ops) == "int":
            return "(data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64)"
        return "data.iter().sum::<f64>() / data.len() as f64"

    def _find_avg_var_go(self, ops: list[str], current_idx: int) -> str:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return f"avg{j+1}"
        return "sum / float64(len(data))"

    def _find_avg_idx(self, ops: list[str], current_idx: int) -> int:
        thr = self._contract_threshold()
        if thr is not None:
            return thr
        for j in range(current_idx - 1, -1, -1):
            if ops[j] == "calculate_average":
                return j
        return 0


def create_pipeline_generator() -> PipelineCodeGenerator:
    """Factory function."""
    return PipelineCodeGenerator()
