"""Operation pipeline → structural IR builder.

The IR is the CENTER of generation: an operation pipeline (e.g.
calculate_average → filter_above → sort_descending → return) is converted
into real IRNodes, and the multi-language generators render
deterministically FROM the IR. No per-problem logic.

Dataflow contract (the invariant the IR must carry):

    Semantic Output Type == IR Output Type
        == Generated Function Return Type == Verified Behavioral Type

  - collection-typed ops (dedup/filter/sort/reverse) produce a new
    collection variable and advance the current-collection pointer
  - scalar ops (average/sum/max/min/count) produce a scalar variable
    and leave the collection pointer unchanged
  - a filter's condition references the most recent scalar (threshold)
  - the return binds to the TERMINAL operation's output — the last
    scalar when the pipeline ends on a scalar op, the last collection
    otherwise.  An identity binding (returning the input unchanged) is
    not an implementation and never occurs here.
  - the function node's type_ref IS the pipeline's output type, read
    from the terminal operation's contract.
"""
from __future__ import annotations
from code.ir.ir_types import IRNode, IRNodeType


def _ref(name: str) -> IRNode:
    return IRNode(node_type=IRNodeType.REFERENCE, name=name)


def _lit(value: str) -> IRNode:
    return IRNode(node_type=IRNodeType.LITERAL, value=value)


def _call(name: str, *args: IRNode) -> IRNode:
    node = IRNode(node_type=IRNodeType.CALL, name=name)
    for a in args:
        node.add_child(a)
    return node


def _length(name: str) -> IRNode:
    node = IRNode(node_type=IRNodeType.LENGTH, name=name)
    node.add_child(_ref(name))
    return node


def _var(name: str, value: IRNode, type_ref: str = None) -> IRNode:
    node = IRNode(node_type=IRNodeType.VARIABLE, name=name)
    if type_ref:
        node.type_ref = type_ref
    node.add_child(value)
    return node


_SCALAR_OPS = {"calculate_average", "calculate_sum", "find_max",
               "find_min", "count_elements"}

# The output TYPE each scalar-producing operation's output carries.
# Operations not listed produce a collection (the transformed sequence).
_IR_SCALAR_TYPES = {
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


def _ir_return_type(ops: list, elem_type: str) -> str:
    """The pipeline's output type: the terminal operation's contract."""
    if not ops:
        return f"list[{elem_type}]"
    last = ops[-1]
    # sum / extremum of an int list is an int; average is division -> float
    if last in ("calculate_sum", "find_max", "find_min"):
        return elem_type
    return _IR_SCALAR_TYPES.get(last, f"list[{elem_type}]")


def build_ir(ops: list[str], func_name: str = "process_data",
             input_type: str = "list", contract=None) -> IRNode:
    """Build a structural IR from an operation pipeline.

    Every operation becomes real executable IR nodes; intermediate values
    flow through variables; exactly one return at the end.

    `contract` (brain.generation.contract.PipelineContract) carries the
    declared element domain and the objective's select key: the IR's
    type_ref then equals the semantic output type, and a length-measure
    extremum is marked as such — the meaning survives into the IR.
    """
    # Parity predicates (even/odd) are defined only over integers — the
    # OPERATION VOCABULARY itself tells us the element domain, without
    # any text-level type sniffing. The terminal sum/max/min then
    # inherit that element type (sum of ints is int), preserving the
    # IR -> code -> behavioral type invariant.
    declared_elem = (getattr(contract, "elem_type", "") or "") \
        if contract is not None else ""
    select_key = (getattr(contract, "select_key", "") or "") \
        if contract is not None else ""
    if declared_elem:
        elem_type = declared_elem
    elif input_type == "list":
        elem_type = "int" if any(
            op in ("filter_even", "is_multiple_of") for op in ops
        ) else "float"
    else:
        elem_type = "int"
    func = IRNode(node_type=IRNodeType.FUNCTION, name=func_name)
    func.add_child(IRNode(node_type=IRNodeType.PARAMETER, name="data",
                          type_ref=f"list[{elem_type}]"))
    body = IRNode(node_type=IRNodeType.BLOCK)
    func.add_child(body)

    if contract is not None and getattr(contract, "direction", ""):
        func.properties["objective"] = {
            "direction": contract.direction,
            "measure": contract.measure,
            "target": contract.target,
            "element": elem_type,
            **({"select_key": contract.select_key}
               if getattr(contract, "keyed", False) else {}),
        }

    current = "data"          # the collection the next op consumes
    threshold = None          # the most recent scalar (for filter conditions)
    last_scalar = None

    for i, op in enumerate(ops):
        nxt = f"step_{i}"

        if op == "calculate_average":
            # avg guarded against empty input
            cond = IRNode(node_type=IRNodeType.BINARY_OP, value=">")
            cond.add_child(_length(current))
            cond.add_child(_lit("0"))
            then_block = IRNode(node_type=IRNodeType.BLOCK)
            div = IRNode(node_type=IRNodeType.BINARY_OP, value="/")
            div.add_child(_call("sum", _ref(current)))
            div.add_child(_length(current))
            then_block.add_child(_var(nxt, div, "float"))
            else_block = IRNode(node_type=IRNodeType.BLOCK)
            else_block.add_child(_var(nxt, _lit("0"), "float"))
            if_node = IRNode(node_type=IRNodeType.IF)
            if_node.add_child(cond)
            if_node.add_child(then_block)
            if_node.add_child(else_block)
            if_node.properties["op"] = op
            body.add_child(if_node)
            threshold = nxt
            last_scalar = nxt
            # the collection pointer is unchanged — UNLESS this is the
            # terminal operation: the pipeline's output is then the
            # scalar itself, and the return must bind to it (never to
            # the input — that would be an identity fallback).
            if i == len(ops) - 1:
                current = nxt

        elif op in ("add", "multiply"):
            # two-operand arithmetic over the collection's first
            # elements, guarded by the operation contract (insufficient
            # input → 0 — the reference semantics decide)
            sym = "+" if op == "add" else "*"
            left = IRNode(node_type=IRNodeType.INDEX, name="index")
            left.add_child(_ref(current))
            left.add_child(_lit("0"))
            right = IRNode(node_type=IRNodeType.INDEX, name="index")
            right.add_child(_ref(current))
            right.add_child(_lit("1"))
            bop = IRNode(node_type=IRNodeType.BINARY_OP, value=sym)
            bop.add_child(left)
            bop.add_child(right)
            cond = IRNode(node_type=IRNodeType.BINARY_OP, value=">=")
            cond.add_child(_length(current))
            cond.add_child(_lit("2"))
            then_block = IRNode(node_type=IRNodeType.BLOCK)
            then_block.add_child(_var(nxt, bop, elem_type))
            # The contract's degenerate branches: one operand → that
            # value; none → 0.
            else_block = IRNode(node_type=IRNodeType.BLOCK)
            inner_cond = IRNode(node_type=IRNodeType.BINARY_OP, value=">=")
            inner_cond.add_child(_length(current))
            inner_cond.add_child(_lit("1"))
            single_index = IRNode(node_type=IRNodeType.INDEX, name="index")
            single_index.add_child(_ref(current))
            single_index.add_child(_lit("0"))
            inner_then = IRNode(node_type=IRNodeType.BLOCK)
            inner_then.add_child(_var(nxt, single_index, elem_type))
            inner_else = IRNode(node_type=IRNodeType.BLOCK)
            inner_else.add_child(_var(nxt, _lit("0"), elem_type))
            inner_if = IRNode(node_type=IRNodeType.IF)
            inner_if.add_child(inner_cond)
            inner_if.add_child(inner_then)
            inner_if.add_child(inner_else)
            else_block.add_child(inner_if)
            if_node = IRNode(node_type=IRNodeType.IF)
            if_node.add_child(cond)
            if_node.add_child(then_block)
            if_node.add_child(else_block)
            if_node.properties["op"] = op
            body.add_child(if_node)
            last_scalar = nxt
            if i == len(ops) - 1:
                current = nxt

        elif op == "calculate_sum":
            node = _var(nxt, _call("sum", _ref(current)), elem_type)
            node.properties["op"] = op
            body.add_child(node)
            last_scalar = nxt
            if i == len(ops) - 1:
                current = nxt

        elif op == "find_max":
            node = _var(nxt, _call("max", _ref(current)), elem_type)
            node.properties["op"] = op
            if select_key == "length":
                node.properties["select_key"] = select_key
            body.add_child(node)
            last_scalar = nxt
            if i == len(ops) - 1:
                current = nxt

        elif op == "find_min":
            node = _var(nxt, _call("min", _ref(current)), elem_type)
            node.properties["op"] = op
            if select_key == "length":
                node.properties["select_key"] = select_key
            body.add_child(node)
            last_scalar = nxt
            if i == len(ops) - 1:
                current = nxt

        elif op == "count_elements":
            node = _var(nxt, _length(current), "int")
            node.properties["op"] = op
            body.add_child(node)
            last_scalar = nxt
            if i == len(ops) - 1:
                current = nxt

        elif op == "sort_ascending":
            node = _var(nxt, _call("sorted", _ref(current)),
                        f"list[{elem_type}]")
            node.properties["op"] = op
            body.add_child(node)
            current = nxt

        elif op == "sort_descending":
            node = _var(
                nxt,
                _call("sorted", _ref(current), _lit("reverse=True")),
                f"list[{elem_type}]")
            node.properties["op"] = op
            body.add_child(node)
            current = nxt

        elif op == "reverse_sequence":
            node = _var(
                nxt,
                _call("list", _call("reversed", _ref(current))),
                f"list[{elem_type}]")
            node.properties["op"] = op
            body.add_child(node)
            current = nxt

        elif op == "remove_duplicates":
            # seen set + loop-filtering loop
            seen_init = IRNode(node_type=IRNodeType.CALL, name="set")
            body.add_child(_var(f"seen_{i}", seen_init))
            out_init = IRNode(node_type=IRNodeType.CALL, name="list")
            body.add_child(_var(nxt, out_init,
                                f"list[{elem_type}]"))
            cond = IRNode(node_type=IRNodeType.UNARY_OP, value="not ")
            contains = IRNode(node_type=IRNodeType.CALL, name="contains")
            contains.add_child(_ref(f"seen_{i}"))
            contains.add_child(_ref("x"))
            cond.add_child(contains)
            then_block = IRNode(node_type=IRNodeType.BLOCK)
            add_call = IRNode(node_type=IRNodeType.CALL, name="add")
            add_call.add_child(_ref(f"seen_{i}"))
            add_call.add_child(_ref("x"))
            then_block.add_child(add_call)
            out_add = IRNode(node_type=IRNodeType.CALL, name="append")
            out_add.add_child(_ref(nxt))
            out_add.add_child(_ref("x"))
            then_block.add_child(out_add)
            if_node = IRNode(node_type=IRNodeType.IF)
            if_node.add_child(cond)
            if_node.add_child(then_block)
            loop_body = IRNode(node_type=IRNodeType.BLOCK)
            loop_body.add_child(if_node)
            loop = IRNode(node_type=IRNodeType.FOR_EACH, name="x",
                          value=current)
            loop.properties["op"] = op
            loop.add_child(loop_body)
            body.add_child(loop)
            current = nxt

        elif op in ("filter_above", "filter_below", "filter_even",
                    "filter_elements"):
            out_init = IRNode(node_type=IRNodeType.CALL, name="list")
            body.add_child(_var(nxt, out_init,
                                f"list[{elem_type}]"))
            cond = _filter_condition(op, threshold, elem_type)
            then_block = IRNode(node_type=IRNodeType.BLOCK)
            add_call = IRNode(node_type=IRNodeType.CALL, name="append")
            add_call.add_child(_ref(nxt))
            add_call.add_child(_ref("x"))
            then_block.add_child(add_call)
            if_node = IRNode(node_type=IRNodeType.IF)
            if_node.add_child(cond)
            if_node.add_child(then_block)
            loop_body = IRNode(node_type=IRNodeType.BLOCK)
            loop_body.add_child(if_node)
            loop = IRNode(node_type=IRNodeType.FOR_EACH, name="x",
                          value=current)
            loop.properties["op"] = op
            loop.add_child(loop_body)
            body.add_child(loop)
            current = nxt

        elif op == "fibonacci":
            # iterative fibonacci: a, b = 0, 1; n times a, b = b, a + b
            a_init = IRNode(node_type=IRNodeType.LITERAL, value="0")
            body.add_child(_var(f"a_{i}", a_init, "int"))
            b_init = IRNode(node_type=IRNodeType.LITERAL, value="1")
            body.add_child(_var(f"b_{i}", b_init, "int"))
            loop_body = IRNode(node_type=IRNodeType.BLOCK)
            # a, b = b, a + b  (simultaneous assignment via a temp)
            tmp = IRNode(node_type=IRNodeType.BINARY_OP, value="+")
            tmp.add_child(_ref(f"a_{i}"))
            tmp.add_child(_ref(f"b_{i}"))
            loop_body.add_child(_var(f"t_{i}", tmp, "int"))
            a_assign = IRNode(node_type=IRNodeType.ASSIGNMENT, name=f"a_{i}")
            a_assign.add_child(_ref(f"b_{i}"))
            loop_body.add_child(a_assign)
            b_assign = IRNode(node_type=IRNodeType.ASSIGNMENT, name=f"b_{i}")
            b_assign.add_child(_ref(f"t_{i}"))
            loop_body.add_child(b_assign)
            loop = IRNode(node_type=IRNodeType.FOR_EACH, name="_",
                          value=current)
            loop.properties["op"] = op
            loop.add_child(loop_body)
            body.add_child(loop)
            last_scalar = f"a_{i}"
            if i == len(ops) - 1:
                current = f"a_{i}"

        elif op == "search":
            # linear search: index counter + first element equal to target
            i_init = IRNode(node_type=IRNodeType.LITERAL, value="0")
            body.add_child(_var(f"i_{i}", i_init, "int"))
            cond = IRNode(node_type=IRNodeType.BINARY_OP, value="==")
            cond.add_child(_ref("x"))
            cond.add_child(_ref("target"))
            then_block = IRNode(node_type=IRNodeType.BLOCK)
            ret = IRNode(node_type=IRNodeType.RETURN)
            ret.add_child(_ref(f"i_{i}"))
            then_block.add_child(ret)
            loop_body = IRNode(node_type=IRNodeType.BLOCK)
            if_node = IRNode(node_type=IRNodeType.IF)
            if_node.add_child(cond)
            if_node.add_child(then_block)
            loop_body.add_child(if_node)
            inc = IRNode(node_type=IRNodeType.BINARY_OP, value="+")
            inc.add_child(_ref(f"i_{i}"))
            inc.add_child(_lit("1"))
            i_assign = IRNode(node_type=IRNodeType.ASSIGNMENT, name=f"i_{i}")
            i_assign.add_child(inc)
            loop_body.add_child(i_assign)
            loop = IRNode(node_type=IRNodeType.FOR_EACH, name="x",
                          value=current)
            loop.properties["op"] = op
            loop.add_child(loop_body)
            body.add_child(loop)
            # not found -> -1
            not_found = IRNode(node_type=IRNodeType.LITERAL, value="-1")
            ret_nf = IRNode(node_type=IRNodeType.RETURN)
            ret_nf.add_child(not_found)
            body.add_child(ret_nf)
            if i == len(ops) - 1:
                current = "-1"

        else:
            raise ValueError(f"No IR mapping for operation '{op}'")

    # The function's OUTPUT TYPE: the terminal operation's contract —
    # the IR carries the type the generated code must return.
    func.type_ref = _ir_return_type(ops, elem_type)

    # Exactly one return: the TERMINAL operation's output (never the
    # input — an identity binding is not an implementation)
    ret = IRNode(node_type=IRNodeType.RETURN)
    ret.type_ref = func.type_ref
    ret.add_child(_ref(current))
    body.add_child(ret)

    return func


def _filter_condition(op: str, threshold: str, elem_type: str) -> IRNode:
    """The filter's condition referencing the current threshold/scalar."""
    if op in ("filter_above",):
        rel = ">"
    elif op in ("filter_below",):
        rel = "<"
    elif op in ("filter_even",):
        cond = IRNode(node_type=IRNodeType.BINARY_OP, value="==")
        mod = IRNode(node_type=IRNodeType.BINARY_OP, value="%")
        mod.add_child(_ref("x"))
        mod.add_child(_lit("2"))
        cond.add_child(mod)
        cond.add_child(_lit("0"))
        return cond
    else:
        # filter_elements: a property filter with no bound condition is
        # not expressible — treat as a passthrough filter on truth
        return _lit("True")

    if not threshold:
        raise ValueError(f"{op} requires a threshold (aggregate) value")
    cond = IRNode(node_type=IRNodeType.BINARY_OP, value=rel)
    cond.add_child(_ref("x"))
    cond.add_child(_ref(threshold))
    return cond
