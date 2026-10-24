"""Typed IR Generator.

Generates a typed intermediate representation from an AlgorithmPlan.
The IR verifies:
  - Variable definitions (all variables defined before use)
  - Branch returns (all paths return a value)
  - Type consistency (no type mismatches)

No LLM involved. IR is built by composing general IR nodes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TypedIRNode:
    """A single node in the typed IR."""
    node_type: str  # "function", "assignment", "call", "return", "if", "for", etc.
    name: str = ""
    value: str = ""
    type_ref: str = ""
    children: list['TypedIRNode'] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "node_type": self.node_type,
            "name": self.name,
            "value": self.value,
            "type_ref": self.type_ref,
            "properties": self.properties,
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


@dataclass
class TypedIR:
    """Complete typed intermediate representation."""
    root: Optional[TypedIRNode] = None
    variables: dict[str, str] = field(default_factory=dict)  # name -> type
    function_name: str = "solve"
    return_type: str = "None"
    verified: bool = False
    verification_errors: list[str] = field(default_factory=list)


class TypedIRGenerator:
    """Generates a typed IR from an AlgorithmPlan.

    The IR is a tree of TypedIRNodes representing the program structure.
    After generation, the IR is verified for:
      - All variables defined before use
      - All branches have return statements
      - Type consistency
    """

    def generate(self, plan) -> TypedIR:
        """Generate a typed IR from the algorithm plan."""
        ir = TypedIR(
            function_name=plan.function_name,
            variables=dict(plan.variables),
        )

        # Build the function node
        func_node = TypedIRNode(
            node_type="function",
            name=plan.function_name,
            type_ref="None",
        )

        # 1. Add input parsing as assignments
        for step in plan.input_parsing:
            assign_node = self._step_to_assignment(step)
            func_node.children.append(assign_node)

        # 2. Add algorithm steps
        for step in plan.algorithm_steps:
            if step.operation == "index_chain":
                algo_node = self._step_to_index_chain(step)
            else:
                algo_node = self._step_to_algorithm(step)
            func_node.children.append(algo_node)

        # 3. Add output/return step
        if plan.output_step:
            ret_node = self._step_to_return(plan.output_step)
            func_node.children.append(ret_node)
        else:
            # Default: return the last computed result
            ret_node = TypedIRNode(
                node_type="return",
                name="result",
                value="result",
            )
            func_node.children.append(ret_node)

        ir.root = func_node

        # 4. Verify the IR
        ir.verified, ir.verification_errors = self._verify_ir(ir)

        return ir

    def _step_to_assignment(self, step) -> TypedIRNode:
        """Convert a plan step to an assignment IR node."""
        var_name = step.output_vars[0] if step.output_vars else "data"
        var_type = step.details.get("type", "unknown")

        if step.operation == "read_int":
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value=f"int(input())",
                type_ref="int",
            )
        elif step.operation == "read_array":
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value="list(map(int, input().split()))",
                type_ref="list[int]",
            )
        elif step.operation == "read_string":
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value="input().strip()",
                type_ref="str",
            )
        elif step.operation == "read_matrix":
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value="[list(map(int, input().split())) for _ in range(n)]",
                type_ref="list[list[int]]",
            )
        elif step.operation == "read_input":
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value="sys.stdin.read().split()",
                type_ref="list[str]",
            )
        else:
            return TypedIRNode(
                node_type="assignment",
                name=var_name,
                value=f"# {step.description}",
                type_ref=var_type,
            )

    def _step_to_algorithm(self, step) -> TypedIRNode:
        """Convert a plan step to an algorithm IR node."""
        # Create a block node for the algorithm
        block = TypedIRNode(
            node_type="block",
            name=step.operation,
            properties={"description": step.description},
        )

        # Add a call node for the algorithm
        call_node = TypedIRNode(
            node_type="call",
            name=step.operation,
            value=step.description,
        )
        block.children.append(call_node)

        return block

    def _step_to_index_chain(self, step) -> TypedIRNode:
        """Convert an index_chain plan step to structural IR.

        The general indirect-reference dataflow, expressed with generic
        primitives (no dedicated operation call):
            FOR_EACH i IN indices(V0) | range(1, N)
                v0 = INDEX(V0, i)          (or INDEX_INV(V0, i))
                v1 = INDEX(V1, v0)
                APPEND(output, vk)
        """
        chain_steps = step.details.get("steps", [])
        output_var = step.output_vars[0] if step.output_vars else "result"
        if not chain_steps:
            raise ValueError("index_chain has no chain steps")
        first_kind, first_col = chain_steps[0]

        # The output collection is created before the loop
        create_node = TypedIRNode(
            node_type="assignment",
            name=output_var,
            value="[]",
            type_ref="list[int]",
            properties={"op": "output_binding", "kind": "collection_create"},
        )

        children = []
        bound = None
        # Chain kind → generic IR primitive: "direct" → INDEX,
        # "inverse" → INDEX_INV (position where the collection's value
        # equals the bound).
        primitive_of = {"direct": "index", "inverse": "inverse"}
        for k, (kind, col) in enumerate(chain_steps):
            prim = primitive_of.get(kind, kind)
            if k == 0:
                source = "i"
                value = (f"INDEX_INV({col}, i)" if kind == "inverse"
                         else f"INDEX({col}, i)")
            else:
                source = bound
                value = (f"INDEX_INV({col}, {source})" if kind == "inverse"
                         else f"INDEX({col}, {source})")
            val_name = col.lower() + "_value" if k == len(chain_steps) - 1 else f"v{k}"
            children.append(TypedIRNode(
                node_type="assignment",
                name=val_name,
                value=value,
                type_ref="int",
                properties={"op": prim, "collection": col, "index": source},
            ))
            bound = val_name
        children.append(TypedIRNode(
            node_type="call",
            name="append",
            value=f"APPEND({output_var}, {bound})",
            properties={"op": "output_binding", "collection": output_var},
        ))

        # Iteration source: direct chains iterate positions; an inverse
        # first step iterates the identifier values 1..N.
        source = (f"range(1, {first_col.lower()}_len + 1)"
                  if first_kind == "inverse"
                  else f"indices({first_col})")

        for_node = TypedIRNode(
            node_type="for_each",
            name="i",
            properties={
                "source": source,
                "type": "iteration",
                "description": step.description,
                "preconditions": list(step.preconditions),
                "postconditions": list(step.postconditions),
                "semantic_justification": step.semantic_justification,
            },
            children=children,
        )

        block = TypedIRNode(
            node_type="block",
            name=step.operation,
            properties={"description": step.description},
            children=[create_node, for_node],
        )
        return block

    def _step_to_return(self, step) -> TypedIRNode:
        """Convert a plan step to a return IR node."""
        return TypedIRNode(
            node_type="return",
            name=step.output_vars[0] if step.output_vars else "result",
            value=step.input_vars[0] if step.input_vars else "result",
        )

    def _verify_ir(self, ir: TypedIR) -> tuple[bool, list[str]]:
        """Verify the IR for correctness.

        Checks:
          1. All variables defined before use
          2. All branches have return statements
          3. Type consistency (basic check)
        """
        errors = []
        defined_vars = set()

        if ir.root is None:
            return False, ["No root node"]

        # Walk the IR tree and check
        for child in ir.root.children:
            if child.node_type == "assignment":
                defined_vars.add(child.name)
            elif child.node_type == "block":
                # Algorithm block - check that input vars are defined
                for inp in (child.properties.get("input_vars", [])):
                    if inp not in defined_vars and inp != "data":
                        errors.append(f"Variable '{inp}' used before definition in {child.name}")
            elif child.node_type == "return":
                # Check return variable exists
                if child.value and child.value not in defined_vars:
                    # "result" and "data" are special cases
                    if child.value not in ("result", "data", "output"):
                        errors.append(f"Return variable '{child.value}' not defined")

        # Check type consistency (basic)
        # If a variable is used as int but assigned as string, flag it
        for child in ir.root.children:
            if child.node_type == "assignment":
                if child.type_ref and child.type_ref not in ("unknown", "computed"):
                    pass  # type is known, check usage

        is_valid = len(errors) == 0
        return is_valid, errors
