"""Semantic Operation Planner - Phase 4.

Converts a SemanticRepresentation into an OperationPlan by composing
primitives from the goal, relations, and constraints. This replaces
the keyword-list-based operation selection with reasoning-based planning.

The planner answers: "Given what the user wants, what transformations
are needed to achieve it?" — not "which keyword matches best?"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanStep:
    """A single step in an operation plan."""
    step_id: str
    operation: str
    input_type: str = ""
    output_type: str = ""
    precondition: str = ""
    description: str = ""
    source_primitive: str = ""  # which SemanticComposer primitive this maps to
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "precondition": self.precondition,
            "description": self.description,
            "source_primitive": self.source_primitive,
            "confidence": self.confidence,
        }


@dataclass
class OperationPlan:
    """A structured plan of operations derived from semantic analysis."""
    steps: list[PlanStep] = field(default_factory=list)
    input_type: str = ""
    output_type: str = ""
    goal_description: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "input_type": self.input_type,
            "output_type": self.output_type,
            "goal_description": self.goal_description,
            "confidence": self.confidence,
            "notes": self.notes,
        }

    def get_operation_sequence(self) -> list[str]:
        return [s.operation for s in self.steps]


class OperationPlanner:
    """Plans operations by composing primitives from semantic analysis.

    Priority hierarchy (from user spec):
      Explicit User Semantics > Validated Semantic Inference >
      Contextual Evidence > Retrieved Knowledge > Legacy Mapping >
      Surface Keyword Match
    """

    # Primitive transformations that can be composed
    PRIMITIVES = {
        # Aggregation
        "calculate_average": {"input": "collection", "output": "scalar",
                              "description": "Compute mean of collection"},
        "calculate_sum": {"input": "collection", "output": "scalar",
                          "description": "Compute sum of collection"},
        "find_max": {"input": "collection", "output": "scalar",
                     "description": "Find maximum element"},
        "find_min": {"input": "collection", "output": "scalar",
                     "description": "Find minimum element"},
        "count": {"input": "collection", "output": "scalar",
                  "description": "Count elements"},
        # Comparison
        "compare_greater_than": {"input": "scalar,scalar", "output": "boolean",
                                 "description": "Check if left > right"},
        "compare_less_than": {"input": "scalar,scalar", "output": "boolean",
                              "description": "Check if left < right"},
        "compare_equal": {"input": "any,any", "output": "boolean",
                          "description": "Check equality"},
        # Predicates
        "is_prime": {"input": "scalar", "output": "boolean",
                     "description": "Check if number is prime"},
        "is_multiple_of": {"input": "scalar", "output": "boolean",
                           "description": "Check divisibility"},
        "is_even": {"input": "scalar", "output": "boolean",
                    "description": "Check if number is even"},
        "is_odd": {"input": "scalar", "output": "boolean",
                   "description": "Check if number is odd"},
        # Filtering
        "filter": {"input": "collection,predicate", "output": "collection",
                   "description": "Select elements matching predicate"},
        "remove_duplicates": {"input": "collection", "output": "collection",
                              "description": "Remove duplicate elements"},
        # Sorting
        "sort_descending": {"input": "collection", "output": "ordered_collection",
                            "description": "Sort elements descending"},
        "sort_ascending": {"input": "collection", "output": "ordered_collection",
                           "description": "Sort elements ascending"},
        "reverse": {"input": "collection", "output": "collection",
                    "description": "Reverse element order"},
        # Generation
        "generate_range": {"input": "scalar,scalar", "output": "collection",
                           "description": "Generate number range"},
        # Transform
        "map": {"input": "collection,function", "output": "collection",
                "description": "Apply function to each element"},
    }

    def plan(self, semantic) -> OperationPlan:
        """Generate an OperationPlan from a SemanticRepresentation.

        Uses composition of primitives rather than keyword matching.
        """
        from brain.semantic_representation import GoalType

        plan = OperationPlan()
        plan.goal_description = semantic.raw_text

        # Step 1: Determine input/output types from I/O specs
        if semantic.inputs:
            plan.input_type = semantic.inputs[0].type_signature
        if semantic.outputs:
            plan.output_type = semantic.outputs[0].type_signature

        # Step 2: Use composition if available (preferred path)
        if hasattr(semantic, 'composition') and semantic.composition:
            plan = self._plan_from_composition(semantic, plan)
            return plan

        # Step 3: Fallback — infer plan from goal + entities + constraints
        plan = self._plan_from_goal(semantic, plan)
        return plan

    def _plan_from_composition(self, semantic, plan: OperationPlan) -> OperationPlan:
        """Build plan from SemanticComposer's operation graph."""
        comp = semantic.composition
        graph_ops = comp.get("graph", {}).get("operations", [])
        dependencies = comp.get("graph", {}).get("dependencies", {})

        step_map = {}
        for i, op_info in enumerate(graph_ops):
            name = op_info["name"]
            primitive = self.PRIMITIVES.get(name, {})
            step = PlanStep(
                step_id=f"step_{i}",
                operation=name,
                input_type=primitive.get("input", ""),
                output_type=primitive.get("output", ""),
                description=primitive.get("description", ""),
                source_primitive=name,
                confidence=0.85,
            )
            step_map[name] = step
            plan.steps.append(step)

        # Wire dependencies as preconditions
        for op_name, deps in dependencies.items():
            if op_name in step_map and deps:
                dep_names = [d for d in deps if d in step_map]
                step_map[op_name].precondition = " and ".join(dep_names)

        plan.confidence = 0.85
        plan.notes.append("Plan derived from semantic composition")
        return plan

    def _plan_from_goal(self, semantic, plan: OperationPlan) -> OperationPlan:
        """Infer plan from goal type, entities, and constraints when composition unavailable."""
        from brain.semantic_representation import GoalType

        goal = semantic.goal.goal_type
        entities = semantic.entities
        constraints = semantic.constraints

        if goal == GoalType.CREATE_FUNCTION:
            plan = self._plan_function(entities, constraints, plan)
        elif goal == GoalType.CREATE_PROJECT:
            plan = self._plan_project(entities, constraints, plan)
        elif goal == GoalType.CREATE_CLASS:
            plan = self._plan_class(entities, constraints, plan)
        elif goal in (GoalType.QUESTION, GoalType.EXPLAIN):
            plan = self._plan_explain(plan)
        else:
            plan.confidence = 0.3
            plan.notes.append("Low confidence: could not infer plan from goal")

        return plan

    def _plan_function(self, entities, constraints, plan):
        """Plan for function creation based on entity types and constraints."""
        # Detect data operations from entity context
        has_collection = any(e.kind == "collection" for e in entities)
        has_numeric = any("number" in e.type_name.lower() or "int" in e.type_name.lower()
                         for e in entities)

        # Default function plan: input → transform → output
        plan.steps.append(PlanStep(
            step_id="step_0", operation="receive_input",
            input_type=plan.input_type or "any",
            description="Accept input parameters",
            confidence=0.7,
        ))
        plan.steps.append(PlanStep(
            step_id="step_1", operation="process",
            input_type=plan.input_type or "any",
            output_type=plan.output_type or "any",
            description="Apply transformation logic",
            confidence=0.5,
        ))
        plan.steps.append(PlanStep(
            step_id="step_2", operation="return_result",
            input_type=plan.output_type or "any",
            description="Return result",
            confidence=0.7,
        ))
        plan.confidence = 0.5
        plan.notes.append("Generic function plan: specific operations depend on composition")
        return plan

    def _plan_project(self, entities, constraints, plan):
        plan.steps.append(PlanStep(
            step_id="step_0", operation="setup_project",
            description="Initialize project structure",
            confidence=0.6,
        ))
        plan.confidence = 0.4
        plan.notes.append("Project plan: detailed steps depend on platform/framework")
        return plan

    def _plan_class(self, entities, constraints, plan):
        plan.steps.append(PlanStep(
            step_id="step_0", operation="define_class",
            description="Define class with attributes and methods",
            confidence=0.6,
        ))
        plan.confidence = 0.5
        return plan

    def _plan_explain(self, plan):
        plan.steps.append(PlanStep(
            step_id="step_0", operation="explain",
            description="Generate explanation",
            confidence=0.6,
        ))
        plan.confidence = 0.6
        return plan
