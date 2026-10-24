"""English Grammar - Grammatical composition for explanations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import re

from brain.decision_trace import DecisionTrace, Alternative
from brain.explanation_types import ClassPurpose, FunctionPurpose, LinePurpose
from brain.explanation_events import ExplanationType
from brain.think_engine import Constraint


class EnglishGrammar:
    """Generates grammatically correct English from semantic structures."""
    
    def compose_explanation(self, purpose: LinePurpose, decision: Optional[DecisionTrace], context: dict) -> str:
        """Compose line-level explanation from semantic components."""
        parts = []
        
        # Main clause
        main = self._compose_line_main(purpose)
        parts.append(main)
        
        # Reason clause
        if decision and decision.evidence:
            reason = self._compose_reason_clause(decision)
            parts.append(f"→ Needed to {reason}.")
        
        # Constraint clause
        if decision and decision.constraints_applied:
            constraint = self._compose_constraint_clause(decision.constraints_applied)
            parts.append(f"({constraint})")
        
        # Alternative mention
        if decision and decision.alternatives_considered:
            alt = self._compose_alternative_clause(decision.alternatives_considered)
            parts.append(f" {alt}")
        
        return " ".join(parts)
    
    def _compose_line_main(self, purpose: LinePurpose) -> str:
        """Compose main clause for a code line."""
        verb_map = {
            "define_class": "defines a class",
            "define_function": "defines a function",
            "pipeline_operation": "performs a pipeline operation",
            "return_result": "returns the result",
            "assign_variable": "assigns a value to a variable",
            "function_call": "calls a function",
            "conditional": "branches based on condition",
            "loop": "performs iteration",
            "computation": "computes a value",
            "variable_declaration": "declares a variable",
            "parameter": "defines a parameter",
            "property": "defines a property",
            "constructor": "defines a constructor",
            "statement": "executes a statement",
        }
        verb = verb_map.get(purpose.line_type, "executes a statement")
        
        subject = self._get_subject(purpose)
        if subject:
            return f"{subject} {verb}."
        return f"It {verb}."
    
    def _get_subject(self, purpose: LinePurpose) -> str:
        if purpose.line_type == "define_class":
            return purpose.node.name
        elif purpose.line_type == "define_function":
            return purpose.node.name
        elif purpose.context.get("current_class"):
            return purpose.context["current_class"]
        elif purpose.context.get("current_function"):
            return purpose.context["current_function"]
        return "The code"
    
    def _compose_reason_clause(self, decision: DecisionTrace) -> str:
        evidence_text = " ".join(decision.evidence[:3])
        return evidence_text.lower()
    
    def _compose_constraint_clause(self, constraints: List[str]) -> str:
        constraint_map = {
            "language": "language constraint",
            "platform": "platform constraint",
            "performance": "performance requirement",
            "security": "security requirement",
        }
        mapped = [constraint_map.get(c, c) for c in constraints]
        return f"with {', '.join(mapped)}"
    
    def _compose_alternative_clause(self, alternatives: List[Alternative]) -> str:
        if not alternatives:
            return ""
        alt_texts = [f"{a.option} ({a.reason_rejected})" for a in alternatives[:2]]
        return f"Alternatives considered: {', '.join(alt_texts)} but rejected."
    
    def explain_file_generation(self, file_path: str, state) -> str:
        parts = [f"Generating {file_path}."]
        
        if "model" in file_path.lower() or "entity" in file_path.lower():
            parts.append("Needed to define the data model.")
        elif "service" in file_path.lower():
            parts.append("Needed to encapsulate business logic.")
        elif "controller" in file_path.lower() or "activity" in file_path.lower():
            parts.append("Needed to connect UI and logic.")
        elif "main" in file_path.lower() or "entry" in file_path.lower():
            parts.append("Needed to provide program entry point.")
        else:
            parts.append("Needed to implement requested functionality.")
        
        return " ".join(parts)
    
    def explain_generation_start(self, project_name: str, state) -> str:
        return f"Starting generation of {project_name} project. Analyzing requirements and composing necessary files."
    
    def explain_generation_complete(self, project_name: str, state) -> str:
        return f"{project_name} project generation completed. All files and dependencies configured."
    
    def explain_class_creation(self, purpose: ClassPurpose) -> str:
        parts = [f"Added {purpose.name} class."]
        
        if purpose.reason:
            parts.append(f"→ Needed to {purpose.reason.lower()}.")
        
        if purpose.properties:
            props = ", ".join(purpose.properties)
            parts.append(f"Properties: {props}.")
        
        return " ".join(parts)
    
    def explain_function_creation(self, purpose: FunctionPurpose) -> str:
        parts = [f"Added {purpose.name} function."]
        
        if purpose.reason:
            parts.append(f"→ Needed to {purpose.reason.lower()}.")
        
        if purpose.params:
            params = ", ".join([f"{n} ({t})" for n, t in purpose.params])
            parts.append(f"Parameters: {params}.")
        
        if purpose.operations:
            ops = " → ".join(purpose.operations)
            parts.append(f"Pipeline: {ops}.")
        
        return " ".join(parts)
    
    def explain_pipeline_stage(self, stage_name: str, operation, operation_order: List[str]) -> str:
        purpose_map = {
            "remove_duplicates": "remove duplicate values",
            "calculate_average": "calculate average as threshold",
            "filter_above": "filter values above average",
            "filter_below": "filter values below average",
            "sort_descending": "sort in descending order",
            "sort_ascending": "sort in ascending order",
            "filter_even": "filter even numbers",
            "sum_list": "calculate sum of all values",
        }
        
        purpose_en = purpose_map.get(stage_name, f"perform {stage_name} operation")
        
        parts = [f"Added {stage_name} stage."]
        parts.append(f"→ To {purpose_en}.")
        
        if stage_name in operation_order:
            idx = operation_order.index(stage_name)
            if idx > 0:
                parts.append(f"Takes input from {operation_order[idx-1]}.")
            if idx < len(operation_order) - 1:
                parts.append(f"Passes output to {operation_order[idx+1]}.")
        
        return " ".join(parts)
    
    def explain_structure_completion(self, struct_type: str, name: str) -> str:
        return f"Completed {struct_type} {name} implementation."
    
    def explain_alternatives(self, alternatives: List[Alternative]) -> str:
        if not alternatives:
            return "No alternatives were considered."
        
        parts = ["Alternatives considered:"]
        for alt in alternatives:
            parts.append(f"- {alt.option}: {alt.reason_rejected} (confidence: {alt.confidence:.0%})")
        return " ".join(parts)
    
    def explain_constraint(self, constraint: Constraint) -> str:
        constraint_map = {
            "language": f"Must be implemented in {constraint.value}.",
            "platform": f"Must run on {constraint.value} platform.",
            "performance": "High performance is required.",
            "security": "Security requirements must be met.",
        }
        return constraint_map.get(constraint.kind, f"Constraint: {constraint.name}={constraint.value}.")
    
    def explain_decision(self, decision: DecisionTrace) -> str:
        parts = []
        
        action_map = {
            "add_class": f"Added {decision.chosen_option} class",
            "add_function": f"Added {decision.chosen_option} function",
            "choose_algorithm": f"Selected {decision.chosen_option} algorithm",
            "choose_structure": f"Chose {decision.chosen_option} structure",
            "add_pipeline_stage": f"Added {decision.chosen_option} stage",
            "select_hypothesis": f"Selected {decision.chosen_option} approach",
            "plan_action": f"Planned {decision.chosen_option} action",
        }
        action_text = action_map.get(decision.decision_type.value, f"Decided on {decision.chosen_option}")
        parts.append(action_text + ".")
        
        if decision.evidence:
            reason = " ".join(decision.evidence[:3])
            parts.append(f"Reason: {reason}.")
        
        if decision.constraints_applied:
            parts.append(f"Constraints: {', '.join(decision.constraints_applied)}.")
        
        if decision.alternatives_considered:
            parts.append(self.explain_alternatives(decision.alternatives_considered))
        
        return " ".join(parts)