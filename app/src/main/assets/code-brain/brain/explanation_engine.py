"""Explanation Engine - Generates genuine explanations from decision traces."""
from __future__ import annotations
from typing import Optional, List
from datetime import datetime

from brain.decision_trace import DecisionTrace, Alternative, DecisionType
from brain.think_engine import ReasoningState, Entity, Operation, ProblemModel, GoalType
from code.ir.ir_types import IRNode, IRNodeType
from brain.korean_grammar import KoreanGrammar
from brain.english_grammar import EnglishGrammar
from brain.explanation_events import ExplanationEvent, ExplanationType
from brain.explanation_types import LinePurpose, ClassPurpose, FunctionPurpose, PipelinePurpose
from brain.concept_store import ConceptStore
from brain.relation_store import RelationStore
from brain.self_knowledge import SelfKnowledge


class ExplanationEngine:
    """Generates honest, data-grounded explanations from decision traces."""

    def __init__(self, 
                 korean_grammar: KoreanGrammar,
                 english_grammar: EnglishGrammar,
                 concept_store: ConceptStore,
                 relation_store: RelationStore,
                 self_knowledge: SelfKnowledge):
        self.korean = korean_grammar
        self.english = english_grammar
        self.concepts = concept_store
        self.relations = relation_store
        self.self_knowledge = self_knowledge

    def explain_code_line(self, 
                          line: str,
                          line_number: int,
                          node: IRNode,
                          structure_context: dict,
                          reasoning_state: ReasoningState) -> ExplanationEvent:
        """Generate explanation for a specific code line."""
        line_purpose = self._analyze_line_purpose(line, node, structure_context)
        decision = self._find_relevant_decision(line_purpose, reasoning_state)
        
        korean = self.korean.compose_explanation(line_purpose, decision, structure_context)
        english = self.english.compose_explanation(line_purpose, decision, structure_context)
        
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="LINE",
            file_number=line_number,
            file_path=structure_context.get("file", ""),
            line_number=line_number,
            code=line,
            explanation_type=self._determine_explanation_type(line_purpose),
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=decision,
            structure_context=structure_context
        )

    def _analyze_line_purpose(self, line: str, node: IRNode, context: dict) -> LinePurpose:
        purpose_map = {
            IRNodeType.CLASS: "define_class",
            IRNodeType.FUNCTION: "define_function", 
            IRNodeType.RETURN: "return_result",
            IRNodeType.ASSIGNMENT: "assign_variable",
            IRNodeType.CALL: "function_call",
            IRNodeType.IF: "conditional",
            IRNodeType.FOR: "loop",
            IRNodeType.WHILE: "loop",
            IRNodeType.BINARY_OP: "computation",
            IRNodeType.BLOCK: "block",
            IRNodeType.VARIABLE: "variable_declaration",
            IRNodeType.PARAMETER: "parameter",
            IRNodeType.PROPERTY: "property",
            IRNodeType.CONSTRUCTOR: "constructor",
        }
        return LinePurpose(
            line_type=purpose_map.get(node.node_type, "statement"),
            node=node,
            context=context,
            detail=self._extract_detail(line, node)
        )

    def _extract_detail(self, line: str, node: IRNode) -> str:
        """Extract semantic detail from line and node."""
        stripped = line.strip()
        if stripped.startswith("return"):
            return "반환문"
        elif stripped.startswith("if"):
            return "조건문"
        elif stripped.startswith("for") or stripped.startswith("while"):
            return "반복문"
        elif "=" in stripped and not stripped.startswith("if") and not stripped.startswith("for"):
            return "할당"
        elif "(" in stripped and ")" in stripped:
            return "함수 호출"
        return "명령문"

    def _find_relevant_decision(self, purpose: LinePurpose, state: ReasoningState) -> Optional[DecisionTrace]:
        """Find the decision trace that caused this line."""
        for decision in state.decisions:
            if self._decision_matches_purpose(decision, purpose):
                return decision
        return None

    def _decision_matches_purpose(self, decision: DecisionTrace, purpose: LinePurpose) -> bool:
        """Check if a decision matches the line purpose."""
        if purpose.line_type in ["define_class", "define_function"]:
            return decision.decision_type in [DecisionType.ADD_CLASS, DecisionType.ADD_FUNCTION]
        elif purpose.line_type == "pipeline_operation":
            return decision.decision_type == DecisionType.ADD_PIPELINE_STAGE
        return False

    def _determine_explanation_type(self, purpose: LinePurpose) -> ExplanationType:
        type_map = {
            "define_class": ExplanationType.CLASS_START,
            "define_function": ExplanationType.FUNCTION_START,
            "pipeline_operation": ExplanationType.PIPELINE_STAGE,
            "return_result": ExplanationType.LINE,
            "assign_variable": ExplanationType.LINE,
            "function_call": ExplanationType.LINE,
            "conditional": ExplanationType.LINE,
            "loop": ExplanationType.LINE,
            "computation": ExplanationType.LINE,
        }
        return type_map.get(purpose.line_type, ExplanationType.LINE)

    # Specific explainers
    def explain_file_start(self, file_ir, state: ReasoningState) -> ExplanationEvent:
        purpose = f"Generate {file_ir.path}"
        korean = self.korean.explain_file_generation(file_ir.path, state)
        english = self.english.explain_file_generation(file_ir.path, state)
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="FILE_START",
            file_path=file_ir.path,
            line_number=0,
            code="",
            explanation_type=ExplanationType.FILE_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"file": file_ir.path}
        )

    def explain_generation_start(self, project_ir, state: ReasoningState) -> ExplanationEvent:
        korean = self.korean.explain_generation_start(project_ir.name, state)
        english = self.english.explain_generation_start(project_ir.name, state)
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="START",
            file_path="",
            line_number=0,
            code="",
            explanation_type=ExplanationType.GENERATION_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"purpose": f"Generate {project_ir.name} project"}
        )

    def explain_generation_complete(self, project_ir, state: ReasoningState) -> ExplanationEvent:
        korean = self.korean.explain_generation_complete(project_ir.name, state)
        english = self.english.explain_generation_complete(project_ir.name, state)
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="COMPLETE",
            file_path="",
            line_number=0,
            code="",
            explanation_type=ExplanationType.GENERATION_COMPLETE,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"purpose": "complete"}
        )

    def explain_class_start(self, node: IRNode, state: ReasoningState) -> ExplanationEvent:
        purpose = ClassPurpose(
            name=node.name,
            properties=[c.name for c in node.children if c.node_type == IRNodeType.PROPERTY],
            reason=self._find_class_reason(node.name, state),
            alternatives=self._get_class_alternatives(node.name, state)
        )
        korean = self.korean.explain_class_creation(purpose)
        english = self.english.explain_class_creation(purpose)
        
        decision = self._find_relevant_decision(
            LinePurpose("define_class", node, {}, ""), state
        )
        
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="CLASS_START",
            file_path=state.model.language,
            line_number=0,
            code="",
            explanation_type=ExplanationType.CLASS_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=decision,
            structure_context={"current_class": node.name}
        )

    def explain_function_start(self, node: IRNode, state: ReasoningState) -> ExplanationEvent:
        purpose = FunctionPurpose(
            name=node.name,
            params=[(c.name, c.type_ref) for c in node.children if c.node_type == IRNodeType.PARAMETER],
            return_type=node.type_ref or "void",
            operations=state.model.operation_order,
            reason=self._find_function_reason(node.name, state),
            alternatives=self._get_function_alternatives(node.name, state)
        )
        korean = self.korean.explain_function_creation(purpose)
        english = self.english.explain_function_creation(purpose)
        
        decision = self._find_relevant_decision(
            LinePurpose("define_function", node, {}, ""), state
        )
        
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="FUNCTION_START",
            file_path=state.model.language,
            line_number=0,
            code="",
            explanation_type=ExplanationType.FUNCTION_START,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=decision,
            structure_context={"current_function": node.name}
        )

    def explain_pipeline_stage(self, node: IRNode, state: ReasoningState) -> ExplanationEvent:
        stage_name = node.name
        operation = next((op for op in state.model.operations if op.name == stage_name), None)
        
        korean = self.korean.explain_pipeline_stage(stage_name, operation, state.model.operation_order)
        english = self.english.explain_pipeline_stage(stage_name, operation, state.model.operation_order)
        
        decision = self._find_relevant_decision(
            LinePurpose("pipeline_operation", node, {}, stage_name), state
        )
        
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="PIPELINE_STAGE",
            file_path=state.model.language,
            line_number=0,
            code="",
            explanation_type=ExplanationType.PIPELINE_STAGE,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=decision,
            structure_context={"current_pipeline_stage": stage_name}
        )

    def explain_structure_end(self, struct_type: str, name: str, state: ReasoningState) -> ExplanationEvent:
        korean = self.korean.explain_structure_completion(struct_type, name)
        english = self.english.explain_structure_completion(struct_type, name)
        return ExplanationEvent(
            timestamp=datetime.now(),
            event_type="STRUCTURE_END",
            file_path=state.model.language,
            line_number=0,
            code="",
            explanation_type=ExplanationType.STRUCTURE_END,
            korean_explanation=korean,
            english_explanation=english,
            decision_trace=None,
            structure_context={"completed": f"{struct_type}:{name}"}
        )

    def _find_class_reason(self, class_name: str, state: ReasoningState) -> str:
        """Find why this class was needed from reasoning state."""
        # Check entities
        for entity in state.model.entities:
            if entity.name.lower() == class_name.lower() or entity.name.lower() in class_name.lower():
                return f"{entity.name} 데이터를 통합 관리하기"
        
        # Check domain
        if state.model.domain:
            return f"{state.model.domain} 도메인 모델링을 위해"
        
        # Check operations that might need this class
        for op in state.model.operations:
            if class_name.lower() in op.name.lower():
                return f"{op.name} 연산을 위한 데이터 구조로"
        
        return "데이터 구조를 정의하기"

    def _find_function_reason(self, func_name: str, state: ReasoningState) -> str:
        """Find why this function was needed."""
        # Check if function matches any operation
        for op in state.model.operations:
            if op.name == func_name or func_name in op.name:
                return f"{op.name} 연산 처리를 위해"
        
        # Check operation order
        for op in state.model.operation_order:
            if op == func_name or func_name in op:
                return f"{op} 파이프라인 단계로"
        
        return "기능 구현을 위해"

    def _get_class_alternatives(self, class_name: str, state: ReasoningState) -> list:
        """Get alternative designs considered for this class."""
        # Could be expanded to track actual alternatives from reasoning
        return []

    def _get_function_alternatives(self, func_name: str, state: ReasoningState) -> list:
        """Get alternative implementations considered."""
        return []