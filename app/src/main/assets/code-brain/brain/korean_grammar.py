"""Korean Grammar - Grammatical composition for explanations (NO TEMPLATES)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import re

from brain.decision_trace import DecisionTrace, Alternative
from brain.explanation_events import ExplanationType
from brain.think_engine import Constraint
from brain.explanation_types import ClassPurpose, FunctionPurpose, LinePurpose


class KoreanGrammar:
    """Generates grammatically correct Korean from semantic structures (NO TEMPLATES)."""
    
    def __init__(self):
        self.particles = ParticleRules()
        self.honorifics = HonorificRules()
        self.connectives = ConnectiveRules()
        self.verb_endings = VerbEndingRules()
    
    def compose_explanation(self, purpose: LinePurpose, decision: Optional[DecisionTrace], context: dict) -> str:
        """Compose line-level explanation from semantic components."""
        parts = []
        
        # Main clause: What was done
        main = self._compose_line_main(purpose)
        parts.append(main)
        
        # Reason clause: Why (from decision trace)
        if decision and decision.evidence:
            reason = self._compose_reason_clause(decision)
            parts.append(f"→ {reason} 위해 필요합니다.")
        
        # Constraint clause
        if decision and decision.constraints_applied:
            constraint = self._compose_constraint_clause(decision.constraints_applied)
            parts.append(f"({constraint})")
        
        # Alternative mention
        if decision and decision.alternatives_considered:
            alt = self._compose_alternative_clause(decision.alternatives_considered)
            parts.append(f" ({alt})")
        
        return " ".join(parts)
    
    def _compose_line_main(self, purpose: LinePurpose) -> str:
        """Compose main clause for a code line."""
        verb_map = {
            "define_class": "클래스를 정의합니다",
            "define_function": "함수를 정의합니다",
            "pipeline_operation": "파이프라인 단계를 수행합니다",
            "return_result": "결과를 반환합니다",
            "assign_variable": "변수에 값을 할당합니다",
            "function_call": "함수를 호출합니다",
            "conditional": "조건에 따라 분기합니다",
            "loop": "반복을 수행합니다",
            "computation": "연산을 수행합니다",
            "variable_declaration": "변수를 선언합니다",
            "parameter": "매개변수를 정의합니다",
            "property": "속성을 정의합니다",
            "constructor": "생성자를 정의합니다",
            "statement": "명령문을 실행합니다",
        }
        verb = verb_map.get(purpose.line_type, "처리합니다")
        
        # Get subject based on context
        subject = self._get_subject(purpose)
        if subject:
            return f"{subject}{self.particles.topic(subject)} {verb}."
        return f"{verb}."
    
    def _get_subject(self, purpose: LinePurpose) -> str:
        """Get appropriate subject for the line."""
        if purpose.line_type == "define_class":
            return purpose.node.name
        elif purpose.line_type == "define_function":
            return purpose.node.name
        elif purpose.context.get("current_class"):
            return purpose.context["current_class"]
        elif purpose.context.get("current_function"):
            return purpose.context["current_function"]
        return ""
    
    def _compose_reason_clause(self, decision: DecisionTrace) -> str:
        """Compose reason clause from decision evidence."""
        # Convert evidence to gerund form for "위해"
        evidence_text = " ".join(decision.evidence[:3])
        return self._to_gerund(evidence_text)
    
    def _to_gerund(self, text: str) -> str:
        """Convert verb/adjective to gerund form for '위해'."""
        # Simple conversion rules
        text = text.strip()
        # Already in gerund form
        if text.endswith(("기", "기 위해")):
            return text
        
        # Convert common patterns
        conversions = {
            "처리": "처리하기",
            "계산": "계산하기",
            "생성": "생성하기",
            "정의": "정의하기",
            "관리": "관리하기",
            "처리": "처리하기",
            "수행": "수행하기",
            "실행": "실행하기",
            "구현": "구현하기",
            "처리": "처리하기",
        }
        
        for k, v in conversions.items():
            if text.endswith(k):
                return text[:-len(k)] + v
        
        # Default: add 기
        if text.endswith("다"):
            return text[:-1] + "기"
        return text + "기"
    
    def _compose_constraint_clause(self, constraints: List[str]) -> str:
        """Compose constraint clause."""
        constraint_map = {
            "language": "언어 제약",
            "platform": "플랫폼 제약",
            "performance": "성능 요구",
            "security": "보안 요구",
        }
        mapped = [constraint_map.get(c, c) for c in constraints]
        return ", ".join(mapped) + " 적용"
    
    def _compose_alternative_clause(self, alternatives: List[Alternative]) -> str:
        """Compose alternative mention clause."""
        if not alternatives:
            return ""
        alt_texts = [f"{a.option}({a.reason_rejected})" for a in alternatives[:2]]
        return f"다른 방법({', '.join(alt_texts)})도 고려했으나 제외했습니다."
    
    def explain_file_generation(self, file_path: str, state) -> str:
        parts = []
        parts.append(f"{self.particles.topic(file_path)} 파일을 생성합니다.")
        
        # Infer purpose from file path
        if "model" in file_path.lower() or "entity" in file_path.lower():
            parts.append("데이터 모델을 정의하기 위해 필요합니다.")
        elif "service" in file_path.lower():
            parts.append("비즈니스 로직을 캡슐화하기 위해 필요합니다.")
        elif "controller" in file_path.lower() or "activity" in file_path.lower():
            parts.append("사용자 인터페이스와 로직을 연결하기 위해 필요합니다.")
        elif "main" in file_path.lower() or "entry" in file_path.lower():
            parts.append("프로그램 진입점을 제공하기 위해 필요합니다.")
        else:
            parts.append("요청된 기능을 구현하기 위해 필요합니다.")
        
        return " ".join(parts)
    
    def explain_generation_start(self, project_name: str, state) -> str:
        return f"{self.particles.topic(project_name)} 프로젝트 생성을 시작합니다. 요청된 기능을 분석하여 필요한 파일들을 구성합니다."
    
    def explain_generation_complete(self, project_name: str, state) -> str:
        return f"{project_name} 프로젝트 생성이 완료되었습니다. 모든 파일과 의존성이 구성되었습니다."
    
    def explain_class_creation(self, purpose: ClassPurpose) -> str:
        """'Student 클래스를 추가했습니다. → 학생의 이름, ID, 점수를 하나의 구조로 관리하기 위해 필요합니다.'"""
        parts = []
        
        # Main clause
        main = f"{self.particles.topic(purpose.name)} 클래스를 추가했습니다."
        parts.append(main)
        
        # Reason clause
        if purpose.reason:
            reason = self._to_gerund(purpose.reason)
            parts.append(f"→ {reason} 위해 필요합니다.")
        
        # Properties detail
        if purpose.properties:
            props = ", ".join(purpose.properties)
            parts.append(f"속성: {props}를 포함합니다.")
        
        return " ".join(parts)
    
    def explain_function_creation(self, purpose: FunctionPurpose) -> str:
        """'calculateAverage 함수를 추가했습니다. → 평균 점수를 계산하는 요구사항을 처리하기 위해 필요합니다.'"""
        parts = []
        
        # Main clause
        main = f"{self.particles.topic(purpose.name)} 함수를 추가했습니다."
        parts.append(main)
        
        # Reason clause
        if purpose.reason:
            parts.append(f"→ {self._to_gerund(purpose.reason)} 위해 필요합니다.")
        
        # Parameters
        if purpose.params:
            params = ", ".join([f"{n}({t})" for n, t in purpose.params])
            parts.append(f"매개변수: {params}를 받습니다.")
        
        # Operations pipeline
        if purpose.operations:
            ops = " → ".join(purpose.operations)
            parts.append(f"내부 처리: {ops} 순서로 수행합니다.")
        
        return " ".join(parts)
    
    def explain_pipeline_stage(self, stage_name: str, operation, operation_order: List[str]) -> str:
        """'{stage_name} 단계를 추가했습니다. → {reason} 위해 필요합니다.'"""
        purpose_map = {
            "remove_duplicates": "중복 데이터를 제거하여 고유한 값만 남기기",
            "calculate_average": "평균값을 계산하여 기준점으로 사용하기",
            "filter_above": "평균 이상의 값만 필터링하기",
            "filter_below": "평균 미만의 값만 필터링하기",
            "sort_descending": "내림차순으로 정렬하여 큰 값이 먼저 오게 하기",
            "sort_ascending": "오름차순으로 정렬하여 작은 값이 먼저 오게 하기",
            "filter_even": "짝수 값만 선별하기",
            "sum_list": "모든 값의 합계를 계산하기",
        }
        
        korean_purpose = purpose_map.get(stage_name, f"{stage_name} 연산 수행")
        
        parts = [f"{self.particles.topic(stage_name)} 단계를 추가했습니다."]
        parts.append(f"→ {korean_purpose} 위해 필요합니다.")
        
        # Pipeline context
        if stage_name in operation_order:
            idx = operation_order.index(stage_name)
            if idx > 0:
                parts.append(f"이전 단계({operation_order[idx-1]})의 결과를 입력으로 받습니다.")
            if idx < len(operation_order) - 1:
                parts.append(f"다음 단계({operation_order[idx+1]})로 결과를 전달합니다.")
        
        return " ".join(parts)
    
    def explain_structure_completion(self, struct_type: str, name: str) -> str:
        """Explain completion of class/function"""
        type_kr = {"class": "클래스", "function": "함수", "file": "파일"}.get(struct_type, struct_type)
        return f"{type_kr} {name} 구현을 완료했습니다."
    
    def explain_alternatives(self, alternatives: List[Alternative]) -> str:
        """Explain why alternatives were rejected"""
        if not alternatives:
            return "대안이 고려되지 않았습니다."
        
        parts = ["다른 방법들도 고려했습니다:"]
        for alt in alternatives:
            parts.append(f"- {alt.option}: {alt.reason_rejected} (신뢰도: {alt.confidence:.0%})")
        return " ".join(parts)
    
    def explain_constraint(self, constraint: Constraint) -> str:
        """Explain a constraint that drove a decision"""
        constraint_map = {
            "language": f"{constraint.value} 언어로 구현해야 합니다.",
            "platform": f"{constraint.value} 플랫폼에서 동작해야 합니다.",
            "performance": "높은 성능이 요구됩니다.",
            "security": "보안 요구사항을 충족해야 합니다.",
        }
        return constraint_map.get(constraint.kind, f"{constraint.name}={constraint.value} 제약이 있습니다.")
    
    def explain_decision(self, decision: DecisionTrace) -> str:
        """Generate explanation for a decision."""
        parts = []
        
        # What was decided
        action_map = {
            "add_class": f"{decision.chosen_option} 클래스를 추가했습니다",
            "add_function": f"{decision.chosen_option} 함수를 추가했습니다",
            "choose_algorithm": f"{decision.chosen_option} 알고리즘을 선택했습니다",
            "choose_structure": f"{decision.chosen_option} 구조를 선택했습니다",
            "add_pipeline_stage": f"{decision.chosen_option} 단계를 추가했습니다",
            "select_hypothesis": f"{decision.chosen_option} 접근법을 선택했습니다",
            "plan_action": f"{decision.chosen_option} 액션을 계획했습니다",
        }
        action_text = action_map.get(decision.decision_type.value, f"{decision.chosen_option}을(를) 결정했습니다")
        parts.append(action_text + ".")
        
        # Why
        if decision.evidence:
            reason = " ".join(decision.evidence[:3])
            parts.append(f"→ {self._to_gerund(reason)} 위해 필요합니다.")
        
        # Constraints
        if decision.constraints_applied:
            parts.append(f"({self._compose_constraint_clause(decision.constraints_applied)})")
        
        # Alternatives
        if decision.alternatives_considered:
            parts.append(self._compose_alternative_clause(decision.alternatives_considered))
        
        return " ".join(parts)


class ParticleRules:
    """Korean particle selection rules."""
    
    def topic(self, noun: str) -> str:
        """Select topic particle (은/는)."""
        if not noun:
            return ""
        # Check if noun ends with consonant (받침)
        last_char = noun[-1]
        if '가' <= last_char <= '힣':
            # Has jongseong if (ord(last_char) - ord('가')) % 28 != 0
            has_jongseong = (ord(last_char) - ord('가')) % 28 != 0
            return f"{noun}은" if has_jongseong else f"{noun}는"
        return f"{noun}은"  # Default for non-Korean
    
    def subject(self, noun: str) -> str:
        """Select subject particle (이/가)."""
        if not noun:
            return ""
        last_char = noun[-1]
        if '가' <= last_char <= '힣':
            has_jongseong = (ord(last_char) - ord('가')) % 28 != 0
            return f"{noun}이" if has_jongseong else f"{noun}가"
        return f"{noun}이"
    
    def object(self, noun: str) -> str:
        """Select object particle (을/를)."""
        if not noun:
            return ""
        last_char = noun[-1]
        if '가' <= last_char <= '힣':
            has_jongseong = (ord(last_char) - ord('가')) % 28 != 0
            return f"{noun}을" if has_jongseong else f"{noun}를"
        return f"{noun}을"


class HonorificRules:
    """Honorific level rules."""
    
    def verb_ending(self, verb: str, politeness: str = "polite") -> str:
        """Apply appropriate verb ending."""
        endings = {
            "casual": {"declarative": "아/어", "interrogative": "아/어?"},
            "polite": {"declarative": "아요/어요", "interrogative": "아요/어요?"},
            "formal": {"declarative": "습니다", "interrogative": "습니까?"},
            "honorific": {"declarative": "십니다", "interrogative": "십니까?"},
        }
        return endings.get(politeness, {}).get("declarative", "아요/어요")


class ConnectiveRules:
    """Connective rules for clause linking."""
    
    def reason(self) -> str:
        return "→"
    
    def contrast(self) -> str:
        return "그러나 "
    
    def addition(self) -> str:
        return "그리고 "
    
    def alternative(self) -> str:
        return "또는 "


class VerbEndingRules:
    """Verb ending rules."""
    
    def gerund(self, verb: str) -> str:
        """Convert to gerund form (for '위해')."""
        # Simple rules for common verbs
        gerund_map = {
            "처리": "처리하기",
            "계산": "계산하기",
            "생성": "생성하기",
            "정의": "정의하기",
            "관리": "관리하기",
            "수행": "수행하기",
            "실행": "실행하기",
            "구현": "구현하기",
            "처리": "처리하기",
            "제거": "제거하기",
            "필터링": "필터링하기",
            "정렬": "정렬하기",
            "저장": "저장하기",
            "불러오기": "불러오기",
            "검색": "검색하기",
            "조회": "조회하기",
            "수정": "수정하기",
            "삭제": "삭제하기",
            "추가": "추가하기",
        }
        
        for k, v in gerund_map.items():
            if verb.endswith(k):
                return verb[:-len(k)] + v
        
        if verb.endswith("다"):
            return verb[:-1] + "기"
        return verb + "기"
    
    def past(self, verb: str) -> str:
        """Convert to past tense."""
        if verb.endswith("다"):
            stem = verb[:-1]
            if stem.endswith(("하", "되")):
                return stem + "였습니다"
            return stem + "었습니다"
        return verb + "었습니다"