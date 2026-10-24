"""QA Engine - Answers user questions based on reasoning trace and event stream."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from datetime import datetime

from brain.decision_trace import DecisionTrace, Alternative, DecisionType
from brain.explanation_events import ExplanationEvent, ExplanationType, GenerationEvent
from brain.think_engine import ReasoningState
from brain.korean_grammar import KoreanGrammar
from brain.english_grammar import EnglishGrammar


class QuestionType(Enum):
    WHY_ADDED = "why_added"
    WHY_MODIFIED = "why_modified"
    WHY_STRUCTURE = "why_structure"
    WHY_ALGORITHM = "why_algorithm"
    WHAT_CHANGED = "what_changed"
    WHAT_ADDED = "what_added"
    ALTERNATIVES = "alternatives"
    CONSTRAINTS = "constraints"
    VERIFICATION_RESULT = "verification_result"
    REPAIR_REASON = "repair_reason"
    LINE_EXPLANATION = "line_explanation"


@dataclass
class ParsedQuestion:
    question_type: QuestionType
    target: str
    language: str
    context: dict = field(default_factory=dict)


class QAEngine:
    """Answers user questions based on reasoning trace and event stream."""

    def __init__(self, 
                 explanation_engine,
                 reasoning_state: ReasoningState,
                 event_stream: List[dict]):
        self.explanation_engine = explanation_engine
        self.reasoning_state = reasoning_state
        self.event_stream = event_stream
        self.conversation_history = []
        self.korean = KoreanGrammar()
        self.english = EnglishGrammar()
    
    def answer(self, question: str, language: str = "ko") -> str:
        """Answer user question using actual reasoning data."""
        
        # Parse the question
        parsed = self._parse_question(question, language)
        
        # Route to appropriate handler
        handlers = {
            QuestionType.WHY_ADDED: self._answer_why_added,
            QuestionType.WHY_MODIFIED: self._answer_why_modified,
            QuestionType.WHY_STRUCTURE: self._answer_why_structure,
            QuestionType.WHY_ALGORITHM: self._answer_why_algorithm,
            QuestionType.WHAT_CHANGED: self._answer_what_changed,
            QuestionType.WHAT_ADDED: self._answer_what_added,
            QuestionType.ALTERNATIVES: self._answer_alternatives,
            QuestionType.CONSTRAINTS: self._answer_constraints,
            QuestionType.VERIFICATION_RESULT: self._answer_verification,
            QuestionType.REPAIR_REASON: self._answer_repair,
            QuestionType.LINE_EXPLANATION: self._answer_line_explanation,
        }
        
        handler = handlers.get(parsed.question_type)
        if not handler:
            return self._default_answer(language)
        
        answer = handler(parsed, language)
        
        # Record in conversation history
        self.conversation_history.append({"q": question, "a": answer})
        
        return answer
    
    def _parse_question(self, question: str, language: str) -> ParsedQuestion:
        """Parse natural language question to structured type."""
        q_lower = question.lower()
        
        # Korean patterns
        if language == "ko":
            if any(p in q_lower for p in ["왜 추가", "왜 넣", "왜 만들", "왜 생"]):
                return ParsedQuestion(QuestionType.WHY_ADDED, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["왜 수정", "왜 변경", "왜 고치", "왜 바꿨"]):
                return ParsedQuestion(QuestionType.WHY_MODIFIED, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["왜 이 구조", "왜 이렇게", "구조 선택", "왜 이 아키텍처"]):
                return ParsedQuestion(QuestionType.WHY_STRUCTURE, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["왜 이 알고리즘", "왜 이 방식", "알고리즘 선택", "왜 이 방식"]):
                return ParsedQuestion(QuestionType.WHY_ALGORITHM, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["뭐가 변경", "뭐가 바뀌", "무엇을 수정", "무엇을 바꿨", "뭐 수정"]):
                return ParsedQuestion(QuestionType.WHAT_CHANGED, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["뭐 추가", "무엇을 추가", "뭐 넣", "뭐 만들"]):
                return ParsedQuestion(QuestionType.WHAT_ADDED, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["다른 방법", "대안", "다른 방식", "다른 알고리즘", "다른 길"]):
                return ParsedQuestion(QuestionType.ALTERNATIVES, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["제약", "조건", "제한", "요구사항", "컨스트레인트"]):
                return ParsedQuestion(QuestionType.CONSTRAINTS, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["검증", "테스트", "빌드", "컴파일", "버전"]):
                return ParsedQuestion(QuestionType.VERIFICATION_RESULT, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["수정 이유", "왜 고쳤", "에러 수정", "버그 수정", "왜 고쳤어"]):
                return ParsedQuestion(QuestionType.REPAIR_REASON, self._extract_target(question), language, {})
            if any(p in q_lower for p in ["이 줄", "이 코드", "이 라인", "이 줄 설명", "코드 설명"]):
                return ParsedQuestion(QuestionType.LINE_EXPLANATION, self._extract_target(question), language, {})
        
        # English patterns
        if any(p in q_lower for p in ["why add", "why create", "why include", "why did you add"]):
            return ParsedQuestion(QuestionType.WHY_ADDED, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["why modif", "why chang", "why edit", "why did you change"]):
            return ParsedQuestion(QuestionType.WHY_MODIFIED, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["why this structure", "why this architecture", "why structured"]):
            return ParsedQuestion(QuestionType.WHY_STRUCTURE, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["why this algorithm", "why this approach", "why this method"]):
            return ParsedQuestion(QuestionType.WHY_ALGORITHM, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["what changed", "what modified", "what was changed"]):
            return ParsedQuestion(QuestionType.WHAT_CHANGED, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["what was added", "what did you add", "what did you create"]):
            return ParsedQuestion(QuestionType.WHAT_ADDED, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["alternative", "other way", "other approach", "other option", "other method"]):
            return ParsedQuestion(QuestionType.ALTERNATIVES, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["constraint", "requirement", "condition", "limitation"]):
            return ParsedQuestion(QuestionType.CONSTRAINTS, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["verification", "test", "build", "compile", "check"]):
            return ParsedQuestion(QuestionType.VERIFICATION_RESULT, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["why fix", "why repair", "why did you fix", "why did you repair"]):
            return ParsedQuestion(QuestionType.REPAIR_REASON, self._extract_target(question), language, {})
        if any(p in q_lower for p in ["this line", "this code", "explain this line", "what does this line"]):
            return ParsedQuestion(QuestionType.LINE_EXPLANATION, self._extract_target(question), language, {})
        
        return ParsedQuestion(QuestionType.WHY_ADDED, "", language, {})
    
    def _extract_target(self, question: str) -> str:
        """Extract what the user is asking about from the question."""
        import re
        # Match quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        if quoted:
            return quoted[0]
        # Match Korean particles
        for particle in ["을", "를", "이", "가", "은", "는", "에", "에서"]:
            match = re.search(rf'(\S+){particle}', question)
            if match:
                return match.group(1)
        # Match English patterns
        for particle in ["the", "a", "an"]:
            match = re.search(rf'{particle}\s+(\w+)', question)
            if match:
                return match.group(1)
        return ""
    
    # Answer handlers
    def _answer_why_added(self, parsed: ParsedQuestion, language: str) -> str:
        target = parsed.target
        
        # Search in generation events - check class_start, function_start, pipeline_stage events
        for event in self.event_stream:
            if event.get("explanation"):
                ctx = event.get("structure_context", {})
                # Check if target matches current class, function, or pipeline stage
                if (target and (
                    target in ctx.get("current_class", "") or
                    target in ctx.get("current_function", "") or
                    target in ctx.get("current_pipeline_stage", "") or
                    target in event.get("code", "")
                )):
                    if language == "ko":
                        return event["explanation"].get("korean_explanation", "")
                    return event["explanation"].get("english_explanation", "")
        
        # Search in decisions
        for decision in self.reasoning_state.decisions:
            if target in decision.chosen_option:
                if language == "ko":
                    return self.explanation_engine.korean.explain_decision(decision)
                return self.explanation_engine.english.explain_decision(decision)
        
        return f"'{target}'에 대한 추가 이유를 찾을 수 없습니다." if language == "ko" else f"Could not find reason for adding '{target}'."
    
    def _answer_why_modified(self, parsed: ParsedQuestion, language: str) -> str:
        target = parsed.target
        for event in self.event_stream:
            if event.get("explanation") and event["explanation"].get("explanation_type") in [
                ExplanationType.WHY_MODIFIED.value, ExplanationType.REPAIR.value
            ]:
                if target in event.get("code", ""):
                    if language == "ko":
                        return event["explanation"].get("korean_explanation", "")
                    return event["explanation"].get("english_explanation", "")
        
        return f"'{target}'에 대한 수정 이유를 찾을 수 없습니다." if language == "ko" else f"Could not find modification reason for '{target}'."
    
    def _answer_why_structure(self, parsed: ParsedQuestion, language: str) -> str:
        # Find architecture decisions
        for decision in self.reasoning_state.decisions:
            if decision.decision_type in [DecisionType.CHOOSE_STRUCTURE, DecisionType.ADD_PIPELINE_STAGE]:
                if language == "ko":
                    return self.explanation_engine.korean.explain_decision(decision)
                return self.explanation_engine.english.explain_decision(decision)
        
        # Fallback to pipeline explanation
        if self.reasoning_state.model.operation_order:
            ops = " → ".join(self.reasoning_state.model.operation_order)
            if language == "ko":
                return f"요청된 작업을 처리하기 위해 {ops} 순서의 파이프라인으로 구성했습니다."
            return f"Structured as pipeline: {ops} to handle the requested operations."
        
        return "구조 선택 이유를 찾을 수 없습니다." if language == "ko" else "Could not find structure rationale."
    
    def _answer_why_algorithm(self, parsed: ParsedQuestion, language: str) -> str:
        for decision in self.reasoning_state.decisions:
            if decision.decision_type == DecisionType.CHOOSE_ALGORITHM:
                if language == "ko":
                    return self.explanation_engine.korean.explain_decision(decision)
                return self.explanation_engine.english.explain_decision(decision)
        
        # Fallback to selected hypothesis
        if self.reasoning_state.selected_hypothesis:
            hyp = self.reasoning_state.selected_hypothesis
            if language == "ko":
                return f"선택된 접근법: {hyp.description} (신뢰도: {hyp.confidence:.0%}). 근거: {', '.join(hyp.evidence[:3])}"
            return f"Selected approach: {hyp.description} (confidence: {hyp.confidence:.0%}). Evidence: {', '.join(hyp.evidence[:3])}"
        
        return "알고리즘 선택 이유를 찾을 수 없습니다." if language == "ko" else "Could not find algorithm rationale."
    
    def _answer_what_changed(self, parsed: ParsedQuestion, language: str) -> str:
        changes = [e for e in self.event_stream 
                   if e.get("explanation") and e["explanation"].get("explanation_type") in 
                   (ExplanationType.WHY_MODIFIED.value, ExplanationType.REPAIR.value)]
        
        if not changes:
            return "수정된 내용이 없습니다." if language == "ko" else "No changes detected."
        
        if language == "ko":
            parts = ["다음 내용이 수정되었습니다:"]
            for c in changes[:5]:
                parts.append(f"- {c.get('file_path', '')}:{c.get('line_number', '')} - {c['explanation'].get('korean_explanation', '')}")
            return "\n".join(parts)
        
        parts = ["The following changes were made:"]
        for c in changes[:5]:
            parts.append(f"- {c.get('file_path', '')}:{c.get('line_number', '')} - {c['explanation'].get('english_explanation', '')}")
        return "\n".join(parts)
    
    def _answer_what_added(self, parsed: ParsedQuestion, language: str) -> str:
        additions = [e for e in self.event_stream 
                     if e.get("event_type") in ("class_start", "function_start", "pipeline_stage")]
        
        if language == "ko":
            parts = ["다음 구성요소가 추가되었습니다:"]
            for a in additions:
                if a.get("explanation"):
                    parts.append(f"- {a['explanation'].get('korean_explanation', '')}")
            return "\n".join(parts)
        
        parts = ["The following components were added:"]
        for a in additions:
            if a.get("explanation"):
                parts.append(f"- {a['explanation'].get('english_explanation', '')}")
        return "\n".join(parts)
    
    def _answer_alternatives(self, parsed: ParsedQuestion, language: str) -> str:
        all_alternatives = []
        for decision in self.reasoning_state.decisions:
            all_alternatives.extend(decision.alternatives_considered)
        
        if not all_alternatives:
            return "고려된 대안이 없습니다." if language == "ko" else "No alternatives were considered."
        
        if language == "ko":
            return self.explanation_engine.korean.explain_alternatives(all_alternatives)
        return self.explanation_engine.english.explain_alternatives(all_alternatives)
    
    def _answer_constraints(self, parsed: ParsedQuestion, language: str) -> str:
        constraints = self.reasoning_state.model.constraints
        
        if not constraints:
            return "적용된 제약사항이 없습니다." if language == "ko" else "No constraints applied."
        
        if language == "ko":
            parts = ["적용된 제약사항:"]
            for c in constraints:
                parts.append(f"- {self.explanation_engine.korean.explain_constraint(c)}")
            return "\n".join(parts)
        
        parts = ["Applied constraints:"]
        for c in constraints:
            parts.append(f"- {self.explanation_engine.english.explain_constraint(c)}")
        return "\n".join(parts)
    
    def _answer_verification(self, parsed: ParsedQuestion, language: str) -> str:
        for event in self.event_stream:
            if event.get("explanation") and event["explanation"].get("explanation_type") == ExplanationType.VERIFICATION.value:
                if language == "ko":
                    return event["explanation"].get("korean_explanation", "")
                return event["explanation"].get("english_explanation", "")
        
        return "검증 결과가 없습니다." if language == "ko" else "No verification results available."
    
    def _answer_repair(self, parsed: ParsedQuestion, language: str) -> str:
        for event in self.event_stream:
            if event.get("explanation") and event["explanation"].get("explanation_type") == ExplanationType.REPAIR.value:
                if parsed.target in event.get("code", ""):
                    if language == "ko":
                        return event["explanation"].get("korean_explanation", "")
                    return event["explanation"].get("english_explanation", "")
        
        return "수정 이유를 찾을 수 없습니다." if language == "ko" else "Could not find repair reason."
    
    def _answer_line_explanation(self, parsed: ParsedQuestion, language: str) -> str:
        target = parsed.target
        
        for event in self.event_stream:
            if event.get("event_type") == "line":
                if target in event.get("code", "") or target in str(event.get("line_number", "")):
                    if language == "ko":
                        return f"[{event.get('file_path', '')}:{event.get('line_number', '')}] {event.get('code', '')}\n→ {event['explanation'].get('korean_explanation', '')}"
                    return f"[{event.get('file_path', '')}:{event.get('line_number', '')}] {event.get('code', '')}\n→ {event['explanation'].get('english_explanation', '')}"
        
        return "해당 줄을 찾을 수 없습니다." if language == "ko" else "Could not find that line."
    
    def _default_answer(self, language: str) -> str:
        if language == "ko":
            return "질문을 이해하지 못했습니다. '왜 추가했어?', '뭐가 변경됐어?', '다른 방법은 없었어?', '3번째 줄 설명해줘' 같은 질문을 해보세요."
        return "I didn't understand the question. Try asking 'Why was this added?', 'What changed?', 'Were there alternatives?', or 'Explain line 3'."