"""Self Knowledge - structured knowledge about RENA-BR itself."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KnowledgeItem:
    key: str
    value: str
    category: str
    language: str = "universal"
    detail_level: str = "standard"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "language": self.language,
            "detail_level": self.detail_level,
            "metadata": self.metadata,
        }


class SelfKnowledgeInterface:
    """Interface for self-knowledge queries."""
    
    def get_identity(self) -> dict:
        raise NotImplementedError
    
    def get_capabilities(self) -> list[str]:
        raise NotImplementedError
    
    def get_architecture_description(self) -> str:
        raise NotImplementedError
    
    def get_supported_languages(self) -> list[str]:
        raise NotImplementedError
    
    def get_code_knowledge(self, topic: str) -> dict | None:
        raise NotImplementedError


class SelfKnowledge(SelfKnowledgeInterface):
    def __init__(self):
        self.knowledge: dict[str, KnowledgeItem] = {}
        self._init_knowledge()
    
    def _init_knowledge(self):
        # Identity
        self._add("identity", "name", "RENA-BR", "identity")
        self._add("identity", "version", "RENA-BR 0.5 Beta", "identity")
        self._add("identity", "type", "AI coding assistant", "identity")
        self._add("identity", "architecture", "Knowledge Graph + Rule Engine + Memory + Reasoning + Planning", "identity")
        self._add("identity", "philosophy", "No transformers, no LLM APIs, no pretrained models", "identity")
        
        self._add("purpose", "main", "코드 생성, 프로젝트 생성, 디버깅, 코드 설명, 학습을 돕는 AI 코딩 어시스턴트", "purpose")
        self._add("purpose", "goal", "사용자의 의도를 이해하고 최적의 코드와 솔루션을 제공", "purpose")
        
        # Capabilities
        self._add("capabilities", "code_generation", "Python, Kotlin, Java, C, C++, JavaScript, Go, Rust에서 함수/클래스 코드 생성", "capabilities")
        self._add("capabilities", "project_generation", "Android, Web, CLI 프로젝트 전체 생성 (Gradle, Manifest, 소스 코드 포함)", "capabilities")
        self._add("capabilities", "debugging", "오류 진단, 원인 분석, 자동 수정 제안", "capabilities")
        self._add("capabilities", "code_explanation", "코드 분석 및 설명, 프로그래밍 개념 설명", "capabilities")
        self._add("capabilities", "learning", "코드 패턴 학습, 사용자 피드백 학습, 오류 패턴 학습", "capabilities")
        self._add("capabilities", "memory", "대화 맥락 기억, 사용자 선호 기억, 과거 오류 기억", "capabilities")
        self._add("capabilities", "refactoring", "코드 리팩토링 및 개선", "capabilities")
        self._add("capabilities", "verification", "코드 문법 검증 및 구조 검증", "capabilities")
        self._add("capabilities", "computer_agent", "범용 Computer Agent: 화면 관찰, UI 의미 분석, 마우스/키보드/창/파일/터미널/브라우저 조작, 프로그램 다운로드/설치, 이미지 재현 (Windows)", "capabilities")

        # Computer Agent system
        self._add("computer_system", "screen_vision", "UI Automation/Accessibility/OCR/Vision을 결합한 구조적 화면 표현 (ScreenRepresentation)", "computer_system")
        self._add("computer_system", "action_ir", "언어 독립적 Action IR (preconditions/expected_effect/postconditions/semantic_justification 포함)", "computer_system")
        self._add("computer_system", "loop", "UNDERSTAND → OBSERVE → REASON → PLAN → ACT → OBSERVE → VERIFY → REPAIR → LEARN", "computer_system")
        self._add("computer_system", "no_fixed_coordinates", "UI 요소는 role/text/state 의미로 현재 위치를 재해석 — 좌표 고정 없음", "computer_system")
        self._add("computer_system", "security", "위험도 분류(LOW/MEDIUM/HIGH): 높은 위험 작업은 사용자 확인 필요, 몰래 권한 상승 없음", "computer_system")
        # Voice Agent system
        self._add("voice_system", "stt", "음성 인식: Vosk(오프라인, 영어) / SpeechRecognition(마이크) — 백엔드 자동 감지", "voice_system")
        self._add("voice_system", "tts", "음성 합성: espeak-ng(Linux) / pyttsx3(Windows SAPI) — 백엔드 자동 감지", "voice_system")
        self._add("voice_system", "loop", "listen → STT → 의미 파싱 → 응답 생성 → TTS → 반복 (종료: 'quit'/'종료')", "voice_system")
        self._add("voice_system", "self_probe", "백엔드 존재를 스스로 확인 — 없는 기능은 가정하지 않고 보고", "voice_system")
        
        # Architecture
        self._add("architecture", "reasoning", "Reasoner: 사용자 입력을 분석하여 의도와 계획을 수립", "architecture")
        self._add("architecture", "planner", "Planner: IR 노드를 생성하고 실행 계획 수립", "architecture")
        self._add("architecture", "generator", "Generator: IR을 각 언어별 코드로 변환", "architecture")
        self._add("architecture", "verifier", "Verifier: 생성된 코드의 문법과 구조 검증", "architecture")
        self._add("architecture", "knowledge_graph", "Knowledge Graph: 개념과 관계를 구조화하여 저장", "architecture")
        self._add("architecture", "memory_system", "Memory: 장단기 기억, 사용자 선호, 대화 맥락 관리", "architecture")
        
        # Supported languages
        self._add("supported_languages", "languages", "Python, Kotlin, Java, C, C++, JavaScript, Go, Rust", "supported_languages")
        self._add("supported_languages", "python", "파이썬: 함수, 클래스, 모듈 생성 지원", "supported_languages")
        self._add("supported_languages", "kotlin", "코틀린: 함수, 클래스, Android 앱 생성 지원", "supported_languages")
        self._add("supported_languages", "java", "자바: 함수, 클래스, Android 앱 생성 지원", "supported_languages")
        self._add("supported_languages", "c", "C: 함수, 구조체, 포인터 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "cpp", "C++: 클래스, 템플릿, STL 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "javascript", "자바스크립트: 함수, 클래스, 웹 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "go", "고: 함수, 구조체, 고루틴 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "rust", "러스트: 함수, 구조체, 트레잇 관련 코드 생성 지원", "supported_languages")
        
        # Project system
        self._add("project_system", "android", "Android 프로젝트: Gradle, Manifest, Activity, Layout XML 생성", "project_system")
        self._add("project_system", "web", "웹 프로젝트: HTML, CSS, JavaScript 파일 구조 생성", "project_system")
        self._add("project_system", "cli", "CLI 프로젝트: 명령줄 인터페이스 프로젝트 생성", "project_system")
        
        # Debugging system
        self._add("debugging_system", "error_analysis", "오류 패턴 분석 및 원인 진단", "debugging_system")
        self._add("debugging_system", "auto_fix", "과거 수정 패턴 기반 자동 수정 제안", "debugging_system")
        self._add("debugging_system", "explanation", "오류 발생 원인에 대한 자연어 설명", "debugging_system")
        
        # Limitations
        self._add("limitations", "compilation", "실제 컴파일/빌드는 수행하지 않음 (환경 제한)", "limitations")
        self._add("limitations", "runtime", "실제 런타임 테스트는 수행하지 않음", "limitations")
        self._add("limitations", "gui", "GUI 기반 개발은 지원하지 않음", "limitations")
        self._add("limitations", "real_time", "실시간 코드 실행은 지원하지 않음", "limitations")
    
    def _add(self, category: str, key: str, value: str, lang: str = "universal"):
        full_key = f"{category}.{key}"
        self.knowledge[full_key] = KnowledgeItem(
            key=key, value=value, category=category, language=lang,
        )
    
    # SelfKnowledgeInterface implementation
    def get_identity(self) -> dict:
        identity_items = self.get_category("identity")
        return {item.key: item.value for item in identity_items}
    
    def get_capabilities(self) -> list[str]:
        caps = self.get_category("capabilities")
        return [item.value for item in caps]
    
    def get_architecture_description(self) -> str:
        arch = self.get_category("architecture")
        return "; ".join(item.value for item in arch)
    
    def get_supported_languages(self) -> list[str]:
        langs = self.get_category("supported_languages")
        return [item.value for item in langs if item.key != "languages"]
    
    def get_code_knowledge(self, topic: str) -> dict | None:
        """Get code knowledge for a topic."""
        knowledge_items = self.search(topic)
        if not knowledge_items:
            return None
        return {
            "topic": topic,
            "explanations": [item.value for item in knowledge_items[:3]],
            "sources": [item.category for item in knowledge_items[:3]]
        }
    
    # Original methods
    def __init__(self):
        self.knowledge: dict[str, KnowledgeItem] = {}
        self._init_knowledge()
    
    def _init_knowledge(self):
        # Identity
        self._add("identity", "name", "RENA-BR", "identity")
        self._add("identity", "version", "RENA-BR 0.5 Beta", "identity")
        self._add("identity", "type", "AI coding assistant", "identity")
        self._add("identity", "architecture", "Knowledge Graph + Rule Engine + Memory + Reasoning + Planning", "identity")
        self._add("identity", "philosophy", "No transformers, no LLM APIs, no pretrained models", "identity")
        
        self._add("purpose", "main", "코드 생성, 프로젝트 생성, 디버깅, 코드 설명, 학습을 돕는 AI 코딩 어시스턴트", "purpose")
        self._add("purpose", "goal", "사용자의 의도를 이해하고 최적의 코드와 솔루션을 제공", "purpose")
        
        # Capabilities
        self._add("capabilities", "code_generation", "Python, Kotlin, Java, C, C++, JavaScript, Go, Rust에서 함수/클래스 코드 생성", "capabilities")
        self._add("capabilities", "project_generation", "Android, Web, CLI 프로젝트 전체 생성 (Gradle, Manifest, 소스 코드 포함)", "capabilities")
        self._add("capabilities", "debugging", "오류 진단, 원인 분석, 자동 수정 제안", "capabilities")
        self._add("capabilities", "code_explanation", "코드 분석 및 설명, 프로그래밍 개념 설명", "capabilities")
        self._add("capabilities", "learning", "코드 패턴 학습, 사용자 피드백 학습, 오류 패턴 학습", "capabilities")
        self._add("capabilities", "memory", "대화 맥락 기억, 사용자 선호 기억, 과거 오류 기억", "capabilities")
        self._add("capabilities", "refactoring", "코드 리팩토링 및 개선", "capabilities")
        self._add("capabilities", "verification", "코드 문법 검증 및 구조 검증", "capabilities")
        self._add("capabilities", "computer_agent", "범용 Computer Agent: 화면 관찰, UI 의미 분석, 마우스/키보드/창/파일/터미널/브라우저 조작, 프로그램 다운로드/설치, 이미지 재현 (Windows)", "capabilities")
        self._add("capabilities", "voice_conversation", "실시간 음성 대화: 음성 입력(STT) → 의미 이해 → 응답 → 음성 출력(TTS). 영어 듣기평가, 일상 대화 지원", "capabilities")

        # Computer Agent system
        self._add("computer_system", "screen_vision", "UI Automation/Accessibility/OCR/Vision을 결합한 구조적 화면 표현 (ScreenRepresentation)", "computer_system")
        self._add("computer_system", "action_ir", "언어 독립적 Action IR (preconditions/expected_effect/postconditions/semantic_justification 포함)", "computer_system")
        self._add("computer_system", "loop", "UNDERSTAND → OBSERVE → REASON → PLAN → ACT → OBSERVE → VERIFY → REPAIR → LEARN", "computer_system")
        self._add("computer_system", "no_fixed_coordinates", "UI 요소는 role/text/state 의미로 현재 위치를 재해석 — 좌표 고정 없음", "computer_system")
        self._add("computer_system", "security", "위험도 분류(LOW/MEDIUM/HIGH): 높은 위험 작업은 사용자 확인 필요, 몰래 권한 상승 없음", "computer_system")
        
        # Architecture
        self._add("architecture", "reasoning", "Reasoner: 사용자 입력을 분석하여 의도와 계획을 수립", "architecture")
        self._add("architecture", "planner", "Planner: IR 노드를 생성하고 실행 계획 수립", "architecture")
        self._add("architecture", "generator", "Generator: IR을 각 언어별 코드로 변환", "architecture")
        self._add("architecture", "verifier", "Verifier: 생성된 코드의 문법과 구조 검증", "architecture")
        self._add("architecture", "knowledge_graph", "Knowledge Graph: 개념과 관계를 구조화하여 저장", "architecture")
        self._add("architecture", "memory_system", "Memory: 장단기 기억, 사용자 선호, 대화 맥락 관리", "architecture")
        
        # Supported languages
        self._add("supported_languages", "languages", "Python, Kotlin, Java, C, C++, JavaScript, Go, Rust", "supported_languages")
        self._add("supported_languages", "python", "파이썬: 함수, 클래스, 모듈 생성 지원", "supported_languages")
        self._add("supported_languages", "kotlin", "코틀린: 함수, 클래스, Android 앱 생성 지원", "supported_languages")
        self._add("supported_languages", "java", "자바: 함수, 클래스, Android 앱 생성 지원", "supported_languages")
        self._add("supported_languages", "c", "C: 함수, 구조체, 포인터 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "cpp", "C++: 클래스, 템플릿, STL 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "javascript", "자바스크립트: 함수, 클래스, 웹 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "go", "고: 함수, 구조체, 고루틴 관련 코드 생성 지원", "supported_languages")
        self._add("supported_languages", "rust", "러스트: 함수, 구조체, 트레잇 관련 코드 생성 지원", "supported_languages")
        
        # Project system
        self._add("project_system", "android", "Android 프로젝트: Gradle, Manifest, Activity, Layout XML 생성", "project_system")
        self._add("project_system", "web", "웹 프로젝트: HTML, CSS, JavaScript 파일 구조 생성", "project_system")
        self._add("project_system", "cli", "CLI 프로젝트: 명령줄 인터페이스 프로젝트 생성", "project_system")
        
        # Debugging system
        self._add("debugging_system", "error_analysis", "오류 패턴 분석 및 원인 진단", "debugging_system")
        self._add("debugging_system", "auto_fix", "과거 수정 패턴 기반 자동 수정 제안", "debugging_system")
        self._add("debugging_system", "explanation", "오류 발생 원인에 대한 자연어 설명", "debugging_system")
        
        # Limitations
        self._add("limitations", "compilation", "실제 컴파일/빌드는 수행하지 않음 (환경 제한)", "limitations")
        self._add("limitations", "runtime", "실제 런타임 테스트는 수행하지 않음", "limitations")
        self._add("limitations", "gui", "GUI 기반 개발은 지원하지 않음", "limitations")
        self._add("limitations", "real_time", "실시간 코드 실행은 지원하지 않음", "limitations")
    
    def _add(self, category: str, key: str, value: str, lang: str = "universal"):
        full_key = f"{category}.{key}"
        self.knowledge[full_key] = KnowledgeItem(
            key=key, value=value, category=category, language=lang,
        )
    
    def get(self, category: str, key: str = "") -> Optional[str]:
        if key:
            item = self.knowledge.get(f"{category}.{key}")
            return item.value if item else None
        items = [v for v in self.knowledge.values() if v.category == category]
        if items:
            return "; ".join(i.value for i in items)
        return None
    
    def get_category(self, category: str) -> list[KnowledgeItem]:
        return [v for v in self.knowledge.values() if v.category == category]
    
    def get_all_categories(self) -> list[str]:
        return list(set(v.category for v in self.knowledge.values()))
    
    def search(self, query: str) -> list[KnowledgeItem]:
        query_lower = query.lower()
        results = []
        for item in self.knowledge.values():
            if (query_lower in item.value.lower() or
                    query_lower in item.key.lower() or
                    query_lower in item.category.lower()):
                results.append(item)
        return results
    
    def get_capabilities_summary(self) -> str:
        caps = self.get_category("capabilities")
        return "; ".join(c.value for c in caps)
    
    def get_supported_languages_list(self) -> list[str]:
        langs = self.get_category("supported_languages")
        return [i.key for i in langs if i.key != "languages"]
    
    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.knowledge.items()}
