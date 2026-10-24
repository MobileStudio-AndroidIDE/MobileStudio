"""Capability Registry - tracks what RENA-BR can actually do."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Capability:
    name: str
    description: str
    category: str
    is_implemented: bool = True
    is_active: bool = True
    languages: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "is_implemented": self.is_implemented,
            "is_active": self.is_active,
            "languages": self.languages,
            "platforms": self.platforms,
            "confidence": self.confidence,
        }


class CapabilityRegistry:
    def __init__(self):
        self.capabilities: dict[str, Capability] = {}
        self._init_capabilities()

    def _init_capabilities(self):
        self._add(
            "code_generation", "Code Generation",
            "Generate code in various languages",
            "code_generation",
            languages=["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"],
        )
        self._add(
            "project_generation", "Project Generation",
            "Generate complete projects (Android, Web, CLI)",
            "project_generation",
            platforms=["android", "web", "cli"],
        )
        self._add(
            "debugging", "Debugging",
            "Error diagnosis, root cause analysis, auto-fix suggestions",
            "debugging",
        )
        self._add(
            "code_explanation", "Code Explanation",
            "Analyze and explain code, programming concepts",
            "explanation",
        )
        self._add(
            "learning", "Learning",
            "Learn code patterns and user feedback",
            "learning",
        )
        self._add(
            "memory", "Memory",
            "Remember conversation context and user preferences",
            "memory",
        )
        self._add(
            "refactoring", "Refactoring",
            "Refactor and improve code",
            "refactoring",
        )
        self._add(
            "verification", "Verification",
            "Verify code syntax and structure",
            "verification",
        )
        self._add(
            "android_generation", "Android Generation",
            "Generate Gradle, Manifest, Activity, Layout XML",
            "project_generation",
            platforms=["android"],
            languages=["kotlin", "java"],
        )
        self._add(
            "web_generation", "Web Generation",
            "Generate HTML, CSS, JavaScript file structure",
            "project_generation",
            platforms=["web"],
            languages=["javascript"],
        )
        self._add(
            "cli_generation", "CLI Generation",
            "Generate command-line interface projects",
            "project_generation",
            platforms=["cli"],
        )
        self._add(
            "error_diagnosis", "오류 진단",
            "오류 패턴 분석 및 원인 진단",
            "debugging",
        )
        self._add(
            "auto_fix", "자동 수정",
            "과거 수정 패턴 기반 자동 수정 제안",
            "debugging",
        )
        self._add(
            "multi_language", "다중 언어",
            "8개 언어 지원: Python, Kotlin, Java, C, C++, JavaScript, Go, Rust",
            "code_generation",
            languages=["python", "kotlin", "java", "c", "cpp", "javascript", "go", "rust"],
        )
        # ── Computer Agent capabilities (General Computer Agent) ──
        self._add(
            "computer_agent", "범용 컴퓨터 조작",
            "화면 관찰 + UI 의미 분석 + Action IR 실행 루프로 데스크톱 전체를 조작",
            "computer_agent",
            platforms=["windows"],
        )
        self._add(
            "screen_vision", "화면 관찰",
            "UI Automation/Accessibility/OCR/Vision을 결합한 구조적 화면 표현",
            "computer_agent",
            platforms=["windows"],
        )
        self._add(
            "mouse_keyboard", "마우스/키보드 조작",
            "UI 요소 의미 기반 Find→Resolve→Act (좌표 고정 없음)",
            "computer_agent",
            platforms=["windows"],
        )
        self._add(
            "window_management", "창/응용프로그램 관리",
            "응용프로그램 실행/종료/포커스/배치 (설치 프로그램 탐색 포함)",
            "computer_agent",
            platforms=["windows"],
        )
        self._add(
            "file_operations", "파일 탐색/편집",
            "메타데이터(MIME/type/size)를 포함한 파일 시스템 조작",
            "computer_agent",
        )
        self._add(
            "terminal_execution", "터미널 실행",
            "빌드 시스템 감지 → 명령 계획 → 실행 → 해석 → 검증",
            "computer_agent",
        )
        self._add(
            "browser_automation", "브라우저 사용",
            "웹 탐색/페이지 조작/다운로드 (DOM/접근성 정보 결합)",
            "computer_agent",
        )
        self._add(
            "software_install", "프로그램 다운로드/설치",
            "공식 배포 경로 우선 + 다운로드 검증 + 사용자 승인 설치",
            "computer_agent",
        )
        self._add(
            "image_reproduction", "이미지 이해/재현",
            "Vision → 장면 표현 → 드로잉 계획 → 그림판 재현 → 비교 검증",
            "computer_agent",
        )
        self._add(
            "ui_understanding", "UI 의미 추론",
            "처음 보는 버튼도 텍스트/컨테이너/창 맥락으로 목적 추론 (기억에 UI 구조 저장)",
            "computer_agent",
            platforms=["windows"],
        )
        self._add(
            "android_backend", "Android 기기/에뮬레이터 조작",
            "ADB 기반 백엔드: uiautomator 접근성 트리 관찰, input tap/swipe/text, am start/force-stop (동일 Action IR/루프)",
            "computer_agent",
            platforms=["android"],
        )
        self._add(
            "voice_conversation", "실시간 음성 대화",
            "STT(음성→텍스트) + TTS(텍스트→음성) 백엔드 자동 감지 + "
            "실시간 대화 루프 (Vosk 오프라인 / espeak-ng / Windows SAPI)",
            "voice_agent",
        )
        self._add(
            "speech_recognition", "음성 인식 (듣기)",
            "마이크/오디오 파일 → 텍스트 (영어 듣기평가 포함)",
            "voice_agent",
        )
        self._add(
            "speech_synthesis", "음성 합성 (말하기)",
            "텍스트 → 음성 출력 (영어/한국어)",
            "voice_agent",
        )

    def _add(self, name: str, description: str, detail: str, category: str,
             languages: list[str] = None, platforms: list[str] = None,
             is_implemented: bool = True):
        full_desc = f"{description}: {detail}" if detail else description
        self.capabilities[name] = Capability(
            name=name,
            description=full_desc,
            category=category,
            is_implemented=is_implemented,
            languages=languages or [],
            platforms=platforms or [],
        )

    def get(self, name: str) -> Optional[Capability]:
        return self.capabilities.get(name)

    def get_by_category(self, category: str) -> list[Capability]:
        return [c for c in self.capabilities.values() if c.category == category]

    def get_implemented(self) -> list[Capability]:
        return [c for c in self.capabilities.values() if c.is_implemented]

    def get_active(self) -> list[Capability]:
        return [c for c in self.capabilities.values() if c.is_active and c.is_implemented]

    def get_for_language(self, language: str) -> list[Capability]:
        return [c for c in self.capabilities.values()
                if language in c.languages and c.is_implemented]

    def get_for_platform(self, platform: str) -> list[Capability]:
        return [c for c in self.capabilities.values()
                if platform in c.platforms and c.is_implemented]

    def can_do(self, task: str) -> bool:
        task_lower = task.lower()
        for cap in self.capabilities.values():
            if cap.is_implemented and cap.is_active:
                if task_lower in cap.description.lower() or task_lower in cap.name.lower():
                    return True
        return False

    def summary(self) -> str:
        active = self.get_active()
        return f"RENA-BR은 {len(active)}개의 기능을 활성화하여 지원합니다."

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.capabilities.items()}
