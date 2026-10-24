"""Reasoning Event System - structured events for explanation generation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime

from brain.explanation_events import GenerationEvent, GenerationEventType, ExplanationEvent


class ReasoningStage(Enum):
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    TECHNOLOGY_SELECTION = "TECHNOLOGY_SELECTION"
    FILE_CREATION = "FILE_CREATION"
    DEPENDENCY_ADDITION = "DEPENDENCY_ADDITION"
    CODE_CHANGE = "CODE_CHANGE"
    BUILD_START = "BUILD_START"
    ERROR_DETECTED = "ERROR_DETECTED"
    ERROR_ANALYSIS = "ERROR_ANALYSIS"
    FIX_APPLIED = "FIX_APPLIED"
    VERIFICATION = "VERIFICATION"
    COMPLETION = "COMPLETION"


class EventSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ReasoningEvent:
    stage: ReasoningStage = ReasoningStage.UNDERSTANDING
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    details: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    file_path: str = ""
    severity: EventSeverity = EventSeverity.INFO
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
            "evidence": self.evidence,
            "file_path": self.file_path,
            "severity": self.severity.value,
            "language": self.language,
        }


@dataclass
class UnderstandingEvent(ReasoningEvent):
    domain: str = ""
    platform: str = ""
    features: list[str] = field(default_factory=list)
    data_models: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stage = ReasoningStage.UNDERSTANDING


@dataclass
class PlanningEvent(ReasoningEvent):
    project_name: str = ""
    architecture: str = ""
    modules: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stage = ReasoningStage.PLANNING


@dataclass
class TechnologySelectionEvent(ReasoningEvent):
    language: str = ""
    platform: str = ""
    framework: str = ""
    build_tool: str = ""
    reason: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.TECHNOLOGY_SELECTION


@dataclass
class FileCreationEvent(ReasoningEvent):
    file_path: str = ""
    file_type: str = ""
    template: str = ""
    responsibility: str = ""
    dependencies: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stage = ReasoningStage.FILE_CREATION


@dataclass
class DependencyEvent(ReasoningEvent):
    dependency: str = ""
    required_by: str = ""
    feature: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.DEPENDENCY_ADDITION


@dataclass
class BuildEvent(ReasoningEvent):
    command: str = ""
    target: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.BUILD_START


@dataclass
class ErrorEvent(ReasoningEvent):
    error_type: str = ""
    error_message: str = ""
    file_path: str = ""
    line: int = 0
    build_output: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.ERROR_DETECTED
        self.severity = EventSeverity.ERROR


@dataclass
class ErrorAnalysisEvent(ReasoningEvent):
    error_type: str = ""
    cause: str = ""
    evidence: list[str] = field(default_factory=list)
    fix_plan: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.ERROR_ANALYSIS
        self.severity = EventSeverity.WARNING


@dataclass
class FixEvent(ReasoningEvent):
    fix_type: str = ""
    file_path: str = ""
    description: str = ""

    def __post_init__(self):
        self.stage = ReasoningStage.FIX_APPLIED
        self.severity = EventSeverity.INFO


@dataclass
class VerificationEvent(ReasoningEvent):
    checks_passed: int = 0
    checks_failed: int = 0
    status: str = ""
    issues: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stage = ReasoningStage.VERIFICATION


@dataclass
class CompletionEvent(ReasoningEvent):
    success: bool = True
    project_name: str = ""
    files_generated: int = 0

    def __post_init__(self):
        self.stage = ReasoningStage.COMPLETION
        self.severity = EventSeverity.SUCCESS if self.success else EventSeverity.ERROR


@dataclass
class GenerationEventWrapper(ReasoningEvent):
    """Wrapper for GenerationEvent to fit in the reasoning event stream."""
    generation_event: 'GenerationEvent' = None
    
    def __post_init__(self):
        self.stage = ReasoningStage.FILE_CREATION


class ReasoningEventCollector:
    def __init__(self, language: str = "en"):
        self.events: list[ReasoningEvent] = []
        self.language = language

    def add_event(self, event: ReasoningEvent):
        event.language = self.language
        self.events.append(event)

    def add_generation_event(self, generation_event: GenerationEvent):
        """Add a generation event to the event stream."""
        wrapper = GenerationEventWrapper(
            message=f"[{generation_event.event_type.value}] {generation_event.file_path}:{generation_event.line_number}",
            details={
                "generation_event": generation_event.to_dict()
            },
            evidence=[f"type={generation_event.event_type.value}", f"file={generation_event.file_path}"],
            file_path=generation_event.file_path,
        )
        wrapper.generation_event = generation_event
        self.add_event(wrapper)

    def add_understanding(self, domain: str, platform: str, features: list[str],
                          data_models: list[str], language: str = ""):
        msg = self._format_understanding(domain, platform, features, language or self.language)
        self.add_event(UnderstandingEvent(
            message=msg,
            domain=domain,
            platform=platform,
            features=features,
            data_models=data_models,
            evidence=[f"domain={domain}", f"platform={platform}"]
        ))

    def add_planning(self, project_name: str, architecture: str, modules: list[str], language: str = ""):
        msg = self._format_planning(project_name, architecture, modules, language or self.language)
        self.add_event(PlanningEvent(
            message=msg,
            project_name=project_name,
            architecture=architecture,
            modules=modules,
            evidence=[f"project={project_name}", f"arch={architecture}"]
        ))

    def add_technology_selection(self, language: str, platform: str, framework: str,
                                  build_tool: str, reason: str, ui_lang: str = ""):
        msg = self._format_tech_selection(language, platform, framework, build_tool, ui_lang or self.language)
        self.add_event(TechnologySelectionEvent(
            message=msg,
            language=language,
            platform=platform,
            framework=framework,
            build_tool=build_tool,
            reason=reason,
            evidence=[f"lang={language}", f"platform={platform}"]
        ))

    def add_file_creation(self, file_path: str, file_type: str, template: str,
                          responsibility: str, dependencies: list[str], language: str = ""):
        msg = self._format_file_creation(file_path, file_type, template, responsibility, language or self.language)
        self.add_event(FileCreationEvent(
            message=msg,
            file_path=file_path,
            file_type=file_type,
            template=template,
            responsibility=responsibility,
            dependencies=dependencies,
            evidence=[f"file={file_path}", f"type={file_type}"]
        ))

    def add_dependency(self, dependency: str, required_by: str, feature: str, language: str = ""):
        msg = self._format_dependency(dependency, required_by, feature, language or self.language)
        self.add_event(DependencyEvent(
            message=msg,
            dependency=dependency,
            required_by=required_by,
            feature=feature,
            evidence=[f"dep={dependency}", f"for={required_by}"]
        ))

    def add_build_start(self, command: str, target: str, language: str = ""):
        msg = self._format_build_start(command, target, language or self.language)
        self.add_event(BuildEvent(
            message=msg,
            command=command,
            target=target,
            evidence=[f"cmd={command}", f"target={target}"]
        ))

    def add_error(self, error_type: str, error_message: str, file_path: str,
                  line: int, build_output: str, language: str = ""):
        msg = self._format_error(error_type, error_message, file_path, language or self.language)
        self.add_event(ErrorEvent(
            message=msg,
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line=line,
            build_output=build_output,
            evidence=[f"error={error_type}", f"file={file_path}"]
        ))

    def add_error_analysis(self, error_type: str, cause: str, evidence: list[str],
                           fix_plan: str, language: str = ""):
        msg = self._format_error_analysis(error_type, cause, fix_plan, language or self.language)
        self.add_event(ErrorAnalysisEvent(
            message=msg,
            error_type=error_type,
            cause=cause,
            evidence=evidence,
            fix_plan=fix_plan,
        ))

    def add_fix(self, fix_type: str, file_path: str, description: str, language: str = ""):
        msg = self._format_fix(fix_type, file_path, description, language or self.language)
        self.add_event(FixEvent(
            message=msg,
            fix_type=fix_type,
            file_path=file_path,
            description=description,
            evidence=[f"fix={fix_type}", f"file={file_path}"]
        ))

    def add_verification(self, checks_passed: int, checks_failed: int, status: str,
                         issues: list[str], language: str = ""):
        msg = self._format_verification(checks_passed, checks_failed, status, language or self.language)
        self.add_event(VerificationEvent(
            message=msg,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            status=status,
            issues=issues,
            evidence=[f"passed={checks_passed}", f"failed={checks_failed}"]
        ))

    def add_completion(self, success: bool, project_name: str, files_generated: int, language: str = ""):
        msg = self._format_completion(success, project_name, files_generated, language or self.language)
        self.add_event(CompletionEvent(
            message=msg,
            success=success,
            project_name=project_name,
            files_generated=files_generated,
            evidence=[f"project={project_name}", f"files={files_generated}"]
        ))

    def get_events(self) -> list[ReasoningEvent]:
        return self.events

    def get_events_by_stage(self, stage: ReasoningStage) -> list[ReasoningEvent]:
        return [e for e in self.events if e.stage == stage]

    def clear(self):
        self.events.clear()

    def _format_understanding(self, domain: str, platform: str, features: list[str], lang: str) -> str:
        if lang == "ko":
            return f"[이해] {domain} 도메인의 {platform} 애플리케이션이 요청되었습니다. 주요 기능: {', '.join(features)}"
        return f"[Understanding] {domain} domain {platform} application requested. Features: {', '.join(features)}"

    def _format_planning(self, project_name: str, architecture: str, modules: list[str], lang: str) -> str:
        if lang == "ko":
            return f"[계획] {project_name} 프로젝트 계획 수립: {architecture} 아키텍처, 모듈: {', '.join(modules)}"
        return f"[Planning] {project_name} project plan: {architecture} architecture, modules: {', '.join(modules)}"

    def _format_tech_selection(self, language: str, platform: str, framework: str,
                               build_tool: str, lang: str) -> str:
        if lang == "ko":
            parts = [f"언어: {language}", f"플랫폼: {platform}"]
            if framework:
                parts.append(f"프레임워크: {framework}")
            if build_tool:
                parts.append(f"빌드 도구: {build_tool}")
            return f"[기술 선택] {', '.join(parts)}"
        parts = [f"Language: {language}", f"Platform: {platform}"]
        if framework:
            parts.append(f"Framework: {framework}")
        if build_tool:
            parts.append(f"Build tool: {build_tool}")
        return f"[Technology] {', '.join(parts)}"

    def _format_file_creation(self, file_path: str, file_type: str, template: str,
                              responsibility: str, lang: str) -> str:
        if lang == "ko":
            return f"[파일 생성] {file_path} - {responsibility} (템플릿: {template})"
        return f"[Creating] {file_path} - {responsibility} (template: {template})"

    def _format_dependency(self, dependency: str, required_by: str, feature: str, lang: str) -> str:
        if lang == "ko":
            return f"[의존성 추가] {dependency} - {required_by}의 '{feature}' 기능에 필요"
        return f"[Dependency] {dependency} - required by {required_by} for {feature} feature"

    def _format_build_start(self, command: str, target: str, lang: str) -> str:
        if lang == "ko":
            return f"[빌드] {target} 검증 시작: {command}"
        return f"[Building] Verifying {target}: {command}"

    def _format_error(self, error_type: str, error_message: str, file_path: str, lang: str) -> str:
        if lang == "ko":
            return f"[오류] {error_type}: {error_message} ({file_path})"
        return f"[Error] {error_type}: {error_message} ({file_path})"

    def _format_error_analysis(self, error_type: str, cause: str, fix_plan: str, lang: str) -> str:
        if lang == "ko":
            return f"[분석] {error_type} 원인: {cause}. 수정: {fix_plan}"
        return f"[Analysis] {error_type} cause: {cause}. Fix: {fix_plan}"

    def _format_fix(self, fix_type: str, file_path: str, description: str, lang: str) -> str:
        if lang == "ko":
            return f"[수정] {file_path}: {description}"
        return f"[Fix] {file_path}: {description}"

    def _format_verification(self, passed: int, failed: int, status: str, lang: str) -> str:
        if lang == "ko":
            return f"[검증] 통과: {passed}, 실패: {failed} - {status}"
        return f"[Verification] Passed: {passed}, Failed: {failed} - {status}"

    def _format_completion(self, success: bool, project_name: str, files_generated: int, lang: str) -> str:
        if lang == "ko":
            status = "성공" if success else "실패"
            return f"[완료] {project_name} 프로젝트 {status}: {files_generated}개 파일 생성됨"
        status = "success" if success else "failed"
        return f"[Complete] {project_name} project {status}: {files_generated} files generated"