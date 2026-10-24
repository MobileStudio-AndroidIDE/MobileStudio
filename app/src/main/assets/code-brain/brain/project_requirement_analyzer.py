"""Project Requirement Analyzer - extracts structured requirements from natural language."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataModel:
    name: str
    fields: list[dict] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)


@dataclass
class Feature:
    name: str
    description: str
    category: str = "functional"
    priority: str = "high"
    methods: list[str] = field(default_factory=list)
    # The requirement's OWN id: assigned once, when the requirements are
    # analyzed, and carried unchanged through plan -> IR -> code -> test.
    # Every stage records THIS id, so the same requirement can be read
    # across the pipeline instead of being re-matched by name.
    req_id: str = ""


@dataclass
class ProjectRequirements:
    project_name: str
    description: str
    domain: str
    platform: str
    language: str
    features: list[Feature] = field(default_factory=list)
    data_models: list[DataModel] = field(default_factory=list)
    storage_needed: bool = False
    storage_type: str = ""
    api_needed: bool = False
    ui_needed: bool = False




def _detect_domain(text: str) -> Optional[str]:
    # Domain detection based on semantic analysis, not keyword patterns.
    # Features should be extracted from actual request text.
    return "generic"


def _extract_project_name(text: str, domain: Optional[str]) -> str:
    if domain:
        name_map = {
            "student_management": "StudentManager",
            "calculator": "Calculator",
            "todo": "TodoApp",
            "contact_manager": "ContactManager",
            "file_organizer": "FileOrganizer",
        }
        return name_map.get(domain, "MyApp")

    m = re.search(r'["\']?(\w+)["\']?\s*(앱|app|프로그램|program|프로젝트|project)', text)
    if m:
        return m.group(1).capitalize()

    m = re.search(r'(\w+)\s*(앱|app|프로그램|program)', text)
    if m:
        return m.group(1).capitalize()

    return "MyApp"


# LEGACY: Platform/language/feature keyword detection
# source: legacy, usage: fallback+compatibility, confidence: 0.5
# Primary path: SemanticParser constraint extraction
def _detect_platform(text: str) -> str:
    text_lower = text.lower()
    platform_keywords = {
        "android": ["android", "안드로이드", "mobile", "모바일"],
        "web": ["web", "웹", "browser", "브라우저", "frontend"],
        "cli": ["cli", "command line", "명령줄", "console", "콘솔", "terminal", "터미널"],
        "backend": ["backend", "서버", "server", "api"],
    }
    for platform, keywords in platform_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return platform
    return "cli"

# LEGACY: Language detection from keywords.
# source: legacy, usage: fallback+compatibility, confidence: 0.5
# Primary path: SemanticParser constraint extraction
def _detect_language(text: str) -> str:
    text_lower = text.lower()
    language_keywords = {
        "python": ["python", "파이썬"],
        "kotlin": ["kotlin", "코틀린"],
        "java": ["java", "자바"],
        "javascript": ["javascript", "js", "자바스크립트"],
        "go": ["go", "고"],
        "rust": ["rust", "러스트"],
        "c": ["c language", "c 언어"],
        "cpp": ["c++", "cpp"],
    }
    for lang, keywords in language_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return lang
    return ""


# LEGACY: Feature extraction from keywords.
# source: legacy, usage: fallback+compatibility, confidence: 0.5
# Primary path: SemanticComposer operation detection
def _extract_features_from_text(text: str, domain: Optional[str]) -> list[Feature]:
    features = []
    text_lower = text.lower()

    feature_keywords = {
        "add": ["add", "create", "register", "등록", "추가", "생성"],
        "remove": ["remove", "delete", "삭제", "제거"],
        "update": ["update", "edit", "modify", "수정", "편집"],
        "search": ["search", "find", "lookup", "검색", "찾기"],
        "list": ["list", "show", "display", "목록", "보기"],
        "sort": ["sort", "order", "정렬"],
        "filter": ["filter", "筛选", "필터"],
        "save": ["save", "persist", "저장"],
        "load": ["load", "read", "불러오기"],
        "export": ["export", "내보내기"],
        "import": ["import", "가져오기"],
        "calculate": ["calculate", "compute", "계산"],
        "validate": ["validate", "check", "검증"],
    }

    for feat_name, keywords in feature_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                features.append(Feature(
                    name=feat_name,
                    description=f"{feat_name.capitalize()} functionality",
                    category="functional",
                ))
                break

    return features


def _infer_data_model(text: str, domain: Optional[str], features: list[Feature]) -> Optional[DataModel]:
    # Extract data model from actual request text, not from domain patterns.
    return None

    text_lower = text.lower()
    entity_patterns = [
        (r"(\w+)\s*(?:management|manager|관리)", 1),
        (r"(\w+)\s*(?:list|목록)", 1),
        (r"(\w+)\s*(?:data|데이터)", 1),
    ]

    for pattern, group in entity_patterns:
        m = re.search(pattern, text_lower)
        if m:
            entity_name = m.group(group).capitalize()
            return DataModel(
                name=entity_name,
                fields=[
                    {"name": "id", "type": "int", "description": f"{entity_name} ID"},
                    {"name": "name", "type": "String", "description": f"{entity_name} name"},
                ],
                methods=[f.name for f in features],
            )

    return None


def _needs_storage(text: str, features: list[Feature]) -> tuple[bool, str]:
    text_lower = text.lower()
    storage_keywords = ["save", "store", "persist", "database", "db", "파일", "file", "저장", "ERSIST"]
    for kw in storage_keywords:
        if kw in text_lower:
            return True, "file"

    feature_names = [f.name for f in features]
    if "save" in feature_names or "load" in feature_names:
        return True, "file"

    return False, ""


def _assign_req_ids(features: list[Feature]) -> list[Feature]:
    """Give each requirement its id, in extraction order — unique, stable
    for the same requirements, and never re-derived downstream."""
    used = set()
    for position, feature in enumerate(features, 1):
        candidate = (feature.req_id or "").strip() or f"R{position}"
        while candidate in used:
            candidate = f"{candidate}.{len(used) + 1}"
        feature.req_id = candidate
        used.add(candidate)
    return features


def analyze_project_request(request: str) -> ProjectRequirements:
    domain = _detect_domain(request)
    project_name = _extract_project_name(request, domain)
    platform = _detect_platform(request)
    language = _detect_language(request)
    features = _assign_req_ids(_extract_features_from_text(request, domain))
    data_model = _infer_data_model(request, domain, features)
    storage_needed, storage_type = _needs_storage(request, features)

    data_models = [data_model] if data_model else []

    description = f"{project_name}: "
    if domain:
        domain_desc = {
            "student_management": "Student management system",
            "calculator": "Calculator application",
            "todo": "Todo list application",
            "contact_manager": "Contact management system",
            "file_organizer": "File organization tool",
        }
        description += domain_desc.get(domain, "Application")
    else:
        description += request[:100] if len(request) > 100 else request

    return ProjectRequirements(
        project_name=project_name,
        description=description,
        domain=domain or "generic",
        platform=platform,
        language=language or "python",
        features=features,
        data_models=data_models,
        storage_needed=storage_needed,
        storage_type=storage_type,
        api_needed=platform == "backend",
        ui_needed=platform in ["web", "android"],
    )
