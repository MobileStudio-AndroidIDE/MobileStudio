"""Requirements Analyzer - extracts structured requirements from natural language."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Requirement:
    name: str
    description: str
    priority: str = "high"
    category: str = "functional"


@dataclass
class AnalysisResult:
    project_name: str
    description: str
    features: list[Requirement]
    entities: list[dict]
    platform: str = ""
    language: str = ""


# LEGACY: Platform/language keyword detection
# source: legacy, usage: fallback+compatibility, confidence: 0.5
PLATFORM_KW = {
    "android": ["안드로이드", "android", "모바일", "핸드폰", "카메라 앱", "카메라"],
    "web": ["웹", "web", "브라우저", "browser", "프론트엔드"],
    "cli": ["명령줄", "cli", "콘솔", "터미널"],
    "backend": ["백엔드", "서버", "server", "api"],
}

LANGUAGE_KW = {
    "python": ["python", "파이썬"],
    "kotlin": ["kotlin", "코틀린"],
    "java": ["java", "자바"],
    "javascript": ["javascript", "js", "자바스크립트"],
    "c": ["c 언어", "c language"],
    "cpp": ["c++", "cpp"],
    "go": ["go", "고"],
    "rust": ["rust", "러스트"],
}


class RequirementsAnalyzer:
    def __init__(self):
        pass

    def analyze(self, text: str, entities: list[dict] = None) -> AnalysisResult:
        entities = entities or []
        project_name = self._extract_project_name(text)
        features = self._extract_features(text)
        platform = self._extract_platform(text, entities)
        language = self._extract_language(text, entities)
        description = self._generate_description(text, features)

        return AnalysisResult(
            project_name=project_name,
            description=description,
            features=features,
            entities=entities,
            platform=platform,
            language=language,
        )

    def _extract_project_name(self, text: str) -> str:
        # LEGACY: Project name mapping from keywords.
        # source: legacy, usage: fallback+compatibility, confidence: 0.4
        # Primary path: SemanticParser entity extraction
        name_kw = {
            "카메라": "CameraApp", "calculator": "CalculatorApp",
            "계산기": "CalculatorApp", "메모": "MemoApp",
            "todo": "TodoApp", "할일": "TodoApp",
            "채팅": "ChatApp", "뉴스": "NewsApp",
            "날씨": "WeatherApp", "음악": "MusicApp",
            "학생": "StudentApp", "직원": "EmployeeApp",
            "상품": "ProductApp", "주문": "OrderApp",
        }
        for kw, name in name_kw.items():
            if kw in text.lower():
                return name

        m = re.search(r'(\w+\s+\w+)\s*(앱|app|프로그램|program|프로젝트|project)', text)
        if m:
            full_name = m.group(1).strip()
            first_word = full_name.split()[0]
            return first_word.capitalize()

        m = re.search(r'(\w+)\s*(앱|app|프로그램|program|프로젝트|project)', text)
        if m:
            return m.group(1).capitalize()

        return "MyApp"

    def _extract_features(self, text: str) -> list[Requirement]:
        features = []
        # LEGACY: Feature extraction from keywords.
        # source: legacy, usage: fallback+compatibility, confidence: 0.4
        # Primary path: SemanticComposer operation detection
        feature_map = {
            "검색": ("Search", "Search functionality"),
            "로그인": ("Login", "User authentication"),
            "회원가입": ("Registration", "User registration"),
            "관리": ("Management", "Data management"),
            "등록": ("Registration", "Data registration"),
            "삭제": ("Delete", "Data deletion"),
            "수정": ("Edit", "Data editing"),
            "조회": ("View", "Data viewing"),
            "목록": ("List", "List view"),
            "상세": ("Detail", "Detail view"),
            "추천": ("Recommendation", "Recommendation system"),
            "알림": ("Notification", "Notification system"),
            "설정": ("Settings", "Settings page"),
            "프로필": ("Profile", "User profile"),
            "계산": ("Calculation", "Calculation functionality"),
            "정렬": ("Sort", "Sorting functionality"),
            "필터": ("Filter", "Filtering functionality"),
            "내보내기": ("Export", "Export functionality"),
            "가져오기": ("Import", "Import functionality"),
        }
        for kw, (name, desc) in feature_map.items():
            if kw in text:
                features.append(Requirement(name=name, description=desc))
        return features

    def _extract_platform(self, text: str, entities: list[dict]) -> str:
        for e in entities:
            if e.get("type") == "platform":
                return e["value"]
        text_lower = text.lower()
        for platform, keywords in PLATFORM_KW.items():
            for kw in keywords:
                if kw in text_lower:
                    return platform
        return "cli"

    def _extract_language(self, text: str, entities: list[dict]) -> str:
        for e in entities:
            if e.get("type") == "language":
                return e["value"]
        text_lower = text.lower()
        for lang, keywords in LANGUAGE_KW.items():
            for kw in keywords:
                if kw in text_lower:
                    return lang
        return ""

    def _generate_description(self, text: str, features: list[Requirement]) -> str:
        if features:
            feature_names = [f.name for f in features]
            return f"A project with {', '.join(feature_names)} functionality."
        return text[:100] if len(text) > 100 else text
