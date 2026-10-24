"""Technology Selector - selects language and platform based on requirements."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from brain.requirements_analyzer import AnalysisResult


@dataclass
class TechnologySelection:
    language: str
    platform: str
    framework: str = ""
    build_tool: str = ""
    reason: str = ""


PLATFORM_LANGUAGE_MAP = {
    "android": {"language": "kotlin", "framework": "Android SDK", "build_tool": "gradle"},
    "ios": {"language": "swift", "framework": "UIKit", "build_tool": "xcode"},
    "web": {"language": "javascript", "framework": "React", "build_tool": "vite"},
    "backend": {"language": "python", "framework": "FastAPI", "build_tool": "pip"},
    "cli": {"language": "python", "framework": "", "build_tool": "pip"},
    "library": {"language": "python", "framework": "", "build_tool": "pip"},
}

LANGUAGE_PREFERENCES = {
    "android": ["kotlin", "java"],
    "ios": ["swift"],
    "web": ["javascript", "typescript", "python"],
    "backend": ["python", "java", "go", "rust"],
    "cli": ["python", "go", "rust"],
    "data_science": ["python", "r"],
    "systems": ["c", "cpp", "rust"],
    "game": ["c++", "c#", "python"],
}


class TechnologySelector:
    def __init__(self):
        pass

    def select(self, analysis: AnalysisResult,
               requested_language: str = "") -> TechnologySelection:
        platform = analysis.platform or "cli"
        language = requested_language or analysis.language or ""

        if not language:
            language = self._select_language_for_platform(platform, analysis)

        platform_config = PLATFORM_LANGUAGE_MAP.get(platform, {})
        framework = platform_config.get("framework", "")
        build_tool = platform_config.get("build_tool", "")
        reason = self._generate_reason(platform, language, framework)

        return TechnologySelection(
            language=language,
            platform=platform,
            framework=framework,
            build_tool=build_tool,
            reason=reason,
        )

    def _select_language_for_platform(self, platform: str,
                                      analysis: AnalysisResult) -> str:
        prefs = LANGUAGE_PREFERENCES.get(platform, ["python"])
        if prefs:
            return prefs[0]
        return "python"

    def _generate_reason(self, platform: str, language: str,
                         framework: str) -> str:
        parts = []
        if platform:
            parts.append(f"Platform: {platform}")
        if language:
            parts.append(f"Language: {language}")
        if framework:
            parts.append(f"Framework: {framework}")
        return ", ".join(parts) if parts else "Default selection"
