"""Target detection for multi-platform code generation.

Determines what the user wants to create based on their natural language request,
and maps it to the appropriate target platform + language + generator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Target(Enum):
    ANDROID = "android"
    JVM = "jvm"
    IOS = "ios"
    DESKTOP = "desktop"
    CLI = "cli"
    WEB = "web"
    SERVER = "server"
    DATABASE = "database"
    SCRIPTING = "scripting"
    CONFIGURATION = "configuration"
    LIBRARY = "library"
    UNKNOWN = "unknown"


@dataclass
class TargetSpec:
    target: Target
    language: str
    format: str  # "source", "html", "css", "xml", "json", "yaml", "sql", etc.
    generator_name: str  # key in generator registry
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)

    def to_semantic_dict(self) -> dict:
        """Serialize for SemanticRepresentation.target (spec #1).

        Only the DECLARED target is emitted.  The default fallback — the
        request mentioned no platform and no language — emits {} so an
        absent target stays absent instead of becoming a guess that
        later stages would treat as a user declaration.
        """
        if self.evidence.get("default"):
            return {}
        if self.target in (Target.UNKNOWN, Target.CLI, Target.SCRIPTING):
            return {
                "kind": "language",
                "language": self.language,
                "confidence": self.confidence,
                "evidence": dict(self.evidence),
            }
        return {
            "kind": "platform",
            "platform": self.target.value,
            "language": self.language,
            "format": self.format,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


# Keyword → (Target, preferred languages, format)
_TARGET_PATTERNS = [
    # Android
    (r'\b(android|안드로이드|apk|activity|fragment|compose|jetpack)\b',
     Target.ANDROID, ["kotlin", "java", "xml"], "source"),
    # Kotlin Compose
    (r'\b(compose|@\s*composable|composable\s+function)\b',
     Target.ANDROID, ["kotlin"], "source"),
    # XML Layout
    (r'\b(xml\s+layout|layout\s+xml|linearlayout|relativelayout|constraintlayout)\b',
     Target.ANDROID, ["xml"], "xml"),
    # JVM
    (r'\b(jvm|java\s+virtual\s+jvm|bytecode|spring\s+boot|gradle|maven)\b',
     Target.JVM, ["java", "kotlin"], "source"),
    # iOS
    (r'\b(ios|iphone|ipad|swiftui|uikit|xcode)\b',
     Target.IOS, ["swift"], "source"),
    # Desktop
    (r'\b(desktop|gui|window|electron|qt|gtk|swing|javafx|wpf)\b',
     Target.DESKTOP, ["python", "javascript", "java", "csharp", "dart"], "source"),
    # Web frontend
    (r'\b(web\s+page|html\s+page|website|frontend|front-end|react|vue|angular|dom)\b',
     Target.WEB, ["html", "css", "javascript", "typescript"], "html"),
    # CSS
    (r'\b(css\s+style|stylesheet|styling|responsive\s+design|flexbox|grid\s+layout)\b',
     Target.WEB, ["css"], "css"),
    # Server / Backend
    (r'\b(server|backend|back-end|api\s+endpoint|rest\s+api|graphql|microservice)\b',
     Target.SERVER, ["python", "java", "go", "javascript", "typescript"], "source"),
    # Database / SQL
    (r'\b(sql|database|db|table|query|select|insert|update|delete|create\s+table|schema)\b',
     Target.DATABASE, ["sql"], "sql"),
    # CLI / Script
    (r'\b(cli|command\s+line|terminal|shell\s+script|bash\s+script|script|cron)\b',
     Target.SCRIPTING, ["bash", "powershell", "python", "lua"], "source"),
    # PowerShell
    (r'\b(powershell|cmdlet|ps1|pwsh)\b',
     Target.SCRIPTING, ["powershell"], "source"),
    # Lua
    (r'\b(lua|neovim\s+plugin|redis\s+script)\b',
     Target.SCRIPTING, ["lua"], "source"),
    # Configuration
    (r'\b(config|configuration|yaml\s+file|json\s+config|docker|kubernetes|terraform)\b',
     Target.CONFIGURATION, ["yaml", "json"], "yaml"),
    # Library
    (r'\b(library|lib|package|module|sdk|toolkit)\b',
     Target.LIBRARY, ["python", "java", "kotlin", "typescript"], "source"),
    # JSON
    (r'\b(json|json\s+data|json\s+format|api\s+response)\b',
     Target.CONFIGURATION, ["json"], "json"),
    # XML
    (r'\b(xml|xml\s+file|xml\s+format|xml\s+document)\b',
     Target.CONFIGURATION, ["xml"], "xml"),
]

# Language detection from request (order matters — longer/more specific first)
# NOTE: bare Korean "고" is a grammatical connective ("제거하고") — the
# explicit forms only ("고 언어", "고랭").
_LANG_KEYWORDS = {
    "python": ["python", "pip", "django", "flask", "fastapi", "pylint", "파이썬"],
    "kotlin": ["kotlin", "jetpack", "compose", "coroutine", "코틀린"],
    "java": ["java ", "spring", "gradle", "maven", "자바"],
    "javascript": ["javascript", "node", "npm", "react", "vue", "express", "자바스크립트"],
    "typescript": ["typescript", "angular", " nest ", "nestjs"],
    "c": [" c ", " ansi c", "c language", "gcc", "clang", "c언어"],
    "cpp": ["c++", "cpp", "cmake", "stl"],
    "go": ["golang", "go ", "goroutine", "go module", "고 언어", "고랭"],
    "rust": ["rust", "cargo", "rustc", "러스트"],
    "swift": ["swift", "swiftui", "uikit", "xcode"],
    "dart": ["dart", "flutter"],
    "ruby": ["ruby", "rails", "gem", "erb"],
    "php": ["php", "laravel"],
    "csharp": ["c#", "csharp", ".net", "dotnet", "nuget"],
    "bash": ["bash", "shell script"],
    "powershell": ["powershell", "pwsh", "ps1", "cmdlet"],
    "lua": ["lua", "neovim"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "html": ["html", "web page", "webpage"],
    "css": ["css", "stylesheet", "styling"],
    "xml": ["xml", "layout xml", "android xml"],
    "json": ["json"],
    "yaml": ["yaml", "yml"],
}


def detect_target(request: str) -> TargetSpec:
    """Detect the target platform, language, and format from a natural language request.

    Returns a TargetSpec with the best match. If no specific target is detected,
    returns a default CLI target with the most likely language.
    """
    import re
    text = request.lower()

    # Phase 1: Detect target platform
    best_target = None
    best_score = 0
    for pattern, target, preferred_langs, fmt in _TARGET_PATTERNS:
        if re.search(pattern, text):
            # Score based on specificity of match
            matches = re.findall(pattern, text)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_target = (target, preferred_langs, fmt)

    # Phase 2: Detect language from keywords
    #
    # The language NAME is matched after the Korean case particles that
    # attach to it ("Java로", "C를") are removed — a morphological
    # normalization of the request, so the declared language is read in
    # any of the request's phrasings and never guessed from context.
    lang_text = re.sub(
        r"(?<=[a-z0-9+#])(?:으로|에서|보다|이랑|로|를|을|이|가|의|과|와|은|는"
        r"|도|만|랑)(?=[\s,.!?]|$)",
        "", text)
    # padded, so a keyword anchored to a word edge (" c ") matches the
    # request's first and last word too — and never the middle of one.
    lang_text = f" {lang_text} "
    detected_lang = None
    best_lang_score = 0
    for lang, keywords in _LANG_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lang_text)
        if score > best_lang_score:
            best_lang_score = score
            detected_lang = lang

    # Phase 3: Combine target + language
    if best_target:
        target, preferred_langs, fmt = best_target
        if detected_lang and detected_lang in preferred_langs:
            lang = detected_lang
        else:
            lang = preferred_langs[0]
        return TargetSpec(
            target=target,
            language=lang,
            format=fmt,
            generator_name=lang,
            confidence=min(0.5 + best_score * 0.15 + best_lang_score * 0.1, 0.95),
            evidence={"target_matches": best_score, "lang_score": best_lang_score},
        )

    # Phase 4: Language-only detection (no explicit target)
    if detected_lang:
        fmt = "sql" if detected_lang == "sql" else "html" if detected_lang == "html" else "css" if detected_lang == "css" else "source"
        return TargetSpec(
            target=Target.SCRIPTING if detected_lang in ("bash", "powershell", "lua") else Target.UNKNOWN,
            language=detected_lang,
            format=fmt,
            generator_name=detected_lang,
            confidence=min(0.3 + best_lang_score * 0.15, 0.7),
            evidence={"lang_score": best_lang_score},
        )

    # Phase 5: Default — use Python CLI
    return TargetSpec(
        target=Target.CLI,
        language="python",
        format="source",
        generator_name="python",
        confidence=0.3,
        evidence={"default": True},
    )
