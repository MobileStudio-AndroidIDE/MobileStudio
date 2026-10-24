"""Error Analyzer - analyzes build/verification errors and suggests fixes."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from code.ir.project_ir import ProjectIR, FileIR, FileType
from code.project.project_verifier import VerificationIssue


@dataclass
class ErrorAnalysis:
    original_error: str
    error_type: str = ""
    affected_files: list[str] = field(default_factory=list)
    root_cause: str = ""
    suggestions: list[str] = field(default_factory=list)
    auto_fixable: bool = False

    def to_dict(self) -> dict:
        return {
            "original_error": self.original_error,
            "error_type": self.error_type,
            "affected_files": self.affected_files,
            "root_cause": self.root_cause,
            "suggestions": self.suggestions,
            "auto_fixable": self.auto_fixable,
        }


class ErrorAnalyzer:
    def __init__(self):
        self._error_patterns = {
            "missing_file": {
                "pattern": r"Missing required file:\s*(.+)",
                "type": "structure",
                "suggestion_template": "Create the required file: {file}",
            },
            "unresolved_import": {
                "pattern": r"Unresolved import|cannot resolve import",
                "type": "import",
                "suggestion_template": "Add the missing import or dependency",
            },
            "missing_permission": {
                "pattern": r"Missing.*permission",
                "type": "permission",
                "suggestion_template": "Add the required permission to AndroidManifest.xml",
            },
            "build_error": {
                "pattern": r"Build failed|compilation error|syntax error",
                "type": "build",
                "suggestion_template": "Fix the syntax error in the affected file",
            },
            "type_error": {
                "pattern": r"type mismatch|incompatible types|expected.*got",
                "type": "type",
                "suggestion_template": "Fix the type mismatch",
            },
            "missing_declaration": {
                "pattern": r"not defined|undefined|unresolved reference",
                "type": "declaration",
                "suggestion_template": "Define the missing declaration",
            },
            "gradle_error": {
                "pattern": r"Could not.*gradle|gradle.*error",
                "type": "gradle",
                "suggestion_template": "Fix the Gradle configuration",
            },
        }

    def analyze(self, error: str, project_ir: ProjectIR = None) -> ErrorAnalysis:
        analysis = ErrorAnalysis(original_error=error)
        for error_type, config in self._error_patterns.items():
            if re.search(config["pattern"], error, re.IGNORECASE):
                analysis.error_type = error_type
                break

        if project_ir:
            analysis.affected_files = self._find_affected_files(error, project_ir)

        analysis.root_cause = self._infer_root_cause(analysis)
        analysis.suggestions = self._generate_suggestions(analysis)
        analysis.auto_fixable = analysis.error_type in ("structure", "missing_file", "permission", "manifest")
        return analysis

    def analyze_issues(self, issues: list[VerificationIssue],
                       project_ir: ProjectIR = None) -> list[ErrorAnalysis]:
        analyses = []
        for issue in issues:
            if issue.severity == "error":
                analysis = self.analyze(issue.message, project_ir)
                if issue.file_path:
                    if issue.file_path not in analysis.affected_files:
                        analysis.affected_files.append(issue.file_path)
                analyses.append(analysis)
        return analyses

    def _find_affected_files(self, error: str, project_ir: ProjectIR) -> list[str]:
        affected = []
        for m in project_ir.modules:
            for f in m.files:
                if f.path in error or f.path.split("/")[-1] in error:
                    affected.append(f.path)
        return affected

    def _infer_root_cause(self, analysis: ErrorAnalysis) -> str:
        if analysis.error_type == "structure":
            return "Project structure is missing required files"
        if analysis.error_type == "import":
            return "A file references code that is not imported"
        if analysis.error_type == "permission":
            return "Required permission is not declared in manifest"
        if analysis.error_type == "build":
            return "Source code contains syntax or compilation errors"
        if analysis.error_type == "type":
            return "Type mismatch in code"
        if analysis.error_type == "declaration":
            return "A referenced symbol is not defined"
        if analysis.error_type == "gradle":
            return "Gradle configuration has errors"
        return "Unknown error"

    def _generate_suggestions(self, analysis: ErrorAnalysis) -> list[str]:
        suggestions = []
        if analysis.error_type in ("structure", "missing_file"):
            for f in analysis.affected_files:
                suggestions.append(f"Create file: {f}")
            if not suggestions:
                import re
                m = re.search(r"Missing required file:\s*(.+)", analysis.original_error)
                if m:
                    suggestions.append(f"Create the required file: {m.group(1).strip()}")
        elif analysis.error_type == "import":
            suggestions.append("Check that all imported files exist in the project")
            suggestions.append("Add missing files or remove unused imports")
        elif analysis.error_type == "permission":
            suggestions.append("Add permission to AndroidManifest.xml")
        elif analysis.error_type == "build":
            suggestions.append("Review the source file for syntax errors")
            suggestions.append("Check for missing semicolons, brackets, or keywords")
        elif analysis.error_type == "type":
            suggestions.append("Check variable types and function signatures")
        elif analysis.error_type == "declaration":
            suggestions.append("Define the missing function, class, or variable")
        elif analysis.error_type == "gradle":
            suggestions.append("Check build.gradle.kts for syntax errors")
            suggestions.append("Verify plugin versions and dependencies")
        return suggestions

    def suggest_fix(self, analysis: ErrorAnalysis, project_ir: ProjectIR) -> Optional[dict]:
        if not analysis.auto_fixable:
            return None
        if analysis.error_type in ("structure", "missing_file"):
            for f in analysis.affected_files:
                return {"action": "create_file", "path": f, "content": f"# Generated file: {f}\n"}
            # If no affected files but we have a suggestion, try to extract file from original error
            import re
            m = re.search(r"Missing required file:\s*(.+)", analysis.original_error)
            if m:
                path = m.group(1).strip()
                return {"action": "create_file", "path": path, "content": f"# Generated file: {path}\n"}
        if analysis.error_type == "permission":
            return {"action": "update_manifest", "permission": "android.permission.CAMERA"}
        return None
