"""Project Verifier - validates project structure, files, dependencies, and build configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import shutil
import subprocess

from code.ir.project_ir import (
    ProjectIR, ModuleIR, FileIR, FileType, Platform,
    ModuleDependency, DependencyType,
)
from code.generator.project_generator import GeneratedFile, ProjectGenerationResult


STATUS_STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
STATUS_BUILD_SUCCESS = "BUILD_SUCCESS"
STATUS_BUILD_FAILED = "BUILD_FAILED"
STATUS_TOOLCHAIN_MISSING = "TOOLCHAIN_MISSING"
STATUS_ENVIRONMENT_UNSUPPORTED = "ENVIRONMENT_UNSUPPORTED"
STATUS_NOT_BUILT = "NOT_BUILT"
STATUS_BUILD_ATTEMPTED = "BUILD_ATTEMPTED"
STATUS_REQUIREMENTS_MISSING = "REQUIREMENTS_MISSING"


@dataclass
class VerificationIssue:
    severity: str
    category: str
    message: str
    file_path: str = ""
    line: int = 0
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "suggestion": self.suggestion,
        }


@dataclass
class ProjectVerificationResult:
    success: bool = True
    issues: list[VerificationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    verifier_status: str = STATUS_STRUCTURAL_ONLY

    def add_issue(self, severity: str, category: str, message: str,
                  file_path: str = "", line: int = 0, suggestion: str = ""):
        issue = VerificationIssue(
            severity=severity, category=category, message=message,
            file_path=file_path, line=line, suggestion=suggestion,
        )
        self.issues.append(issue)
        if severity == "error":
            self.success = False
            self.checks_failed += 1
        else:
            self.checks_passed += 1

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "verifier_status": self.verifier_status,
            "issues": [i.to_dict() for i in self.issues],
        }


class ProjectVerifier:
    def __init__(self):
        self._required_android_files = [
            "settings.gradle.kts",
            "build.gradle.kts",
            "app/build.gradle.kts",
            "app/src/main/AndroidManifest.xml",
        ]
        self._required_web_files = [
            "frontend/package.json",
            "frontend/index.html",
        ]

    def verify(self, project_ir: ProjectIR,
               generation_result: ProjectGenerationResult = None,
               success_criteria: list = None,
               requirements: list = None) -> ProjectVerificationResult:
        result = ProjectVerificationResult()
        self._check_project_structure(project_ir, result)
        self._check_required_files(project_ir, result)
        self._check_imports(project_ir, generation_result, result)
        self._check_dependencies(project_ir, generation_result, result)
        self._check_manifest(project_ir, result)
        self._check_gradle_config(project_ir, result)
        self._check_source_files(project_ir, generation_result, result)
        self._check_kotlin_syntax(project_ir, generation_result, result)
        self._check_package_consistency(project_ir, result)
        if generation_result:
            self._check_generation_errors(generation_result, result)

        # Check success criteria if provided
        if success_criteria and generation_result:
            self._check_success_criteria(project_ir, generation_result, success_criteria, result)

        if project_ir.platform == Platform.ANDROID:
            toolchain = self._detect_toolchain("gradle")
            if toolchain:
                result.verifier_status = STATUS_BUILD_SUCCESS
            else:
                result.verifier_status = STATUS_STRUCTURAL_ONLY
        elif project_ir.platform == Platform.CLI:
            lang = project_ir.language
            toolchain = self._detect_toolchain(lang)
            if toolchain:
                result.verifier_status = STATUS_BUILD_SUCCESS
            else:
                result.verifier_status = STATUS_STRUCTURAL_ONLY
        else:
            result.verifier_status = STATUS_STRUCTURAL_ONLY

        # Try actual compilation if generation_result is available
        if generation_result:
            compile_results = self._try_compile_all(generation_result, project_ir.language)
            if compile_results:
                any_success = any(r.get("status") == "success" for r in compile_results)
                any_attempted = any(r.get("status") not in ("toolchain_missing", "error") for r in compile_results)
                if any_success:
                    result.verifier_status = STATUS_BUILD_SUCCESS
                elif any_attempted:
                    result.verifier_status = STATUS_BUILD_ATTEMPTED
                    for cr in compile_results:
                        if cr.get("status") == "failed":
                            result.add_issue("error", "compilation",
                                f" Compilation failed for {cr.get('file', 'unknown')}: {cr.get('error', '')[:200]}")

        # Check requirement coverage
        if requirements and generation_result:
            missing = self._check_requirement_coverage(requirements, generation_result)
            if missing:
                result.verifier_status = STATUS_REQUIREMENTS_MISSING
                result.add_issue("warning", "requirements",
                    f"Missing operations: {', '.join(missing)}")

        return result

    def requirement_coverage(self, requirements: list,
                              generation_result: ProjectGenerationResult) -> list[dict]:
        """Per-requirement coverage: the requirement's OWN id with what
        the generated code satisfies — the check verify() runs, reported
        against the requirement so plan, IR, code and test read one set
        of ids."""
        if not requirements or not generation_result:
            return []
        missing = set(self._check_requirement_coverage(requirements,
                                                       generation_result))
        coverage = []
        for req in requirements:
            name = getattr(req, "name", "") or str(req)
            rid = str(getattr(req, "id", "") or "") or name
            coverage.append({
                "req_id": rid,
                "name": name,
                "satisfied": name not in missing,
            })
        return coverage

    def _detect_toolchain(self, tool: str) -> bool:
        tool_map = {
            "gradle": "gradle",
            "python": "python3",
            "java": "javac",
            "kotlin": "kotlinc",
            "javascript": "node",
            "go": "go",
            "rust": "rustc",
            "c": "gcc",
            "cpp": "g++",
        }
        cmd = tool_map.get(tool, tool)
        return shutil.which(cmd) is not None

    def _check_project_structure(self, project_ir: ProjectIR, result: ProjectVerificationResult):
        if not project_ir.name:
            result.add_issue("error", "structure", "Project has no name")
            return
        result.checks_passed += 1

        if not project_ir.modules:
            result.add_issue("error", "structure", "Project has no modules")
            return
        result.checks_passed += 1

        for module in project_ir.modules:
            if not module.files:
                result.add_issue("warning", "structure",
                    f"Module '{module.name}' has no files", suggestion="Add source files to the module")
            else:
                result.checks_passed += 1

    def _check_required_files(self, project_ir: ProjectIR, result: ProjectVerificationResult):
        required = []
        if project_ir.platform == Platform.ANDROID:
            required = self._required_android_files
        elif project_ir.platform == Platform.WEB:
            required = self._required_web_files

        all_paths = [f.path for m in project_ir.modules for f in m.files]
        for req in required:
            if req not in all_paths:
                result.add_issue("error", "required_file",
                    f"Missing required file: {req}",
                    suggestion=f"Add {req} to the project")
            else:
                result.checks_passed += 1

    def _check_imports(self, project_ir: ProjectIR,
                       generation_result: ProjectGenerationResult,
                       result: ProjectVerificationResult):
        all_files = project_ir.get_all_files()
        for file_ir in all_files:
            if file_ir.file_type != FileType.SOURCE:
                continue
            for imp in file_ir.imports:
                if imp.startswith("android.") or imp.startswith("java.") or imp.startswith("javax."):
                    continue
                if imp.startswith("androidx.") or imp.startswith("com.google."):
                    continue
                if imp.startswith(".") or imp.startswith("src.") or imp.startswith("tests."):
                    continue
                if "/" in imp and imp.endswith(".py"):
                    continue
        result.checks_passed += 1

    def _check_dependencies(self, project_ir: ProjectIR,
                            generation_result: ProjectGenerationResult,
                            result: ProjectVerificationResult):
        all_files = project_ir.get_all_files()
        all_paths = set(f.path for f in all_files)
        all_decls = set()
        for f in all_files:
            all_decls.update(f.declarations)

        for file_ir in all_files:
            for dep in file_ir.dependencies:
                if dep.target_file not in all_paths and dep.target_file not in all_decls:
                    result.add_issue("warning", "dependency",
                        f"Unresolved dependency: {file_ir.path} -> {dep.target_file}",
                        file_path=file_ir.path,
                        suggestion=f"Add {dep.target_file} or resolve the dependency")
                else:
                    result.checks_passed += 1

    def _check_manifest(self, project_ir: ProjectIR, result: ProjectVerificationResult):
        if project_ir.platform != Platform.ANDROID:
            result.checks_passed += 1
            return

        manifest_files = []
        for m in project_ir.modules:
            for f in m.files:
                if f.file_type == FileType.MANIFEST:
                    manifest_files.append(f)

        if not manifest_files:
            result.add_issue("error", "manifest", "No AndroidManifest.xml found")
            return

        for manifest in manifest_files:
            meta = manifest.metadata
            if not meta.get("package_name"):
                result.add_issue("error", "manifest",
                    "Manifest missing package name",
                    file_path=manifest.path,
                    suggestion="Add package_name to manifest metadata")
            else:
                result.checks_passed += 1

            if not meta.get("activities"):
                result.add_issue("warning", "manifest",
                    "Manifest has no activities declared",
                    file_path=manifest.path,
                    suggestion="Add at least one activity")
            else:
                result.checks_passed += 1

            if manifest.permissions:
                result.checks_passed += 1

    def _check_gradle_config(self, project_ir: ProjectIR, result: ProjectVerificationResult):
        if project_ir.platform != Platform.ANDROID:
            result.checks_passed += 1
            return

        gradle_files = []
        for m in project_ir.modules:
            for f in m.files:
                if f.file_type == FileType.BUILD and f.content_type in ("gradle", "root_gradle", "settings_gradle"):
                    gradle_files.append(f)

        if not gradle_files:
            result.add_issue("error", "gradle", "No Gradle build files found")
            return

        result.checks_passed += 1

        app_gradle = [f for f in gradle_files if f.content_type == "gradle"]
        if not app_gradle:
            result.add_issue("error", "gradle", "No app/build.gradle.kts found")
        else:
            result.checks_passed += 1

    def _check_source_files(self, project_ir: ProjectIR,
                            generation_result: ProjectGenerationResult,
                            result: ProjectVerificationResult):
        source_files = project_ir.get_all_source_files()
        if not source_files:
            result.add_issue("warning", "source", "No source files found",
                suggestion="Add at least one source file")
            return

        result.checks_passed += 1

        placeholder_patterns = [
            "// TODO: Implement",
            "pass",
            "// Generated file",
            "// Generated Kotlin file",
            "// Generated Python file",
            "// Generated JavaScript file",
            "// Generated Java file",
        ]
        
        has_meaningful_code = False
        
        for file_ir in source_files:
            if not file_ir.declarations and file_ir.content_type not in ("init",):
                result.add_issue("warning", "source",
                    f"Source file {file_ir.path} has no declarations",
                    file_path=file_ir.path)
            else:
                result.checks_passed += 1
            
            if generation_result:
                gen_file = generation_result.get_file(file_ir.path)
                if gen_file:
                    content = gen_file.content
                    is_placeholder = False
                    for pattern in placeholder_patterns:
                        if pattern in content:
                            is_placeholder = True
                            break
                    
                    if is_placeholder:
                        result.add_issue("warning", "source",
                            f"Source file {file_ir.path} contains placeholder code",
                            file_path=file_ir.path,
                            suggestion="Replace placeholder code with actual implementation")
                    else:
                        has_meaningful_code = True
        
        if generation_result and source_files and not has_meaningful_code:
            result.add_issue("error", "source",
                "All source files contain only placeholder code",
                suggestion="Implement actual functionality in source files")

    def _check_kotlin_syntax(self, project_ir: ProjectIR,
                             generation_result: ProjectGenerationResult,
                             result: ProjectVerificationResult):
        if not generation_result:
            return
        for gf in generation_result.files:
            if not gf.path.endswith(".kt"):
                continue
            content = gf.content
            if "??" in content:
                result.add_issue("error", "syntax",
                    f"Invalid Kotlin syntax '??' in {gf.path}",
                    file_path=gf.path,
                    suggestion="Use '?' for nullable types, not '??'")
            brace_count = content.count("{") - content.count("}")
            if brace_count != 0:
                result.add_issue("warning", "syntax",
                    f"Unbalanced braces in {gf.path} (diff: {brace_count})",
                    file_path=gf.path)
        result.checks_passed += 1

    def _check_package_consistency(self, project_ir: ProjectIR,
                                   result: ProjectVerificationResult):
        if project_ir.platform != Platform.ANDROID:
            result.checks_passed += 1
            return
        manifest_pkg = None
        for m in project_ir.modules:
            for f in m.files:
                if f.file_type == FileType.MANIFEST:
                    manifest_pkg = f.metadata.get("package_name")
                    break
        if not manifest_pkg:
            result.checks_passed += 1
            return
        for m in project_ir.modules:
            for f in m.files:
                if f.file_type == FileType.SOURCE and f.language == "kotlin":
                    if f.path.startswith("app/src/main/java/"):
                        path_pkg_parts = f.path.replace("app/src/main/java/", "").split("/")
                        path_pkg = ".".join(path_pkg_parts[:-1])
                        if path_pkg and path_pkg != manifest_pkg:
                            result.add_issue("warning", "package",
                                f"Package mismatch: {f.path} has path package '{path_pkg}' but manifest has '{manifest_pkg}'",
                                file_path=f.path,
                                suggestion="Ensure package matches directory structure")
        result.checks_passed += 1

    def _check_generation_errors(self, generation_result: ProjectGenerationResult,
                                 result: ProjectVerificationResult):
        if generation_result.errors:
            for err in generation_result.errors:
                result.add_issue("error", "generation", err)
        else:
            result.checks_passed += 1

    def _check_success_criteria(self, project_ir: ProjectIR,
                                generation_result: ProjectGenerationResult,
                                criteria: list,
                                result: ProjectVerificationResult):
        """Check that generated code actually implements the required operations."""
        # Collect all generated code
        all_code = ""
        for gf in generation_result.files:
            all_code += gf.content + "\n"

        all_code_lower = all_code.lower()

        # Detect platform-specific patterns
        is_android = project_ir.platform == Platform.ANDROID
        is_cli = project_ir.platform == Platform.CLI

        for criterion in criteria:
            if isinstance(criterion, dict):
                check_type = criterion.get("check_type", "")
                params = criterion.get("params", {})
                description = criterion.get("description", "")
            else:
                # Support SuccessCriteria dataclass
                check_type = criterion.check_type if hasattr(criterion, 'check_type') else ""
                params = criterion.params if hasattr(criterion, 'params') else {}
                description = criterion.description if hasattr(criterion, 'description') else str(criterion)

            if check_type == "contains_operation":
                op_name = params.get("operation", "")
                if op_name:
                    # Check if the operation is actually implemented in the code
                    op_implemented = self._check_operation_implemented(op_name, all_code_lower)
                    if op_implemented:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "error", "success_criteria",
                            f"Operation '{op_name}' ({description}) is not implemented in generated code",
                            suggestion=f"Implement {op_name.replace('_', ' ')} in the generated code"
                        )

            elif check_type == "sort_order":
                order = params.get("order", "descending")
                if order == "descending":
                    if "reverse=True" in all_code or "reverse = true" in all_code_lower or "desc" in all_code_lower:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "warning", "success_criteria",
                            f"Sort order '{order}' may not be correctly implemented",
                            suggestion="Ensure sorting uses reverse=True for descending order"
                        )
                else:
                    result.checks_passed += 1

            elif check_type == "has_function":
                if "def " in all_code or "fun " in all_code or "function " in all_code:
                    result.checks_passed += 1
                else:
                    result.add_issue(
                        "error", "success_criteria",
                        "No function definition found in generated code",
                        suggestion="Add a function definition"
                    )

            elif check_type == "has_model":
                # For Android apps, check for Activity or data class
                if is_android:
                    if "Activity" in all_code or "class " in all_code or "@Entity" in all_code:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "error", "success_criteria",
                            "No Activity or data model found in generated code",
                            suggestion="Add an Activity or data model class"
                        )
                else:
                    if "class " in all_code or "@dataclass" in all_code or "@Entity" in all_code:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "error", "success_criteria",
                            "No data model/class found in generated code",
                            suggestion="Add a data model class"
                        )

            elif check_type == "has_entry_point":
                # For Android apps, check for Activity
                if is_android:
                    if "Activity" in all_code or "onCreate" in all_code:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "error", "success_criteria",
                            "No Activity found in generated code",
                            suggestion="Add an Activity class"
                        )
                else:
                    if 'if __name__' in all_code or "def main" in all_code or "fun main" in all_code:
                        result.checks_passed += 1
                    else:
                        result.add_issue(
                            "error", "success_criteria",
                            "No entry point found in generated code",
                            suggestion="Add a main function or entry point"
                        )
            else:
                result.checks_passed += 1

    def _check_operation_implemented(self, op_name: str, code: str) -> bool:
        """Check if a specific operation is actually implemented in code."""
        op_checks = {
            "remove_duplicates": [
                "seen", "set()", "unique", "distinct", "not in",
            ],
            "calculate_average": [
                "average", "mean", "sum(", "len(",
            ],
            "filter_above": [
                ">", "filter", "if x >", "for x in",
            ],
            "filter_below": [
                "<", "filter", "if x <", "for x in",
            ],
            "sort_descending": [
                "sorted(", "reverse=True", "sort",
            ],
            "sort_ascending": [
                "sorted(", "sort",
            ],
            "sort": [
                "sorted(", "sort",
            ],
            "add": [
                "def add", "append", "add(",
            ],
            "remove": [
                "def remove", "remove(",
            ],
            "search": [
                "def search", "search(",
            ],
            "list": [
                "def list", "list_all", "list(",
            ],
            "store": [
                "save", "store", "persist",
            ],
            "calculate": [
                "def calc", "calculate", "compute",
            ],
        }

        keywords = op_checks.get(op_name, [])
        if not keywords:
            return True  # No specific check available, assume implemented

        # Check if at least some of the keywords are present
        found = sum(1 for kw in keywords if kw in code)
        return found >= len(keywords) // 2 or found >= 2

    def _try_compile_all(self, generation_result: ProjectGenerationResult,
                         language: str) -> list[dict]:
        """Attempt to compile all generated source files."""
        results = []
        compilers = {
            "python": ["python3", "-m", "py_compile"],
            "kotlin": ["kotlinc"] if shutil.which("kotlinc") else None,
            "java": ["javac"] if shutil.which("javac") else None,
            "c": ["gcc", "-fsyntax-only"] if shutil.which("gcc") else None,
            "cpp": ["g++", "-fsyntax-only"] if shutil.which("g++") else None,
            "javascript": ["node", "--check"] if shutil.which("node") else None,
            "go": ["go", "build"] if shutil.which("go") else None,
            "rust": ["cargo", "check"] if shutil.which("cargo") else None,
        }
        cmd = compilers.get(language)
        if not cmd:
            return [{"status": "toolchain_missing", "language": language}]

        for f in generation_result.files:
            if not hasattr(f, 'path') or not hasattr(f, 'content'):
                continue
            if not f.content:
                continue
            ext = f.path.rsplit(".", 1)[-1] if "." in f.path else ""
            lang_exts = {
                "python": ["py"], "kotlin": ["kt"], "java": ["java"],
                "c": ["c"], "cpp": ["cpp", "cc", "cxx"],
                "javascript": ["js"], "go": ["go"], "rust": ["rs"],
            }
            if ext not in lang_exts.get(language, []):
                continue

            import tempfile, os
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{ext}',
                                                  delete=False) as tmp:
                    tmp.write(f.content)
                    tmp_path = tmp.name
                proc = subprocess.run(cmd + [tmp_path],
                                       capture_output=True, timeout=10, text=True)
                os.unlink(tmp_path)
                if proc.returncode == 0:
                    results.append({"status": "success", "file": f.path, "language": language})
                else:
                    results.append({"status": "failed", "file": f.path,
                                    "error": proc.stderr[:500], "language": language})
            except subprocess.TimeoutExpired:
                results.append({"status": "timeout", "file": f.path, "language": language})
            except Exception as e:
                results.append({"status": "error", "file": f.path,
                                "error": str(e), "language": language})

        return results

    def _check_requirement_coverage(self, requirements: list,
                                     generation_result: ProjectGenerationResult) -> list[str]:
        """Check if generated code covers all requested requirements."""
        missing = []
        if not requirements:
            return missing

        all_code = " ".join(
            f.content for f in generation_result.files
            if hasattr(f, 'content') and f.content
        ).lower()

        for req in requirements:
            if hasattr(req, 'name'):
                req_name = req.name
                req_value = req.value
                req_category = req.category
            else:
                req_name = req.lower().replace(" ", "_")
                req_value = ""
                req_category = ""

            satisfied = False

            # Check by requirement type
            if req_category == "numeric" or "zoom" in req_name:
                # Check for zoom/camera implementation
                if any(kw in all_code for kw in ["zoom", "camera", "capture", "camerax"]):
                    satisfied = True
            elif req_category == "quality" or "complete" in req_name:
                # Check for complete implementation (not just stubs)
                if len(all_code) > 500:  # Substantial code
                    satisfied = True
            elif req_category == "feature":
                # Check for feature implementation
                if req_name.replace("camera_", "") in all_code:
                    satisfied = True
            elif req_category == "dependency":
                # Check for dependency usage
                if req_value.lower() in all_code:
                    satisfied = True
            elif req_category == "ui":
                # Check for UI implementation
                if any(kw in all_code for kw in ["activity", "fragment", "compose", "layout"]):
                    satisfied = True
            elif req_category == "operation":
                # Check for operation implementation
                if req_name in all_code:
                    satisfied = True
            else:
                # Generic check
                if req_name in all_code or (hasattr(req, 'value') and req.value in all_code):
                    satisfied = True

            if not satisfied:
                missing.append(req_name)

        return missing
