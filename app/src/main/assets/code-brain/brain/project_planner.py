"""Project Planner - generates Project IR from natural language requests with dependency reasoning."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from brain.reasoner import ReasoningResult, StructuredIntent, ClassInfo
from code.ir.project_ir import (
    ProjectIR, ModuleIR, FileIR, FileType, Platform,
    ModuleDependency, FileDependency, DependencyType,
)


# LEGACY: Platform detection patterns
# source: legacy, usage: fallback+compatibility, confidence: 0.6
# Primary path: SemanticParser constraint extraction
PLATFORM_PATTERNS = {
    Platform.ANDROID: {
        "keywords_ko": ["안드로이드", "카메라 앱", "모바일 앱", "핸드폰 앱"],
        "keywords_en": ["android", "camera app", "mobile app"],
        "language": "kotlin",
    },
    Platform.WEB: {
        "keywords_ko": ["웹 앱", "웹사이트", "프론트엔드", "웹페이지"],
        "keywords_en": ["web app", "website", "frontend", "webpage"],
        "language": "javascript",
    },
    Platform.CLI: {
        "keywords_ko": ["명령줄", "CLI", "스크립트", "도구"],
        "keywords_en": ["cli", "command line", "script", "tool"],
        "language": "python",
    },
    Platform.BACKEND: {
        "keywords_ko": ["백엔드", "서버", "API", "REST"],
        "keywords_en": ["backend", "server", "api", "rest"],
        "language": "python",
    },
    Platform.LIBRARY: {
        "keywords_ko": ["라이브러리", "패키지", "모듈"],
        "keywords_en": ["library", "package", "module"],
        "language": "python",
    },
}


def _detect_platform(request: str) -> Platform:
    request_lower = request.lower()
    for platform, patterns in PLATFORM_PATTERNS.items():
        for kw in patterns["keywords_ko"]:
            if kw in request:
                return platform
        for kw in patterns["keywords_en"]:
            if kw in request_lower:
                return platform
    return Platform.CLI


def _infer_dependencies(files: list[FileIR]) -> list[FileDependency]:
    deps = []
    for f in files:
        if f.file_type != FileType.SOURCE:
            continue
        for other in files:
            if other.id == f.id:
                continue
            if other.file_type != FileType.SOURCE:
                continue
            other_name = other.path.split("/")[-1].replace(".kt", "").replace(".java", "").replace(".py", "").replace(".js", "").replace(".jsx", "").replace(".go", "").replace(".rs", "").replace(".c", "").replace(".cpp", "")
            if other_name in f.imports or other_name in f.declarations:
                deps.append(FileDependency(
                    source_file=f.path,
                    target_file=other.path,
                    dependency_type=DependencyType.COMPILE,
                    reason=f"{f.path.split('/')[-1]} references {other_name}",
                ))
    return deps


def _extract_class_info_from_request(request: str) -> Optional[ClassInfo]:
    name = None
    m = re.search(r'(\w+)\s*클래스', request)
    if m:
        name = m.group(1)
    m = re.search(r'class\s+(\w+)', request, re.IGNORECASE)
    if m:
        name = m.group(1)
    if name:
        return ClassInfo(name=name, properties=[], constructor_params=[])
    return None


class ProjectPlanner:
    def __init__(self):
        self._plan_counter = 0
        from brain.project_requirement_analyzer import analyze_project_request
        self._requirement_analyzer = analyze_project_request

    @staticmethod
    def _requirement_entries(requirements, reasoning=None) -> list:
        """The PLAN's requirement records: one entry per requirement,
        each with the id the rest of the pipeline reads.

        The analyzer's requirements carry their own ids (the plan's
        requirement vocabulary); the pipeline's requirement objects join
        when they add a requirement the analyzer did not name — every
        id is recorded once and never re-derived downstream."""
        entries = []
        seen = set()
        for feature in (getattr(requirements, "features", None) or []):
            rid = str(getattr(feature, "req_id", "") or "")
            if not rid:
                continue
            name = str(feature.name)
            entries.append({
                "req_id": rid,
                "name": name,
                "category": str(getattr(feature, "category", "") or ""),
                "value": "",
                "description": str(getattr(feature, "description", "") or ""),
                "source": "requirements",
            })
            seen.add(name.lower())
        model = getattr(reasoning, "model", None)
        for req in (getattr(model, "requirements", None) or []):
            rid = str(getattr(req, "id", "") or "")
            name = str(getattr(req, "name", "") or "")
            if not rid or not name or name.lower() in seen:
                continue
            entries.append({
                "req_id": rid,
                "name": name,
                "category": str(getattr(req, "category", "") or ""),
                "value": str(getattr(req, "value", "") or ""),
                "description": str(getattr(req, "description", "") or ""),
                "source": "reasoning",
            })
            seen.add(name.lower())
        return entries

    @staticmethod
    def _attach_requirement_ids(project: ProjectIR, requirements,
                                reasoning=None) -> ProjectIR:
        """plan -> IR: the requirement ids enter the plan's metadata and
        reach every file that implements one, so the generated code can
        carry the SAME ids the plan recorded."""
        entries = ProjectPlanner._requirement_entries(requirements, reasoning)
        project.metadata["requirements"] = entries
        project.metadata["requirement_ids"] = [e["req_id"] for e in entries]
        by_name = {}
        for entry in entries:
            name = str(entry.get("name") or "")
            if name:
                by_name[name.lower()] = entry["req_id"]
        if not by_name:
            return project
        for module in project.modules:
            for file_ir in module.files:
                meta = file_ir.metadata
                if not isinstance(meta, dict):
                    continue
                named = []
                for key in ("features", "methods"):
                    for name in (meta.get(key) or []):
                        rid = by_name.get(str(name).lower())
                        if rid and rid not in named:
                            named.append(rid)
                if named:
                    meta["requirement_ids"] = named
        return project

    def create_project_plan(self, request: str, reasoning: ReasoningResult = None,
                            language: str = "python", event_collector=None) -> ProjectIR:
        self._event_collector = event_collector
        requirements = self._requirement_analyzer(request)
        
        if self._event_collector:
            self._event_collector.add_understanding(
                requirements.domain,
                requirements.platform,
                [f.name for f in requirements.features],
                [dm.name for dm in requirements.data_models],
                language
            )
        
        if requirements.domain != "generic":
            return self._attach_requirement_ids(
                self._build_from_requirements(requirements, language,
                                              original_request=request),
                requirements, reasoning)
        
        # Even for generic domain, check if there are pipeline operations
        pipeline_ops = self._detect_pipeline_operations(requirements, request)
        if pipeline_ops:
            return self._attach_requirement_ids(
                self._build_from_requirements(requirements, language,
                                              original_request=request),
                requirements, reasoning)
        
        platform = _detect_platform(request)

        return self._attach_requirement_ids(
            self._build_generic_project(request, platform, language),
            requirements, reasoning)

    def _build_from_requirements(self, requirements, language: str,
                                  original_request: str = None) -> ProjectIR:
        lang = language or requirements.language

        # Check if this is a pipeline operation project using original request
        pipeline_ops = self._detect_pipeline_operations(requirements, original_request)

        project = ProjectIR(
            name=requirements.project_name,
            language=lang,
            platform=Platform.CLI if requirements.platform == "cli" else Platform.WEB if requirements.platform == "web" else Platform.ANDROID if requirements.platform == "android" else Platform.CLI,
            metadata={
                "domain": requirements.domain,
                "description": requirements.description,
                "features": [f.name for f in requirements.features],
                "data_models": [dm.name for dm in requirements.data_models],
                "pipeline_operations": pipeline_ops,
            },
        )

        if self._event_collector:
            self._event_collector.add_planning(
                requirements.project_name,
                "layered" if requirements.data_models else "simple",
                ["src"],
                lang
            )

        if pipeline_ops and lang == "python":
            return self._build_python_pipeline_project(project, requirements, pipeline_ops)
        elif lang == "kotlin":
            return self._build_kotlin_project(project, requirements)
        elif lang == "java":
            return self._build_java_project(project, requirements)
        elif lang == "python":
            return self._build_python_project(project, requirements)
        elif lang == "javascript":
            return self._build_javascript_project(project, requirements)
        else:
            return self._build_python_project(project, requirements)

    def _detect_pipeline_operations(self, requirements, original_request: str = None) -> list:
        """Detect if this project has pipeline operations (filter, sort, aggregate)."""
        pipeline_keywords = {
            "remove_duplicates": ["duplicate", "deduplicate", "unique", "distinct", "no duplicate"],
            "calculate_average": ["average", "mean", "average score", "average value"],
            "filter_above": ["above", "greater than", "more than", "over", "higher than",
                            "above average", "greater than average", "values above",
                            "eligible", "qualifying", "meets criteria"],
            "filter_below": ["below", "less than", "under", "lower than",
                            "below average", "less than average"],
            "sort_descending": ["descending", "largest first", "highest first",
                               "sort descending", "order descending", "by score descending"],
            "sort_ascending": ["ascending", "smallest first", "lowest first",
                              "sort ascending", "order ascending"],
            "sort": ["sort", "order", "sorted", "ordered"],
            "calculate": ["calculate", "compute", "calculates", "computes"],
            "filter": ["filter", "filters", "identify", "identifies", "select", "selects"],
        }

        # Check both the requirements description and the original request
        texts_to_check = [requirements.description.lower()]
        if original_request:
            texts_to_check.append(original_request.lower())
        features = [f.name.lower() for f in requirements.features]

        ops = []
        for op_name, keywords in pipeline_keywords.items():
            for kw in keywords:
                for text in texts_to_check:
                    if kw in text or kw in " ".join(features):
                        if op_name not in ops:
                            ops.append(op_name)
                        break
                else:
                    continue
                break

        return ops

    def _build_python_pipeline_project(self, project: ProjectIR, requirements,
                                       pipeline_ops: list) -> ProjectIR:
        """Build a Python project with a data processing pipeline."""
        module = ModuleIR(name="src", path="src")

        module.add_file(FileIR(
            path="src/__init__.py",
            file_type=FileType.SOURCE,
            language="python",
            content_type="init",
            metadata={"template": "python_init"},
        ))

        # Generate pipeline function file
        pipeline_path = "src/pipeline.py"
        pipeline_file = FileIR(
            path=pipeline_path,
            file_type=FileType.SOURCE,
            language="python",
            content_type="module",
            imports=[],
            declarations=["process_data"],
            metadata={
                "template": "python_pipeline",
                "function_name": "process_data",
                "operations": pipeline_ops,
                "inputs": [{"name": "data", "type": "List[float]"}],
                "outputs": [{"name": "result", "type": "List[float]"}],
                "conditions": [],
            },
        )
        module.add_file(pipeline_file)

        # Generate data model if needed
        if requirements.data_models:
            dm = requirements.data_models[0]
            model_path = f"src/{dm.name.lower()}.py"
            model_file = FileIR(
                path=model_path,
                file_type=FileType.SOURCE,
                language="python",
                content_type="class",
                imports=["dataclasses"],
                declarations=[dm.name],
                metadata={
                    "template": "python_data_class",
                    "class_name": dm.name,
                    "fields": dm.fields,
                    "methods": dm.methods,
                },
            )
            module.add_file(model_file)

        # Generate entry point
        main_path = "src/main.py"
        main_file = FileIR(
            path=main_path,
            file_type=FileType.SOURCE,
            language="python",
            content_type="entry_point",
            imports=["src.pipeline"],
            declarations=["main"],
            metadata={
                "template": "python_main",
                "functions": ["main"],
                "is_entry_point": True,
            },
        )
        module.add_file(main_file)

        if self._event_collector:
            self._event_collector.add_file_creation(
                pipeline_path, "source", "python_pipeline",
                f"Data processing pipeline: {' -> '.join(pipeline_ops)}", [],
                project.language
            )
            self._event_collector.add_file_creation(
                main_path, "source", "python_main",
                f"Application entry point", [pipeline_path],
                project.language
            )

        project.add_module(module)
        return project

    def _build_kotlin_project(self, project: ProjectIR, requirements) -> ProjectIR:
        module = ModuleIR(name="src", path="src")
        pkg = f"com.example.{project.name.lower()}"

        main_class = "Main"
        model_files = []
        service_files = []

        for dm in requirements.data_models:
            model_path = f"src/main/kotlin/{pkg.replace('.', '/')}/model/{dm.name}.kt"
            model_file = FileIR(
                path=model_path,
                file_type=FileType.SOURCE,
                language="kotlin",
                content_type="data_class",
                imports=[],
                declarations=[dm.name],
                metadata={
                    "template": "kotlin_data_class",
                    "class_name": dm.name,
                    "fields": dm.fields,
                    "methods": dm.methods,
                },
            )
            model_files.append(model_file)
            module.add_file(model_file)

        service_path = f"src/main/kotlin/{pkg.replace('.', '/')}/service/{project.name}Service.kt"
        service_file = FileIR(
            path=service_path,
            file_type=FileType.SOURCE,
            language="kotlin",
            content_type="class",
            imports=[f"model.{dm.name}" for dm in requirements.data_models],
            declarations=[f"{project.name}Service"],
            metadata={
                "template": "kotlin_service",
                "class_name": f"{project.name}Service",
                "data_models": [dm.name for dm in requirements.data_models],
                "features": [f.name for f in requirements.features],
                "methods": [f.name for f in requirements.features],
            },
        )
        service_files.append(service_file)
        module.add_file(service_file)

        main_path = f"src/main/kotlin/{pkg.replace('.', '/')}/Main.kt"
        main_file = FileIR(
            path=main_path,
            file_type=FileType.SOURCE,
            language="kotlin",
            content_type="entry_point",
            imports=[f"service.{project.name}Service"],
            declarations=[main_class],
            metadata={
                "template": "kotlin_main",
                "class_name": main_class,
                "service_class": f"{project.name}Service",
                "features": [f.name for f in requirements.features],
            },
        )
        module.add_file(main_file)

        if self._event_collector:
            for dm in requirements.data_models:
                self._event_collector.add_file_creation(
                    model_path, "source", "kotlin_data_class",
                    f"Data model for {dm.name}", [f"service.{project.name}Service"],
                    project.language
                )
            self._event_collector.add_file_creation(
                service_path, "source", "kotlin_service",
                f"Business logic for {project.name}", [f"model.{dm.name}" for dm in requirements.data_models],
                project.language
            )
            self._event_collector.add_file_creation(
                main_path, "source", "kotlin_main",
                f"Application entry point", [f"service.{project.name}Service"],
                project.language
            )

        project.add_module(module)
        return project

    def _build_java_project(self, project: ProjectIR, requirements) -> ProjectIR:
        module = ModuleIR(name="src", path="src")
        pkg = f"com.example.{project.name.lower()}"

        for dm in requirements.data_models:
            model_path = f"src/main/java/{pkg.replace('.', '/')}/model/{dm.name}.java"
            model_file = FileIR(
                path=model_path,
                file_type=FileType.SOURCE,
                language="java",
                content_type="class",
                imports=[],
                declarations=[dm.name],
                metadata={
                    "template": "java_data_class",
                    "class_name": dm.name,
                    "fields": dm.fields,
                    "methods": dm.methods,
                },
            )
            module.add_file(model_file)

        service_path = f"src/main/java/{pkg.replace('.', '/')}/service/{project.name}Service.java"
        service_file = FileIR(
            path=service_path,
            file_type=FileType.SOURCE,
            language="java",
            content_type="class",
            imports=[f"model.{dm.name}" for dm in requirements.data_models],
            declarations=[f"{project.name}Service"],
            metadata={
                "template": "java_service",
                "class_name": f"{project.name}Service",
                "data_models": [dm.name for dm in requirements.data_models],
                "features": [f.name for f in requirements.features],
                "methods": [f.name for f in requirements.features],
            },
        )
        module.add_file(service_file)

        main_path = f"src/main/java/{pkg.replace('.', '/')}/Main.java"
        main_file = FileIR(
            path=main_path,
            file_type=FileType.SOURCE,
            language="java",
            content_type="entry_point",
            imports=[f"service.{project.name}Service"],
            declarations=["Main"],
            metadata={
                "template": "java_main",
                "class_name": "Main",
                "service_class": f"{project.name}Service",
                "features": [f.name for f in requirements.features],
            },
        )
        module.add_file(main_file)

        if self._event_collector:
            for dm in requirements.data_models:
                self._event_collector.add_file_creation(
                    model_path, "source", "java_data_class",
                    f"Data model for {dm.name}", [f"service.{project.name}Service"],
                    project.language
                )
            self._event_collector.add_file_creation(
                service_path, "source", "java_service",
                f"Business logic for {project.name}", [f"model.{dm.name}" for dm in requirements.data_models],
                project.language
            )
            self._event_collector.add_file_creation(
                main_path, "source", "java_main",
                f"Application entry point", [f"service.{project.name}Service"],
                project.language
            )

        project.add_module(module)
        return project

    def _build_python_project(self, project: ProjectIR, requirements) -> ProjectIR:
        module = ModuleIR(name="src", path="src")

        module.add_file(FileIR(
            path="src/__init__.py",
            file_type=FileType.SOURCE,
            language="python",
            content_type="init",
            metadata={"template": "python_init"},
        ))

        for dm in requirements.data_models:
            model_path = f"src/{dm.name.lower()}.py"
            model_file = FileIR(
                path=model_path,
                file_type=FileType.SOURCE,
                language="python",
                content_type="class",
                imports=["dataclasses"],
                declarations=[dm.name],
                metadata={
                    "template": "python_data_class",
                    "class_name": dm.name,
                    "fields": dm.fields,
                    "methods": dm.methods,
                },
            )
            module.add_file(model_file)

        service_path = f"src/{project.name.lower()}_service.py"
        service_file = FileIR(
            path=service_path,
            file_type=FileType.SOURCE,
            language="python",
            content_type="module",
            imports=[dm.name.lower() for dm in requirements.data_models],
            declarations=[f"{project.name}Service"],
            metadata={
                "template": "python_service",
                "class_name": f"{project.name}Service",
                "data_models": [dm.name for dm in requirements.data_models],
                "features": [f.name for f in requirements.features],
                "methods": [f.name for f in requirements.features],
            },
        )
        module.add_file(service_file)

        main_path = "src/main.py"
        main_file = FileIR(
            path=main_path,
            file_type=FileType.SOURCE,
            language="python",
            content_type="entry_point",
            imports=[f"src.{project.name.lower()}_service"],
            declarations=["main"],
            metadata={
                "template": "python_main_with_service",
                "service_class": f"{project.name}Service",
                "features": [f.name for f in requirements.features],
            },
        )
        module.add_file(main_file)

        if self._event_collector:
            for dm in requirements.data_models:
                self._event_collector.add_file_creation(
                    model_path, "source", "python_data_class",
                    f"Data model for {dm.name}", [service_path],
                    project.language
                )
            self._event_collector.add_file_creation(
                service_path, "source", "python_service",
                f"Business logic for {project.name}", [dm.name.lower() for dm in requirements.data_models],
                project.language
            )
            self._event_collector.add_file_creation(
                main_path, "source", "python_main_with_service",
                f"Application entry point", [service_path],
                project.language
            )

        project.add_module(module)
        return project

    def _build_javascript_project(self, project: ProjectIR, requirements) -> ProjectIR:
        module = ModuleIR(name="src", path="src")

        for dm in requirements.data_models:
            model_path = f"src/{dm.name.lower()}.js"
            model_file = FileIR(
                path=model_path,
                file_type=FileType.SOURCE,
                language="javascript",
                content_type="module",
                imports=[],
                declarations=[dm.name],
                metadata={
                    "template": "js_data_class",
                    "class_name": dm.name,
                    "fields": dm.fields,
                    "methods": dm.methods,
                },
            )
            module.add_file(model_file)

        service_path = f"src/{project.name.lower()}_service.js"
        service_file = FileIR(
            path=service_path,
            file_type=FileType.SOURCE,
            language="javascript",
            content_type="module",
            imports=[dm.name.lower() for dm in requirements.data_models],
            declarations=[f"{project.name}Service"],
            metadata={
                "template": "js_service",
                "class_name": f"{project.name}Service",
                "data_models": [dm.name for dm in requirements.data_models],
                "features": [f.name for f in requirements.features],
                "methods": [f.name for f in requirements.features],
            },
        )
        module.add_file(service_file)

        main_path = "src/main.js"
        main_file = FileIR(
            path=main_path,
            file_type=FileType.SOURCE,
            language="javascript",
            content_type="entry_point",
            imports=[f"./{project.name.lower()}_service"],
            declarations=["main"],
            metadata={
                "template": "js_main",
                "service_class": f"{project.name}Service",
                "features": [f.name for f in requirements.features],
            },
        )
        module.add_file(main_file)

        if self._event_collector:
            for dm in requirements.data_models:
                self._event_collector.add_file_creation(
                    model_path, "source", "js_data_class",
                    f"Data model for {dm.name}", [service_path],
                    project.language
                )
            self._event_collector.add_file_creation(
                service_path, "source", "js_service",
                f"Business logic for {project.name}", [dm.name.lower() for dm in requirements.data_models],
                project.language
            )
            self._event_collector.add_file_creation(
                main_path, "source", "js_main",
                f"Application entry point", [service_path],
                project.language
            )

        project.add_module(module)
        return project

    def _build_generic_project(self, request: str, platform: Platform, language: str) -> ProjectIR:
        project_name = self._extract_project_name(request)
        lang = language or PLATFORM_PATTERNS.get(platform, {}).get("language", "python")

        project = ProjectIR(
            name=project_name,
            language=lang,
            platform=platform,
        )

        if platform == Platform.ANDROID:
            return self._build_android_project(project, request)
        elif platform == Platform.WEB:
            return self._build_web_project(project, request, lang)
        else:
            return self._build_generic_cli_project(project, request, lang)

    def _build_android_project(self, project: ProjectIR, request: str) -> ProjectIR:
        module = ModuleIR(name="app", path="app")
        pkg = f"com.example.{project.name.lower()}"

        module.add_file(FileIR(
            path="settings.gradle.kts",
            file_type=FileType.BUILD,
            language="kotlin",
            content_type="settings_gradle",
            metadata={"template": "settings_gradle_kts", "project_name": project.name, "modules": [":app"]},
        ))
        module.add_file(FileIR(
            path="build.gradle.kts",
            file_type=FileType.BUILD,
            language="kotlin",
            content_type="root_gradle",
            metadata={"template": "root_gradle_kts", "plugins_list": [
                "com.android.application version 8.2.0 apply false",
                "org.jetbrains.kotlin.android version 1.9.22 apply false",
            ]},
        ))
        module.add_file(FileIR(
            path="app/build.gradle.kts",
            file_type=FileType.BUILD,
            language="kotlin",
            content_type="gradle",
            metadata={"template": "app_gradle_kts", "min_sdk": "24", "target_sdk": "34",
                      "dependencies_list": [
                          "androidx.core:core-ktx:1.12.0",
                          "androidx.appcompat:appcompat:1.6.1",
                          "com.google.android.material:material:1.11.0",
                      ]},
        ))
        module.add_file(FileIR(
            path=f"app/src/main/AndroidManifest.xml",
            file_type=FileType.MANIFEST,
            language="xml",
            content_type="manifest",
            permissions=["android.permission.CAMERA"],
            metadata={"template": "android_manifest", "package_name": pkg, "min_sdk": "24",
                      "target_sdk": "34", "activities": ["MainActivity"]},
        ))

        activity_name = self._extract_main_class(request) or "MainActivity"
        main_activity_path = f"app/src/main/java/{pkg.replace('.', '/')}/{activity_name}.kt"
        module.add_file(FileIR(
            path=main_activity_path,
            file_type=FileType.SOURCE,
            language="kotlin",
            content_type="activity",
            imports=["android.os.Bundle", "androidx.activity.ComponentActivity"],
            declarations=[activity_name],
            metadata={"template": "kotlin_activity", "class_name": activity_name,
                      "parent_class": "ComponentActivity", "functions": ["onCreate"]},
        ))

        project.add_module(module)
        return project

    def _build_web_project(self, project: ProjectIR, request: str, language: str) -> ProjectIR:
        module = ModuleIR(name="frontend", path="frontend")
        module.add_file(FileIR(
            path="frontend/package.json",
            file_type=FileType.CONFIG,
            language="json",
            content_type="package_json",
            metadata={"template": "package_json", "name": f"{project.name.lower()}-frontend",
                      "scripts": {"dev": "vite", "build": "vite build"},
                      "dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"},
                      "devDependencies": {"vite": "^5.0.0"}},
        ))
        module.add_file(FileIR(
            path="frontend/src/App.jsx",
            file_type=FileType.SOURCE,
            language="javascript",
            content_type="component",
            imports=["./App.css"],
            declarations=["App"],
            metadata={"template": "react_component", "class_name": "App"},
        ))
        module.add_file(FileIR(
            path="frontend/index.html",
            file_type=FileType.RESOURCE,
            language="html",
            content_type="html",
            metadata={"template": "html_index", "title": project.name, "scripts": ["src/main.jsx"]},
        ))
        project.add_module(module)
        return project

    def _build_generic_cli_project(self, project: ProjectIR, request: str, language: str) -> ProjectIR:
        main_ext = {"python": ".py", "javascript": ".js", "go": ".go", "rust": ".rs",
                     "kotlin": ".kt", "java": ".java", "c": ".c", "cpp": ".cpp"}
        ext = main_ext.get(language, ".py")

        module = ModuleIR(name="src", path="src")
        module.add_file(FileIR(
            path=f"src/__init__{ext if language == 'python' else ''}",
            file_type=FileType.SOURCE,
            language=language,
            content_type="init",
            metadata={"template": f"{language}_init"},
        ))
        main_file_path = f"src/main{ext}"
        module.add_file(FileIR(
            path=main_file_path,
            file_type=FileType.SOURCE,
            language=language,
            content_type="entry_point",
            imports=[],
            declarations=["main"],
            metadata={"template": f"{language}_main", "functions": ["main"]},
        ))
        project.add_module(module)
        return project

    def _extract_project_name(self, request: str) -> str:
        m = re.search(r'["\']?(\w+)["\']?\s*(앱|app|프로젝트|project)', request)
        if m:
            return m.group(1).capitalize()
        m = re.search(r'(\w+)\s*(앱|app)', request)
        if m:
            return m.group(1).capitalize()
        name_keywords = {
            "카메라": "CameraApp", "calculator": "CalculatorApp",
            "메모": "MemoApp", "todo": "TodoApp", "채팅": "ChatApp",
            "뉴스": "NewsApp", "날씨": "WeatherApp", "음악": "MusicApp",
        }
        for kw, name in name_keywords.items():
            if kw in request.lower():
                return name
        return "MyApp"

    def _extract_main_class(self, request: str) -> Optional[str]:
        m = re.search(r'(\w+)\s*(액티비티|Activity)', request)
        if m:
            return m.group(1) + "Activity"
        return None

    def plan_to_dict(self, project: ProjectIR) -> dict:
        return project.to_dict()
