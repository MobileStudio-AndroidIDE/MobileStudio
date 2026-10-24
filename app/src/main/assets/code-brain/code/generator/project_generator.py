"""Project Generator - generates actual file contents from Project IR."""
from __future__ import annotations
import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from code.ir.project_ir import (
    ProjectIR, ModuleIR, FileIR, FileType, Platform,
)
from code.ir.ir_types import IRNode, IRNodeType, create_class_ir, create_function_ir, create_return_ir
from languages.python.python_gen import PythonGenerator
from languages.kotlin.kotlin_gen import KotlinGenerator
from languages.java.java_gen import JavaGenerator
from languages.c.c_gen import CGenerator
from languages.cpp.cpp_gen import CppGenerator
from languages.javascript.js_gen import JavaScriptGenerator
from languages.go.go_gen import GoGenerator
from languages.rust.rust_gen import RustGenerator


@dataclass
class GeneratedFile:
    path: str
    content: str
    file_type: str = "source"
    language: str = ""
    # The requirement ids this file implements — the SAME ids the plan
    # recorded — so the generated code (and the tests run against it)
    # can be read back to the requirement they satisfy.
    requirement_ids: list = field(default_factory=list)


@dataclass
class ProjectGenerationResult:
    project_ir: ProjectIR
    files: list[GeneratedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # per-file python symbols (functions / classes / imports) for the
    # cross-file dependency check and downstream tooling
    file_symbols: dict = field(default_factory=dict)
    # non-fatal findings: syntax that survived repair, unresolved
    # project-local imports
    warnings: list[str] = field(default_factory=list)

    def get_file(self, path: str) -> Optional[GeneratedFile]:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def get_files_by_type(self, file_type: str) -> list[GeneratedFile]:
        return [f for f in self.files if f.file_type == file_type]


class ProjectGenerator:
    def __init__(self):
        self._generators = {
            "python": PythonGenerator(),
            "kotlin": KotlinGenerator(),
            "java": JavaGenerator(),
            "c": CGenerator(),
            "cpp": CppGenerator(),
            "javascript": JavaScriptGenerator(),
            "go": GoGenerator(),
            "rust": RustGenerator(),
        }

    def generate_project(self, project_ir: ProjectIR) -> ProjectGenerationResult:
        result = ProjectGenerationResult(project_ir=project_ir)
        for module in project_ir.modules:
            for file_ir in module.files:
                try:
                    content = self.generate_file(file_ir, module, project_ir)
                    if file_ir.language == "python":
                        content, issue = self._validate_python_file(
                            content, file_ir, module, project_ir)
                        if issue:
                            result.warnings.append(
                                f"{file_ir.path}: {issue}")
                        result.file_symbols[file_ir.path] = \
                            self._python_symbols(content)
                    result.files.append(GeneratedFile(
                        path=file_ir.path,
                        content=content,
                        file_type=file_ir.file_type.value,
                        language=file_ir.language,
                        requirement_ids=list(
                            (file_ir.metadata or {}).get("requirement_ids")
                            or []),
                    ))
                except Exception as e:
                    result.errors.append(f"Error generating {file_ir.path}: {e}")
        self._check_local_imports(result, project_ir)
        return result

    @staticmethod
    def _python_symbols(content: str) -> dict:
        """The file's own symbol table: top-level functions, classes
        with their methods, and the imports — extracted structurally
        from the AST, never by pattern-matching the text."""
        try:
            tree = ast.parse(content or "")
        except SyntaxError:
            return {}
        symbols = {"functions": [], "classes": {}, "imports": []}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                members = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef,
                                             ast.AsyncFunctionDef))]
                symbols["classes"][node.name] = members
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbols["imports"].append(
                        {"module": alias.name, "name": ""})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    symbols["imports"].append(
                        {"module": module, "name": alias.name})
        return symbols

    def _validate_python_file(self, content: str, file_ir: FileIR,
                              module: ModuleIR,
                              project_ir: ProjectIR) -> tuple:
        """The per-file verification and its BOUNDED repair: a syntax
        failure triggers one regeneration with sanitized identifiers
        (the common template failure), then it is reported as it
        stands — never silently accepted, never looped forever."""
        try:
            ast.parse(content or "")
            return content, ""
        except SyntaxError as exc:
            issue = (f"syntax error: {exc.msg}"
                     + (f" (line {exc.lineno})" if exc.lineno else ""))
        original_meta = dict(file_ir.metadata or {})
        repaired_meta = dict(original_meta)
        changed = False

        def sanitize(name: str) -> str:
            cleaned = re.sub(r"\W+", "_", name).strip("_")
            if cleaned and cleaned[0].isdigit():
                cleaned = f"R_{cleaned}"
            return cleaned

        for key in ("class_name", "function_name", "module_name",
                    "name", "package_name", "test_class"):
            value = repaired_meta.get(key)
            if isinstance(value, str) and value and \
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
                cleaned = sanitize(value)
                if cleaned:
                    repaired_meta[key] = cleaned
                    changed = True
        for key in ("functions", "test_functions"):
            values = repaired_meta.get(key)
            if isinstance(values, list):
                cleaned_values = []
                value_changed = False
                for value in values:
                    if isinstance(value, str) and \
                            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",
                                         value) is None:
                        cleaned = sanitize(value)
                        if cleaned:
                            value_changed = True
                            cleaned_values.append(cleaned)
                            continue
                    cleaned_values.append(value)
                if value_changed:
                    repaired_meta[key] = cleaned_values
                    changed = True
        fields = repaired_meta.get("fields")
        if isinstance(fields, list):
            cleaned_fields = []
            fields_changed = False
            for field in fields:
                if isinstance(field, dict) and \
                        isinstance(field.get("name"), str) and \
                        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",
                                     field["name"]) is None:
                    cleaned = sanitize(field["name"])
                    if cleaned:
                        rebuilt = dict(field)
                        rebuilt["name"] = cleaned
                        cleaned_fields.append(rebuilt)
                        fields_changed = True
                        continue
                cleaned_fields.append(field)
            if fields_changed:
                repaired_meta["fields"] = cleaned_fields
                changed = True
        if not changed:
            return content, issue
        file_ir.metadata = repaired_meta
        try:
            repaired = self.generate_file(file_ir, module, project_ir)
        finally:
            file_ir.metadata = original_meta
        try:
            ast.parse(repaired or "")
            return repaired, f"{issue}; repaired by identifier sanitization"
        except SyntaxError as exc2:
            return repaired, f"{issue}; repair failed: {exc2.msg}"

    def _check_local_imports(self, result: ProjectGenerationResult,
                             project_ir: ProjectIR) -> None:
        """The cross-file dependency check: an import of a
        project-local module must resolve to a symbol that module
        actually defines.  Stdlib / third-party imports are left
        alone — their validity is not knowable from the IR."""
        local_symbols = dict(result.file_symbols)
        module_files: dict[str, str] = {}
        for path in local_symbols:
            if path.endswith(".py"):
                stem = path[:-3]
                module_files[stem.replace("/", ".")] = path
                if "." in stem:
                    module_files[stem.rsplit(".", 1)[-1]] = path
        for path, symbols in local_symbols.items():
            for entry in (symbols.get("imports") or []):
                module_name = str(entry.get("module") or "")
                name = str(entry.get("name") or "")
                target = module_files.get(module_name)
                if target is None or target == path:
                    continue
                target_symbols = local_symbols.get(target) or {}
                defined = set(target_symbols.get("functions") or []) | \
                    set((target_symbols.get("classes") or {}).keys())
                if name and name != "*" and name not in defined:
                    result.warnings.append(
                        f"{path}: unresolved local import "
                        f"{module_name}.{name}")

    def generate_file(self, file_ir: FileIR, module: ModuleIR,
                      project_ir: ProjectIR) -> str:
        ct = file_ir.content_type
        if ct == "settings_gradle":
            return self._gen_settings_gradle(file_ir, project_ir)
        elif ct == "root_gradle":
            return self._gen_root_gradle(file_ir, project_ir)
        elif ct == "gradle":
            return self._gen_app_gradle(file_ir, project_ir)
        elif ct == "manifest":
            return self._gen_manifest(file_ir, project_ir)
        elif ct == "layout":
            return self._gen_layout(file_ir)
        elif ct == "package_json":
            return self._gen_package_json(file_ir)
        elif ct == "html":
            return self._gen_html(file_ir)
        elif ct == "gradle_properties":
            return self._gen_gradle_properties(file_ir)
        elif ct == "gradle_wrapper":
            return self._gen_gradle_wrapper(file_ir)
        elif ct == "gradle_wrapper_jar":
            return self._gen_gradle_wrapper_jar(file_ir)
        elif ct == "gradlew":
            return self._gen_gradlew(file_ir)
        elif ct == "gradlew_bat":
            return self._gen_gradlew_bat(file_ir)
        elif ct == "readme":
            return self._gen_readme(file_ir, project_ir)
        elif ct == "init":
            return self._gen_init(file_ir)
        elif ct in ("data_pipeline", "data_cli", "data_tests"):
            # Requirements-driven data applications (CSV -> JSON etc.):
            # the renderer reads the structured spec from the project IR.
            from brain.data_app_generator import render_data_app_file
            return render_data_app_file(ct, file_ir, project_ir)
        elif ct in ("source", "entry_point", "module", "component", "activity", "class", "data_class", "test"):
            return self._gen_source(file_ir, module)
        elif file_ir.language == "json":
            return self._gen_json_config(file_ir)
        elif file_ir.language == "xml" and file_ir.file_type == FileType.RESOURCE:
            return self._gen_xml_resource(file_ir)
        return self._gen_source(file_ir, module)

    def _gen_settings_gradle(self, file_ir: FileIR, project_ir: ProjectIR) -> str:
        meta = file_ir.metadata
        project_name = meta.get("project_name", project_ir.name)
        modules = meta.get("modules", [])
        lines = [
            "pluginManagement {",
            "    repositories {",
            "        google()",
            "        mavenCentral()",
            "        gradlePluginPortal()",
            "    }",
            "}",
            "dependencyResolutionManagement {",
            "    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)",
            "    repositories {",
            "        google()",
            "        mavenCentral()",
            "    }",
            "}",
            "",
            f'rootProject.name = "{project_name}"',
        ]
        for mod in modules:
            lines.append(f"include({mod!r})")
        return "\n".join(lines) + "\n"

    def _gen_root_gradle(self, file_ir: FileIR, project_ir: ProjectIR) -> str:
        meta = file_ir.metadata
        plugins_list = meta.get("plugins_list", [])
        lines = ["plugins {"]
        for p in plugins_list:
            lines.append(f"    {p}")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _gen_app_gradle(self, file_ir: FileIR, project_ir: ProjectIR) -> str:
        meta = file_ir.metadata
        min_sdk = meta.get("min_sdk", "24")
        target_sdk = meta.get("target_sdk", "34")
        deps = meta.get("dependencies_list", [])
        pkg = meta.get("package_name", f"com.example.{project_ir.name.lower()}")
        lines = [
            "plugins {",
            '    id("com.android.application")',
            '    id("org.jetbrains.kotlin.android")',
            "}",
            "",
            "android {",
            f'    namespace = "{pkg}"',
            f"    compileSdk = {target_sdk}",
            "",
            "    defaultConfig {",
            f'        applicationId = "{pkg}"',
            f"        minSdk = {min_sdk}",
            f"        targetSdk = {target_sdk}",
            '        versionCode = 1',
            '        versionName = "1.0"',
            "    }",
            "",
            "    buildTypes {",
            "        release {",
            "            isMinifyEnabled = false",
            "        }",
            "    }",
            "    compileOptions {",
            "        sourceCompatibility = JavaVersion.VERSION_1_8",
            "        targetCompatibility = JavaVersion.VERSION_1_8",
            "    }",
            "    kotlinOptions {",
            "        jvmTarget = '1.8'",
            "    }",
            "}",
            "",
            "dependencies {",
        ]
        for d in deps:
            lines.append(f"    implementation({d!r})")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _gen_manifest(self, file_ir: FileIR, project_ir: ProjectIR) -> str:
        meta = file_ir.metadata
        pkg = meta.get("package_name", f"com.example.{project_ir.name.lower()}")
        min_sdk = meta.get("min_sdk", "24")
        target_sdk = meta.get("target_sdk", "34")
        activities = meta.get("activities", ["MainActivity"])
        permissions = meta.get("permissions_list", file_ir.permissions)
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<manifest xmlns:android="http://schemas.android.com/apk/res/android">',
        ]
        for perm in permissions:
            lines.append(f'    <uses-permission android:name="{perm}" />')
        lines.append("")
        lines.append("    <application")
        lines.append('        android:allowBackup="true"')
        lines.append('        android:label="@string/app_name"')
        lines.append('        android:supportsRtl="true"')
        lines.append('        android:theme="@style/Theme.MaterialComponents.Light.DarkActionBar">')
        for act in activities:
            lines.append(f"        <activity android:name=\".{act}\"")
            lines.append('            android:exported="true">')
            lines.append("            <intent-filter>")
            lines.append('                <action android:name="android.intent.action.MAIN" />')
            lines.append('                <category android:name="android.intent.category.LAUNCHER" />')
            lines.append("            </intent-filter>")
            lines.append("        </activity>")
        lines.append("    </application>")
        lines.append("</manifest>")
        return "\n".join(lines) + "\n"

    def _gen_layout(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        layout_type = meta.get("layout_type", "LinearLayout")
        components = meta.get("components", [])
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            f'<{layout_type}',
            '    xmlns:android="http://schemas.android.com/apk/res/android"',
            '    xmlns:app="http://schemas.android.com/apk/res-auto"',
            '    android:layout_width="match_parent"',
            '    android:layout_height="match_parent"',
            '    android:orientation="vertical">',
            "",
        ]
        for comp in components:
            comp_type = comp.get("type", "TextView")
            name = comp.get("name", "view")
            text = comp.get("text", "")
            raw = self._resolve_component_tag(comp_type, name)
            lines.append(f'    <{raw}')
            lines.append(f'        android:id="@+id/{name}"')
            lines.append(f'        android:layout_width="{comp.get("width", "wrap_content")}"')
            lines.append(f'        android:layout_height="{comp.get("height", "wrap_content")}"')
            if text:
                lines.append(f'        android:text="{text}" />')
            else:
                lines.append(f'        />')
        lines.append(f"</{layout_type}>")
        return "\n".join(lines) + "\n"

    def _resolve_component_tag(self, comp_type: str, name: str) -> str:
        if comp_type == "PreviewView":
            return "androidx.camera.view.PreviewView"
        if comp_type == "ImageView":
            return "ImageView"
        if comp_type == "Button":
            return "Button"
        if comp_type == "EditText":
            return "EditText"
        if comp_type == "TextView":
            return "TextView"
        return comp_type

    def _gen_package_json(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        pkg = {
            "name": meta.get("name", "my-app"),
            "version": "1.0.0",
            "scripts": meta.get("scripts", {}),
            "dependencies": meta.get("dependencies", {}),
            "devDependencies": meta.get("devDependencies", {}),
        }
        return json.dumps(pkg, indent=2) + "\n"

    def _gen_html(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        title = meta.get("title", "App")
        scripts = meta.get("scripts", ["src/main.jsx"])
        lines = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '    <meta charset="UTF-8" />',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
            f"    <title>{title}</title>",
            "</head>",
            "<body>",
            '    <div id="root"></div>',
        ]
        for s in scripts:
            lines.append(f'    <script type="module" src="{s}"></script>')
        lines.append("</body>")
        lines.append("</html>")
        return "\n".join(lines) + "\n"

    def _gen_gradle_properties(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        props = meta.get("properties", {})
        lines = []
        for k, v in props.items():
            lines.append(f"{k}={v}")
        return "\n".join(lines) + "\n"

    def _gen_gradle_wrapper(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        distribution_url = meta.get("distribution_url",
            "https\\://services.gradle.org/distributions/gradle-8.2-bin.zip")
        lines = [
            "distributionBase=GRADLE_USER_HOME",
            "distributionPath=wrapper/dists",
            f"distributionUrl={distribution_url}",
            "networkTimeout=10000",
            "validateDistributionUrl=true",
            "zipStoreBase=GRADLE_USER_HOME",
            "zipStorePath=wrapper/dists",
        ]
        return "\n".join(lines) + "\n"

    def _gen_gradle_wrapper_jar(self, file_ir: FileIR) -> str:
        return (
            "# gradle-wrapper.jar is a binary file and cannot be generated as text.\n"
            "# Run `gradle wrapper` in the project root to generate it, or\n"
            "# download it from the Gradle distribution.\n"
        )

    def _gen_gradlew(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        return (
            "#!/bin/sh\n"
            "# Gradle wrapper script (generated by RENA-BR).\n"
            '# APP_ARGS and DEFAULT_JVM_OPTS are standard Gradle wrapper variables.\n'
            "DEFAULT_JVM_OPTS='\"-Xmx64m\" \"-Xms64m\"'\n"
            "APP_NAME=\"Gradle\"\n"
            "exec gradle \"$@\"\n"
        )

    def _gen_gradlew_bat(self, file_ir: FileIR) -> str:
        return (
            "@rem Gradle wrapper (generated by RENA-BR).\n"
            "@if \"%DEBUG%\"==\"\" @echo off\n"
            "@gradle %*\n"
        )

    def _gen_readme(self, file_ir: FileIR, project_ir: ProjectIR) -> str:
        meta = file_ir.metadata
        name = meta.get("project_name", project_ir.name)
        desc = meta.get("description", f"{name} project")
        features = meta.get("features", [])
        build_cmd = meta.get("build_instructions", "")
        lines = [
            f"# {name}",
            "",
            desc,
            "",
        ]
        if features:
            lines.append("## Features")
            lines.append("")
            for f in features:
                lines.append(f"- {f}")
            lines.append("")
        if build_cmd:
            lines.append("## Build")
            lines.append("")
            lines.append(f"```bash\n{build_cmd}\n```")
            lines.append("")
        lines.append("## Structure")
        lines.append("")
        lines.append("```")
        for m in project_ir.modules:
            lines.append(f"{m.path}/")
            for f in m.files:
                lines.append(f"  {f.path.split('/')[-1]}")
        lines.append("```")
        return "\n".join(lines) + "\n"

    def _gen_init(self, file_ir: FileIR) -> str:
        if file_ir.language == "python":
            return ""
        return ""

    def _gen_json_config(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        if meta:
            return json.dumps(meta, indent=2) + "\n"
        return "{}\n"

    def _gen_xml_resource(self, file_ir: FileIR) -> str:
        meta = file_ir.metadata
        if meta.get("template") == "strings_xml":
            lines = [
                '<?xml version="1.0" encoding="utf-8"?>',
                "<resources>",
                f'    <string name="app_name">{meta.get("app_name", "App")}</string>',
                "</resources>",
            ]
            return "\n".join(lines) + "\n"
        return '<?xml version="1.0" encoding="utf-8"?>\n<resources />\n'

    def _gen_source(self, file_ir: FileIR, module: ModuleIR) -> str:
        lang = file_ir.language
        meta = file_ir.metadata
        template = meta.get("template", "")

        if lang == "kotlin":
            return self._gen_kotlin_source(file_ir, meta)
        elif lang == "java":
            return self._gen_java_source(file_ir, meta)
        elif lang == "python":
            return self._gen_python_source(file_ir, meta)
        elif lang == "javascript":
            return self._gen_javascript_source(file_ir, meta)
        elif lang == "go":
            return self._gen_go_source(file_ir, meta)
        elif lang == "rust":
            return self._gen_rust_source(file_ir, meta)
        elif lang == "c":
            return self._gen_c_source(file_ir, meta)
        elif lang == "cpp":
            return self._gen_cpp_source(file_ir, meta)
        return f"// Generated file: {file_ir.path}\n"

    def _package_from_path(self, path: str) -> str:
        parts = path.split("/")
        for marker in ("java", "kotlin", "src"):
            if marker in parts:
                idx = parts.index(marker)
                pkg_parts = parts[idx + 1:-1]
                return ".".join(pkg_parts)
        return ""

    def _nullable_type(self, ptype: str) -> str:
        if not ptype:
            return "Any?"
        if ptype.endswith("?"):
            return ptype
        return f"{ptype}?"

    def _gen_kotlin_source(self, file_ir: FileIR, meta: dict) -> str:
        template = meta.get("template", "")
        class_name = meta.get("class_name", "")
        pkg_str = meta.get("package_name", "") or self._package_from_path(file_ir.path)

        imports = file_ir.imports or meta.get("imports", [])
        lines = []
        if pkg_str:
            lines.append(f"package {pkg_str}")
            lines.append("")
        if imports:
            for imp in imports:
                lines.append(f"import {imp}")
            lines.append("")

        if template == "kotlin_activity" or file_ir.content_type == "activity":
            lines.extend(self._gen_kotlin_activity(class_name, meta))
        elif meta.get("camera_controller"):
            lines.extend(self._gen_camera_controller(class_name, meta))
        elif template == "kotlin_data_class" or file_ir.content_type == "data_class":
            lines.extend(self._gen_kotlin_data_class(class_name, meta))
        elif template == "kotlin_service" or (file_ir.content_type == "class" and "Service" in class_name):
            lines.extend(self._gen_kotlin_service(class_name, meta))
        elif template == "kotlin_main" or file_ir.content_type == "entry_point":
            lines.extend(self._gen_kotlin_main(class_name, meta))
        elif template == "kotlin_class" or file_ir.content_type == "class":
            lines.extend(self._gen_kotlin_class(class_name, meta))
        else:
            if class_name:
                lines.append(f"class {class_name} {{")
                lines.append("}")
            else:
                lines.append("// Generated Kotlin file")
        return "\n".join(lines) + "\n"

    def _gen_kotlin_data_class(self, class_name: str, meta: dict) -> list[str]:
        fields = meta.get("fields", [])
        methods = meta.get("methods", [])
        
        params = []
        for f in fields:
            fname = f.get("name", "field")
            ftype = f.get("type", "String")
            if ftype == "int":
                params.append(f"val {fname}: Int = 0")
            elif ftype == "boolean":
                params.append(f"val {fname}: Boolean = false")
            else:
                params.append(f"val {fname}: String = \"\"")
        
        params_str = ", ".join(params)
        lines = [f"data class {class_name}({params_str})"]
        
        if methods:
            lines.append("")
            for method in methods:
                if method == "to_dict":
                    lines.append(f"    fun toMap(): Map<String, Any> {{")
                    lines.append(f"        return mapOf(")
                    for f in fields:
                        fname = f.get("name", "field")
                        lines.append(f"            \"{fname}\" to {fname},")
                    lines.append(f"        )")
                    lines.append(f"    }}")
                elif method == "search":
                    lines.append(f"    fun matches(query: String): Boolean {{")
                    lines.append(f"        val queryLower = query.lowercase()")
                    for f in fields:
                        if f.get("type") == "String":
                            fname = f.get("name", "field")
                            lines.append(f"        if (queryLower in {fname}.lowercase()) return true")
                    lines.append(f"        return false")
                    lines.append(f"    }}")
        
        return lines

    def _gen_kotlin_service(self, class_name: str, meta: dict) -> list[str]:
        data_models = meta.get("data_models", [])
        features = meta.get("features", [])
        
        lines = [
            f"class {class_name} {{",
            f"    private val items = mutableListOf<Any>()",
            f"    private val storageFile = \"data.json\"",
            "",
            f"    init {{",
            f"        load()",
            f"    }}",
            "",
            f"    private fun load() {{",
            f"        // Load from file",
            f"    }}",
            "",
            f"    private fun save() {{",
            f"        // Save to file",
            f"    }}",
        ]
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"    fun add(item: Any): Boolean {{",
                    f"        items.add(item)",
                    f"        save()",
                    f"        return true",
                    f"    }}",
                ])
            elif feature == "remove":
                lines.extend([
                    f"    fun remove(identifier: Any): Boolean {{",
                    f"        val iterator = items.iterator()",
                    f"        while (iterator.hasNext()) {{",
                    f"            val item = iterator.next()",
                    f"            if (getId(item) == identifier) {{",
                    f"                iterator.remove()",
                    f"                save()",
                    f"                return true",
                    f"            }}",
                    f"        }}",
                    f"        return false",
                    f"    }}",
                ])
            elif feature == "update":
                lines.extend([
                    f"    fun update(identifier: Any, updates: Map<String, Any>): Boolean {{",
                    f"        for (item in items) {{",
                    f"            if (getId(item) == identifier) {{",
                    f"                // Apply updates",
                    f"                save()",
                    f"                return true",
                    f"            }}",
                    f"        }}",
                    f"        return false",
                    f"    }}",
                ])
            elif feature == "search":
                lines.extend([
                    f"    fun search(query: String): List<Any> {{",
                    f"        return items.filter {{ item ->",
                    f"            item.toString().contains(query, ignoreCase = true)",
                    f"        }}",
                    f"    }}",
                ])
            elif feature == "list":
                lines.extend([
                    f"    fun listAll(): List<Any> {{",
                    f"        return items.toList()",
                    f"    }}",
                ])
        
        lines.extend([
            f"    private fun getId(item: Any): Any {{",
            f"        return item.hashCode()",
            f"    }}",
            f"    fun count(): Int {{",
            f"        return items.size",
            f"    }}",
            f"}}",
        ])
        
        return lines

    def _gen_kotlin_main(self, class_name: str, meta: dict) -> list[str]:
        service_class = meta.get("service_class", "Service")
        features = meta.get("features", [])
        
        lines = [
            f"fun main() {{",
            f"    val service = {service_class}()",
            f"    println(\"Welcome to {service_class.replace('Service', '')} Manager!\")",
            f"    println(\"Available commands:\")",
        ]
        
        cmd_map = {
            "add": "add - Add a new item",
            "remove": "remove - Remove an item",
            "update": "update - Update an item",
            "search": "search - Search items",
            "list": "list - List all items",
        }
        
        for feature in features:
            if feature in cmd_map:
                lines.append(f"    println(\"  {cmd_map[feature]}\")")
        
        lines.extend([
            f"    println(\"  quit - Exit the program\")",
            f"    println()",
            f"",
            f"    while (true) {{",
            f"        print(\"Enter command: \")",
            f"        val command = readLine()?.trim()?.lowercase() ?: continue",
            f"        when (command) {{",
            f"            \"quit\" -> {{",
            f"                println(\"Goodbye!\")",
            f"                break",
            f"            }}",
        ])
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"            \"add\" -> {{",
                    f"                print(\"Enter name: \")",
                    f"                val name = readLine()?.trim() ?: continue",
                    f"                if (name.isNotEmpty()) {{",
                    f"                    service.add(mapOf(\"name\" to name))",
                    f"                    println(\"Added: $name\")",
                    f"                }}",
                    f"            }}",
                ])
            elif feature == "list":
                lines.extend([
                    f"            \"list\" -> {{",
                    f"                val items = service.listAll()",
                    f"                if (items.isNotEmpty()) {{",
                    f"                    items.forEach {{ println(\"  - $it\") }}",
                    f"                }} else {{",
                    f"                    println(\"No items found\")",
                    f"                }}",
                    f"            }}",
                ])
        
        lines.extend([
            f"            else -> println(\"Unknown command. Type a command from the list above.\")",
            f"        }}",
            f"        println()",
            f"    }}",
            f"}}",
        ])
        
        return lines

    def _gen_kotlin_activity(self, class_name: str, meta: dict) -> list[str]:
        parent = meta.get("parent_class", "ComponentActivity")
        functions = meta.get("functions", ["onCreate"])
        controller = meta.get("controller", "")
        lines = [f"class {class_name} : {parent}() {{"]
        for fn in functions:
            if fn == "onCreate":
                lines.append("    override fun onCreate(savedInstanceState: Bundle?) {")
                lines.append("        super.onCreate(savedInstanceState)")
                if controller:
                    lines.append(f"        val controller = {controller}(this)")
                    lines.append("        controller.bind(previewView)")
                    lines.append("        controller.startCamera()")
                else:
                    lines.append("        setContentView(R.layout.activity_main)")
                lines.append("    }")
            else:
                lines.append(f"    fun {fn}() {{")
                lines.append("        // TODO: Implement")
                lines.append("    }")
        lines.append("}")
        return lines

    def _gen_kotlin_class(self, class_name: str, meta: dict) -> list[str]:
        props = meta.get("properties", [])
        ctor_params = meta.get("constructor_params", [])
        functions = meta.get("functions", [])
        ctor_str = ""
        if ctor_params:
            params_str = ", ".join(f"val {n}: {t}" for n, t in ctor_params)
            ctor_str = f"({params_str})"
        lines = [f"class {class_name}{ctor_str} {{"]
        for pname, ptype in props:
            if not any(pname == cn for cn, _ in ctor_params):
                lines.append(f"    var {pname}: {self._nullable_type(ptype)} = null")
        for fn in functions:
            lines.append(f"    fun {fn}() {{")
            lines.append("        // TODO: Implement")
            lines.append("    }")
        lines.append("}")
        return lines

    def _gen_camera_controller(self, class_name: str, meta: dict) -> list[str]:
        props = meta.get("properties", [])
        ctor_params = meta.get("constructor_params", [])
        functions = meta.get("functions", [])
        ctor_str = ""
        if ctor_params:
            params_str = ", ".join(f"val {n}: {t}" for n, t in ctor_params)
            ctor_str = f"({params_str})"
        lines = [f"class {class_name}{ctor_str} {{"]
        lines.append("    private var camera: Camera? = null")
        lines.append("    private var imageCapture: ImageCapture? = null")
        lines.append("")
        lines.append("    fun startCamera() {")
        lines.append("        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)")
        lines.append("        cameraProviderFuture.addListener({")
        lines.append("            val cameraProvider = cameraProviderFuture.get()")
        lines.append("            val preview = Preview.Builder().build()")
        lines.append("            preview.setSurfaceProvider(previewView.surfaceProvider)")
        lines.append("            imageCapture = ImageCapture.Builder().build()")
        lines.append("            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA")
        lines.append("            cameraProvider.unbindAll()")
        lines.append("            camera = cameraProvider.bindToLifecycle(")
        lines.append("                context as ComponentActivity, cameraSelector, preview, imageCapture")
        lines.append("            )")
        lines.append("        }, ContextCompat.getMainExecutor(context))")
        lines.append("    }")
        lines.append("")
        lines.append("    fun takePhoto() {")
        lines.append("        imageCapture?.takePicture(")
        lines.append("            ContextCompat.getMainExecutor(context),")
        lines.append("            object : ImageCapture.OnImageSavedCallback {")
        lines.append("                override fun onImageSaved(output: ImageCapture.OutputFileResults) {")
        lines.append("                    Log.d(\"CameraController\", \"Photo saved\")")
        lines.append("                }")
        lines.append("                override fun onError(exc: ImageCaptureException) {")
        lines.append("                    Log.e(\"CameraController\", \"Photo capture failed\", exc)")
        lines.append("                }")
        lines.append("            }")
        lines.append("        )")
        lines.append("    }")
        lines.append("")
        lines.append("    fun stopCamera() {")
        lines.append("        camera?.cameraControl?.enableTorch(false)")
        lines.append("        camera = null")
        lines.append("    }")
        lines.append("}")
        return lines

    def _gen_java_source(self, file_ir: FileIR, meta: dict) -> str:
        class_name = meta.get("class_name", "Main")
        imports = file_ir.imports or meta.get("imports", [])
        template = meta.get("template", "")
        
        lines = []
        if imports:
            for imp in imports:
                lines.append(f"import {imp};")
            lines.append("")
        
        if template == "java_data_class" or file_ir.content_type == "data_class":
            lines.extend(self._gen_java_data_class(class_name, meta))
        elif template == "java_service" or (file_ir.content_type == "class" and "Service" in class_name):
            lines.extend(self._gen_java_service(class_name, meta))
        elif template == "java_main" or file_ir.content_type == "entry_point":
            lines.extend(self._gen_java_main(class_name, meta))
        else:
            lines.append(f"public class {class_name} {{")
            props = meta.get("properties", [])
            for pname, ptype in props:
                lines.append(f"    private {ptype} {pname};")
            ctor_params = meta.get("constructor_params", [])
            if ctor_params:
                params_str = ", ".join(f"{t} {n}" for n, t in ctor_params)
                lines.append(f"    public {class_name}({params_str}) {{")
                for pname, _ in ctor_params:
                    lines.append(f"        this.{pname} = {pname};")
                lines.append("    }")
            functions = meta.get("functions", [])
            for fn in functions:
                lines.append(f"    public void {fn}() {{")
                lines.append("        // TODO: Implement")
                lines.append("    }")
            lines.append("}")
        
        return "\n".join(lines) + "\n"

    def _gen_java_data_class(self, class_name: str, meta: dict) -> list[str]:
        fields = meta.get("fields", [])
        methods = meta.get("methods", [])
        
        lines = [f"public class {class_name} {{"]
        
        for f in fields:
            fname = f.get("name", "field")
            ftype = f.get("type", "String")
            if ftype == "int":
                lines.append(f"    private int {fname};")
            elif ftype == "boolean":
                lines.append(f"    private boolean {fname};")
            else:
                lines.append(f"    private String {fname};")
        
        lines.append("")
        
        params = []
        for f in fields:
            fname = f.get("name", "field")
            ftype = f.get("type", "String")
            if ftype == "int":
                params.append(f"int {fname}")
            elif ftype == "boolean":
                params.append(f"boolean {fname}")
            else:
                params.append(f"String {fname}")
        
        if params:
            params_str = ", ".join(params)
            lines.append(f"    public {class_name}({params_str}) {{")
            for f in fields:
                fname = f.get("name", "field")
                lines.append(f"        this.{fname} = {fname};")
            lines.append(f"    }}")
            lines.append("")
        
        for f in fields:
            fname = f.get("name", "field")
            ftype = f.get("type", "String")
            cap_name = fname.capitalize()
            if ftype == "boolean":
                lines.append(f"    public boolean is{cap_name}() {{ return {fname}; }}")
            else:
                if ftype == "int":
                    lines.append(f"    public int get{cap_name}() {{ return {fname}; }}")
                else:
                    lines.append(f"    public String get{cap_name}() {{ return {fname}; }}")
            lines.append(f"    public void set{cap_name}({ftype} {fname}) {{ this.{fname} = {fname}; }}")
            lines.append("")
        
        if "to_dict" in methods:
            lines.extend([
                f"    public Map<String, Object> toMap() {{",
                f"        Map<String, Object> map = new HashMap<>();",
            ])
            for f in fields:
                fname = f.get("name", "field")
                lines.append(f"        map.put(\"{fname}\", {fname});")
            lines.extend([
                f"        return map;",
                f"    }}",
                "",
            ])
        
        lines.append("}")
        return lines

    def _gen_java_service(self, class_name: str, meta: dict) -> list[str]:
        data_models = meta.get("data_models", [])
        features = meta.get("features", [])
        
        lines = [
            f"import java.util.ArrayList;",
            f"import java.util.List;",
            f"import java.util.stream.Collectors;",
            "",
            f"public class {class_name} {{",
            f"    private List<Object> items = new ArrayList<>();",
            f"    private String storageFile = \"data.json\";",
            "",
            f"    public {class_name}() {{",
            f"        load();",
            f"    }}",
            "",
            f"    private void load() {{",
            f"        // Load from file",
            f"    }}",
            "",
            f"    private void save() {{",
            f"        // Save to file",
            f"    }}",
        ]
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"    public boolean add(Object item) {{",
                    f"        items.add(item);",
                    f"        save();",
                    f"        return true;",
                    f"    }}",
                ])
            elif feature == "remove":
                lines.extend([
                    f"    public boolean remove(Object identifier) {{",
                    f"        return items.removeIf(item -> getId(item).equals(identifier));",
                    f"    }}",
                ])
            elif feature == "search":
                lines.extend([
                    f"    public List<Object> search(String query) {{",
                    f"        return items.stream()",
                    f"            .filter(item -> item.toString().toLowerCase().contains(query.toLowerCase()))",
                    f"            .collect(Collectors.toList());",
                    f"    }}",
                ])
            elif feature == "list":
                lines.extend([
                    f"    public List<Object> listAll() {{",
                    f"        return new ArrayList<>(items);",
                    f"    }}",
                ])
        
        lines.extend([
            f"    private Object getId(Object item) {{",
            f"        return item.hashCode();",
            f"    }}",
            f"    public int count() {{",
            f"        return items.size();",
            f"    }}",
            f"}}",
        ])
        
        return lines

    def _gen_java_main(self, class_name: str, meta: dict) -> list[str]:
        service_class = meta.get("service_class", "Service")
        features = meta.get("features", [])
        
        lines = [
            f"import java.util.Scanner;",
            "",
            f"public class {class_name} {{",
            f"    public static void main(String[] args) {{",
            f"        {service_class} service = new {service_class}();",
            f"        Scanner scanner = new Scanner(System.in);",
            f"        System.out.println(\"Welcome to {service_class.replace('Service', '')} Manager!\");",
            f"        System.out.println(\"Available commands:\");",
        ]
        
        cmd_map = {
            "add": "add - Add a new item",
            "remove": "remove - Remove an item",
            "update": "update - Update an item",
            "search": "search - Search items",
            "list": "list - List all items",
        }
        
        for feature in features:
            if feature in cmd_map:
                lines.append(f"        System.out.println(\"  {cmd_map[feature]}\");")
        
        lines.extend([
            f"        System.out.println(\"  quit - Exit the program\");",
            f"        System.out.println();",
            f"",
            f"        while (true) {{",
            f"            System.out.print(\"Enter command: \");",
            f"            String command = scanner.nextLine().trim().toLowerCase();",
            f"            switch (command) {{",
            f"                case \"quit\":",
            f"                    System.out.println(\"Goodbye!\");",
            f"                    return;",
        ])
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"                case \"add\":",
                    f"                    System.out.print(\"Enter name: \");",
                    f"                    String name = scanner.nextLine().trim();",
                    f"                    if (!name.isEmpty()) {{",
                    f"                        service.add(new Object() {{ String n = name; }});",
                    f"                        System.out.println(\"Added: \" + name);",
                    f"                    }}",
                    f"                    break;",
                ])
            elif feature == "list":
                lines.extend([
                    f"                case \"list\":",
                    f"                    var items = service.listAll();",
                    f"                    if (!items.isEmpty()) {{",
                    f"                        items.forEach(item -> System.out.println(\"  - \" + item));",
                    f"                    }} else {{",
                    f"                        System.out.println(\"No items found\");",
                    f"                    }}",
                    f"                    break;",
                ])
        
        lines.extend([
            f"                default:",
            f"                    System.out.println(\"Unknown command. Type a command from the list above.\");",
            f"            }}",
            f"            System.out.println();",
            f"        }}",
            f"    }}",
            f"}}",
        ])
        
        return lines

    def _gen_python_source(self, file_ir: FileIR, meta: dict) -> str:
        template = meta.get("template", "")
        if template == "python_init" or file_ir.content_type == "init":
            return ""
        if template == "python_data_class" or file_ir.content_type == "data_class":
            return self._gen_python_data_class(file_ir, meta)
        if template == "python_pipeline":
            return self._gen_python_pipeline(file_ir, meta)
        if template == "python_service" or (file_ir.content_type == "module" and "Service" in meta.get("class_name", "")):
            return self._gen_python_service(file_ir, meta)
        if template == "python_main_with_service" or file_ir.content_type == "entry_point":
            return self._gen_python_main_with_service(file_ir, meta)
        if template == "python_main" or file_ir.content_type == "entry_point":
            functions = meta.get("functions", ["main"])
            lines = []
            for fn in functions:
                lines.append(f"def {fn}():")
                lines.append('    """Main entry point."""')
                lines.append("    print('Hello from {}')".format(file_ir.path.split("/")[-1].replace(".py", "")))
                lines.append("")
            lines.append('if __name__ == "__main__":')
            lines.append("    main()")
            return "\n".join(lines) + "\n"
        if template == "python_module" or file_ir.content_type == "module":
            functions = meta.get("functions", [])
            lines = []
            for fn in functions:
                lines.append(f"def {fn}():")
                lines.append(f'    """Utility function {fn}."""')
                lines.append("    pass")
                lines.append("")
            return "\n".join(lines) + "\n"
        if template == "python_test" or file_ir.content_type == "test":
            test_class = meta.get("test_class", "TestMain")
            test_funcs = meta.get("test_functions", ["test_main"])
            lines = [
                "import unittest",
                "",
                f"class {test_class}(unittest.TestCase):",
            ]
            for tf in test_funcs:
                lines.append(f"    def {tf}(self):")
                lines.append("        self.assertTrue(True)")
            lines.append("")
            lines.append('if __name__ == "__main__":')
            lines.append("    unittest.main()")
            return "\n".join(lines) + "\n"
        return f"# Generated Python file: {file_ir.path}\n"

    def _gen_python_data_class(self, file_ir: FileIR, meta: dict) -> str:
        class_name = meta.get("class_name", "DataClass")
        fields = meta.get("fields", [])
        methods = meta.get("methods", [])
        
        lines = [
            "from dataclasses import dataclass, field",
            "from typing import List, Optional",
            "",
            "",
            "@dataclass",
            f"class {class_name}:",
        ]
        
        for f in fields:
            fname = f.get("name", "field")
            ftype = f.get("type", "str")
            fdesc = f.get("description", "")
            if fdesc:
                lines.append(f"    # {fdesc}")
            if ftype == "int":
                lines.append(f"    {fname}: int = 0")
            elif ftype == "boolean":
                lines.append(f"    {fname}: bool = False")
            elif ftype == "List[str]":
                lines.append(f"    {fname}: List[str] = field(default_factory=list)")
            else:
                lines.append(f"    {fname}: str = \"\"")
        
        if methods:
            lines.append("")
            for method in methods:
                if method == "add":
                    lines.append(f"    @staticmethod")
                    lines.append(f"    def create(**kwargs) -> \"{class_name}\":")
                    lines.append(f"        return {class_name}(**kwargs)")
                elif method == "update":
                    lines.append(f"    def update(self, **kwargs):")
                    lines.append(f"        for key, value in kwargs.items():")
                    lines.append(f"            if hasattr(self, key):")
                    lines.append(f"                setattr(self, key, value)")
                elif method == "to_dict":
                    lines.append(f"    def to_dict(self) -> dict:")
                    lines.append(f"        return {{")
                    for f in fields:
                        lines.append(f"            \"{f.get('name', 'field')}\": self.{f.get('name', 'field')},")
                    lines.append(f"        }}")
                elif method == "search":
                    lines.append(f"    def matches(self, query: str) -> bool:")
                    lines.append(f"        query_lower = query.lower()")
                    for f in fields:
                        if f.get("type") == "str":
                            lines.append(f"        if query_lower in self.{f.get('name', 'field')}.lower():")
                            lines.append(f"            return True")
                    lines.append(f"        return False")
        
        return "\n".join(lines) + "\n"

    def _gen_python_service(self, file_ir: FileIR, meta: dict) -> str:
        class_name = meta.get("class_name", "Service")
        data_models = meta.get("data_models", [])
        features = meta.get("features", [])
        
        lines = [
            "from typing import List, Dict, Optional, Any",
            "import json",
            "import os",
            "",
        ]
        
        for dm in data_models:
            lines.append(f"from .{dm.lower()} import {dm}")
        
        lines.extend([
            "",
            "",
            f"class {class_name}:",
            f'    """Service for managing {class_name.replace("Service", "").lower()} data."""',
            "",
            "    def __init__(self, storage_file: str = None):",
            "        self.items: List[Any] = []",
            '        self.storage_file = storage_file or "data.json"',
            "        self._load()",
            "",
            "    def _load(self):",
            '        """Load data from file."""',
            "        if os.path.exists(self.storage_file):",
            "            try:",
            "                with open(self.storage_file, 'r') as f:",
            "                    data = json.load(f)",
            "                    self.items = [self._from_dict(item) for item in data]",
            "            except (json.JSONDecodeError, Exception):",
            "                self.items = []",
            "",
            "    def _save(self):",
            '        """Save data to file."""',
            "        with open(self.storage_file, 'w') as f:",
            "            json.dump([self._to_dict(item) for item in self.items], f, indent=2)",
            "",
            "    def _from_dict(self, data: dict) -> Any:",
            '        """Convert dict to data model."""',
            f"        return {data_models[0] if data_models else 'dict'}(**data) if isinstance(data, dict) else data",
            "",
            "    def _to_dict(self, item: Any) -> dict:",
            '        """Convert data model to dict."""',
            "        if hasattr(item, '__dict__'):",
            "            return item.__dict__",
            "        return item",
            "",
        ])
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    "    def add(self, item: Any) -> bool:",
                    '        """Add a new item."""',
                    "        self.items.append(item)",
                    "        self._save()",
                    "        return True",
                    "",
                ])
            elif feature == "remove":
                lines.extend([
                    "    def remove(self, identifier: Any) -> bool:",
                    '        """Remove an item by identifier."""',
                    "        for i, item in enumerate(self.items):",
                    "            if self._get_id(item) == identifier:",
                    "                self.items.pop(i)",
                    "                self._save()",
                    "                return True",
                    "        return False",
                    "",
                ])
            elif feature == "update":
                lines.extend([
                    "    def update(self, identifier: Any, **kwargs) -> bool:",
                    '        """Update an item."""',
                    "        for item in self.items:",
                    "            if self._get_id(item) == identifier:",
                    "                for key, value in kwargs.items():",
                    "                    if hasattr(item, key):",
                    "                        setattr(item, key, value)",
                    "                self._save()",
                    "                return True",
                    "        return False",
                    "",
                ])
            elif feature == "search":
                lines.extend([
                    "    def search(self, query: str) -> List[Any]:",
                    '        """Search items by query."""',
                    "        results = []",
                    "        for item in self.items:",
                    "            if hasattr(item, 'matches') and item.matches(query):",
                    "                results.append(item)",
                    "            elif query.lower() in str(item).lower():",
                    "                results.append(item)",
                    "        return results",
                    "",
                ])
            elif feature == "list":
                lines.extend([
                    "    def list_all(self) -> List[Any]:",
                    '        """List all items."""',
                    "        return self.items.copy()",
                    "",
                ])
            elif feature == "complete":
                lines.extend([
                    "    def complete(self, identifier: Any) -> bool:",
                    '        """Mark item as complete."""',
                    "        for item in self.items:",
                    "            if self._get_id(item) == identifier:",
                    "                if hasattr(item, 'completed'):",
                    "                    item.completed = True",
                    "                    self._save()",
                    "                    return True",
                    "        return False",
                    "",
                ])
            elif feature == "history":
                lines.extend([
                    "    def get_history(self) -> List[Any]:",
                    '        """Get calculation history."""',
                    "        return self.items.copy()",
                    "",
                ])
            elif feature == "scan":
                lines.extend([
                    "    def scan(self, path: str) -> List[str]:",
                    '        """Scan directory for files."""',
                    "        import os",
                    "        files = []",
                    "        if os.path.exists(path):",
                    "            for item in os.listdir(path):",
                    "                files.append(item)",
                    "        return files",
                    "",
                ])
            elif feature == "categorize":
                lines.extend([
                    "    def categorize(self, files: List[str]) -> Dict[str, List[str]]:",
                    '        """Categorize files by type."""',
                    "        categories = {}",
                    "        for f in files:",
                    "            ext = os.path.splitext(f)[1].lower()",
                    "            if ext not in categories:",
                    "                categories[ext] = []",
                    "            categories[ext].append(f)",
                    "        return categories",
                    "",
                ])
            elif feature == "move":
                lines.extend([
                    "    def move(self, src: str, dst: str) -> bool:",
                    '        """Move a file."""',
                    "        import shutil",
                    "        try:",
                    "            shutil.move(src, dst)",
                    "            return True",
                    "        except Exception:",
                    "            return False",
                    "",
                ])
            elif feature == "rename":
                lines.extend([
                    "    def rename(self, old_name: str, new_name: str) -> bool:",
                    '        """Rename a file."""',
                    "        try:",
                    "            os.rename(old_name, new_name)",
                    "            return True",
                    "        except Exception:",
                    "            return False",
                    "",
                ])
            elif feature == "preview":
                lines.extend([
                    "    def preview(self, operations: List[Dict]) -> List[Dict]:",
                    '        """Preview operations before applying."""',
                    "        return operations",
                    "",
                ])
        
        lines.extend([
            "    def _get_id(self, item: Any) -> Any:",
            '        """Get identifier from item."""',
            "        if hasattr(item, 'id'):",
            "            return item.id",
            "        return id(item)",
            "",
            "    def count(self) -> int:",
            '        """Get count of items."""',
            "        return len(self.items)",
            "",
        ])
        
        return "\n".join(lines) + "\n"

    def _gen_python_main_with_service(self, file_ir: FileIR, meta: dict) -> str:
        service_class = meta.get("service_class", "Service")
        features = meta.get("features", [])
        
        service_module = service_class.lower().replace("service", "_service")
        
        lines = [
            f"from .{service_module} import {service_class}",
            "",
            "",
            "def main():",
            f"    service = {service_class}()",
            f"    print(\"Welcome to {service_class.replace('Service', '')} Manager!\")",
            '    print("Available commands:")',
        ]
        
        cmd_map = {
            "add": "add - Add a new item",
            "remove": "remove - Remove an item",
            "update": "update - Update an item",
            "search": "search - Search items",
            "list": "list - List all items",
            "complete": "complete - Mark item as complete",
            "history": "history - View history",
            "scan": "scan - Scan directory",
            "categorize": "categorize - Categorize files",
            "move": "move - Move files",
            "rename": "rename - Rename files",
            "preview": "preview - Preview changes",
        }
        
        for feature in features:
            if feature in cmd_map:
                lines.append(f'    print("  {cmd_map[feature]}")')
        
        lines.extend([
            '    print("  quit - Exit the program")',
            '    print()',
            '',
            '    while True:',
            '        command = input("Enter command: ").strip().lower()',
            '        if command == "quit":',
            '            print("Goodbye!")',
            '            break',
        ])
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    '        elif command == "add":',
                    '            name = input("Enter name: ").strip()',
                    '            if name:',
                    '                item = {"name": name}',
                    '                service.add(item)',
                    '                print(f"Added: {name}")',
                ])
            elif feature == "remove":
                lines.extend([
                    '        elif command == "remove":',
                    '            name = input("Enter name to remove: ").strip()',
                    '            if service.remove(name):',
                    '                print(f"Removed: {name}")',
                    '            else:',
                    '                print("Item not found")',
                ])
            elif feature == "list":
                lines.extend([
                    '        elif command == "list":',
                    '            items = service.list_all()',
                    '            if items:',
                    '                for item in items:',
                    '                    print(f"  - {item}")',
                    '            else:',
                    '                print("No items found")',
                ])
            elif feature == "search":
                lines.extend([
                    '        elif command == "search":',
                    '            query = input("Enter search query: ").strip()',
                    '            results = service.search(query)',
                    '            if results:',
                    '                for item in results:',
                    '                    print(f"  - {item}")',
                    '            else:',
                    '                print("No results found")',
                ])
            elif feature == "scan":
                lines.extend([
                    '        elif command == "scan":',
                    '            path = input("Enter directory path: ").strip()',
                    '            print(f"Scanning {path}...")',
                ])
            elif feature == "categorize":
                lines.extend([
                    '        elif command == "categorize":',
                    '            print("Categorizing files...")',
                ])
            elif feature == "move":
                lines.extend([
                    '        elif command == "move":',
                    '            print("Moving files...")',
                ])
            elif feature == "rename":
                lines.extend([
                    '        elif command == "rename":',
                    '            print("Renaming files...")',
                ])
            elif feature == "preview":
                lines.extend([
                    '        elif command == "preview":',
                    '            print("Previewing changes...")',
                ])
        
        lines.extend([
            '        else:',
            '            print("Unknown command. Type a command from the list above.")',
            '        print()',
            '',
            '',
            'if __name__ == "__main__":',
            '    main()',
        ])
        
        return "\n".join(lines) + "\n"

    def _gen_javascript_source(self, file_ir: FileIR, meta: dict) -> str:
        template = meta.get("template", "")
        class_name = meta.get("class_name", "")
        
        if template == "js_data_class" or file_ir.content_type == "data_class":
            return self._gen_js_data_class(class_name, meta)
        elif template == "js_service" or (file_ir.content_type == "module" and "Service" in class_name):
            return self._gen_js_service(class_name, meta)
        elif template == "js_main" or file_ir.content_type == "entry_point":
            return self._gen_js_main(class_name, meta)
        elif template == "react_component" or file_ir.content_type == "component":
            imports = file_ir.imports or []
            lines = []
            if imports:
                for imp in imports:
                    lines.append(f"import '{imp}';")
                lines.append("")
            lines.append(f"export default function {class_name}() {{")
            lines.append("    return (")
            lines.append("        <div>")
            lines.append(f"            <h1>{class_name}</h1>")
            lines.append("        </div>")
            lines.append("    );")
            lines.append("}")
            return "\n".join(lines) + "\n"
        return f"// Generated JavaScript file: {file_ir.path}\n"

    def _gen_js_data_class(self, class_name: str, meta: dict) -> str:
        fields = meta.get("fields", [])
        methods = meta.get("methods", [])
        
        lines = [
            f"class {class_name} {{",
            f"    constructor({', '.join(f.get('name', 'field') for f in fields)}) {{",
        ]
        
        for f in fields:
            fname = f.get("name", "field")
            lines.append(f"        this.{fname} = {fname};")
        
        lines.extend([
            f"    }}",
            f"",
            f"    toObject() {{",
            f"        return {{",
        ])
        
        for f in fields:
            fname = f.get("name", "field")
            lines.append(f"            {fname}: this.{fname},")
        
        lines.extend([
            f"        }};",
            f"    }}",
            f"",
            f"    static fromObject(obj) {{",
            f"        return new {class_name}(",
        ])
        
        for f in fields:
            fname = f.get("name", "field")
            lines.append(f"            obj.{fname},")
        
        lines.extend([
            f"        );",
            f"    }}",
            f"}}",
            f"",
            f"module.exports = {class_name};",
        ])
        
        return "\n".join(lines) + "\n"

    def _gen_js_service(self, class_name: str, meta: dict) -> str:
        data_models = meta.get("data_models", [])
        features = meta.get("features", [])
        
        lines = [
            f"class {class_name} {{",
            f"    constructor() {{",
            f"        this.items = [];",
            f"        this.storageFile = 'data.json';",
            f"        this.load();",
            f"    }}",
            f"",
            f"    load() {{",
            f"        // Load from file",
            f"    }}",
            f"",
            f"    save() {{",
            f"        // Save to file",
            f"    }}",
        ]
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"    add(item) {{",
                    f"        this.items.push(item);",
                    f"        this.save();",
                    f"        return true;",
                    f"    }}",
                ])
            elif feature == "remove":
                lines.extend([
                    f"    remove(identifier) {{",
                    f"        const index = this.items.findIndex(item => this.getId(item) === identifier);",
                    f"        if (index !== -1) {{",
                    f"            this.items.splice(index, 1);",
                    f"            this.save();",
                    f"            return true;",
                    f"        }}",
                    f"        return false;",
                    f"    }}",
                ])
            elif feature == "search":
                lines.extend([
                    f"    search(query) {{",
                    f"        const queryLower = query.toLowerCase();",
                    f"        return this.items.filter(item =>",
                    f"            JSON.stringify(item).toLowerCase().includes(queryLower)",
                    f"        );",
                    f"    }}",
                ])
            elif feature == "list":
                lines.extend([
                    f"    listAll() {{",
                    f"        return [...this.items];",
                    f"    }}",
                ])
        
        lines.extend([
            f"    getId(item) {{",
            f"        return item.id || JSON.stringify(item);",
            f"    }}",
            f"    count() {{",
            f"        return this.items.length;",
            f"    }}",
            f"}}",
            f"",
            f"module.exports = {class_name};",
        ])
        
        return "\n".join(lines) + "\n"

    def _gen_js_main(self, class_name: str, meta: dict) -> str:
        service_class = meta.get("service_class", "Service")
        features = meta.get("features", [])
        
        lines = [
            f"const {service_class} = require('./{service_class.toLowerCase()}');",
            f"",
            f"function main() {{",
            f"    const service = new {service_class}();",
            f"    const readline = require('readline');",
            f"    const rl = readline.createInterface({{",
            f"        input: process.stdin,",
            f"        output: process.stdout",
            f"    }});",
            f"",
            f"    console.log('Welcome to {service_class.replace('Service', '')} Manager!');",
            f"    console.log('Available commands:');",
        ]
        
        cmd_map = {
            "add": "add - Add a new item",
            "remove": "remove - Remove an item",
            "update": "update - Update an item",
            "search": "search - Search items",
            "list": "list - List all items",
        }
        
        for feature in features:
            if feature in cmd_map:
                lines.append(f"    console.log('  {cmd_map[feature]}');")
        
        lines.extend([
            f"    console.log('  quit - Exit the program');",
            f"    console.log();",
            f"",
            f"    const promptUser = () => {{",
            f"        rl.question('Enter command: ', (command) => {{",
            f"            command = command.trim().toLowerCase();",
            f"            switch (command) {{",
            f"                case 'quit':",
            f"                    console.log('Goodbye!');",
            f"                    rl.close();",
            f"                    return;",
        ])
        
        for feature in features:
            if feature == "add":
                lines.extend([
                    f"                case 'add':",
                    f"                    rl.question('Enter name: ', (name) => {{",
                    f"                        if (name.trim()) {{",
                    f"                            service.add({{ name: name.trim() }});",
                    f"                            console.log('Added: ' + name);",
                    f"                        }}",
                    f"                        promptUser();",
                    f"                    }});",
                    f"                    return;",
                ])
            elif feature == "list":
                lines.extend([
                    f"                case 'list':",
                    f"                    const items = service.listAll();",
                    f"                    if (items.length > 0) {{",
                    f"                        items.forEach(item => console.log('  - ' + item));",
                    f"                    }} else {{",
                    f"                        console.log('No items found');",
                    f"                    }}",
                    f"                    break;",
                ])
        
        lines.extend([
            f"                default:",
            f"                    console.log('Unknown command. Type a command from the list above.');",
            f"            }}",
            f"            console.log();",
            f"            promptUser();",
            f"        }});",
            f"    }};",
            f"",
            f"    promptUser();",
            f"}}",
            f"",
            f"main();",
        ])
        
        return "\n".join(lines) + "\n"

    def _gen_python_pipeline(self, file_ir: FileIR, meta: dict) -> str:
        """Generate Python function that implements an operation pipeline."""
        operations = meta.get("operations", [])
        inputs = meta.get("inputs", [])
        outputs = meta.get("outputs", [])
        conditions = meta.get("conditions", [])
        success_criteria = meta.get("success_criteria", [])
        func_name = meta.get("function_name", "process_data")

        lines = [
            "from typing import List, Any",
            "",
            "",
        ]

        # Build the function signature
        input_type = "List[float]"
        if inputs:
            inp = inputs[0]
            if isinstance(inp, dict):
                input_type = inp.get("type", "List[float]")
            else:
                input_type = str(inp)
        output_type = "List[float]"
        if outputs:
            out = outputs[0]
            if isinstance(out, dict):
                output_type = out.get("type", "List[float]")
            else:
                output_type = str(out)

        lines.append(f"def {func_name}(data: {input_type}) -> {output_type}:")
        lines.append(f'    """')
        lines.append(f"    Process data through operation pipeline:")
        for i, op in enumerate(operations):
            lines.append(f"        {i+1}. {op.replace('_', ' ')}")
        lines.append(f'    """')

        # Generate operation implementations in order
        var_name = "data"
        for i, op in enumerate(operations):
            next_var = f"step_{i+1}"
            if op == "remove_duplicates":
                lines.append(f"    # Step {i+1}: Remove duplicates (preserve order)")
                lines.append(f"    seen = set()")
                lines.append(f"    {next_var} = []")
                lines.append(f"    for x in {var_name}:")
                lines.append(f"        if x not in seen:")
                lines.append(f"            seen.add(x)")
                lines.append(f"            {next_var}.append(x)")
                lines.append("")
            elif op == "calculate_average":
                lines.append(f"    # Step {i+1}: Calculate average")
                lines.append(f"    avg_{i+1} = sum({var_name}) / len({var_name}) if {var_name} else 0")
                lines.append(f"    {next_var} = {var_name}  # Pass through for filtering")
                lines.append("")
            elif op == "filter_above":
                ref = "average"
                for cond in conditions:
                    if cond.get("type") == "filter" and cond.get("comparison") == "greater_than":
                        ref = cond.get("reference", "average")
                        break
                lines.append(f"    # Step {i+1}: Filter values above {ref}")
                if ref == "average":
                    lines.append(f"    {next_var} = [x for x in {var_name} if x > avg_{i}]" if i > 0 else f"    {next_var} = [x for x in {var_name} if x > sum({var_name}) / len({var_name})]")
                else:
                    lines.append(f"    {next_var} = [x for x in {var_name} if x > {ref}]")
                lines.append("")
            elif op == "filter_below":
                ref = "average"
                for cond in conditions:
                    if cond.get("type") == "filter" and cond.get("comparison") == "less_than":
                        ref = cond.get("reference", "average")
                        break
                lines.append(f"    # Step {i+1}: Filter values below {ref}")
                if ref == "average":
                    lines.append(f"    {next_var} = [x for x in {var_name} if x < avg_{i}]" if i > 0 else f"    {next_var} = [x for x in {var_name} if x < sum({var_name}) / len({var_name})]")
                else:
                    lines.append(f"    {next_var} = [x for x in {var_name} if x < {ref}]")
                lines.append("")
            elif op == "sort_descending":
                lines.append(f"    # Step {i+1}: Sort in descending order")
                lines.append(f"    {next_var} = sorted({var_name}, reverse=True)")
                lines.append("")
            elif op == "sort_ascending":
                lines.append(f"    # Step {i+1}: Sort in ascending order")
                lines.append(f"    {next_var} = sorted({var_name})")
                lines.append("")
            elif op == "sort":
                lines.append(f"    # Step {i+1}: Sort")
                lines.append(f"    {next_var} = sorted({var_name})")
                lines.append("")
            elif op == "add":
                lines.append(f"    # Step {i+1}: Add/collect items")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy")
                lines.append("")
            elif op == "remove":
                lines.append(f"    # Step {i+1}: Remove items")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy (removal handled by caller)")
                lines.append("")
            elif op == "search":
                lines.append(f"    # Step {i+1}: Search/filter")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy (search handled by caller)")
                lines.append("")
            elif op == "list":
                lines.append(f"    # Step {i+1}: List items")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy")
                lines.append("")
            elif op == "store":
                lines.append(f"    # Step {i+1}: Store results")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy")
                lines.append("")
            elif op == "calculate":
                lines.append(f"    # Step {i+1}: Calculate")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy")
                lines.append("")
            elif op == "manage":
                lines.append(f"    # Step {i+1}: Manage data")
                lines.append(f"    {next_var} = {var_name}[:]  # Copy")
                lines.append("")
            else:
                lines.append(f"    # Step {i+1}: {op.replace('_', ' ')}")
                lines.append(f"    {next_var} = {var_name}  # Pass through")
                lines.append("")

            var_name = next_var

        lines.append(f"    return {var_name}")
        lines.append("")

        # Add example usage if it's a main file
        if meta.get("is_entry_point", False):
            lines.append("")
            lines.append('if __name__ == "__main__":')
            lines.append("    # Example usage")
            lines.append("    sample_data = [5, 2, 5, 8, 3, 8, 2, 1, 9, 4]")
            lines.append(f"    result = {func_name}(sample_data)")
            lines.append("    print(f'Input: {sample_data}')")
            lines.append("    print(f'Result: {result}')")

        return "\n".join(lines) + "\n"

    def _gen_go_source(self, file_ir: FileIR, meta: dict) -> str:
        return f"package main\n\n// Generated Go file: {file_ir.path}\n"

    def _gen_rust_source(self, file_ir: FileIR, meta: dict) -> str:
        return f"// Generated Rust file: {file_ir.path}\n"

    def _gen_c_source(self, file_ir: FileIR, meta: dict) -> str:
        return f"// Generated C file: {file_ir.path}\n"

    def _gen_cpp_source(self, file_ir: FileIR, meta: dict) -> str:
        return f"// Generated C++ file: {file_ir.path}\n"
