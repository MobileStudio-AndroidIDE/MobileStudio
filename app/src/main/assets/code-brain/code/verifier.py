"""Verifier - validates generated code using language tools."""
from __future__ import annotations
import subprocess
import os
import tempfile
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerificationResult:
    language: str
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output: str = ""
    details: str = ""


class Verifier:
    def __init__(self):
        self._verifiers = {
            "python": self._verify_python,
            "javascript": self._verify_javascript,
            "go": self._verify_go,
            "rust": self._verify_rust,
            "java": self._verify_java,
            "kotlin": self._verify_kotlin,
            "c": self._verify_c,
            "cpp": self._verify_cpp,
        }

    def verify(self, code: str, language: str) -> VerificationResult:
        verifier = self._verifiers.get(language)
        if not verifier:
            return VerificationResult(
                language=language,
                success=False,
                errors=[f"No verifier available for {language}"],
            )
        return verifier(code)

    def _verify_python(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python3", "-c", f"import ast; ast.parse(open('{f.name}').read())"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    try:
                        compile_result = subprocess.run(
                            ["python3", "-m", "py_compile", f.name],
                            capture_output=True, text=True, timeout=10
                        )
                        if compile_result.returncode == 0:
                            return VerificationResult(language="python", success=True, output="Syntax OK")
                        else:
                            errors = compile_result.stderr.strip().split("\n")
                            return VerificationResult(language="python", success=False, errors=errors)
                    except Exception:
                        return VerificationResult(language="python", success=True, output="Syntax OK (AST parse)")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="python", success=False, errors=errors)
            except subprocess.TimeoutExpired:
                return VerificationResult(language="python", success=False, errors=["Timeout"])
            except FileNotFoundError:
                return VerificationResult(language="python", success=False, errors=["Python not found"])
            finally:
                os.unlink(f.name)

    def _verify_javascript(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["node", "--check", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return VerificationResult(language="javascript", success=True, output="Syntax OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="javascript", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return VerificationResult(language="javascript", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)

    def _verify_go(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["gofmt", "-e", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return VerificationResult(language="go", success=True, output="Syntax OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="go", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                if "No such file" in str(e):
                    return VerificationResult(language="go", success=False, errors=["gofmt not installed"])
                return VerificationResult(language="go", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)

    def _verify_rust(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["rustfmt", "--check", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return VerificationResult(language="rust", success=True, output="Syntax OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="rust", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                if "No such file" in str(e):
                    return VerificationResult(language="rust", success=False, errors=["rustfmt not installed"])
                return VerificationResult(language="rust", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)

    def _verify_java(self, code: str) -> VerificationResult:
        import re as re_mod
        class_match = re_mod.search(r'public\s+class\s+(\w+)', code)
        class_name = class_match.group(1) if class_match else "Main"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", dir=os.path.dirname(tempfile.gettempdir()), delete=False) as f:
            actual_path = f.name
            temp_dir = os.path.dirname(actual_path)
            target_path = os.path.join(temp_dir, f"{class_name}.java")
            f.write(code)
            f.flush()
        try:
            import shutil
            if actual_path != target_path:
                shutil.copy2(actual_path, target_path)
            result = subprocess.run(
                ["javac", "-Xlint:none", target_path],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return VerificationResult(language="java", success=True, output="Compilation OK")
            else:
                errors = result.stderr.strip().split("\n")
                filtered = [e for e in errors if e.strip()]
                if not filtered:
                    return VerificationResult(language="java", success=True, output="Compilation OK (minor warnings)")
                return VerificationResult(language="java", success=False, errors=filtered)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if "No such file" in str(e):
                return VerificationResult(language="java", success=False, errors=["javac not installed"])
            return VerificationResult(language="java", success=False, errors=[str(e)])
        finally:
            for p in [actual_path, target_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _verify_kotlin(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".kt", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["kotlinc", "-script", f.name],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return VerificationResult(language="kotlin", success=True, output="Compilation OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="kotlin", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                if "No such file" in str(e):
                    return VerificationResult(language="kotlin", success=False, errors=["kotlinc not installed"])
                return VerificationResult(language="kotlin", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)

    def _verify_c(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["gcc", "-fsyntax-only", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return VerificationResult(language="c", success=True, output="Syntax OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="c", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return VerificationResult(language="c", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)

    def _verify_cpp(self, code: str) -> VerificationResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["g++", "-fsyntax-only", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    return VerificationResult(language="cpp", success=True, output="Syntax OK")
                else:
                    errors = result.stderr.strip().split("\n")
                    return VerificationResult(language="cpp", success=False, errors=errors)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                return VerificationResult(language="cpp", success=False, errors=[str(e)])
            finally:
                os.unlink(f.name)
