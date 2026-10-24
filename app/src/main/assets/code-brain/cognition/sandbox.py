"""Bounded execution and guarded self-modification.

Two separate safety layers:

* :class:`Sandbox` runs small Python artifacts in a fresh temporary
  directory with a static AST permission scan, a timeout and (where the
  platform supports it) CPU/address-space rlimits.  This contains
  accidents; it is not an adversarial OS sandbox.
* :class:`SelfModifier` applies a patch only when self-modification is
  explicitly enabled, the target is inside an allowed write root, the
  patch parses, the whole test command passes, and a backup exists.
  Any failure restores the byte-identical original.

Every action produces a machine-readable result, including failures and
the reason.  Nothing is reported as applied before verification passes.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from cognition.config import GrowthConfig
from cognition.util import atomic_write_json, read_json, stable_hash

_FORBIDDEN_MODULES = {
    "subprocess", "socket", "ctypes", "multiprocessing", "http",
    "urllib", "ftplib", "telnetlib", "paramiko", "shutil",
}
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "os.system", "os.popen", "os.remove", "os.unlink", "os.rmdir",
    "os.removedirs", "os.rename", "os.replace", "shutil.rmtree",
}
_ALLOWED_IMPORTS = {
    "math", "statistics", "itertools", "functools", "collections",
    "json", "re", "datetime", "fractions", "decimal", "random", "typing",
    "dataclasses", "string", "operator", "heapq", "bisect",
}


class PermissionDenied(Exception):
    pass


@dataclass
class SandboxResult:
    ok: bool
    timed_out: bool = False
    violations: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    seconds: float = 0.0
    results: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "timed_out": self.timed_out,
            "violations": list(self.violations),
            "stdout": self.stdout[-2000:],
            "stderr": self.stderr[-2000:],
            "returncode": self.returncode,
            "seconds": round(self.seconds, 4),
            "results": self.results,
            "reason": self.reason,
        }


class PermissionPolicy:
    def __init__(self, config: GrowthConfig):
        self.config = config
        self.read_roots = [os.path.realpath(p)
                           for p in config.allowed_read_roots]
        self.write_roots = [os.path.realpath(p)
                            for p in config.allowed_write_roots]
        self.allow_network = bool(config.allow_network)

    @staticmethod
    def _inside(path: str, roots: Iterable[str]) -> bool:
        real = os.path.realpath(path)
        for root in roots:
            try:
                if os.path.commonpath([real, root]) == root:
                    return True
            except ValueError:
                continue
        return False

    def check_read(self, path: str) -> str:
        real = os.path.realpath(path)
        if not self._inside(real, self.read_roots):
            raise PermissionDenied(f"read outside allowed roots: {real}")
        return real

    def check_write(self, path: str) -> str:
        real = os.path.realpath(path)
        if not self._inside(real, self.write_roots):
            raise PermissionDenied(f"write outside allowed roots: {real}")
        return real

    def check_network(self) -> None:
        if not self.allow_network:
            raise PermissionDenied("network access is disabled by policy")


class Sandbox:
    def __init__(self, config: GrowthConfig):
        self.config = config
        self.policy = PermissionPolicy(config)

    # ------------------------------------------------------------------
    # Static checks
    # ------------------------------------------------------------------
    def static_violations(self, code: str,
                          allow_network: bool = False) -> list[str]:
        violations: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax_error: {exc.msg} (line {exc.lineno})"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _FORBIDDEN_MODULES:
                        violations.append(f"forbidden import: {alias.name}")
                    elif root not in _ALLOWED_IMPORTS:
                        violations.append(f"unexpected import: {alias.name}")
                    if root in ("socket", "urllib", "http") and allow_network:
                        violations = [v for v in violations
                                      if not v.endswith(alias.name)]
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    violations.append(f"forbidden import: {node.module}")
                elif root not in _ALLOWED_IMPORTS:
                    violations.append(f"unexpected import: {node.module}")
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in _FORBIDDEN_CALLS:
                    violations.append(f"forbidden call: {name}")
                if name == "open" and not allow_network:
                    # file I/O in a synthesized artifact is a denied
                    # capability, not a silent permission grant
                    if node.args:
                        mode = "r"
                        if len(node.args) > 1 and isinstance(
                                node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        if any(ch in mode for ch in "wax+"):
                            violations.append(
                                "forbidden file write in sandbox")
        return violations

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_python_source(self, code: str, timeout: Optional[float] = None,
                          extra_files: Optional[dict[str, str]] = None,
                          allow_network: bool = False) -> SandboxResult:
        timeout = float(timeout or self.config.sandbox_timeout)
        started = time.time()
        violations = self.static_violations(code, allow_network=allow_network)
        if violations:
            return SandboxResult(False, violations=violations,
                                 seconds=time.time() - started,
                                 reason="static_permission_violation")
        workdir = tempfile.mkdtemp(prefix="rena_cog_", dir=self.config.data_dir)
        script = os.path.join(workdir, "artifact.py")
        try:
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)
            for name, content in (extra_files or {}).items():
                safe = os.path.basename(name)
                with open(os.path.join(workdir, safe), "w",
                          encoding="utf-8") as handle:
                    handle.write(content)
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": workdir,
                "TMPDIR": workdir,
                "RENA_SANDBOX": "1",
            }
            command = [sys.executable, "-I", "-S", script]
            preexec = _limit_setter(self.config.sandbox_memory_mb)
            proc = subprocess.run(
                command, cwd=workdir, env=env, capture_output=True,
                text=True, timeout=timeout, preexec_fn=preexec)
            return SandboxResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
                seconds=time.time() - started,
                reason="ok" if proc.returncode == 0 else "nonzero_exit")
        except subprocess.TimeoutExpired:
            return SandboxResult(False, timed_out=True,
                                 seconds=time.time() - started,
                                 reason="timeout")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def verify_python_function(self, code: str, function_name: str,
                               cases: list[dict],
                               timeout: Optional[float] = None
                               ) -> SandboxResult:
        """Run a generated function against JSON-serializable cases.

        Each case is ``{"args": [...], "expected": ...}``.  The harness
        writes its verdict to a result file; output formatting of the
        artifact cannot forge it.
        """
        timeout = float(timeout or self.config.sandbox_timeout)
        if not cases:
            return SandboxResult(False, reason="no_test_cases")
        marker = stable_hash(code, function_name, time.time(), length=16)
        result_file = f"__rena_result_{marker}.json"
        return self._retained_verification(code, function_name, cases,
                                           timeout, result_file)

    def _retained_verification(self, code: str, function_name: str,
                               cases: list[dict], timeout: float,
                               result_file: str) -> SandboxResult:
        workdir = tempfile.mkdtemp(prefix="rena_verify_",
                                   dir=self.config.data_dir)
        started = time.time()
        try:
            violations = self.static_violations(code)
            if violations:
                return SandboxResult(False, violations=violations,
                                     seconds=time.time() - started,
                                     reason="static_permission_violation")
            script = os.path.join(workdir, "artifact.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(code)
                handle.write("\n\n")
                handle.write(_verification_harness(function_name, cases,
                                                   result_file))
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": workdir,
                "TMPDIR": workdir,
                "RENA_SANDBOX": "1",
            }
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", script], cwd=workdir,
                    env=env, capture_output=True, text=True,
                    timeout=timeout,
                    preexec_fn=_limit_setter(
                        self.config.sandbox_memory_mb))
                timed_out = False
            except subprocess.TimeoutExpired:
                proc = None
                timed_out = True
            verdict_path = os.path.join(workdir, result_file)
            rows: list[dict] = []
            if os.path.exists(verdict_path):
                try:
                    with open(verdict_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    rows = list(payload.get("results") or [])
                except Exception:
                    rows = []
            passed = bool(rows) and all(row.get("passed") for row in rows)
            ok = bool(proc and proc.returncode == 0 and passed)
            return SandboxResult(
                ok=ok,
                timed_out=timed_out,
                stdout=(proc.stdout or "") if proc else "",
                stderr=(proc.stderr or "") if proc else "",
                returncode=proc.returncode if proc else -1,
                seconds=time.time() - started,
                results=rows,
                reason=("ok" if ok else
                        "timeout" if timed_out else
                        "tests_failed" if not passed else "nonzero_exit"))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # File operations (policy checked)
    # ------------------------------------------------------------------
    def write_text(self, path: str, text: str) -> dict:
        safe = self.policy.check_write(path)
        os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
        tmp = safe + ".tmp_" + stable_hash(time.time(), length=8)
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, safe)
        with open(safe, "rb") as handle:
            digest = stable_hash(handle.read(), length=32)
        return {"path": safe, "bytes": os.path.getsize(safe),
                "sha256": digest}

    def read_text(self, path: str) -> str:
        safe = self.policy.check_read(path)
        with open(safe, "r", encoding="utf-8") as handle:
            return handle.read()

    def delete_file(self, path: str) -> bool:
        safe = self.policy.check_write(path)
        if os.path.isfile(safe):
            os.remove(safe)
            return True
        return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _verification_harness(function_name: str, cases: list[dict],
                          result_file: str) -> str:
    cases_json = json.dumps(cases)
    return f"""
import json as _rena_json

def _rena_verify():
    cases = _rena_json.loads({cases_json!r})
    rows = []
    for index, case in enumerate(cases):
        try:
            args = case.get("args") or []
            expected = case.get("expected")
            actual = {function_name}(*args)
            passed = _rena_equal(actual, expected)
            rows.append({{"index": index, "passed": passed,
                          "actual": repr(actual),
                          "expected": repr(expected), "error": ""}})
        except Exception as exc:
            rows.append({{"index": index, "passed": False,
                          "actual": "", "expected": repr(
                              case.get("expected")),
                          "error": type(exc).__name__ + ": " + str(exc)[:300]}})
    with open({result_file!r}, "w", encoding="utf-8") as handle:
        _rena_json.dump({{"results": rows}}, handle)

def _rena_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual) == bool(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-7 * max(
            1.0, abs(float(expected)))
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _rena_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected

_rena_verify()
"""


def _limit_setter(memory_mb: int):
    try:
        import resource
    except Exception:
        return None

    android = bool(os.environ.get("ANDROID_ROOT")
                   or os.path.exists("/system/build.prop"))

    def set_limits():
        if not android:
            try:
                bytes_limit = int(memory_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS,
                                   (bytes_limit, bytes_limit))
            except Exception:
                pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except Exception:
            pass

    return set_limits


# --------------------------------------------------------------------- #
# Guarded self-modification
# --------------------------------------------------------------------- #
@dataclass
class PatchPlan:
    target: str
    old_text: str
    new_text: str
    reason: str
    test_command: list[str] = field(default_factory=list)
    author: str = "growth_engine"


class SelfModifier:
    """Apply a patch only if every guard passes; otherwise roll back."""

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.policy = PermissionPolicy(config)
        self.backup_dir = os.path.join(config.data_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.backup_dir, "manifest.json")
        self.manifest = read_json(self.manifest_path, {"changes": []})

    def apply(self, plan: PatchPlan, timeout: float = 120.0) -> dict:
        result = {
            "applied": False,
            "target": plan.target,
            "reason": plan.reason,
            "change_id": "",
            "backup": "",
            "test_passed": False,
            "error": "",
        }
        if not self.config.allow_self_modification:
            result["error"] = "self_modification_disabled"
            return result
        try:
            target = self.policy.check_write(plan.target)
        except PermissionDenied as exc:
            result["error"] = str(exc)
            return result
        if not os.path.isfile(target):
            result["error"] = "target_not_found"
            return result
        with open(target, "r", encoding="utf-8") as handle:
            original = handle.read()
        if plan.old_text not in original:
            result["error"] = "old_text_not_found"
            return result
        patched = original.replace(plan.old_text, plan.new_text, 1)
        if target.endswith(".py"):
            try:
                ast.parse(patched)
            except SyntaxError as exc:
                result["error"] = f"syntax_error: {exc.msg}"
                return result
            violations = Sandbox(self.config).static_violations(
                patched, allow_network=self.config.allow_network)
            if violations:
                result["error"] = "permission_violation: " + \
                                  "; ".join(violations[:5])
                return result
        change_id = "chg_" + stable_hash(target, time.time(),
                                         plan.reason, length=12)
        backup = os.path.join(
            self.backup_dir,
            f"{change_id}_{os.path.basename(target)}.bak")
        with open(backup, "w", encoding="utf-8") as handle:
            handle.write(original)
        result["change_id"] = change_id
        result["backup"] = backup
        try:
            tmp = target + ".rena_patch_" + change_id
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(patched)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            test_passed, test_detail = self._run_tests(plan.test_command,
                                                       timeout)
            result["test_passed"] = test_passed
            result["test_detail"] = test_detail
            if not test_passed:
                self._restore(target, original)
                result["error"] = "tests_failed_rolled_back"
                self._record(result, plan, applied=False)
                return result
            result["applied"] = True
            self._record(result, plan, applied=True)
            return result
        except Exception as exc:
            self._restore(target, original)
            result["error"] = f"exception_rolled_back: {exc}"
            self._record(result, plan, applied=False)
            return result

    def rollback(self, change_id: str) -> bool:
        for change in reversed(self.manifest.get("changes") or []):
            if change.get("change_id") != change_id:
                continue
            backup = change.get("backup") or ""
            target = change.get("target") or ""
            if not backup or not os.path.isfile(backup):
                return False
            try:
                safe_target = self.policy.check_write(target)
                with open(backup, "r", encoding="utf-8") as handle:
                    original = handle.read()
                self._restore(safe_target, original)
                change["rolled_back_at"] = time.time()
                atomic_write_json(self.manifest_path, self.manifest)
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def _restore(target: str, original: str) -> None:
        tmp = target + ".rena_rollback"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)

    def _run_tests(self, command: list[str],
                   timeout: float) -> tuple[bool, str]:
        if not command:
            return False, "no_test_command_refused"
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, env=env)
            detail = ((proc.stdout or "") + (proc.stderr or ""))[-3000:]
            return proc.returncode == 0, detail
        except subprocess.TimeoutExpired:
            return False, "test_timeout"
        except Exception as exc:
            return False, f"test_error: {exc}"

    def _record(self, result: dict, plan: PatchPlan,
                applied: bool) -> None:
        self.manifest.setdefault("changes", []).append({
            "change_id": result.get("change_id"),
            "target": result.get("target"),
            "backup": result.get("backup"),
            "reason": plan.reason,
            "author": plan.author,
            "applied": bool(applied),
            "test_passed": bool(result.get("test_passed")),
            "error": result.get("error", ""),
            "at": time.time(),
        })
        self.manifest["changes"] = self.manifest.get("changes", [])[-200:]
        atomic_write_json(self.manifest_path, self.manifest)
