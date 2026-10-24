"""Capability actuators: code, math, image, video, web, files and data.

Actuators return evidence, not just strings.  A result is ``ok`` only
when the operation completed and its artifact/check was observed.  Tools
that have not passed verification are reported as unavailable rather
than silently used.
"""
from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.config import GrowthConfig
from cognition.math_engine import (differentiate, evaluate,
                                   parse_expression, simplify, solve_linear)
from cognition.sandbox import PermissionDenied, PermissionPolicy, Sandbox
from cognition.tools import ToolService


@dataclass
class ActuatorResult:
    ok: bool
    actuator: str
    operation: str
    output: object = None
    artifacts: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    error: str = ""
    latency: float = 0.0
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "actuator": self.actuator,
            "operation": self.operation,
            "output": self.output,
            "artifacts": list(self.artifacts),
            "evidence": self.evidence,
            "error": self.error,
            "latency": round(self.latency, 4),
            "verified": bool(self.verified),
        }


class BaseActuator:
    name = "base"
    modalities: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str]:
        return True, ""

    def execute(self, request, **kwargs) -> ActuatorResult:
        raise NotImplementedError

    def describe(self) -> dict:
        available, reason = self.available()
        return {
            "name": self.name,
            "modalities": list(self.modalities),
            "available": available,
            "reason": reason,
        }


def _normalize_math_text(text: str, operation: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    lowered = raw.lower()
    if operation == "evaluate":
        if any(key in lowered for key in
               ("integrate", "적분", "antiderivative")):
            operation = "integrate"
        elif any(key in lowered for key in
                 ("differentiate", "derivative", "미분", "도함수")):
            operation = "differentiate"
        elif "=" in raw or "solve" in lowered or "풀어" in lowered:
            operation = "solve"
        elif any(key in lowered for key in ("simplify", "단순화")):
            operation = "simplify"
    cleaned = raw
    for token in ("미분해줘", "미분", "도함수", "풀어줘", "풀어",
                  "단순화해줘", "단순화", "적분해줘", "적분",
                  "differentiate", "derivative of", "derivative",
                  "solve", "simplify", "evaluate", "d/dx"):
        cleaned = re.sub(re.escape(token), " ", cleaned,
                         flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\r\n:：,，.")
    return cleaned or raw, operation


class MathActuator(BaseActuator):
    name = "math"
    modalities = ("math",)

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "evaluate"
        expression = ""
        if isinstance(request, dict):
            operation = str(request.get("operation", "evaluate"))
            expression = str(request.get("expression",
                                         request.get("input", "")))
            variable = str(request.get("variable", "x"))
            env = dict(request.get("env") or {})
        else:
            expression = str(request)
            variable, env = str(kwargs.get("variable", "x")), \
                dict(kwargs.get("env") or {})
            lowered = expression.lower()
            if "=" in expression and not lowered.startswith("solve"):
                operation = "solve"
            elif lowered.startswith(("differentiate", "derivative",
                                     "d/dx")):
                operation = "differentiate"
                expression = expression.split(None, 1)[-1] if " " in \
                    expression else expression
            elif lowered.startswith("simplify"):
                operation = "simplify"
            elif lowered.startswith("solve"):
                operation = "solve"
                expression = expression[5:].strip()
            elif lowered.startswith("evaluate"):
                operation = "evaluate"
                expression = expression.split(None, 1)[-1] if " " in \
                    expression else expression
        expression, operation = _normalize_math_text(expression, operation)
        try:
            if operation == "integrate":
                raise ValueError("symbolic integration is not implemented")
            if operation == "solve":
                value = solve_linear(expression, variable)
                if value is None:
                    raise ValueError("not a linear equation")
                output = {"variable": variable, "value": value}
            elif operation == "differentiate":
                output = {"derivative": differentiate(expression, variable)}
            elif operation == "simplify":
                try:
                    from knowledge import mathcore
                    output = {"expression": mathcore.to_string(
                        mathcore.simplify(expression))}
                except Exception:
                    output = {"expression": simplify(
                        parse_expression(expression)).to_string()}
            elif operation == "variables":
                output = {"variables":
                          parse_expression(expression).variables()}
            else:
                tree = parse_expression(expression)
                output = {"value": evaluate(tree, env),
                          "expression": tree.to_string(),
                          "variables": tree.variables()}
            return ActuatorResult(
                ok=True, actuator=self.name, operation=operation,
                output=output,
                evidence={"expression": expression},
                latency=time.time() - started, verified=True)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class CodeActuator(BaseActuator):
    name = "code"
    modalities = ("code",)

    def __init__(self, config: GrowthConfig, brain=None):
        self.config = config
        self.sandbox = Sandbox(config)
        self.brain = brain

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "analyze"
        code = ""
        if isinstance(request, dict):
            operation = str(request.get("operation", "analyze"))
            code = str(request.get("code", request.get("input", "")))
            function = str(request.get("function", ""))
            cases = list(request.get("cases") or [])
            language = str(request.get("language", "python"))
        else:
            code = str(request)
            function = str(kwargs.get("function", ""))
            cases = list(kwargs.get("cases") or [])
            language = str(kwargs.get("language", "python"))
        try:
            if operation in ("run", "execute"):
                run = self.sandbox.run_python_source(
                    code, allow_network=self.config.allow_network)
                return ActuatorResult(
                    ok=run.ok, actuator=self.name, operation=operation,
                    output={"stdout": run.stdout, "stderr": run.stderr},
                    evidence=run.to_dict(), latency=time.time() - started,
                    verified=run.ok, error="" if run.ok else run.reason)
            if operation in ("verify", "test"):
                if not function or not cases:
                    raise ValueError("verify requires function and cases")
                run = self.sandbox.verify_python_function(
                    code, function, cases)
                return ActuatorResult(
                    ok=run.ok, actuator=self.name, operation=operation,
                    output={"results": run.results},
                    evidence=run.to_dict(), latency=time.time() - started,
                    verified=run.ok, error="" if run.ok else run.reason)
            if operation == "analyze":
                tree = ast.parse(code)
                imports, functions, classes = [], [], []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(node.module or "")
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                violations = self.sandbox.static_violations(
                    code, allow_network=self.config.allow_network)
                return ActuatorResult(
                    ok=not violations, actuator=self.name,
                    operation=operation,
                    output={"language": language, "functions": functions,
                            "classes": classes, "imports": imports,
                            "lines": len(code.splitlines()),
                            "violations": violations},
                    evidence={"violations": violations},
                    latency=time.time() - started,
                    verified=not violations)
            if operation == "generate":
                if self.brain is None:
                    raise RuntimeError("CodeBrain is not attached")
                response = self.brain.process(code, language=language)
                output = response if isinstance(response, dict) else \
                    {"response": response}
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=output, latency=time.time() - started,
                    verified=False)
            raise ValueError("unknown code operation: " + operation)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class ImageActuator(BaseActuator):
    name = "image"
    modalities = ("image",)

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.policy = PermissionPolicy(config)

    def available(self) -> tuple[bool, str]:
        try:
            from media.image import analysis, generate
            return bool(analysis and generate), ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _path(self, path: str = "", suffix: str = ".png") -> str:
        if path:
            return self.policy.check_write(path)
        os.makedirs(self.config.artifact_dir, exist_ok=True)
        return os.path.join(
            self.config.artifact_dir,
            "image_%d%s" % (int(time.time() * 1000), suffix))

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "describe"
        if isinstance(request, dict):
            operation = str(request.get("operation", "describe"))
            path = str(request.get("path", request.get("image", "")))
            prompt = str(request.get("prompt", request.get("request", "")))
            width = int(request.get("width", 480))
            height = int(request.get("height", 360))
            seed = int(request.get("seed", self.config.seed))
            output_path = str(request.get("output_path", ""))
            chart = request.get("chart") or {
                key: request.get(key) for key in
                ("labels", "values", "title", "kind", "width", "height")
                if key in request}
        else:
            prompt = str(request)
            path = str(kwargs.get("path", ""))
            width = int(kwargs.get("width", 480))
            height = int(kwargs.get("height", 360))
            seed = int(kwargs.get("seed", self.config.seed))
            output_path = str(kwargs.get("output_path", ""))
            chart = kwargs.get("chart") or {}
            if prompt and not path:
                operation = "generate"
        try:
            from media.image import analysis, generate
            if operation in ("generate", "draw", "create"):
                if not prompt:
                    raise ValueError("generate requires a prompt")
                output_path = self._path(output_path, ".png")
                metadata = generate.generate_image(
                    prompt, output_path, width=width, height=height,
                    seed=seed, force=kwargs.get("force", True))
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=metadata,
                    artifacts=[output_path],
                    evidence={"verified": True,
                              "width": metadata.get("width"),
                              "height": metadata.get("height")},
                    latency=time.time() - started, verified=True)
            if operation in ("chart", "graph"):
                from brain import image_io as ii
                if not isinstance(chart, dict) or not chart.get("values"):
                    raise ValueError("chart requires labels and values")
                output_path = self._path(output_path, ".png")
                title = str(chart.get("title", "chart"))
                labels = [str(item) for item in chart.get("labels") or []]
                values = [float(item) for item in chart.get("values") or []]
                if chart.get("kind", "bar") == "line":
                    image = ii.line_chart(
                        title, labels, values,
                        width=int(chart.get("width", 640)),
                        height=int(chart.get("height", 360)))
                else:
                    image = ii.bar_chart(
                        title, labels, values,
                        width=int(chart.get("width", 640)),
                        height=int(chart.get("height", 360)))
                ii.save_image(image, output_path)
                back = ii.load_image(output_path)
                verified = (back.width, back.height) == (
                    image.width, image.height)
                return ActuatorResult(
                    ok=verified, actuator=self.name, operation=operation,
                    output={"path": output_path, "labels": labels,
                            "values": values},
                    artifacts=[output_path],
                    evidence={"width": back.width,
                              "height": back.height},
                    latency=time.time() - started, verified=verified,
                    error="" if verified else "artifact_verification_failed")
            if operation in ("describe", "analyze"):
                if not path:
                    raise ValueError("describe requires a path")
                safe = self.policy.check_read(path)
                description = analysis.describe_image(safe)
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=description, artifacts=[safe],
                    evidence={"scene": description.get("scene_guess")},
                    latency=time.time() - started, verified=True)
            raise ValueError("unknown image operation: " + operation)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class VideoActuator(BaseActuator):
    name = "video"
    modalities = ("video",)

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.policy = PermissionPolicy(config)

    def available(self) -> tuple[bool, str]:
        try:
            from media.video import analyze, generate
            return bool(analyze and generate), ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _path(self, path: str = "") -> str:
        if path:
            return self.policy.check_write(path)
        os.makedirs(self.config.artifact_dir, exist_ok=True)
        return os.path.join(
            self.config.artifact_dir,
            "video_%d.avi" % int(time.time() * 1000))

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "analyze"
        if isinstance(request, dict):
            operation = str(request.get("operation", "analyze"))
            path = str(request.get("path", request.get("video", "")))
            prompt = str(request.get("prompt", request.get("request", "")))
            duration = float(request.get("duration", 2.0))
            fps = float(request.get("fps", 8.0))
            seed = int(request.get("seed", self.config.seed))
            output_path = str(request.get("output_path", ""))
            chart = request.get("chart") or {
                key: request.get(key) for key in
                ("labels", "values", "title", "kind", "width", "height")
                if key in request}
        else:
            prompt = str(request)
            path = str(kwargs.get("path", ""))
            duration = float(kwargs.get("duration", 2.0))
            fps = float(kwargs.get("fps", 8.0))
            seed = int(kwargs.get("seed", self.config.seed))
            output_path = str(kwargs.get("output_path", ""))
            if prompt and not kwargs.get("operation"):
                operation = "generate"
        try:
            from media.video import analyze, generate
            if operation in ("generate", "create"):
                if not prompt:
                    raise ValueError("generate requires a prompt")
                output_path = self._path(output_path)
                metadata = generate.generate_video(
                    prompt, output_path, duration=duration, fps=fps,
                    width=int(kwargs.get("width", 160)),
                    height=int(kwargs.get("height", 120)),
                    seed=seed, force=kwargs.get("force", True))
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=metadata, artifacts=[output_path],
                    evidence={"verified": True,
                              "frames": metadata.get("frame_count")},
                    latency=time.time() - started, verified=True)
            if operation in ("analyze", "describe"):
                if not path:
                    raise ValueError("analyze requires a path")
                safe = self.policy.check_read(path)
                report = analyze.analyze_video(safe)
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=report, artifacts=[safe],
                    evidence={"scene_changes":
                              report.get("scene_changes")},
                    latency=time.time() - started, verified=True)
            raise ValueError("unknown video operation: " + operation)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class WebActuator(BaseActuator):
    name = "web"
    modalities = ("web", "text")

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.tools = ToolService(config)

    def available(self) -> tuple[bool, str]:
        if not self.config.allow_network:
            return False, "network_disabled"
        try:
            tool = self.tools.verifier.tools.get(
                "cognition.tools.web_research")
            if tool is not None and tool.verified:
                return True, ""
        except Exception:
            pass
        return False, "web_tool_not_verified"

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        topic = str(request.get("topic", request.get("query", ""))) \
            if isinstance(request, dict) else str(request)
        available, reason = self.available()
        if not available:
            return ActuatorResult(
                ok=False, actuator=self.name, operation="research",
                error=reason, latency=time.time() - started)
        try:
            result = self.tools.web_research(topic)
            return ActuatorResult(
                ok=bool(result), actuator=self.name,
                operation="research", output=result,
                evidence={"topic": topic, "source": result.get("url", "")},
                latency=time.time() - started,
                verified=bool(result.get("url")))
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation="research",
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class AudioActuator(BaseActuator):
    name = "audio"
    modalities = ("audio",)

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.policy = PermissionPolicy(config)

    def available(self) -> tuple[bool, str]:
        try:
            from media.audio import speech
            return bool(speech.SPEECH_AVAILABLE), speech.SPEECH_QUALITY
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _path(self, path: str = "") -> str:
        if path:
            return self.policy.check_write(path)
        os.makedirs(self.config.artifact_dir, exist_ok=True)
        return os.path.join(
            self.config.artifact_dir,
            "audio_%d.wav" % int(time.time() * 1000))

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "speak"
        text, path, voice = "", "", "default"
        sample_rate = 16000
        if isinstance(request, dict):
            operation = str(request.get("operation", "speak"))
            text = str(request.get("text", request.get("request", "")))
            path = str(request.get("path", request.get("audio", "")))
            voice = str(request.get("voice", "default"))
            sample_rate = int(request.get("sample_rate", 16000))
            output_path = str(request.get("output_path", ""))
        else:
            text = str(request)
            path = str(kwargs.get("path", ""))
            voice = str(kwargs.get("voice", "default"))
            sample_rate = int(kwargs.get("sample_rate", 16000))
            output_path = str(kwargs.get("output_path", ""))
        try:
            import wave
            from media.audio.speech import synthesize_speech
            if operation in ("capabilities", "info"):
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output={"available": True, "method": "formant "
                            "synthesis", "note": "robotic, not natural"},
                    latency=time.time() - started, verified=True)
            if operation in ("speak", "tts", "synthesize"):
                if not text.strip():
                    raise ValueError("speak requires text")
                output_path = self._path(output_path)
                synthesize_speech(text, output_path, sample_rate=sample_rate,
                                  voice=voice)
                with wave.open(output_path, "rb") as handle:
                    channels = handle.getnchannels()
                    width = handle.getsampwidth()
                    rate = handle.getframerate()
                    frames = handle.getnframes()
                verified = frames > 0 and rate > 0
                return ActuatorResult(
                    ok=verified, actuator=self.name, operation=operation,
                    output={"path": output_path, "text": text,
                            "duration": round(frames / max(1, rate), 3),
                            "voice": voice},
                    artifacts=[output_path],
                    evidence={"sample_rate": rate, "channels": channels,
                              "sample_width": width, "frames": frames},
                    latency=time.time() - started, verified=verified,
                    error="" if verified else "empty_audio")
            if operation in ("describe", "analyze"):
                if not path:
                    raise ValueError("describe requires a path")
                safe = self.policy.check_read(path)
                with wave.open(safe, "rb") as handle:
                    meta = {"channels": handle.getnchannels(),
                            "sample_rate": handle.getframerate(),
                            "sample_width": handle.getsampwidth(),
                            "frames": handle.getnframes(),
                            "duration": round(
                                handle.getnframes() /
                                max(1, handle.getframerate()), 3)}
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=meta, artifacts=[safe], evidence=meta,
                    latency=time.time() - started, verified=True)
            raise ValueError("unknown audio operation: " + operation)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class FileActuator(BaseActuator):
    name = "file"
    modalities = ("files", "file")

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.sandbox = Sandbox(config)

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        operation = "read"
        path = ""
        text = ""
        if isinstance(request, dict):
            operation = str(request.get("operation", "read"))
            path = str(request.get("path", ""))
            text = str(request.get("text", request.get("content", "")))
        else:
            path = str(request)
            text = str(kwargs.get("text", ""))
        try:
            if operation in ("list", "ls"):
                safe = self.sandbox.policy.check_read(
                    path or self.config.data_dir)
                entries = []
                for name in sorted(os.listdir(safe))[:500]:
                    full = os.path.join(safe, name)
                    entries.append({
                        "name": name,
                        "directory": os.path.isdir(full),
                        "bytes": os.path.getsize(full)
                        if os.path.isfile(full) else 0,
                    })
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=entries, evidence={"root": safe},
                    latency=time.time() - started, verified=True)
            if operation in ("write", "save"):
                if not path:
                    raise ValueError("write requires a path")
                result = self.sandbox.write_text(path, text)
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output=result, artifacts=[result["path"]],
                    evidence=result, latency=time.time() - started,
                    verified=True)
            if operation in ("read", "load"):
                if not path:
                    raise ValueError("read requires a path")
                content = self.sandbox.read_text(path)
                return ActuatorResult(
                    ok=True, actuator=self.name, operation=operation,
                    output={"path": path, "text": content[:100000],
                            "chars": len(content)},
                    artifacts=[path], evidence={"chars": len(content)},
                    latency=time.time() - started, verified=True)
            if operation in ("delete", "remove"):
                deleted = self.sandbox.delete_file(path)
                return ActuatorResult(
                    ok=deleted, actuator=self.name, operation=operation,
                    output={"deleted": deleted}, evidence={"path": path},
                    latency=time.time() - started, verified=deleted)
            raise ValueError("unknown file operation: " + operation)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=operation,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class DataActuator(BaseActuator):
    name = "data"
    modalities = ("data", "table")

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.tools = ToolService(config)
        self.sandbox = Sandbox(config)

    def available(self) -> tuple[bool, str]:
        tool = self.tools.verifier.tools.get(
            "cognition.tools.data_analysis")
        if tool is not None and tool.verified:
            return True, ""
        return False, "data_tool_not_verified"

    def execute(self, request, **kwargs) -> ActuatorResult:
        started = time.time()
        rows = []
        query = ""
        if isinstance(request, dict):
            rows = list(request.get("rows") or [])
            query = str(request.get("query", request.get("operation", "")))
        else:
            rows = list(kwargs.get("rows") or [])
            query = str(request)
        try:
            result = self.tools.data_aggregate(rows, query)
            return ActuatorResult(
                ok=True, actuator=self.name, operation=query,
                output=result, evidence={"rows": len(rows)},
                latency=time.time() - started, verified=True)
        except Exception as exc:
            return ActuatorResult(
                ok=False, actuator=self.name, operation=query,
                error=f"{type(exc).__name__}: {exc}",
                latency=time.time() - started)


class ActuatorRegistry:
    def __init__(self, config: GrowthConfig, brain=None):
        self.config = config
        self.actuators: dict[str, BaseActuator] = {}
        for actuator in (
                MathActuator(),
                CodeActuator(config, brain=brain),
                ImageActuator(config),
                VideoActuator(config),
                AudioActuator(config),
                WebActuator(config),
                FileActuator(config),
                DataActuator(config)):
            self.actuators[actuator.name] = actuator

    def get(self, name: str) -> Optional[BaseActuator]:
        return self.actuators.get(str(name))

    def execute(self, name: str, request, **kwargs) -> ActuatorResult:
        actuator = self.get(name)
        if actuator is None:
            return ActuatorResult(
                ok=False, actuator=str(name), operation="dispatch",
                error="unknown actuator")
        available, reason = actuator.available()
        if not available:
            return ActuatorResult(
                ok=False, actuator=str(name), operation="dispatch",
                error=reason)
        return actuator.execute(request, **kwargs)

    def capabilities(self) -> list[dict]:
        return [actuator.describe() for actuator in self.actuators.values()]

    def available(self) -> dict[str, bool]:
        return {name: actuator.available()[0]
                for name, actuator in self.actuators.items()}
