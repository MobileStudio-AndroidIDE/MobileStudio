"""Verified tool capabilities: web, video, image, tabular data.

A tool is registered only after fresh, independent checks pass.  A
failed or skipped probe is persisted as such and is not advertised as a
capability.  This keeps the existing RENA-BR tool stack honest: the
capability registry describes what was demonstrated in this environment,
not what the source tree merely contains.
"""
from __future__ import annotations

import ast
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.config import GrowthConfig
from cognition.domains import DataAnalysisDomain, _DATA_CODE, _data_args
from cognition.sandbox import Sandbox
from cognition.util import atomic_write_json, json_safe, read_json, stable_hash


@dataclass
class ToolCapability:
    name: str
    label: str
    category: str
    description: str
    status: str = "unverified"
    checks: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "status": self.status,
            "checks": self.checks,
            "limitations": self.limitations,
            "evidence": self.evidence,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCapability":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


class ToolVerifier:
    FILE = "tools.json"

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.config.ensure_dirs()
        self.path = os.path.join(self.config.data_dir, self.FILE)
        self.sandbox = Sandbox(config)
        self.tools: dict[str, ToolCapability] = {}
        self.history: list[dict] = []
        self._load()

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("tools") or []:
            try:
                tool = ToolCapability.from_dict(row)
            except Exception:
                continue
            if tool.name:
                self.tools[tool.name] = tool
        self.history = list(data.get("history") or [])[-20:]

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "tools": [t.to_dict() for t in self.tools.values()],
            "history": self.history[-20:],
        })

    def verified(self) -> list[ToolCapability]:
        return sorted([t for t in self.tools.values() if t.verified],
                      key=lambda t: t.name)

    def summarize(self) -> dict:
        counts = {"verified": 0, "failed": 0, "skipped": 0, "unverified": 0}
        for tool in self.tools.values():
            counts[tool.status] = counts.get(tool.status, 0) + 1
        return {
            "tools": len(self.tools),
            "counts": counts,
            "verified": [t.name for t in self.verified()],
        }

    @staticmethod
    def _check(name: str, passed: bool, detail: str = "", **data) -> dict:
        row = {"name": name, "passed": bool(passed), "detail": detail}
        row.update({key: json_safe(value) for key, value in data.items()})
        return row

    def _store(self, name: str, label: str, category: str,
               description: str, checks: list,
               limitations: list, evidence: dict,
               status: str = "") -> ToolCapability:
        if not status:
            status = "verified" if checks and all(
                c.get("passed") for c in checks) else "failed"
        tool = ToolCapability(
            name=name, label=label, category=category,
            description=description, status=status, checks=checks,
            limitations=limitations, evidence=json_safe(evidence),
            updated_at=time.time())
        self.tools[name] = tool
        return tool

    def _web(self) -> ToolCapability:
        name = "cognition.tools.web_research"
        description = ("Retrieve external evidence from Wikipedia (summary "
                       "API) with a source URL and a non-empty extract.")
        limitations = [
            "Wikipedia summary retrieval is not a general web crawl.",
            "The DuckDuckGo endpoint is probed but may be rate-limited or "
            "blocked; it is not advertised unless it returns results.",
        ]
        if not self.config.allow_network:
            return self._store(
                name, "Web research", "web", description, [],
                limitations + ["Network access is disabled by config."],
                {}, status="skipped")
        checks = []
        evidence = {"topics": []}
        try:
            from brain.web_search import (search_web, wikipedia_summary)
        except Exception as exc:
            checks.append(self._check(
                "import", False, f"{type(exc).__name__}: {exc}"))
            return self._store(name, "Web research", "web", description,
                               checks, limitations, evidence)
        for topic in ("Photosynthesis", "Alan Turing"):
            try:
                result = wikipedia_summary(topic)
            except Exception as exc:
                result = None
                checks.append(self._check(
                    "summary:" + topic, False,
                    f"{type(exc).__name__}: {exc}"))
            else:
                passed = bool(result and result.url.startswith("http")
                              and len(result.snippet) >= 40)
                checks.append(self._check(
                    "summary:" + topic, passed,
                    "source url + extract" if passed else "missing evidence",
                    url=getattr(result, "url", ""),
                    chars=len(getattr(result, "snippet", "") or "")))
                evidence["topics"].append({
                    "topic": topic,
                    "url": getattr(result, "url", ""),
                    "chars": len(getattr(result, "snippet", "") or ""),
                })
        try:
            general = search_web("Termux terminal emulator", limit=3)
            evidence["general_results"] = len(general)
            checks.append(self._check(
                "general_search", bool(general),
                "endpoint returned results" if general
                else "endpoint returned no results; not advertised"))
            if not general:
                checks[-1]["informational"] = True
        except Exception as exc:
            evidence["general_results"] = 0
            checks.append(self._check(
                "general_search", False,
                f"{type(exc).__name__}: {exc}; informational"))
            checks[-1]["informational"] = True
        required = [c for c in checks if not c.get("informational")]
        status = "verified" if required and all(
            c["passed"] for c in required) else "failed"
        return self._store(name, "Web research", "web", description,
                           checks, limitations, evidence, status=status)

    def _video(self) -> ToolCapability:
        name = "cognition.tools.video_pipeline"
        description = ("Plan, render and encode a small MJPEG AVI, parse "
                       "the artifact back, decode frames, and compute frame "
                       "statistics and motion from the pixels.")
        limitations = [
            "Rendering is procedural (creative planner + NumPy raster), not "
            "a learned video model.",
            "Analysis estimates motion by frame differencing; it does not "
            "recognize objects.",
        ]
        checks = []
        evidence = {"videos": []}
        try:
            from media.video.analyze import analyze_video
            from media.video.generate import generate_video
        except Exception as exc:
            checks.append(self._check(
                "import", False, f"{type(exc).__name__}: {exc}"))
            return self._store(name, "Video pipeline", "video", description,
                               checks, limitations, evidence)
        out_dir = os.path.join(
            self.config.artifact_dir,
            "tool_checks_" + stable_hash(time.time(), length=8))
        os.makedirs(out_dir, exist_ok=True)
        specs = [
            ("a red sphere over the sea", 11, (64, 48)),
            ("a blue orb in a dark sky", 12, (64, 48)),
        ]
        for request, seed, (width, height) in specs:
            path = os.path.join(out_dir, f"video_{seed}.avi")
            try:
                meta = generate_video(
                    request, path, duration=0.5, fps=4, width=width,
                    height=height, samples=1, seed=seed)
                analysis = analyze_video(path, max_analysis_frames=8)
            except Exception as exc:
                checks.append(self._check(
                    f"video:{seed}", False, f"{type(exc).__name__}: {exc}"))
                continue
            passed = (
                meta.get("frame_count", 0) >= 2
                and meta.get("width") == width
                and meta.get("height") == height
                and os.path.getsize(path) > 1000
                and analysis.get("format") == "mjpeg_avi"
                and analysis.get("frame_count") == meta.get("frame_count")
                and analysis.get("analyzed_frames", 0) >= 1
            )
            checks.append(self._check(
                f"video:{seed}", passed,
                "encoded artifact parsed and analyzed",
                frames=meta.get("frame_count"),
                bytes=meta.get("bytes"),
                analyzed=analysis.get("analyzed_frames"),
                mean_diff=analysis.get("motion", {}).get("mean_diff"),
                path=path))
            evidence["videos"].append({
                "seed": seed,
                "path": path,
                "frames": meta.get("frame_count"),
                "bytes": meta.get("bytes"),
                "mean_diff": analysis.get("motion", {}).get("mean_diff"),
            })
        status = "verified" if len(checks) >= 2 and all(
            c["passed"] for c in checks) else "failed"
        return self._store(name, "Video pipeline", "video", description,
                           checks, limitations, evidence, status=status)

    def _image(self) -> ToolCapability:
        name = "cognition.tools.image_pipeline"
        description = ("Render synthetic images, encode/decode PNG, and "
                       "measure regions, palette and mean color from pixels.")
        limitations = [
            "Image understanding is classical pixel analysis, not a "
            "learned open-vocabulary vision model.",
        ]
        checks = []
        evidence = {"images": []}
        try:
            from brain import image_io as ii
        except Exception as exc:
            checks.append(self._check(
                "import", False, f"{type(exc).__name__}: {exc}"))
            return self._store(name, "Image pipeline", "image", description,
                               checks, limitations, evidence)
        out_dir = os.path.join(
            self.config.artifact_dir,
            "tool_checks_" + stable_hash(time.time(), "image", length=8))
        os.makedirs(out_dir, exist_ok=True)
        specs = [
            ("rectangle", (20, 30, 200), (220, 30, 40), "rectangle"),
            ("circle", (10, 10, 10), (40, 200, 80), "circle"),
        ]
        for index, (shape, background, color, expected_shape) in enumerate(
                specs):
            path = os.path.join(out_dir, f"image_{index}.png")
            try:
                image = ii.new_image(64, 64, background)
                if shape == "circle":
                    ii.draw_circle(image, 32, 32, 13, color, filled=True)
                else:
                    ii.draw_rect(image, 16, 16, 24, 24, color, filled=True)
                ii.save_image(image, path)
                described = ii.describe_image(path)
            except Exception as exc:
                checks.append(self._check(
                    f"image:{shape}", False,
                    f"{type(exc).__name__}: {exc}"))
                continue
            regions = described.get("regions") or []
            top = max(regions, key=lambda r: r.get("area", 0)) if regions \
                else {}
            mean = tuple(described.get("mean_color") or ())
            passed = (
                described.get("width") == 64
                and described.get("height") == 64
                and bool(regions)
                and top.get("shape") == expected_shape
                and top.get("area", 0) > 100
                and len(mean) == 3
            )
            checks.append(self._check(
                f"image:{shape}", passed,
                "PNG round-trip + region measurement",
                shape=top.get("shape"), area=top.get("area"),
                mean_color=list(mean), path=path))
            evidence["images"].append({
                "shape": shape, "path": path,
                "detected": top.get("shape"), "area": top.get("area"),
                "mean_color": list(mean),
            })
        status = "verified" if len(checks) >= 2 and all(
            c["passed"] for c in checks) else "failed"
        return self._store(name, "Image pipeline", "image", description,
                           checks, limitations, evidence, status=status)

    def _data(self) -> ToolCapability:
        name = "cognition.tools.data_analysis"
        description = ("Execute aggregate implementations over tabular rows "
                       "in the sandbox and compare them with an independent "
                       "reference computation.")
        limitations = [
            "The tested table operations are numeric aggregates; joins, "
            "SQL planning and data cleaning are not covered.",
        ]
        checks = []
        evidence = {"queries": []}
        rows = [
            {"group": "A", "value": 5},
            {"group": "B", "value": -2},
            {"group": "A", "value": 9},
            {"group": "C", "value": 4},
            {"group": "B", "value": 11},
        ]
        cases = [
            ("sum", [], 27),
            ("mean", [], 27 / 5),
            ("count_gt", [3], 4),
            ("group_sum", ["A"], 14),
            ("median", [], 5),
        ]
        domain = DataAnalysisDomain()
        for query, extra, expected in cases:
            code = _DATA_CODE.get((query, "builtin")) or _DATA_CODE.get(
                (query, "loop"))
            if not code:
                checks.append(self._check(
                    "data:" + query, False, "no implementation"))
                continue
            args = _data_args({"rows": rows, "extra": extra})
            run = self.sandbox.verify_python_function(
                code, "solve", [{"args": args, "expected": expected}])
            reference = domain._reference(query, *args)
            passed = (run.ok and abs(float(reference) - float(expected))
                      <= 1e-9)
            checks.append(self._check(
                "data:" + query, passed,
                "sandboxed aggregate + independent reference",
                expected=expected, reference=reference,
                sandbox=run.reason))
            evidence["queries"].append({
                "query": query, "expected": expected,
                "reference": reference, "sandbox": run.reason,
            })
        status = "verified" if len(checks) >= 3 and all(
            c["passed"] for c in checks) else "failed"
        return self._store(name, "Tabular data analysis", "data",
                           description, checks, limitations, evidence,
                           status=status)

    def run(self, include_network: Optional[bool] = None) -> dict:
        started = time.time()
        old_network = self.config.allow_network
        if include_network is not None:
            self.config.allow_network = bool(include_network)
        try:
            web = self._web()
            video = self._video()
            image = self._image()
            data = self._data()
        finally:
            self.config.allow_network = old_network
        rows = [web, video, image, data]
        summary = {
            "at": time.time(),
            "seconds": round(time.time() - started, 3),
            "verified": [t.name for t in rows if t.verified],
            "failed": [t.name for t in rows if t.status == "failed"],
            "skipped": [t.name for t in rows if t.status == "skipped"],
        }
        self.history.append(summary)
        self.save()
        return summary

    def register(self, registry) -> list[str]:
        names = []
        for tool in self.verified():
            cap = registry.get(tool.name)
            if cap is None:
                registry._add(
                    tool.name, tool.label, tool.description,
                    "cognition_tool")
            else:
                cap.description = f"{tool.label}: {tool.description}"
                cap.category = "cognition_tool"
                cap.is_implemented = True
                cap.is_active = True
            names.append(tool.name)
        return names


class ToolService:
    """Callable wrappers that refuse tools whose verification did not pass."""

    def __init__(self, config: GrowthConfig):
        self.config = config
        self.verifier = ToolVerifier(config)
        self.sandbox = Sandbox(config)

    def _require(self, name: str) -> ToolCapability:
        tool = self.verifier.tools.get(name)
        if tool is None or not tool.verified:
            status = tool.status if tool else "unverified"
            raise PermissionError(f"tool {name} is {status}; run tools.verify")
        return tool

    def web_research(self, topic: str) -> dict:
        self._require("cognition.tools.web_research")
        from brain.web_search import wikipedia_summary
        result = wikipedia_summary(topic)
        return result.to_dict() if result else {}

    def image_describe(self, path: str) -> dict:
        self._require("cognition.tools.image_pipeline")
        from brain import image_io as ii
        return ii.describe_image(path)

    def video_generate(self, request: str, output_path: str = "",
                       **kwargs) -> dict:
        self._require("cognition.tools.video_pipeline")
        from media.video.generate import generate_video
        return generate_video(request, output_path, **kwargs)

    def video_analyze(self, path: str, **kwargs) -> dict:
        self._require("cognition.tools.video_pipeline")
        from media.video.analyze import analyze_video
        return analyze_video(path, **kwargs)

    def data_aggregate(self, rows: list[dict], query: str,
                       extra: Optional[list] = None):
        self._require("cognition.tools.data_analysis")
        code = _DATA_CODE.get((query, "builtin")) or _DATA_CODE.get(
            (query, "loop"))
        if not code:
            raise ValueError(f"unsupported aggregate: {query}")
        args = _data_args({"rows": rows, "extra": list(extra or [])})
        expected = DataAnalysisDomain()._reference(query, *args)
        run = self.sandbox.verify_python_function(
            code, "solve", [{"args": args, "expected": expected}])
        if not run.ok or not run.results:
            raise RuntimeError(f"aggregate sandbox failed: {run.reason}")
        actual = run.results[0].get("actual")
        try:
            return ast.literal_eval(actual)
        except Exception:
            return actual
