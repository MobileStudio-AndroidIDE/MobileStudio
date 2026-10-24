"""Virtual sensor synthesis from existing measurement channels.

A physical sensor needs a transducer and a driver.  This module cannot
invent hardware.  What it does is invent a new *virtual sensor*: a named,
persisted combination of existing measurement channels (pixels, waveform,
text or file bytes) with a fixed output transform.  Each adopted sensor
is verified only for determinism, finiteness and measurable dispersion on
supplied samples; it is not claimed to have external semantic validity.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from cognition.util import (atomic_write_json, read_json, stable_hash)


def _as_text(source) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8", errors="replace")
    text = str(source or "")
    try:
        if os.path.exists(text) and os.path.isfile(text) and \
                os.path.getsize(text) <= 2_000_000:
            with open(text, "r", encoding="utf-8",
                      errors="replace") as handle:
                return handle.read()
    except (OSError, ValueError):
        pass
    return text


def _image_array(source) -> np.ndarray:
    from media.image import analysis
    if isinstance(source, np.ndarray):
        return analysis._to_rgb_float(source)
    return analysis.load_image(source)


def _audio_signal(source) -> tuple[np.ndarray, int]:
    from media.audio.wav import read_wav
    data, rate, _channels = read_wav(str(source))
    signal = np.asarray(data, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size:
        signal = signal / 32767.0
    return signal, int(rate)


def _text_entropy(text: str) -> float:
    if not text:
        return 0.0
    values, counts = np.unique(
        np.frombuffer(text.encode("utf-8", errors="replace"),
                      dtype=np.uint8), return_counts=True)
    total = float(counts.sum())
    probabilities = counts.astype(np.float64) / total
    return float(-(probabilities * np.log2(probabilities)).sum() /
                 max(1.0, math.log2(max(2.0, total))))


@dataclass(frozen=True)
class SensorChannel:
    name: str
    modalities: tuple[str, ...]
    description: str
    fn: Callable


def _image_brightness(source) -> list[float]:
    rgb = _image_array(source)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return [float(luma.mean() / 255.0)]


def _image_contrast(source) -> list[float]:
    rgb = _image_array(source)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return [float(luma.std() / 255.0)]


def _image_edge(source) -> list[float]:
    rgb = _image_array(source)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if luma.shape[1] > 1:
        horizontal = np.abs(np.diff(luma, axis=1)).mean()
    else:
        horizontal = 0.0
    if luma.shape[0] > 1:
        vertical = np.abs(np.diff(luma, axis=0)).mean()
    else:
        vertical = 0.0
    return [float((horizontal + vertical) / (2.0 * 255.0))]


def _image_colorfulness(source) -> list[float]:
    rgb = _image_array(source)[:, :, :3].astype(np.float32)
    return [float(rgb[:, :, index].std() / 255.0)
            for index in range(3)]


def _image_saturation(source) -> list[float]:
    rgb = _image_array(source)[:, :, :3].astype(np.float32)
    high = np.max(rgb, axis=2)
    low = np.min(rgb, axis=2)
    return [float(np.mean((high - low) / np.maximum(1.0, high)))]


def _image_entropy(source) -> list[float]:
    rgb = _image_array(source)[:, :, :3].astype(np.float32)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hist, _ = np.histogram(luma, bins=32, range=(0, 255))
    probabilities = hist.astype(np.float64) / max(1, int(hist.sum()))
    probabilities = probabilities[probabilities > 0]
    return [float(-(probabilities * np.log2(probabilities)).sum() / 5.0)]


def _image_symmetry(source) -> list[float]:
    rgb = _image_array(source)[:, :, :3].astype(np.float32)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mirror = luma[:, ::-1]
    return [float(1.0 - np.abs(luma - mirror).mean() / 255.0)]


def _image_size(source) -> list[float]:
    rgb = _image_array(source)
    return [float(math.log1p(rgb.shape[0])),
            float(math.log1p(rgb.shape[1]))]


def _audio_rms(source) -> list[float]:
    signal, _ = _audio_signal(source)
    return [float(np.sqrt(np.mean(signal ** 2))) if signal.size else 0.0]


def _audio_peak(source) -> list[float]:
    signal, _ = _audio_signal(source)
    return [float(np.max(np.abs(signal))) if signal.size else 0.0]


def _audio_zcr(source) -> list[float]:
    signal, _ = _audio_signal(source)
    if signal.size < 2:
        return [0.0]
    return [float(np.mean(np.abs(np.diff(np.signbit(signal)))) )]


def _audio_centroid(source) -> list[float]:
    signal, rate = _audio_signal(source)
    if signal.size < 4 or rate <= 0:
        return [0.0]
    spectrum = np.abs(np.fft.rfft(signal))
    if not np.any(spectrum):
        return [0.0]
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / rate)
    return [float((frequencies * spectrum).sum() /
                  spectrum.sum() / rate)]


def _audio_duration(source) -> list[float]:
    signal, rate = _audio_signal(source)
    return [float(signal.size / max(1, rate))]


def _audio_silence(source) -> list[float]:
    signal, _ = _audio_signal(source)
    return [float(np.mean(np.abs(signal) < 1e-3))
            if signal.size else 1.0]


def _text_length(source) -> list[float]:
    return [float(math.log1p(len(_as_text(source))))]


def _text_unique_ratio(source) -> list[float]:
    text = _as_text(source)
    return [float(len(set(text)) / len(text)) if text else 0.0]


def _text_digit_ratio(source) -> list[float]:
    text = _as_text(source)
    return [float(sum(ch.isdigit() for ch in text) / len(text))
            if text else 0.0]


def _text_space_ratio(source) -> list[float]:
    text = _as_text(source)
    return [float(sum(ch.isspace() for ch in text) / len(text))
            if text else 0.0]


def _text_uppercase_ratio(source) -> list[float]:
    text = _as_text(source)
    return [float(sum(ch.isupper() for ch in text) / len(text))
            if text else 0.0]


def _text_entropy_channel(source) -> list[float]:
    return [_text_entropy(_as_text(source))]


def _text_lines(source) -> list[float]:
    return [float(math.log1p(_as_text(source).count("\n") + 1))]


def _file_size(source) -> list[float]:
    path = str(source)
    try:
        return [float(math.log1p(os.path.getsize(path)))]
    except OSError:
        return [float(math.log1p(len(_as_text(source).encode("utf-8"))))]


def _file_binary_ratio(source) -> list[float]:
    try:
        with open(str(source), "rb") as handle:
            data = handle.read(8192)
    except OSError:
        data = _as_text(source).encode("utf-8", errors="replace")[:8192]
    if not data:
        return [0.0]
    printable = sum(32 <= byte < 127 or byte in (9, 10, 13)
                    for byte in data)
    return [float(1.0 - printable / len(data))]


CHANNELS: dict[str, SensorChannel] = {
    "image.brightness": SensorChannel(
        "image.brightness", ("image",), "mean luma",
        _image_brightness),
    "image.contrast": SensorChannel(
        "image.contrast", ("image",), "luma standard deviation",
        _image_contrast),
    "image.edge": SensorChannel(
        "image.edge", ("image",), "mean absolute luma gradient",
        _image_edge),
    "image.colorfulness": SensorChannel(
        "image.colorfulness", ("image",), "per-channel spread",
        _image_colorfulness),
    "image.saturation": SensorChannel(
        "image.saturation", ("image",), "mean HSV-style saturation",
        _image_saturation),
    "image.entropy": SensorChannel(
        "image.entropy", ("image",), "normalized luma entropy",
        _image_entropy),
    "image.symmetry": SensorChannel(
        "image.symmetry", ("image",), "mirror consistency",
        _image_symmetry),
    "image.size": SensorChannel(
        "image.size", ("image",), "log width and height",
        _image_size),
    "audio.rms": SensorChannel(
        "audio.rms", ("audio",), "root mean square", _audio_rms),
    "audio.peak": SensorChannel(
        "audio.peak", ("audio",), "absolute peak", _audio_peak),
    "audio.zcr": SensorChannel(
        "audio.zcr", ("audio",), "zero crossing rate", _audio_zcr),
    "audio.centroid": SensorChannel(
        "audio.centroid", ("audio",), "spectral centroid", _audio_centroid),
    "audio.duration": SensorChannel(
        "audio.duration", ("audio",), "duration in seconds",
        _audio_duration),
    "audio.silence": SensorChannel(
        "audio.silence", ("audio",), "silence ratio", _audio_silence),
    "text.length": SensorChannel(
        "text.length", ("text", "code", "file"), "log character count",
        _text_length),
    "text.unique_ratio": SensorChannel(
        "text.unique_ratio", ("text", "code", "file"),
        "unique character ratio", _text_unique_ratio),
    "text.digit_ratio": SensorChannel(
        "text.digit_ratio", ("text", "code", "file"),
        "digit ratio", _text_digit_ratio),
    "text.space_ratio": SensorChannel(
        "text.space_ratio", ("text", "code", "file"),
        "whitespace ratio", _text_space_ratio),
    "text.uppercase_ratio": SensorChannel(
        "text.uppercase_ratio", ("text", "code", "file"),
        "uppercase ratio", _text_uppercase_ratio),
    "text.entropy": SensorChannel(
        "text.entropy", ("text", "code", "file"),
        "normalized byte entropy", _text_entropy_channel),
    "text.lines": SensorChannel(
        "text.lines", ("text", "code", "file"),
        "log line count", _text_lines),
    "file.size": SensorChannel(
        "file.size", ("file",), "log file size", _file_size),
    "file.binary_ratio": SensorChannel(
        "file.binary_ratio", ("file",), "non-printable byte ratio",
        _file_binary_ratio),
}


def _compose(values: list[list[float]], transform: str) -> list[float]:
    if not values:
        return []
    if transform == "summary":
        output = []
        for value in values:
            array = np.asarray(value, dtype=np.float64)
            output.extend([
                float(array.mean()) if array.size else 0.0,
                float(array.std()) if array.size else 0.0,
                float(array.min()) if array.size else 0.0,
                float(array.max()) if array.size else 0.0,
            ])
        return output
    if transform == "ratio":
        means = [float(np.mean(value)) if value else 0.0
                 for value in values]
        return [float(means[i] / means[j])
                if abs(means[j]) > 1e-9 else 0.0
                for i, j in itertools.permutations(
                    range(len(means)), 2) if i < j]
    return [float(item) for value in values for item in value]


@dataclass
class SensorSpec:
    name: str
    modality: str
    channels: list[str]
    transform: str
    description: str = ""
    kind: str = "virtual"
    created_at: float = field(default_factory=time.time)

    def signature(self) -> tuple:
        return (str(self.modality), tuple(self.channels),
                str(self.transform))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "modality": self.modality,
            "channels": list(self.channels),
            "transform": self.transform,
            "description": self.description,
            "kind": self.kind,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SensorSpec":
        return cls(
            name=str(data.get("name") or ""),
            modality=str(data.get("modality") or "text"),
            channels=[str(item) for item in data.get("channels") or []],
            transform=str(data.get("transform") or "concat"),
            description=str(data.get("description") or ""),
            kind=str(data.get("kind") or "virtual"),
            created_at=float(data.get("created_at", time.time())),
        )


class SensorRegistry:
    FILE = "sensor_library.json"

    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, self.FILE)
        self.sensors: dict[str, SensorSpec] = {}
        self._load()

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("sensors") or []:
            try:
                spec = SensorSpec.from_dict(row)
            except Exception:
                continue
            if spec.name:
                self.sensors[spec.name] = spec

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "sensors": [spec.to_dict()
                        for spec in self.sensors.values()],
        })

    def register(self, spec: SensorSpec) -> bool:
        if not spec.channels:
            return False
        for existing in self.sensors.values():
            if existing.signature() == spec.signature():
                return False
        if not spec.name:
            spec.name = "sensor_" + stable_hash(
                spec.signature(), length=10)
        self.sensors[spec.name] = spec
        self.save()
        return True

    def get(self, name: str) -> Optional[SensorSpec]:
        return self.sensors.get(str(name))

    def all(self) -> list[SensorSpec]:
        return sorted(self.sensors.values(),
                      key=lambda row: (row.modality, row.name))

    def read(self, name: str, source) -> dict:
        spec = self.get(name)
        if spec is None:
            raise ValueError("unknown sensor: " + str(name))
        return self._read_spec(spec, source)

    def _read_spec(self, spec: SensorSpec, source) -> dict:
        values = []
        for channel_name in spec.channels:
            channel = CHANNELS.get(channel_name)
            if channel is None:
                raise ValueError("unknown sensor channel: " +
                                 str(channel_name))
            if spec.modality not in channel.modalities:
                raise ValueError(
                    "channel %s does not support modality %s" %
                    (channel_name, spec.modality))
            value = [float(item) for item in channel.fn(source)]
            if not all(math.isfinite(item) for item in value):
                raise ValueError("non-finite sensor channel: " +
                                 channel_name)
            values.append(value)
        features = _compose(values, spec.transform)
        if not features or not all(math.isfinite(item)
                                   for item in features):
            raise ValueError("sensor produced no finite features")
        return {
            "sensor": spec.name,
            "modality": spec.modality,
            "channels": list(spec.channels),
            "transform": spec.transform,
            "features": [round(float(item), 9) for item in features],
            "dimension": len(features),
        }

    def stats(self) -> dict:
        return {
            "sensors": len(self.sensors),
            "modalities": sorted({spec.modality
                                  for spec in self.sensors.values()}),
            "names": sorted(self.sensors)[-20:],
        }


@dataclass
class SensorInventionReport:
    session_id: str
    started_at: float
    modality: str
    samples: int
    cycles_requested: int
    cycles: int = 0
    candidates_generated: int = 0
    candidates_verified: int = 0
    candidates_adopted: int = 0
    sensor: str = ""
    candidates: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    stopped_reason: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "modality": self.modality,
            "samples": self.samples,
            "cycles_requested": self.cycles_requested,
            "cycles": self.cycles,
            "candidates_generated": self.candidates_generated,
            "candidates_verified": self.candidates_verified,
            "candidates_adopted": self.candidates_adopted,
            "sensor": self.sensor,
            "candidates": list(self.candidates),
            "duration_s": round(self.duration_s, 4),
            "stopped_reason": self.stopped_reason,
            "limitations": list(self.limitations),
        }


class SensorForge:
    def __init__(self, config, registry: Optional[SensorRegistry] = None,
                 seed: Optional[int] = None):
        self.config = config
        self.registry = registry or SensorRegistry(config.data_dir)
        self.rng = random.Random(
            config.seed if seed is None else int(seed))
        self.path = os.path.join(config.data_dir, "invention.json")

    def channels_for(self, modality: str) -> list[str]:
        return sorted(name for name, channel in CHANNELS.items()
                      if modality in channel.modalities)

    def propose(self, modality: str,
                limit: int = 12) -> list[SensorSpec]:
        channels = self.channels_for(modality)
        if not channels:
            return []
        seen = {spec.signature()
                for spec in self.registry.all()}
        specs: list[SensorSpec] = []
        max_size = min(3, len(channels))
        for size in range(2, max_size + 1):
            for subset in itertools.combinations(channels, size):
                for transform in ("concat", "summary", "ratio"):
                    if transform == "ratio" and size < 2:
                        continue
                    signature = (modality, tuple(subset), transform)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    specs.append(SensorSpec(
                        name="candidate_" + stable_hash(
                            signature, length=10),
                        modality=modality,
                        channels=list(subset),
                        transform=transform,
                        description=("virtual %s sensor: %s -> %s" %
                                     (modality, ", ".join(subset),
                                      transform))))
        self.rng.shuffle(specs)
        specs.sort(key=lambda spec: (len(spec.channels),
                                     spec.transform != "concat"))
        return specs[:max(1, int(limit))]

    def _verify(self, spec: SensorSpec,
                samples: list) -> tuple[bool, float, dict]:
        vectors = []
        for source in samples:
            try:
                first = self.registry._read_spec(spec, source)
                second = self.registry._read_spec(spec, source)
            except Exception as exc:
                return False, 0.0, {
                    "reason": "read: " + type(exc).__name__ + ": " +
                              str(exc)[:160]}
            if first["features"] != second["features"]:
                return False, 0.0, {"reason": "non_deterministic"}
            vectors.append(first["features"])
        if len(vectors) < 2:
            return False, 0.0, {"reason": "need_two_samples"}
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] > 64:
            return False, 0.0, {"reason": "invalid_dimension"}
        if not np.isfinite(matrix).all():
            return False, 0.0, {"reason": "non_finite"}
        scale = float(np.mean(np.abs(matrix))) + 1e-9
        spread = float(np.mean(np.std(matrix, axis=0))) / scale
        if spread <= 1e-7:
            return False, 0.0, {"reason": "constant_features"}
        distances = []
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                distances.append(float(np.mean(
                    np.abs(matrix[left] - matrix[right]))) / scale)
        contrast = float(np.mean(distances)) if distances else 0.0
        if contrast <= 1e-7:
            return False, 0.0, {"reason": "no_contrast"}
        score = spread + 0.25 * contrast
        return True, score, {
            "dimension": int(matrix.shape[1]),
            "spread": round(spread, 6),
            "contrast": round(contrast, 6),
            "deterministic": True,
            "finite": True,
        }

    def run(self, modality: str, samples: list,
            cycles: int = 1, candidates: int = 12,
            time_budget: Optional[float] = None) -> SensorInventionReport:
        samples = [sample for sample in (samples or []) if sample is not None]
        budget = float(time_budget if time_budget is not None else
                       max(1.0, float(getattr(
                           self.config, "invention_seconds", 20.0))))
        started = time.time()
        report = SensorInventionReport(
            session_id="sensor_" + stable_hash(
                time.time(), modality, self.config.seed, length=12),
            started_at=started,
            modality=modality,
            samples=len(samples),
            cycles_requested=max(0, int(cycles)),
            limitations=[
                "A physical sensor requires a transducer and driver; this "
                "forge invents virtual sensors only.",
                "A virtual sensor is a named composition of existing "
                "measurement channels, not a new physical modality.",
                "Verification checks determinism, finiteness and dispersion "
                "on the supplied samples; it does not establish external "
                "semantic validity.",
            ])
        if len(samples) < 2:
            report.stopped_reason = "need_two_samples"
            report.duration_s = time.time() - started
            self._save(report)
            return report
        proposed = self.propose(modality, limit=max(1, int(candidates)))
        report.candidates_generated = len(proposed)
        best = None
        best_score = 0.0
        for cycle in range(1, max(0, int(cycles)) + 1):
            if time.time() - started >= budget:
                report.stopped_reason = "time_budget"
                break
            report.cycles = cycle
            for spec in proposed:
                if time.time() - started >= budget:
                    report.stopped_reason = "time_budget"
                    break
                ok, score, evidence = self._verify(spec, samples)
                row = spec.to_dict()
                row["status"] = "verified" if ok else "rejected"
                row["score"] = round(score, 6)
                row["evidence"] = evidence
                report.candidates.append(row)
                if ok:
                    report.candidates_verified += 1
                    if score > best_score:
                        best, best_score = spec, score
            if best is not None:
                break
        if best is not None:
            best.name = "sensor_" + stable_hash(
                best.signature(), length=10)
            if self.registry.register(best):
                report.candidates_adopted = 1
                report.sensor = best.name
        if not report.stopped_reason:
            report.stopped_reason = ("adopted" if report.sensor
                                     else "no_verified_sensor")
        report.duration_s = time.time() - started
        self._save(report)
        return report

    def _save(self, report: SensorInventionReport) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            data = {}
        sessions = data.get("sensor_sessions")
        if not isinstance(sessions, list):
            sessions = []
        sessions.append(report.to_dict())
        data["sensor_sessions"] = sessions[-50:]
        data["last_sensor"] = report.to_dict()
        atomic_write_json(self.path, data)
