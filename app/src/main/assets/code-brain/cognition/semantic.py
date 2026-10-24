"""Multi-modal semantic encoders over the shared VSA substrate.

Everything here is constructed from the input itself: characters,
pixels, AST nodes, expression tokens and graph structure.  There are no
pretrained embeddings.  Text is encoded with exact tokens plus character
n-grams (so spelling variation still overlaps), images with patch
statistics and colour histograms, video as a permuted frame sequence,
code as a typed AST, math as an expression tree and graphs as
role/filler edges.
"""
from __future__ import annotations

import ast
import os
import re
import unicodedata
from typing import Iterable, Optional

import numpy as np

from cognition.vsa import VSA

_TOKEN_RE = re.compile(
    r"[\U0001F000-\U0001FAFF]|[가-힣]+|[A-Za-z]+|"
    r"\d+(?:\.\d+)?|[^\s]", re.UNICODE)


class SemanticEncoder:
    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.last_tokens: list[str] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", str(text or "")).lower()
        return _TOKEN_RE.findall(text)

    def _char_features(self, token: str) -> np.ndarray:
        parts = []
        chars = list(token)
        for char in chars:
            parts.append(self.vsa.symbol("char:" + char))
        for size in (2, 3):
            if len(chars) < size:
                continue
            for index in range(len(chars) - size + 1):
                gram = "".join(chars[index:index + size])
                parts.append(self.vsa.symbol("ngram%d:%s" % (size, gram)))
        if not parts:
            return np.zeros(self.vsa.dimension, dtype=np.float32)
        return self.vsa.bundle(parts)

    def _token_vector(self, token: str) -> np.ndarray:
        exact = self.vsa.symbol("token:" + token)
        chars = self._char_features(token)
        parts = [exact, chars, chars]
        weights = [0.55, 0.8, 0.8]
        if any(unicodedata.category(ch) == "So" for ch in token):
            name = unicodedata.name(token[0], "")
            for word in re.findall(r"[A-Z]+", name):
                parts.append(self.vsa.symbol("emoji:name:" + word.lower()))
                weights.append(0.5)
        return self.vsa.bundle(parts, weights=weights)

    def encode_text(self, text: str, include_order: bool = True) -> np.ndarray:
        tokens = self.tokenize(text)
        self.last_tokens = tokens
        if not tokens:
            return np.zeros(self.vsa.dimension, dtype=np.float32)
        vectors = [self._token_vector(token) for token in tokens]
        parts = []
        for index, vector in enumerate(vectors):
            if include_order and len(tokens) > 1:
                parts.append(self.vsa.bind(
                    vector, self.vsa.symbol("position:%d" % index)))
            else:
                parts.append(vector)
        for index in range(len(vectors) - 1):
            parts.append(self.vsa.bind(
                vectors[index],
                self.vsa.permute(vectors[index + 1], 1)))
            parts.append(self.vsa.bind(
                self.vsa.symbol("bigram:%d" % index),
                vectors[index]))
        return self.vsa.bundle(parts)

    def text_features(self, text: str) -> dict:
        tokens = self.tokenize(text)
        return {
            "tokens": tokens,
            "length": len(tokens),
            "char_length": len(str(text or "")),
            "has_digits": any(token[0].isdigit() for token in tokens
                              if token),
            "has_emoji": any(unicodedata.category(ch) == "So"
                             for ch in str(text or "")),
        }

    @staticmethod
    def _as_array(source) -> np.ndarray:
        from media.image import analysis
        if isinstance(source, np.ndarray):
            return analysis._to_rgb_float(source)
        if hasattr(source, "pixels") and hasattr(source, "width"):
            raw = np.frombuffer(bytes(source.pixels), dtype=np.uint8)
            array = raw.reshape(source.height, source.width,
                                source.channels())
            return analysis._to_rgb_float(array)
        return analysis.load_image(source)

    def encode_image(self, source) -> np.ndarray:
        array = self._as_array(source)
        height, width = array.shape[:2]
        parts = [
            self.vsa.encode_pair("image:width", _size_bucket(width)),
            self.vsa.encode_pair("image:height", _size_bucket(height)),
        ]
        rgb = array[:, :, :3]
        global_mean = rgb.reshape(-1, 3).mean(axis=0)
        for channel, value in zip(("r", "g", "b"), global_mean):
            parts.append(self.vsa.encode_pair(
                "image:mean:" + channel, _color_bucket(value)))
        grid = 4
        for row in range(grid):
            for col in range(grid):
                y0 = row * height // grid
                y1 = max(y0 + 1, (row + 1) * height // grid)
                x0 = col * width // grid
                x1 = max(x0 + 1, (col + 1) * width // grid)
                patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
                if not len(patch):
                    continue
                mean = patch.mean(axis=0)
                spread = patch.std(axis=0)
                luma = patch @ np.array([0.2126, 0.7152, 0.0722],
                                        dtype=np.float32)
                edge = float(np.abs(np.diff(luma)).mean()) \
                    if len(luma) > 1 else 0.0
                role = self.vsa.symbol("image:patch:%d:%d" % (row, col))
                values = [self.vsa.encode_pair(role, _color_bucket(v))
                          for v in mean]
                values.append(self.vsa.encode_pair(
                    role, "spread:" + _size_bucket(float(spread.mean()))))
                values.append(self.vsa.encode_pair(
                    role, "edge:" + _size_bucket(edge)))
                parts.extend(values)
        for channel_index, channel in enumerate(("r", "g", "b")):
            bins = np.histogram(rgb[:, :, channel_index],
                                bins=8, range=(0, 255))[0]
            for index, count in enumerate(bins):
                if count:
                    parts.append(self.vsa.encode_pair(
                        "hist:%s:%d" % (channel, index),
                        _size_bucket(float(count))))
        return self.vsa.bundle(parts)

    def encode_video(self, source, max_frames: int = 8) -> np.ndarray:
        frames = []
        meta = {}
        if isinstance(source, (list, tuple)):
            frames = list(source)[:max_frames]
        else:
            from media.video.analyze import (frame_iterator, inspect_video)
            info = inspect_video(str(source))
            meta = info
            total = max(1, int(info.get("frame_count") or 1))
            step = max(1, total // max(1, max_frames))
            for index, time_value, image in frame_iterator(
                    str(source), every=step):
                frames.append(image)
                if len(frames) >= max_frames:
                    break
        parts = [
            self.vsa.encode_pair("video:frames",
                                 _size_bucket(float(len(frames)))),
        ]
        for key in ("width", "height", "fps", "duration"):
            if key in meta:
                parts.append(self.vsa.encode_pair(
                    "video:" + key, _size_bucket(float(meta[key] or 0))))
        frame_vectors = [self.encode_image(frame) for frame in frames]
        for index, vector in enumerate(frame_vectors):
            role = self.vsa.symbol("video:time:%d" % index)
            parts.append(self.vsa.bind(role, self.vsa.permute(vector, index)))
            if index:
                parts.append(self.vsa.bind(
                    self.vsa.symbol("video:motion:%d" % index),
                    self.vsa.bind(
                        frame_vectors[index - 1],
                        self.vsa.permute(vector, 1))))
        return self.vsa.bundle(parts)

    def encode_code(self, source: str, language: str = "python") -> np.ndarray:
        source = str(source or "")
        parts = [self.vsa.encode_pair("code:language", language)]
        if language in ("python", "py"):
            try:
                tree = ast.parse(source)
            except SyntaxError:
                tree = None
            if tree is not None:
                count = [0]

                def walk(node, role: str, depth: int) -> np.ndarray:
                    if count[0] > 400 or depth > 12:
                        return np.zeros(self.vsa.dimension,
                                        dtype=np.float32)
                    count[0] += 1
                    pieces = [self.vsa.encode_pair(
                        "ast:type", type(node).__name__)]
                    for field_name, value in ast.iter_fields(node):
                        if isinstance(value, ast.AST):
                            pieces.append(self.vsa.bind(
                                self.vsa.symbol("ast:field:" + field_name),
                                walk(value, field_name, depth + 1)))
                        elif isinstance(value, list):
                            children = [walk(child, field_name, depth + 1)
                                        for child in value if isinstance(
                                            child, ast.AST)]
                            if children:
                                pieces.append(self.vsa.bind(
                                    self.vsa.symbol(
                                        "ast:field:" + field_name),
                                    self.vsa.bundle(children)))
                        elif isinstance(value, (str, int, float, bool)):
                            pieces.append(self.vsa.encode_pair(
                                "ast:value:" + field_name, value))
                    return self.vsa.bundle(pieces)

                parts.append(walk(tree, "root", 0))
        for token in self.tokenize(source)[:400]:
            parts.append(self.vsa.encode_pair("code:token", token))
        return self.vsa.bundle(parts)

    def encode_math(self, expression: str) -> np.ndarray:
        text = str(expression or "")
        parts = [self.vsa.encode_pair("math:expr", text.strip())]
        try:
            from cognition.math_engine import parse_expression
            tree = parse_expression(text)
        except Exception:
            tree = None
        if tree is not None:
            parts.append(self._encode_math_node(tree, 0))
        for token in self.tokenize(text)[:200]:
            parts.append(self.vsa.encode_pair("math:token", token))
        return self.vsa.bundle(parts)

    def _encode_math_node(self, node, depth: int) -> np.ndarray:
        if depth > 16:
            return self.vsa.symbol("math:deep")
        pieces = [self.vsa.encode_pair("math:op", node.op)]
        if node.value is not None:
            pieces.append(self.vsa.encode_pair("math:value", node.value))
        for index, child in enumerate(node.args):
            pieces.append(self.vsa.bind(
                self.vsa.symbol("math:arg:%d" % index),
                self._encode_math_node(child, depth + 1)))
        return self.vsa.bundle(pieces)

    def encode_graph(self, nodes: Iterable, edges: Iterable) -> np.ndarray:
        parts = []
        for node in nodes:
            name = node.get("id", node.get("name", node)) \
                if isinstance(node, dict) else node
            attrs = []
            if isinstance(node, dict):
                for key in sorted(node):
                    if key not in ("id", "name"):
                        attrs.append(self.vsa.encode_pair(
                            "graph:attr:" + str(key), node[key]))
            parts.append(self.vsa.bind(
                self.vsa.symbol("graph:node"), self.vsa.symbol(name)))
            parts.extend(attrs)
            self.vsa.register("graph:node:" + str(name))
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get("source", edge.get("from", ""))
                target = edge.get("target", edge.get("to", ""))
                relation = edge.get("relation", edge.get("label", "edge"))
            else:
                source, target = edge[0], edge[1]
                relation = edge[2] if len(edge) > 2 else "edge"
            parts.append(self.vsa.encode_role_fillers([
                ("edge:source", source), ("edge:relation", relation),
                ("edge:target", target)]))
        return self.vsa.bundle(parts)

    def encode(self, modality: str, data, **kwargs) -> np.ndarray:
        modality = str(modality or "text").lower()
        if modality in ("text", "language", "chat"):
            return self.encode_text(data)
        if modality in ("image", "vision", "photo"):
            return self.encode_image(data)
        if modality == "video":
            return self.encode_video(data, **kwargs)
        if modality in ("code", "program"):
            return self.encode_code(data, kwargs.get("language", "python"))
        if modality in ("math", "equation"):
            return self.encode_math(data)
        if modality in ("graph", "network"):
            nodes = kwargs.get("nodes") or []
            edges = kwargs.get("edges") or []
            if isinstance(data, dict):
                nodes = data.get("nodes") or nodes
                edges = data.get("edges") or edges
            return self.encode_graph(nodes, edges)
        return self.encode_features({"modality": modality, "data": str(data)})

    def encode_features(self, features: dict) -> np.ndarray:
        return self.vsa.encode_features(features)

    def decode(self, vector, k: int = 5) -> list[tuple[str, float]]:
        return self.vsa.cleanup(vector, k=k)


def _size_bucket(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    edges = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
    for edge in edges:
        if value <= edge:
            return "bucket:%s" % edge
    return "bucket:large"


def _color_bucket(value) -> str:
    try:
        value = max(0.0, min(255.0, float(value)))
    except (TypeError, ValueError):
        return "color:unknown"
    return "color:%d" % int(value // 32)
