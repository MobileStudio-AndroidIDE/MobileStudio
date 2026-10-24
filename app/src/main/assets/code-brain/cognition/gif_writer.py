"""Pure-Python animated GIF89a writer.

Dependency-free animated GIF output for RENA-BR cognition.  Uses a fixed
6x6x6 RGB color cube (216 colors) and GIF-flavoured LZW compression.  No
ffmpeg, no Pillow, no external library required.

The writer is validated by re-reading the header, frame count and trailer
after writing.  It is used by the video actuator when ffmpeg is absent.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Sequence

import numpy as np


def _build_palette() -> list[tuple[int, int, int]]:
    levels = (0, 51, 102, 153, 204, 255)
    palette = [(r, g, b) for r in levels for g in levels for b in levels]
    palette.extend([(0, 0, 0)] * (256 - len(palette)))
    return palette


_PALETTE = _build_palette()


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer = 0
        self.bits = 0

    def write(self, code: int, width: int) -> None:
        self.buffer |= int(code) << self.bits
        self.bits += width
        while self.bits >= 8:
            self.data.append(self.buffer & 0xFF)
            self.buffer >>= 8
            self.bits -= 8

    def finish(self) -> bytes:
        if self.bits:
            self.data.append(self.buffer & 0xFF)
            self.buffer = 0
            self.bits = 0
        return bytes(self.data)


def _lzw_encode(indices: np.ndarray) -> bytes:
    """GIF-flavoured LZW with clear/end codes and 8-bit symbols.

    The encoder assigns a dictionary entry one code earlier than the
    decoder does (the decoder adds an entry after reading the next code),
    so switching code width exactly at 2**width is one code too early and
    corrupts the stream.  We switch when next_code > 2**width instead.
    """
    values = indices.reshape(-1)
    if values.size == 0:
        return b""

    clear_code = 256
    end_code = 257
    code_width = 9
    next_code = 258
    table: dict[tuple[int, int], int] = {}
    writer = _BitWriter()

    writer.write(clear_code, code_width)
    prefix = int(values[0])
    for raw in values[1:]:
        value = int(raw)
        key = (prefix, value)
        existing = table.get(key)
        if existing is not None:
            prefix = existing
            continue

        writer.write(prefix, code_width)
        if next_code < 4096:
            table[key] = next_code
            next_code += 1
            if next_code > (1 << code_width) and code_width < 12:
                code_width += 1
        else:
            writer.write(clear_code, code_width)
            table.clear()
            next_code = 258
            code_width = 9
        prefix = value

    writer.write(prefix, code_width)
    writer.write(end_code, code_width)
    return writer.finish()


def _blockify(data: bytes) -> bytes:
    output = bytearray()
    for offset in range(0, len(data), 255):
        chunk = data[offset:offset + 255]
        output.append(len(chunk))
        output.extend(chunk)
    output.append(0)
    return bytes(output)


def write_gif(path: str | Path,
              frames: Sequence[np.ndarray],
              fps: int = 12,
              loop: int = 0) -> bool:
    """Write an animated GIF and return True when the file is valid.

    Validation re-reads the header, frame count and trailer.
    """
    if not frames:
        return False
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] < 3:
        return False
    height, width = first.shape[:2]
    if width <= 0 or height <= 0:
        return False
    delay = max(2, int(round(100.0 / max(1, int(fps)))))

    output = bytearray(b"GIF89a")
    output.extend(struct.pack("<HH", width, height))
    # Global color table: 8-bit color resolution, 256 entries.
    output.extend(bytes((0xF7, 0, 0)))
    for color in _PALETTE:
        output.extend(bytes(color))
    # Netscape looping extension.
    output.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01")
    output.extend(struct.pack("<H", max(0, int(loop))))
    output.append(0)

    for frame in frames:
        array = np.asarray(frame)
        if array.shape[0] != height or array.shape[1] != width:
            return False
        indices = quantize_rgb(array)
        # Graphic control extension: disposal 1, no transparency.
        output.extend(b"\x21\xf9\x04\x04")
        output.extend(struct.pack("<H", delay))
        output.extend(b"\x00\x00")
        # Image descriptor: full frame, no local color table.
        output.append(0x2C)
        output.extend(struct.pack("<HHHH", 0, 0, width, height))
        output.append(0)
        output.append(8)
        output.extend(_blockify(_lzw_encode(indices)))

    output.append(0x3B)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(output))

    try:
        raw = target.read_bytes()
    except OSError:
        return False
    return (
        len(raw) > 64
        and raw[:6] in (b"GIF89a", b"GIF87a")
        and raw[-1:] == b"\x3b"
        and raw.count(b"\x21\xf9\x04") >= len(frames)
    )


def quantize_rgb(frame: np.ndarray) -> np.ndarray:
    """Map an HxWx3 array to 216-color-cube indices."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("frame must have shape (height, width, 3)")
    rgb = np.clip(array[..., :3], 0, 255).astype(np.uint16)
    # //51 maps 0..255 to 0..5.  r is the most significant digit,
    # then g, then b — matching _build_palette ordering.
    return ((rgb[..., 0] // 51) * 36 + (rgb[..., 1] // 51) * 6
            + (rgb[..., 2] // 51)).astype(np.uint8)
