"""Image input/output for RENA-BR — self-contained, dependency-free.

IMAGE INPUT
  * PNG decode: color types 0/2/3/4/6, bit depths 1/2/4/8/16
    (palette expanded, 16-bit reduced to 8), all five row filters.
  * BMP decode: uncompressed 24/32-bit.
  * PPM/PGM decode (P5/P6).
  * Vision analysis WITHOUT pretrained models: dominant palette,
    mean color, luminance histogram, connected regions (bounding box,
    area, mean color, geometric shape guess) and a structured
    description.

IMAGE OUTPUT
  * PNG encode (gray/RGB/RGBA, zlib), PPM encode.
  * Raster drawing: points, lines, rectangles, circles, filled shapes,
    a built-in 5x7 bitmap font, and bar/line CHART rendering from data.

This is classical computer vision: everything here is computed from the
pixels themselves — no learned weights, no external services.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# ===================================================================== #
# Raster image
# ===================================================================== #

class ImageError(Exception):
    """Raised when an image cannot be read or written honestly."""


@dataclass
class Image:
    width: int
    height: int
    mode: str                    # "RGB" | "RGBA" | "L"
    pixels: bytearray            # row-major, channels per mode

    def channels(self) -> int:
        return {"L": 1, "RGB": 3, "RGBA": 4}[self.mode]

    def get_pixel(self, x: int, y: int) -> tuple:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return (0, 0, 0, 0)
        c = self.channels()
        i = (y * self.width + x) * c
        p = self.pixels
        if self.mode == "L":
            v = p[i]
            return (v, v, v, 255)
        if self.mode == "RGB":
            return (p[i], p[i + 1], p[i + 2], 255)
        return (p[i], p[i + 1], p[i + 2], p[i + 3])

    def set_pixel(self, x: int, y: int, color) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        c = self.channels()
        i = (y * self.width + x) * c
        p = self.pixels
        if self.mode == "L":
            p[i] = int(color[0])
            return
        p[i] = int(color[0])
        p[i + 1] = int(color[1])
        p[i + 2] = int(color[2])
        if self.mode == "RGBA":
            p[i + 3] = int(color[3]) if len(color) > 3 else 255

    def to_rgb(self) -> "Image":
        if self.mode == "RGB":
            return self
        out = bytearray(self.width * self.height * 3)
        for y in range(self.height):
            for x in range(self.width):
                r, g, b, _ = self.get_pixel(x, y)
                i = (y * self.width + x) * 3
                out[i] = r
                out[i + 1] = g
                out[i + 2] = b
        return Image(self.width, self.height, "RGB", out)

    def copy(self) -> "Image":
        return Image(self.width, self.height, self.mode,
                     bytearray(self.pixels))


def new_image(width: int, height: int, color=(0, 0, 0),
              mode: str = "RGB") -> Image:
    if width <= 0 or height <= 0:
        raise ImageError("image dimensions must be positive")
    img = Image(width, height, mode, bytearray(width * height * (1 if mode == "L" else 3 if mode == "RGB" else 4)))
    img.fill(color)
    return img


def fill(img: Image, color) -> None:
    for y in range(img.height):
        for x in range(img.width):
            img.set_pixel(x, y, color)


Image.fill = fill  # type: ignore[attr-defined]


# ===================================================================== #
# PNG codec
# ===================================================================== #

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(data: bytes) -> Image:
    if data[:8] != _PNG_SIG:
        raise ImageError("not a PNG file (bad signature)")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    palette: list[tuple] = []
    trns: Optional[bytes] = None
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"PLTE":
            palette = [(chunk[i], chunk[i + 1], chunk[i + 2])
                       for i in range(0, len(chunk) - 2, 3)]
        elif ctype == b"tRNS":
            trns = chunk
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    if width == 0 or height == 0:
        raise ImageError("PNG missing IHDR")
    if interlace != 0:
        raise ImageError("interlaced PNG is not supported")
    if color_type not in _CHANNELS:
        raise ImageError(f"unsupported PNG color type {color_type}")
    if bit_depth not in (1, 2, 4, 8, 16):
        raise ImageError(f"unsupported PNG bit depth {bit_depth}")

    raw = zlib.decompress(bytes(idat))
    channels = _CHANNELS[color_type]
    bits_per_pixel = channels * bit_depth
    stride = (width * bits_per_pixel + 7) // 8
    bpp = max(1, bits_per_pixel // 8)

    out = bytearray()
    index_plane = bytearray()   # palette indices (color type 3 only)
    prev = bytearray(stride)
    offset = 0
    for _y in range(height):
        if offset >= len(raw):
            raise ImageError("truncated PNG data")
        ftype = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if ftype == 0:
                pass
            elif ftype == 1:
                line[i] = (line[i] + a) & 0xFF
            elif ftype == 2:
                line[i] = (line[i] + b) & 0xFF
            elif ftype == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
            elif ftype == 4:
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
            else:
                raise ImageError(f"unknown PNG filter {ftype}")
        prev = line

        if bit_depth == 8:
            if color_type == 0:
                out.extend(line)
            elif color_type == 2:
                out.extend(line)
            elif color_type == 3:
                index_plane.extend(line)
                for index in line:
                    if index >= len(palette):
                        raise ImageError("PNG palette index out of range")
                    out.extend(palette[index])
            elif color_type == 4:
                for i in range(0, len(line), 2):
                    out.extend((line[i], line[i], line[i], line[i + 1]))
            else:
                out.extend(line)
        elif bit_depth == 16:
            high = bytearray(line[i] for i in range(0, len(line), 2))
            if color_type == 0:
                out.extend(high)
            elif color_type == 2:
                out.extend(high)
            elif color_type == 4:
                for i in range(0, len(high), 2):
                    out.extend((high[i], high[i], high[i], high[i + 1]))
            else:
                out.extend(high)
        else:  # 1/2/4-bit
            mask = (1 << bit_depth) - 1
            values: list[int] = []
            per_byte = 8 // bit_depth
            for byte in line:
                for shift in range(per_byte - 1, -1, -1):
                    values.append((byte >> (shift * bit_depth)) & mask)
            if color_type == 3:
                index_plane.extend(bytes(values[:width]))
                for index in values[:width]:
                    if index >= len(palette):
                        raise ImageError("PNG palette index out of range")
                    out.extend(palette[index])
            else:
                scale = 255 // mask
                for v in values[:width * height]:
                    g = v * scale
                    out.extend((g, g, g))

    if color_type in (0, 4):
        mode = "L" if color_type == 0 else "RGBA"
    elif color_type == 6:
        mode = "RGBA"
    else:
        mode = "RGB"
    img = Image(width, height, mode, out)
    total = width * height
    if trns and color_type == 3 and palette:
        alphas = [trns[i] if i < len(trns) else 255
                  for i in range(len(palette))]
        if any(a < 255 for a in alphas):
            rgba = bytearray(total * 4)
            for i in range(min(total, len(index_plane))):
                index = index_plane[i]
                rgba[i * 4:i * 4 + 3] = out[i * 3:i * 3 + 3]
                rgba[i * 4 + 3] = alphas[index] if index < len(alphas) \
                    else 255
            return Image(width, height, "RGBA", rgba)
    if trns and color_type == 0 and len(trns) >= 2:
        key = int.from_bytes(trns[:2], "big")
        if bit_depth in (1, 2, 4):
            key = key * (255 // ((1 << bit_depth) - 1))
        elif bit_depth == 16:
            key = trns[0]
        if any(img.pixels[i] == key for i in range(total)):
            rgba = bytearray(total * 4)
            for i in range(total):
                value = img.pixels[i]
                rgba[i * 4:i * 4 + 3] = bytes((value, value, value))
                rgba[i * 4 + 3] = 0 if value == key else 255
            return Image(width, height, "RGBA", rgba)
    if trns and color_type == 2 and len(trns) >= 6:
        # 8-bit samples live in the LOW byte of each 16-bit tRNS value;
        # 16-bit samples were reduced to their high byte above
        if bit_depth == 16:
            key = (trns[0], trns[2], trns[4])
        else:
            key = (trns[1], trns[3], trns[5])
        match = any((img.pixels[i * 3], img.pixels[i * 3 + 1],
                     img.pixels[i * 3 + 2]) == key for i in range(total))
        if match:
            rgba = bytearray(total * 4)
            for i in range(total):
                rgba[i * 4:i * 4 + 3] = out[i * 3:i * 3 + 3]
                rgba[i * 4 + 3] = 0 if (out[i * 3], out[i * 3 + 1],
                                        out[i * 3 + 2]) == key else 255
            return Image(width, height, "RGBA", rgba)
    return img


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def encode_png(img: Image) -> bytes:
    color_type = {"L": 0, "RGB": 2, "RGBA": 6}[img.mode]
    raw = bytearray()
    stride = img.width * img.channels()
    for y in range(img.height):
        raw.append(0)
        raw.extend(img.pixels[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 8,
                       color_type, 0, 0, 0)
    return (_PNG_SIG
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _png_chunk(b"IEND", b""))


# ===================================================================== #
# BMP / PPM codecs
# ===================================================================== #

def decode_bmp(data: bytes) -> Image:
    if data[:2] != b"BM":
        raise ImageError("not a BMP file")
    pixel_offset = struct.unpack("<I", data[10:14])[0]
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size < 40:
        raise ImageError("unsupported BMP header")
    width, height = struct.unpack("<ii", data[18:26])
    bpp = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]
    if compression != 0:
        raise ImageError("compressed BMP is not supported")
    if bpp not in (24, 32):
        raise ImageError(f"unsupported BMP bit depth {bpp}")
    if width <= 0 or height == 0:
        raise ImageError("invalid BMP dimensions")
    flip = height > 0
    height = abs(height)
    channels = bpp // 8
    stride = ((width * channels + 3) // 4) * 4
    mode = "RGB" if bpp == 24 else "RGBA"
    out = bytearray(width * height * channels)
    for row in range(height):
        src_y = (height - 1 - row) if flip else row
        base = pixel_offset + src_y * stride
        for x in range(width):
            i = base + x * channels
            b, g, r = data[i], data[i + 1], data[i + 2]
            a = data[i + 3] if channels == 4 else 255
            j = (row * width + x) * channels
            out[j] = r
            out[j + 1] = g
            out[j + 2] = b
            if channels == 4:
                out[j + 3] = a
    return Image(width, height, mode, out)


def decode_ppm(data: bytes) -> Image:
    if not data[:2] in (b"P5", b"P6"):
        raise ImageError("not a PPM/PGM file")
    binary = data[:2] == b"P6"

    def tokens(blob: bytes):
        current: list[bytes] = []
        i = 0
        while i < len(blob):
            ch = blob[i:i + 1]
            if ch == b"#":
                while i < len(blob) and blob[i:i + 1] not in (b"\n", b"\r"):
                    i += 1
            elif ch.isspace():
                if current:
                    yield b"".join(current)
                    current = []
            else:
                current.append(ch)
            i += 1
        if current:
            yield b"".join(current)

    gen = tokens(data[2:])
    width = int(next(gen))
    height = int(next(gen))
    maxval = int(next(gen))
    if maxval > 255:
        raise ImageError("16-bit PPM is not supported")
    header_end = data.find(bytes(str(maxval), "ascii")) + len(str(maxval))
    body = data[header_end + 1:]
    if binary:
        out = bytearray(body[:width * height * 3])
        return Image(width, height, "RGB", out)
    out = bytearray(body[:width * height])
    return Image(width, height, "L", out)


def decode_image(data: bytes) -> Image:
    if data[:8] == _PNG_SIG:
        return decode_png(data)
    if data[:2] == b"BM":
        return decode_bmp(data)
    if data[:2] in (b"P5", b"P6"):
        return decode_ppm(data)
    raise ImageError("unsupported image format (PNG, BMP, PPM/PGM)")


# ===================================================================== #
# File I/O
# ===================================================================== #

def load_image(path: str) -> Image:
    if not os.path.exists(path):
        raise ImageError(f"image not found: {path}")
    with open(path, "rb") as handle:
        return decode_image(handle.read())


def save_image(img: Image, path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".ppm", ".pgm"):
        with open(path, "wb") as handle:
            handle.write(encode_ppm(img))
        return path
    if ext not in (".png", ""):
        raise ImageError(f"unsupported output format '{ext}' (use .png/.ppm)")
    with open(path, "wb") as handle:
        handle.write(encode_png(img))
    return path


def encode_ppm(img: Image) -> bytes:
    src = img.to_rgb() if img.mode != "RGB" else img
    header = f"P6\n{src.width} {src.height}\n255\n".encode("ascii")
    return header + bytes(src.pixels)


# ===================================================================== #
# Pixel operations
# ===================================================================== #

def _clamp(v: float) -> int:
    return 0 if v < 0 else (255 if v > 255 else int(v + 0.5))


def grayscale(img: Image) -> Image:
    src = img.to_rgb()
    out = bytearray(src.width * src.height)
    for i in range(0, len(src.pixels), 3):
        r, g, b = src.pixels[i], src.pixels[i + 1], src.pixels[i + 2]
        out[i // 3] = (r * 299 + g * 587 + b * 114) // 1000
    return Image(src.width, src.height, "L", out)


def invert(img: Image) -> Image:
    out = bytearray(v ^ 0xFF for v in img.pixels)
    return Image(img.width, img.height, img.mode, out)


def threshold(img: Image, level: int = 128) -> Image:
    gray = img if img.mode == "L" else grayscale(img)
    out = bytearray(255 if v >= level else 0 for v in gray.pixels)
    return Image(gray.width, gray.height, "L", out)


def adjust_brightness(img: Image, delta: int) -> Image:
    out = bytearray(_clamp(v + delta) for v in img.pixels)
    return Image(img.width, img.height, img.mode, out)


def box_blur(img: Image, radius: int = 2) -> Image:
    if radius < 1:
        return img.copy()
    src = img.to_rgb()
    w, h = src.width, src.height
    out = bytearray(len(src.pixels))
    for y in range(h):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        for x in range(w):
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            sr = sg = sb = count = 0
            for yy in range(y0, y1):
                base = yy * w * 3
                for xx in range(x0, x1):
                    i = base + xx * 3
                    sr += src.pixels[i]
                    sg += src.pixels[i + 1]
                    sb += src.pixels[i + 2]
                    count += 1
            j = (y * w + x) * 3
            out[j] = sr // count
            out[j + 1] = sg // count
            out[j + 2] = sb // count
    return Image(w, h, "RGB", out)


def sobel_edges(img: Image, threshold_value: int = 80) -> Image:
    gray = grayscale(img)
    w, h = gray.width, gray.height
    px = gray.pixels
    out = bytearray(w * h)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            i = y * w + x
            gx = (-px[i - w - 1] - 2 * px[i - 1] - px[i + w - 1]
                  + px[i - w + 1] + 2 * px[i + 1] + px[i + w + 1])
            gy = (-px[i - w - 1] - 2 * px[i - w] - px[i - w + 1]
                  + px[i + w - 1] + 2 * px[i + w] + px[i + w + 1])
            mag = int((gx * gx + gy * gy) ** 0.5)
            out[i] = 255 if mag >= threshold_value else 0
    return Image(w, h, "L", out)


def resize(img: Image, width: int, height: int) -> Image:
    if width <= 0 or height <= 0:
        raise ImageError("resize dimensions must be positive")
    src = img.to_rgb()
    out = bytearray(width * height * 3)
    for y in range(height):
        sy = min(src.height - 1, y * src.height // height)
        for x in range(width):
            sx = min(src.width - 1, x * src.width // width)
            i = (sy * src.width + sx) * 3
            j = (y * width + x) * 3
            out[j] = src.pixels[i]
            out[j + 1] = src.pixels[i + 1]
            out[j + 2] = src.pixels[i + 2]
    return Image(width, height, "RGB", out)


def crop(img: Image, x: int, y: int, width: int, height: int) -> Image:
    x = max(0, min(x, img.width - 1))
    y = max(0, min(y, img.height - 1))
    width = max(1, min(width, img.width - x))
    height = max(1, min(height, img.height - y))
    src = img.to_rgb()
    out = bytearray(width * height * 3)
    for row in range(height):
        src_off = ((y + row) * src.width + x) * 3
        out[row * width * 3:(row + 1) * width * 3] = \
            src.pixels[src_off:src_off + width * 3]
    return Image(width, height, "RGB", out)


# ===================================================================== #
# Drawing
# ===================================================================== #

def draw_rect(img: Image, x: int, y: int, width: int, height: int,
              color, filled: bool = True) -> None:
    if filled:
        for yy in range(y, min(y + height, img.height)):
            for xx in range(x, min(x + width, img.width)):
                img.set_pixel(xx, yy, color)
        return
    for xx in range(x, x + width):
        img.set_pixel(xx, y, color)
        img.set_pixel(xx, y + height - 1, color)
    for yy in range(y, y + height):
        img.set_pixel(x, yy, color)
        img.set_pixel(x + width - 1, yy, color)


def draw_line(img: Image, x0: int, y0: int, x1: int, y1: int, color) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        img.set_pixel(x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_circle(img: Image, cx: int, cy: int, r: int, color,
                filled: bool = False) -> None:
    for y in range(-r, r + 1):
        for x in range(-r, r + 1):
            d2 = x * x + y * y
            if filled:
                if d2 <= r * r:
                    img.set_pixel(cx + x, cy + y, color)
            else:
                if (r - 0.8) ** 2 <= d2 <= (r + 0.8) ** 2:
                    img.set_pixel(cx + x, cy + y, color)


def draw_ellipse(img: Image, cx: int, cy: int, rx: int, ry: int, color,
                 filled: bool = False) -> None:
    if rx <= 0 or ry <= 0:
        return
    for y in range(-ry, ry + 1):
        for x in range(-rx, rx + 1):
            t = (x * x) / float(rx * rx) + (y * y) / float(ry * ry)
            if filled:
                if t <= 1.0:
                    img.set_pixel(cx + x, cy + y, color)
            else:
                if 0.72 <= t <= 1.0:
                    img.set_pixel(cx + x, cy + y, color)


def draw_polygon(img: Image, points: Sequence[Sequence[int]], color,
                 filled: bool = False) -> None:
    """Scanline-filled polygon (even-odd) plus its outline."""
    pts = [(int(px), int(py)) for px, py in points]
    if len(pts) < 2:
        return
    min_y = max(0, min(p[1] for p in pts))
    max_y = min(img.height - 1, max(p[1] for p in pts))
    if filled:
        for y in range(min_y, max_y + 1):
            xs: list[int] = []
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                if y0 == y1:
                    continue
                lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
                if lo <= y < hi:
                    xs.append(int(x0 + (y - y0) * (x1 - x0) / (y1 - y0)))
            xs.sort()
            for i in range(0, len(xs) - 1, 2):
                for x in range(max(0, xs[i]), min(img.width - 1, xs[i + 1]) + 1):
                    img.set_pixel(x, y, color)
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        draw_line(img, x0, y0, x1, y1, color)


# 5x7 bitmap font: uppercase letters, digits, and common punctuation.
_FONT: dict[str, tuple] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11100", "10010", "10001", "10001", "10001", "10010", "11100"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "10001", "11001", "10101", "10011", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    ",": ("00000", "00000", "00000", "00000", "00110", "00110", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
}


def text_width(text: str, scale: int = 1) -> int:
    return len(text) * 6 * scale


def draw_text(img: Image, x: int, y: int, text: str, color=(255, 255, 255),
              scale: int = 1) -> None:
    """Draw text with the built-in 5x7 font (lowercase → uppercase)."""
    cx = x
    for ch in text:
        glyph = _FONT.get(ch.upper(), _FONT.get(ch, _FONT[" "]))
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            img.set_pixel(cx + col * scale + dx,
                                          y + row * scale + dy, color)
        cx += 6 * scale


# ===================================================================== #
# Analysis (classical vision)
# ===================================================================== #

def _downsample(img: Image, max_dim: int = 160) -> Image:
    if max(img.width, img.height) <= max_dim:
        return img.to_rgb()
    scale = max_dim / max(img.width, img.height)
    return resize(img, max(1, int(img.width * scale)),
                  max(1, int(img.height * scale)))


def mean_color(img: Image) -> tuple:
    src = img.to_rgb()
    n = src.width * src.height
    if n == 0:
        return (0, 0, 0)
    sr = sg = sb = 0
    for i in range(0, len(src.pixels), 3):
        sr += src.pixels[i]
        sg += src.pixels[i + 1]
        sb += src.pixels[i + 2]
    return (sr // n, sg // n, sb // n)


def dominant_colors(img: Image, limit: int = 6) -> list:
    """The dominant palette by quantized-pixel counts."""
    src = _downsample(img)
    counts: dict[tuple, int] = {}
    for i in range(0, len(src.pixels), 3):
        key = (src.pixels[i] & 0xE0, src.pixels[i + 1] & 0xE0,
               src.pixels[i + 2] & 0xE0)
        counts[key] = counts.get(key, 0) + 1
    total = src.width * src.height or 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [{"rgb": (key[0] + 16, key[1] + 16, key[2] + 16),
             "ratio": round(count / total, 4)}
            for key, count in ranked[:limit]]


def luminance_histogram(img: Image, bins: int = 8) -> list:
    gray = img if img.mode == "L" else grayscale(img)
    hist = [0] * bins
    for v in gray.pixels:
        hist[min(bins - 1, v * bins // 256)] += 1
    total = max(1, gray.width * gray.height)
    return [round(h / total, 4) for h in hist]


@dataclass
class Region:
    x: int
    y: int
    width: int
    height: int
    area: int
    mean_color: tuple
    shape: str = ""

    def to_dict(self) -> dict:
        return {
            "bbox": [self.x, self.y, self.width, self.height],
            "area": self.area,
            "mean_color": list(self.mean_color),
            "shape": self.shape,
        }


def _classify_shape(width: int, height: int, area: int) -> str:
    if width <= 2 or height <= 2:
        return "line"
    fill = area / float(width * height)
    aspect = width / float(height)
    if fill < 0.25:
        return "line"
    if fill > 0.85:
        return "rectangle"
    if 0.55 <= fill <= 0.85 and 0.6 <= aspect <= 1.6:
        return "circle"
    if fill <= 0.6 and 0.6 <= aspect <= 1.6:
        return "triangle"
    return "region"


def connected_regions(img: Image, min_area: int = 8,
                      max_regions: int = 24) -> list:
    """Connected components of non-background pixels (4-connectivity).

    Background is the image's most common quantized color; region color
    is the mean of the ORIGINAL pixels inside the bounding box.
    """
    src = _downsample(img)
    w, h = src.width, src.height
    pixels = src.pixels
    counts: dict[tuple, int] = {}
    for i in range(0, len(pixels), 3):
        key = (pixels[i] & 0xE0, pixels[i + 1] & 0xE0, pixels[i + 2] & 0xE0)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    background = max(counts.items(), key=lambda kv: kv[1])[0]

    def is_foreground(x: int, y: int) -> bool:
        i = (y * w + x) * 3
        key = (pixels[i] & 0xE0, pixels[i + 1] & 0xE0, pixels[i + 2] & 0xE0)
        return key != background

    visited = bytearray(w * h)
    regions: list[Region] = []
    for y0 in range(h):
        for x0 in range(w):
            start = y0 * w + x0
            if visited[start] or not is_foreground(x0, y0):
                continue
            stack = [start]
            visited[start] = 1
            min_x = max_x = x0
            min_y = max_y = y0
            area = 0
            color = img.to_rgb()
            sr = sg = sb = 0
            sx = src.width
            scale_x = img.width / float(src.width)
            scale_y = img.height / float(src.height)
            while stack:
                idx = stack.pop()
                xx, yy = idx % w, idx // w
                area += 1
                if xx < min_x:
                    min_x = xx
                if xx > max_x:
                    max_x = xx
                if yy < min_y:
                    min_y = yy
                if yy > max_y:
                    max_y = yy
                ox = min(color.width - 1, int(xx * scale_x))
                oy = min(color.height - 1, int(yy * scale_y))
                oi = (oy * color.width + ox) * 3
                sr += color.pixels[oi]
                sg += color.pixels[oi + 1]
                sb += color.pixels[oi + 2]
                for nx, ny in ((xx - 1, yy), (xx + 1, yy),
                               (xx, yy - 1), (xx, yy + 1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        nidx = ny * w + nx
                        if not visited[nidx] and is_foreground(nx, ny):
                            visited[nidx] = 1
                            stack.append(nidx)
            if area < min_area:
                continue
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1
            regions.append(Region(
                x=int(min_x * scale_x), y=int(min_y * scale_y),
                width=max(1, int(box_w * scale_x)),
                height=max(1, int(box_h * scale_y)),
                area=area,
                mean_color=(sr // area, sg // area, sb // area),
                shape=_classify_shape(box_w, box_h, area),
            ))
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions[:max_regions]


def describe_image(path: str) -> dict:
    """A structured description of an image (input side of image I/O)."""
    img = load_image(path)
    regions = connected_regions(img)
    dominant = dominant_colors(img)
    mean = mean_color(img)
    summary_parts = [
        f"{img.width}x{img.height} {img.mode} image",
        f"mean color rgb{mean}",
    ]
    if dominant:
        summary_parts.append(
            "dominant color rgb" + str(dominant[0]["rgb"]))
    if regions:
        shapes = ", ".join(f"{r.shape} at ({r.x},{r.y})"
                           for r in regions[:4])
        summary_parts.append(f"{len(regions)} region(s): {shapes}")
    return {
        "path": path,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "mean_color": list(mean),
        "dominant_colors": dominant,
        "luminance_histogram": luminance_histogram(img),
        "regions": [r.to_dict() for r in regions],
        "summary": "; ".join(summary_parts),
    }


# ===================================================================== #
# Charts (image output)
# ===================================================================== #

_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
]


def _axis_max(values: Sequence[float]) -> float:
    peak = max(values) if values else 1.0
    if peak <= 0:
        return 1.0
    exp = 0
    scaled = peak
    while scaled >= 10:
        scaled /= 10
        exp += 1
    step = 1
    if scaled <= 2:
        step = 0.5
    return (int(scaled / step) + 1) * step * (10 ** exp)


def bar_chart(title: str, labels: Sequence[str], values: Sequence[float],
              width: int = 640, height: int = 360,
              background=(250, 250, 252), text_color=(40, 40, 40)) -> Image:
    img = new_image(width, height, background)
    left, right, top, bottom = 64, 24, 46, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = _axis_max(values)
    # axes
    draw_line(img, left, top, left, top + plot_h, text_color)
    draw_line(img, left, top + plot_h, left + plot_w, top + plot_h,
              text_color)
    # horizontal grid + value labels
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - int(plot_h * frac)
        if frac > 0:
            draw_line(img, left, y, left + plot_w, y, (225, 225, 230))
        value = peak * frac
        label = f"{value:.0f}" if abs(value - round(value)) < 1e-9 \
            else f"{value:.1f}"
        draw_text(img, 4, y - 3, label, text_color)
    draw_text(img, left, 14, title, text_color, scale=2)
    # bars
    count = max(1, len(values))
    slot = plot_w / count
    bar_w = max(4, int(slot * 0.6))
    for index, value in enumerate(values):
        bar_h = int(plot_h * (value / peak)) if peak else 0
        x = left + int(slot * index + (slot - bar_w) / 2)
        y = top + plot_h - bar_h
        color = _PALETTE[index % len(_PALETTE)]
        draw_rect(img, x, y, bar_w, bar_h, color, filled=True)
        value_label = f"{value:.0f}" if abs(value - round(value)) < 1e-9 \
            else f"{value:.1f}"
        draw_text(img, x + max(0, (bar_w - text_width(value_label)) // 2),
                  y - 10, value_label, text_color)
        if index < len(labels):
            label = str(labels[index])[:max(1, int(slot / 6))]
            draw_text(img, x + max(0, (bar_w - text_width(label)) // 2),
                      top + plot_h + 8, label, text_color)
    return img


def line_chart(title: str, labels: Sequence[str], values: Sequence[float],
               width: int = 640, height: int = 360,
               background=(250, 250, 252), text_color=(40, 40, 40)) -> Image:
    img = new_image(width, height, background)
    left, right, top, bottom = 64, 24, 46, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = _axis_max(values)
    draw_line(img, left, top, left, top + plot_h, text_color)
    draw_line(img, left, top + plot_h, left + plot_w, top + plot_h,
              text_color)
    for frac in (0.0, 0.5, 1.0):
        y = top + plot_h - int(plot_h * frac)
        if frac > 0:
            draw_line(img, left, y, left + plot_w, y, (225, 225, 230))
        value = peak * frac
        label = f"{value:.0f}" if abs(value - round(value)) < 1e-9 \
            else f"{value:.1f}"
        draw_text(img, 4, y - 3, label, text_color)
    draw_text(img, left, 14, title, text_color, scale=2)
    if not values:
        return img
    points = []
    count = len(values)
    step = plot_w / max(1, count - 1)
    for index, value in enumerate(values):
        x = left + int(step * index)
        y = top + plot_h - int(plot_h * (value / peak))
        points.append((x, y))
    for index in range(len(points) - 1):
        draw_line(img, points[index][0], points[index][1],
                  points[index + 1][0], points[index + 1][1],
                  _PALETTE[0])
    for index, (x, y) in enumerate(points):
        draw_circle(img, x, y, 4, _PALETTE[0], filled=True)
        value_label = f"{values[index]:.0f}" \
            if abs(values[index] - round(values[index])) < 1e-9 \
            else f"{values[index]:.1f}"
        draw_text(img, x - text_width(value_label) // 2, y - 16,
                  value_label, text_color)
        if index < len(labels):
            label = str(labels[index])[:8]
            draw_text(img, x - text_width(label) // 2,
                      top + plot_h + 8, label, text_color)
    return img


def palette_card(colors: Sequence[tuple], names: Optional[Sequence[str]] = None,
                 width: int = 480, height: int = 120) -> Image:
    """An image made of the given colors — the palette as an artifact."""
    img = new_image(width, height, (255, 255, 255))
    count = max(1, len(colors))
    swatch = width // count
    for index, color in enumerate(colors):
        draw_rect(img, index * swatch, 0, swatch, height - 24,
                  color, filled=True)
        if names and index < len(names):
            draw_text(img, index * swatch + 6, height - 18,
                      str(names[index]), (30, 30, 30))
    return img


# ===================================================================== #
# High-level operations (CLI entry points)
# ===================================================================== #

_OPERATIONS = ("gray", "invert", "blur", "edges", "threshold",
               "resize", "brighten", "crop")


def convert_image(input_path: str, output_path: str, op: str,
                  radius: int = 2, level: int = 128, width: int = 0,
                  height: int = 0, delta: int = 40,
                  crop_box: Optional[Sequence[int]] = None) -> str:
    """Apply one named operation and save the result."""
    img = load_image(input_path)
    if op == "gray":
        out = grayscale(img)
    elif op == "invert":
        out = invert(img)
    elif op == "blur":
        out = box_blur(img, radius)
    elif op == "edges":
        out = sobel_edges(img, threshold_value=level)
    elif op == "threshold":
        out = threshold(img, level)
    elif op == "resize":
        target_w = width or max(1, img.width // 2)
        target_h = height or max(1, int(img.height * target_w / img.width))
        out = resize(img, target_w, target_h)
    elif op == "brighten":
        out = adjust_brightness(img, delta)
    elif op == "crop":
        if not crop_box or len(crop_box) != 4:
            raise ImageError("crop requires --box x y width height")
        out = crop(img, *[int(v) for v in crop_box])
    else:
        raise ImageError(f"unknown operation '{op}' (use: "
                         f"{', '.join(_OPERATIONS)})")
    return save_image(out, output_path)


def chart_from_json(input_path: str, output_path: str, kind: str = "bar",
                    title: str = "") -> str:
    """Render a chart from a JSON data file.

    Accepted JSON shapes:
      {"labels": [...], "values": [...]}
      {"A": 85, "B": 70}          (labels are the keys)
      [[label, value], ...]
    """
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    labels: list[str] = []
    values: list[float] = []
    if isinstance(data, dict) and "values" in data:
        values = [float(v) for v in data["values"]]
        labels = [str(x) for x in (data.get("labels") or [])]
        title = title or str(data.get("title") or "")
    elif isinstance(data, dict):
        for key, value in data.items():
            labels.append(str(key))
            values.append(float(value))
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                labels.append(str(entry[0]))
                values.append(float(entry[1]))
    if not values:
        raise ImageError("no numeric data found in the JSON input")
    if kind == "line":
        img = line_chart(title, labels, values)
    else:
        img = bar_chart(title, labels, values)
    return save_image(img, output_path)
