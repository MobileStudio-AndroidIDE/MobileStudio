"""Physics-based 3D rendering — photorealistic images without any model.

This is the honest "realistic generation" route: no pretrained weights,
no ONNX, no neural networks.  A vectorized path tracer computes the
image from PHYSICS:

  * ray-sphere / ray-plane / ray-ring intersections (closed form),
  * Lambertian, metal (roughness), dielectric/water (Fresnel, Snell),
    emissive and textured planet surfaces,
  * procedural materials from value-noise/fBm (continents, clouds,
    marble, sand, gas bands) — computed from the hit point, never from
    a texture file,
  * direct sun lighting with SOFT SHADOWS (area-light sampling),
    sky ambient, specular highlights, multi-bounce global illumination,
  * depth of field, anti-aliasing, ACES tone mapping and gamma.

Rendering happens on numpy arrays (math, not AI).  Scenes are described
declaratively (camera, sky, sun, surfaces); the scene library is a
vocabulary of STAGED SETS, and any staging parameter can be driven by a
seed — the result is a rendered photograph of a synthetic world, not a
drawing.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from brain import image_io as ii


def _require_numpy():
    if np is None:
        raise ii.ImageError(
            "physics-based rendering requires numpy (math library only — "
            "no pretrained models)")


RGB = tuple


# ===================================================================== #
# Noise (value noise + fBm), fully vectorized
# ===================================================================== #

def _hash3(ix, iy, iz, seed: int):
    h = (ix.astype(np.int64) * 374761393
         + iy.astype(np.int64) * 668265263
         + iz.astype(np.int64) * 1442695040
         + np.int64(seed) * 971
         + np.int64(1013904223))
    h = (h ^ (h >> 13)) & np.int64(0x7FFFFFFF)
    h = (h * 1274126177) & np.int64(0x7FFFFFFF)
    h = h ^ (h >> 16)
    return (h & np.int64(0xFFFF)).astype(np.float64) / 65535.0


def value_noise(points, seed: int = 0):
    """3D value noise in [0, 1] for an (N, 3) point array."""
    _require_numpy()
    p = np.asarray(points, dtype=np.float64)
    base = np.floor(p)
    frac = p - base
    frac = frac * frac * (3.0 - 2.0 * frac)
    ix, iy, iz = base[:, 0], base[:, 1], base[:, 2]
    c000 = _hash3(ix, iy, iz, seed)
    c100 = _hash3(ix + 1, iy, iz, seed)
    c010 = _hash3(ix, iy + 1, iz, seed)
    c110 = _hash3(ix + 1, iy + 1, iz, seed)
    c001 = _hash3(ix, iy, iz + 1, seed)
    c101 = _hash3(ix + 1, iy, iz + 1, seed)
    c011 = _hash3(ix, iy + 1, iz + 1, seed)
    c111 = _hash3(ix + 1, iy + 1, iz + 1, seed)
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    x00 = c000 + (c100 - c000) * fx
    x10 = c010 + (c110 - c010) * fx
    x01 = c001 + (c101 - c001) * fx
    x11 = c011 + (c111 - c011) * fx
    y0 = x00 + (x10 - x00) * fy
    y1 = x01 + (x11 - x01) * fy
    return y0 + (y1 - y0) * fz


def fbm(points, seed: int = 0, octaves: int = 5, lacunarity: float = 2.0,
        gain: float = 0.5):
    """Fractal Brownian motion over value noise, normalized to [0, 1]."""
    _require_numpy()
    total = np.zeros(points.shape[0])
    amplitude = 0.5
    frequency = 1.0
    span = 0.0
    for _ in range(octaves):
        total += amplitude * value_noise(points * frequency, seed)
        span += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / span


# ===================================================================== #
# Vector helpers
# ===================================================================== #

def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _mix(a, b, t):
    t = t[:, None] if t.ndim == 1 else t
    return a * (1.0 - t) + b * t


def _orthonormal_basis(normals):
    """A tangent basis (t, b) for each normal (N, 3)."""
    n = normals
    helper = np.where(np.abs(n[:, 1:2]) < 0.99,
                      np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t = _normalize(np.cross(helper, n))
    b = np.cross(n, t)
    return t, b


def _cosine_hemisphere(normals, rng):
    """Cosine-weighted directions around normals, plus their pdf (cos/π)."""
    n = normals.shape[0]
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi),
                      np.sqrt(np.maximum(0.0, 1.0 - u1))], axis=1)
    t, b = _orthonormal_basis(normals)
    return _normalize(t * local[:, 0:1] + b * local[:, 1:2]
                      + normals * local[:, 2:3])


def _reflect(d, n):
    return d - 2.0 * np.einsum("ij,ij->i", d, n)[:, None] * n


def _refract(d, n, eta):
    """Snell refraction; ``eta`` may be a scalar or a per-ray array."""
    eta_arr = np.asarray(eta, dtype=np.float64)
    cosi = -np.einsum("ij,ij->i", d, n)
    k = 1.0 - eta_arr * eta_arr * (1.0 - cosi * cosi)
    out = np.zeros_like(d)
    ok = k > 0
    if ok.any():
        e = eta_arr[ok] if eta_arr.ndim else eta_arr
        if e.ndim == 0:
            e = np.full(ok.sum(), float(e))
        out[ok] = (e[:, None] * d[ok]
                   + (e * cosi[ok] - np.sqrt(k[ok]))[:, None] * n[ok])
    return out, ok


# ===================================================================== #
# Scene model
# ===================================================================== #

@dataclass
class Material:
    """A surface: albedo (color or texture fn), shading model, params."""

    shade: str = "lambert"          # lambert|metal|glass|water|emissive|planet
    color: RGB = (0.8, 0.8, 0.8)
    texture: Optional[Callable] = None
    roughness: float = 0.05
    emission: RGB = (0.0, 0.0, 0.0)
    ior: float = 1.5
    noise_seed: int = 1

    def albedo(self, points):
        if self.texture is not None:
            return self.texture(points)
        return np.tile(np.array(self.color, dtype=np.float64),
                       (points.shape[0], 1))


@dataclass
class Sphere:
    center: Sequence[float]
    radius: float
    material: Material

    def intersect(self, origins, dirs):
        oc = origins - np.asarray(self.center, dtype=np.float64)
        b = 2.0 * np.einsum("ij,ij->i", oc, dirs)
        c = np.einsum("ij,ij->i", oc, oc) - self.radius ** 2
        disc = b * b - 4.0 * c
        valid = disc > 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sq) * 0.5
        t1 = (-b + sq) * 0.5
        t = np.where(t0 > 1e-4, t0, t1)
        hit = valid & (t > 1e-4)
        return hit, t

    def normal(self, points):
        return _normalize(points - np.asarray(self.center, dtype=np.float64))


@dataclass
class Plane:
    point: Sequence[float]
    normal_vec: Sequence[float]
    material: Material
    tex_scale: float = 1.0
    tex_kind: str = "checker"

    def intersect(self, origins, dirs):
        n = np.asarray(self.normal_vec, dtype=np.float64)
        n = n / np.linalg.norm(n)
        denom = dirs @ n
        rel = np.asarray(self.point, dtype=np.float64) - origins
        t = (rel @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        hit = (np.abs(denom) > 1e-9) & (t > 1e-4)
        return hit, t

    def normal(self, points):
        n = np.asarray(self.normal_vec, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return np.tile(n, (points.shape[0], 1))


@dataclass
class Ring:
    """A flat annulus (planet rings): plane + inner/outer radius."""

    center: Sequence[float]
    normal_vec: Sequence[float]
    inner: float
    outer: float
    material: Material
    tilt_basis: Sequence[float] = (1.0, 0.0, 0.0)

    def intersect(self, origins, dirs):
        n = np.asarray(self.normal_vec, dtype=np.float64)
        n = n / np.linalg.norm(n)
        denom = dirs @ n
        rel = np.asarray(self.center, dtype=np.float64) - origins
        t = (rel @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        pts = origins + dirs * t[:, None]
        radial = pts - np.asarray(self.center, dtype=np.float64)
        radial = radial - np.einsum("ij,j->i", radial, n)[:, None] * n
        dist = np.linalg.norm(radial, axis=1)
        hit = ((np.abs(denom) > 1e-9) & (t > 1e-4)
               & (dist > self.inner) & (dist < self.outer))
        return hit, t

    def normal(self, points):
        n = np.asarray(self.normal_vec, dtype=np.float64)
        n = n / np.linalg.norm(n)
        return np.tile(n, (points.shape[0], 1))


@dataclass
class Camera:
    eye: Sequence[float] = (0.0, 0.8, 4.5)
    look_at: Sequence[float] = (0.0, 0.2, 0.0)
    up: Sequence[float] = (0.0, 1.0, 0.0)
    fov_deg: float = 42.0
    aperture: float = 0.0
    focus_dist: float = 0.0


@dataclass
class Scene:
    name: str
    description: str
    camera: Camera
    sun_dir: Sequence[float] = (-0.55, 0.6, 0.35)
    sun_color: RGB = (1.0, 0.98, 0.94)
    sun_intensity: float = 3.0
    sun_radius_deg: float = 2.0
    ambient: float = 0.35
    sky: Optional[Callable] = None
    objects: list = field(default_factory=list)
    max_bounces: int = 3
    seed: int = 7


# ===================================================================== #
# Skies and procedural textures
# ===================================================================== #

def gradient_sky(top: RGB, horizon: RGB, ground: RGB,
                 sun_disc: bool = True, sun_size_deg: float = 3.0,
                 glow: float = 0.35, stars: bool = False,
                 star_density: float = 260.0, seed: int = 3) -> Callable:
    def sky(dirs):
        d = dirs
        h = d[:, 1]
        sky_top = np.array(top, dtype=np.float64)
        sky_horizon = np.array(horizon, dtype=np.float64)
        sky_ground = np.array(ground, dtype=np.float64)
        up_t = np.clip(h, 0.0, 1.0)[:, None]
        color = sky_horizon * (1.0 - up_t) + sky_top * up_t
        down_t = np.clip(-h, 0.0, 1.0)[:, None]
        color = np.where(h[:, None] < 0.0,
                         sky_horizon * (1.0 - down_t) + sky_ground * down_t,
                         color)
        if stars:
            grid = np.floor((d * star_density)).astype(np.int64)
            star_hash = _hash3(grid[:, 0], grid[:, 1], grid[:, 2], seed)
            mask = (star_hash > 0.9985) & (h > -0.05)
            twinkle = 0.5 + 0.5 * _hash3(grid[:, 0] + 7, grid[:, 1] + 3,
                                         grid[:, 2] + 11, seed + 1)
            color = color + mask[:, None] * twinkle[:, None] * 1.6
        return color
    return sky


def sunset_sky(sun_dir, seed: int = 4) -> Callable:
    base = gradient_sky(
        top=(0.16, 0.28, 0.52), horizon=(1.0, 0.62, 0.32),
        ground=(0.08, 0.08, 0.14), stars=False, seed=seed)

    def sky(dirs):
        color = base(dirs)
        sd = np.clip(dirs @ np.asarray(sun_dir, dtype=np.float64)
                     / np.linalg.norm(sun_dir), 0.0, 1.0)
        color = color + (np.power(sd, 24.0) * 0.9
                         + np.power(sd, 220.0) * 2.4)[:, None] \
            * np.array([1.0, 0.68, 0.35])
        return color
    return sky


def nebula_sky(seed: int = 5) -> Callable:
    palette_a = np.array([0.05, 0.02, 0.10])
    palette_b = np.array([0.55, 0.16, 0.55])
    palette_c = np.array([0.10, 0.35, 0.75])

    def sky(dirs):
        d = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True),
                              1e-12)
        n1 = fbm(d * 3.1 + np.array([11.0, 3.0, 7.0]), seed=seed,
                 octaves=5)
        n2 = fbm(d * 6.3 + np.array([2.0, 9.0, 4.0]), seed=seed + 3,
                 octaves=4)
        color = np.tile(palette_a, (d.shape[0], 1))
        color = _mix(color, np.tile(palette_b, (d.shape[0], 1)),
                     np.clip((n1 - 0.42) * 2.4, 0, 1))
        color = _mix(color, np.tile(palette_c, (d.shape[0], 1)),
                     np.clip((n2 - 0.55) * 2.2, 0, 1) * 0.7)
        grid = np.floor(d * 420.0).astype(np.int64)
        star_hash = _hash3(grid[:, 0], grid[:, 1], grid[:, 2], seed + 9)
        stars = (star_hash > 0.9975)
        color = color + stars[:, None] * 1.8
        return color
    return sky


def earth_texture(seed: int = 11) -> Callable:
    """Continents, oceans, ice caps and clouds from fBm."""
    ocean = np.array([0.04, 0.12, 0.30])
    land_low = np.array([0.16, 0.38, 0.14])
    land_high = np.array([0.55, 0.48, 0.30])
    cloud = np.array([0.92, 0.94, 0.98])

    def tex(points):
        p = points / max(1e-9, float(np.max(np.abs(points)) or 1.0))
        continents = fbm(p * 2.6, seed=seed, octaves=6)
        lat = np.abs(points[:, 1]) / np.maximum(
            1e-9, np.linalg.norm(points, axis=1))
        land = continents > 0.52
        height = np.clip((continents - 0.52) * 6.0, 0, 1)
        color = np.where(land[:, None],
                         _mix(np.tile(land_low, (len(p), 1)),
                              np.tile(land_high, (len(p), 1)), height),
                         np.tile(ocean, (len(p), 1)))
        ice = np.clip((lat - 0.72) * 8.0, 0, 1) \
            + np.clip((lat - 0.66) * 8.0, 0, 1) * 0
        color = _mix(color, np.tile(np.array([0.95, 0.97, 1.0]),
                                    (len(p), 1)), np.clip(ice, 0, 1))
        clouds = fbm(p * 3.4 + np.array([5.0, 2.0, 8.0]), seed=seed + 7,
                     octaves=5)
        cloud_amt = np.clip((clouds - 0.62) * 5.0, 0, 1) * 0.75
        return _mix(color, np.tile(cloud, (len(p), 1)), cloud_amt)
    return tex


def checker_texture(scale: float = 1.0,
                    c1: RGB = (0.85, 0.85, 0.85),
                    c2: RGB = (0.15, 0.15, 0.18)) -> Callable:
    def tex(points):
        idx = (np.floor(points[:, 0] / scale)
               + np.floor(points[:, 2] / scale)).astype(np.int64)
        mask = (idx % 2) == 0
        return np.where(mask[:, None],
                        np.array(c1), np.array(c2))
    return tex


def marble_texture(c1: RGB = (0.85, 0.85, 0.88),
                   c2: RGB = (0.20, 0.22, 0.30),
                   seed: int = 21) -> Callable:
    def tex(points):
        n = fbm(points * 1.7, seed=seed, octaves=5)
        bands = 0.5 + 0.5 * np.sin((points[:, 0] + n * 4.0) * 3.0)
        return _mix(np.array(c1), np.array(c2), bands * 0.85)
    return tex


def sand_texture(seed: int = 31) -> Callable:
    def tex(points):
        n = fbm(points * 6.0, seed=seed, octaves=4)
        base = np.array([0.86, 0.76, 0.58]) * (0.75 + 0.45 * n[:, None])
        return base
    return tex


def brushed_metal(color: RGB, seed: int = 41) -> Callable:
    def tex(points):
        streak = fbm(points * np.array([30.0, 1.2, 1.2]), seed=seed,
                     octaves=3)
        return np.tile(np.array(color), (len(points), 1)) \
            * (0.85 + 0.30 * streak[:, None])
    return tex


def gas_giant_texture(seed: int = 51) -> Callable:
    bands = [(0.72, 0.58, 0.42), (0.88, 0.78, 0.60), (0.55, 0.42, 0.32),
             (0.92, 0.85, 0.70), (0.65, 0.50, 0.38)]

    def tex(points):
        lat = np.arctan2(points[:, 1],
                         np.linalg.norm(points[:, [0, 2]], axis=1))
        turbulence = fbm(points * 2.2, seed=seed, octaves=5) - 0.5
        band_pos = (lat / math.pi + 0.5) * len(bands) * 1.5 \
            + turbulence * 0.9
        idx = np.floor(band_pos).astype(np.int64) % len(bands)
        frac = band_pos - np.floor(band_pos)
        palette = np.array(bands)
        a = palette[idx]
        b = palette[(idx + 1) % len(bands)]
        return a * (1.0 - frac[:, None]) + b * frac[:, None]
    return tex


# ===================================================================== #
# Intersection + path tracing
# ===================================================================== #

def _intersect(origins, dirs, scene):
    n = origins.shape[0]
    best_t = np.full(n, np.inf)
    best_idx = np.full(n, -1, dtype=np.int64)
    for index, obj in enumerate(scene.objects):
        hit, t = obj.intersect(origins, dirs)
        closer = hit & (t < best_t)
        best_t[closer] = t[closer]
        best_idx[closer] = index
    hit = best_idx >= 0
    points = origins + dirs * np.where(hit, best_t, 0.0)[:, None]
    normals = np.zeros_like(points)
    for index, obj in enumerate(scene.objects):
        mask = best_idx == index
        if mask.any():
            normals[mask] = obj.normal(points[mask])
    return hit, best_t, best_idx, points, normals


def _sun_dirs(scene, count, rng):
    d = np.asarray(scene.sun_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    if scene.sun_radius_deg <= 0 or count == 0:
        return np.tile(d, (count, 1))
    t, b = _orthonormal_basis(np.tile(d, (count, 1)))
    radius = math.tan(math.radians(scene.sun_radius_deg))
    u1 = rng.random(count)
    u2 = rng.random(count)
    r = np.sqrt(u1) * radius
    phi = 2.0 * math.pi * u2
    jittered = d[None, :] + (t * (r * np.cos(phi))[:, None]
                             + b * (r * np.sin(phi))[:, None])
    return _normalize(jittered)


def _shade_sample(origins, dirs, scene, rng):
    """One full path-trace sample for a batch of rays -> RGB (N, 3).

    Vectorized per MATERIAL: each material's texture runs ONCE over all
    its hit points (a per-hit texture call is thousands of tiny numpy
    calls and dominates the render time).
    """
    n = origins.shape[0]
    color = np.zeros((n, 3))
    throughput = np.ones((n, 3))
    active = np.ones(n, dtype=bool)

    kind_arr = np.array([obj.material.shade for obj in scene.objects])
    emit_arr = np.array([obj.material.emission for obj in scene.objects],
                        dtype=np.float64)
    rough_arr = np.array([obj.material.roughness
                          for obj in scene.objects], dtype=np.float64)
    sun = np.asarray(scene.sun_color, dtype=np.float64) * scene.sun_intensity

    sun_dir = _sun_dirs(scene, 1, rng)[0]
    for _bounce in range(scene.max_bounces + 1):
        if not active.any():
            break
        hit, t, midx, points, normals = _intersect(origins, dirs, scene)
        miss = active & ~hit
        if miss.any():
            color[miss] += throughput[miss] * scene.sky(dirs[miss])
        active = active & hit
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        p = points[idx]
        nrm = normals[idx]
        wo = dirs[idx]
        mid = midx[idx]
        kinds = kind_arr[mid]

        # albedo: one vectorized texture call per material
        albedo = np.zeros((idx.size, 3))
        for material_index in np.unique(mid):
            sel = np.nonzero(mid == material_index)[0]
            mat = scene.objects[int(material_index)].material
            albedo[sel] = mat.albedo(p[sel])

        # emissive: radiance only, path ends
        emit_mask = kinds == "emissive"
        if emit_mask.any():
            color[idx[emit_mask]] += (throughput[idx[emit_mask]]
                                      * emit_arr[mid[emit_mask]])
            active[idx[emit_mask]] = False

        live = ~emit_mask
        if not live.any():
            continue
        lp = p[live]
        ln = nrm[live]
        lw = wo[live]
        lkinds = kinds[live]
        lalbedo = albedo[live]
        lrough = rough_arr[mid[live]]
        lidx = idx[live]

        # ── direct sunlight with soft shadows (one shadow batch) ──
        ndl = np.clip(ln @ sun_dir, 0.0, 1.0)
        shadow_o = lp + ln * 3e-2
        shadow_d = np.tile(sun_dir, (lp.shape[0], 1))
        occ, _occ_t, *_ = _intersect(shadow_o, shadow_d, scene)
        lit = (~occ).astype(np.float64)

        shine = np.where(lkinds == "lambert", 12.0, 96.0)
        spec_strength = np.where(
            lkinds == "lambert", 0.06,
            np.where(lkinds == "glass", 0.5, 0.9))
        half_v = _normalize(sun_dir[None, :] - lw)
        spec = np.power(np.clip(np.einsum("ij,ij->i", ln, half_v), 0, 1),
                        shine)[:, None] * spec_strength[:, None]
        ambient = lalbedo * scene.ambient * (scene.sky(ln) * 0.5 + 0.5)
        color[lidx] += throughput[lidx] * (
            (lalbedo * ndl[:, None] + spec) * lit[:, None] * sun
            + ambient)

        # ── next bounce per shading model ──
        new_dir = lw.copy()
        new_thr = np.ones((lidx.size, 3))

        diffuse = (lkinds == "lambert") | (lkinds == "planet")
        if diffuse.any():
            new_dir[diffuse] = _cosine_hemisphere(ln[diffuse], rng)
            new_thr[diffuse] = lalbedo[diffuse]

        metal = lkinds == "metal"
        if metal.any():
            rough = np.maximum(lrough[metal], 1e-3)
            refl = _normalize(_reflect(lw[metal], ln[metal])
                              + rng.normal(0.0, 1.0, (metal.sum(), 3))
                              * rough[:, None])
            flip = np.einsum("ij,ij->i", refl, ln[metal]) < 0
            refl[flip] = -refl[flip]
            new_dir[metal] = _normalize(refl)
            new_thr[metal] = lalbedo[metal]

        dielectric = (lkinds == "glass") | (lkinds == "water")
        if dielectric.any():
            dw = lw[dielectric]
            dn = ln[dielectric]
            ior = np.where(lkinds[dielectric] == "water", 1.33,
                           np.array([scene.objects[int(m)].material.ior
                                     for m in mid[live][dielectric]]))
            eta = 1.0 / ior
            cosi = -np.einsum("ij,ij->i", dw, dn)
            f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
            fres = f0 + (1 - f0) * np.power(
                1.0 - np.clip(cosi, 0, 1), 5.0)
            refr_dir, refr_ok = _refract(dw, dn, eta)
            choose_reflect = rng.random(dw.shape[0]) < fres
            chosen = np.where(choose_reflect[:, None],
                              _reflect(dw, dn), refr_dir)
            chosen = np.where(refr_ok[:, None], chosen, _reflect(dw, dn))
            new_dir[dielectric] = _normalize(chosen)
            water = lkinds[dielectric] == "water"
            tint = lalbedo[dielectric] * 0.6 + 0.4
            new_thr[dielectric] = np.where(water[:, None], tint, 1.0)

        # Avoid immediate self-hits: rays leave the surface along their
        # own direction with a small offset
        origins[lidx] = lp + new_dir * 5e-3
        dirs[lidx] = new_dir
        throughput[lidx] *= new_thr
    return color


def render_scene(scene: Scene, width: int = 320, height: int = 240,
                 samples: int = 4, seed: Optional[int] = None) -> ii.Image:
    """Path-trace the scene into an image (ACES tonemapped)."""
    _require_numpy()
    if width <= 0 or height <= 0:
        raise ii.ImageError("render dimensions must be positive")
    rng = np.random.default_rng(scene.seed if seed is None else seed)
    cam = scene.camera
    eye = np.array(cam.eye, dtype=np.float64)
    look = np.array(cam.look_at, dtype=np.float64)
    world_up = np.array(cam.up, dtype=np.float64)
    forward = _normalize((look - eye)[None, :])[0]
    right = _normalize(np.cross(forward, world_up)[None, :])[0]
    true_up = np.cross(right, forward)
    half_h = math.tan(math.radians(cam.fov_deg) * 0.5)
    half_w = half_h * width / height

    accum = np.zeros((height * width, 3))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    px = ((xs.reshape(-1) + 0.5) / width * 2.0 - 1.0)
    py = (1.0 - (ys.reshape(-1) + 0.5) / height * 2.0)

    focus = cam.focus_dist if cam.focus_dist > 0 else np.linalg.norm(
        look - eye)
    for _sample in range(max(1, samples)):
        jx = px + (rng.random(px.shape[0]) - 0.5) * (2.0 / width)
        jy = py + (rng.random(py.shape[0]) - 0.5) * (2.0 / height)
        dirs = _normalize(
            forward[None, :]
            + right[None, :] * (jx * half_w)[:, None]
            + true_up[None, :] * (jy * half_h)[:, None])
        origins = np.tile(eye, (dirs.shape[0], 1))
        if cam.aperture > 0:
            u1 = rng.random(dirs.shape[0])
            u2 = rng.random(dirs.shape[0])
            r = np.sqrt(u1) * cam.aperture
            phi = 2.0 * math.pi * u2
            offset = (right[None, :] * (r * np.cos(phi))[:, None]
                      + true_up[None, :] * (r * np.sin(phi))[:, None])
            origins = origins + offset
            target = eye + dirs * focus
            dirs = _normalize(target - origins)
        accum += _shade_sample(origins, dirs, scene, rng)
    accum /= max(1, samples)

    # ACES tone mapping + gamma
    x = accum
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    mapped = np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)
    rgb = (mapped * 255.0 + 0.5).astype(np.uint8).reshape(height, width, 3)
    # accum is indexed y*width+x — already row-major: the byte buffer is
    # the image itself, never a transposed copy.
    return ii.Image(width, height, "RGB", bytearray(rgb.tobytes()))


# ===================================================================== #
# Scene library
# ===================================================================== #

def _space_planet() -> Scene:
    return Scene(
        name="space_planet",
        description="지구형 행성 + 위성 + 별이 있는 우주 (물리 렌더)",
        camera=Camera(eye=(0.0, 0.75, 4.6), look_at=(0.0, 0.0, 0.0),
                      fov_deg=40.0, aperture=0.005, focus_dist=4.6),
        sun_dir=(-0.75, 0.35, -0.25), sun_color=(1.0, 0.97, 0.92),
        sun_intensity=4.2, sun_radius_deg=2.5, ambient=0.06,
        sky=gradient_sky((0.005, 0.006, 0.012), (0.01, 0.012, 0.02),
                         (0.004, 0.004, 0.008), sun_disc=False,
                         stars=True, star_density=380.0, seed=17),
        objects=[
            Sphere((0.0, 0.0, 0.0), 1.30, Material(
                shade="planet", texture=earth_texture(), roughness=0.2)),
            Sphere((2.6, 1.35, -1.6), 0.34, Material(
                shade="lambert", color=(0.62, 0.60, 0.57),
                texture=lambda p: np.tile(
                    np.array([0.58, 0.56, 0.54]), (len(p), 1))
                * (0.75 + 0.5 * fbm(p * 9.0, seed=23, octaves=4))[:, None],
                roughness=1.0)),
        ],
        max_bounces=2, seed=17)


def _sunset_sea() -> Scene:
    sun = (-0.85, 0.10, -0.30)
    return Scene(
        name="sunset_sea",
        description="노을 지는 바다 + 반사광 + 물보라 (물리 렌더)",
        camera=Camera(eye=(0.0, 1.1, 3.6), look_at=(0.0, 0.5, -3.0),
                      fov_deg=44.0, aperture=0.006, focus_dist=6.0),
        sun_dir=sun, sun_color=(1.0, 0.72, 0.45), sun_intensity=3.6,
        sun_radius_deg=3.2, ambient=0.30,
        sky=sunset_sky(sun, seed=29),
        objects=[
            Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), Material(
                shade="water", color=(0.02, 0.06, 0.10), roughness=0.02,
                ior=1.33, noise_seed=29)),
            Sphere((1.9, 0.42, -1.4), 0.42, Material(
                shade="metal", color=(0.25, 0.24, 0.23),
                texture=brushed_metal((0.30, 0.29, 0.28)), roughness=0.35)),
            Sphere((-2.3, 0.55, -2.4), 0.58, Material(
                shade="lambert", color=(0.35, 0.22, 0.16),
                texture=lambda p: np.tile(np.array([0.34, 0.21, 0.15]),
                                          (len(p), 1))
                * (0.8 + 0.4 * fbm(p * 7.0, seed=31, octaves=4))[:, None],
                roughness=1.0)),
        ],
        max_bounces=3, seed=29)


def _marble_room() -> Scene:
    return Scene(
        name="marble_room",
        description="대리석 바닥 + 유리/금속/대리석 구체 (물리 렌더)",
        camera=Camera(eye=(0.0, 1.05, 3.4), look_at=(0.0, 0.55, 0.0),
                      fov_deg=42.0, aperture=0.012, focus_dist=3.4),
        sun_dir=(-0.3, 0.85, 0.42), sun_color=(1.0, 0.99, 0.95),
        sun_intensity=2.4, sun_radius_deg=7.0, ambient=0.42,
        sky=gradient_sky((0.55, 0.62, 0.75), (0.75, 0.78, 0.85),
                         (0.4, 0.4, 0.42), sun_disc=False, seed=37),
        objects=[
            Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), Material(
                shade="lambert",
                texture=checker_texture(1.2, (0.88, 0.88, 0.90),
                                        (0.20, 0.21, 0.26)),
                roughness=0.25)),
            Plane((0.0, 0.0, -3.2), (0.0, 0.0, 1.0), Material(
                shade="lambert",
                texture=marble_texture(), roughness=0.35)),
            Sphere((-1.35, 0.62, -0.25), 0.62, Material(
                shade="metal", texture=brushed_metal((0.85, 0.68, 0.30)),
                roughness=0.16)),
            Sphere((0.05, 0.55, 0.15), 0.55, Material(
                shade="glass", color=(0.95, 0.98, 1.0), ior=1.5,
                roughness=0.0)),
            Sphere((1.35, 0.6, -0.35), 0.6, Material(
                shade="lambert", texture=marble_texture(
                    (0.90, 0.90, 0.93), (0.25, 0.28, 0.42)),
                roughness=0.18)),
            Sphere((0.0, 2.5, 0.6), 0.45, Material(
                shade="emissive", emission=(14.0, 13.4, 12.0))),
        ],
        max_bounces=3, seed=37)


def _metal_spheres() -> Scene:
    return Scene(
        name="metal_spheres",
        description="금속/크롬/구리 구체 스튜디오 렌더 (물리 렌더)",
        camera=Camera(eye=(0.0, 0.9, 3.9), look_at=(0.0, 0.6, 0.0),
                      fov_deg=40.0, aperture=0.008, focus_dist=3.9),
        sun_dir=(-0.45, 0.75, 0.5), sun_color=(1.0, 0.98, 0.95),
        sun_intensity=3.2, sun_radius_deg=5.0, ambient=0.4,
        sky=gradient_sky((0.18, 0.20, 0.26), (0.42, 0.44, 0.50),
                         (0.10, 0.10, 0.12), sun_disc=False, seed=43),
        objects=[
            Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), Material(
                shade="lambert", color=(0.32, 0.33, 0.36),
                roughness=0.4)),
            Sphere((-1.25, 0.7, 0.0), 0.7, Material(
                shade="metal", texture=brushed_metal((0.95, 0.77, 0.35)),
                roughness=0.06)),
            Sphere((0.0, 0.72, -0.35), 0.72, Material(
                shade="metal", color=(0.95, 0.96, 0.98), roughness=0.02)),
            Sphere((1.25, 0.68, 0.05), 0.68, Material(
                shade="metal", color=(0.95, 0.55, 0.35), roughness=0.18)),
        ],
        max_bounces=3, seed=43)


def _nebula() -> Scene:
    return Scene(
        name="nebula",
        description="성운 + 별 (프랙털 성간 구름 렌더)",
        camera=Camera(eye=(0.0, 0.0, 0.0), look_at=(0.0, 0.0, -1.0),
                      fov_deg=60.0, aperture=0.0),
        sun_dir=(0.0, 1.0, 0.0), sun_intensity=0.0, ambient=0.0,
        sky=nebula_sky(seed=53), objects=[], max_bounces=0, seed=53)


SCENES: dict[str, Callable[[], Scene]] = {
    "space_planet": _space_planet,
    "sunset_sea": _sunset_sea,
    "marble_room": _marble_room,
    "metal_spheres": _metal_spheres,
    "nebula": _nebula,
}


def scene_names() -> list[str]:
    return list(SCENES.keys())


def get_scene(name: str) -> Scene:
    factory = SCENES.get(name)
    if factory is None:
        raise ii.ImageError(
            f"unknown scene '{name}' — available: {', '.join(scene_names())}")
    return factory()


def render_scene_name(name: str, width: int = 320, height: int = 240,
                      samples: int = 4, seed: Optional[int] = None,
                      bounces: Optional[int] = None) -> ii.Image:
    scene = get_scene(name)
    if bounces is not None:
        scene.max_bounces = max(0, bounces)
    return render_scene(scene, width, height, samples, seed)


def render_to_file(name: str, output_path: str, width: int = 320,
                   height: int = 240, samples: int = 4,
                   seed: Optional[int] = None,
                   bounces: Optional[int] = None) -> dict:
    image = render_scene_name(name, width, height, samples, seed, bounces)
    ii.save_image(image, output_path)
    scene = get_scene(name)
    return {
        "scene": name,
        "path": output_path,
        "width": width,
        "height": height,
        "samples": samples,
        "summary": f"physics-rendered {scene.description}",
    }


# ===================================================================== #
# Natural-language routing
# ===================================================================== #

_SCENE_WORDS: dict[str, tuple] = {
    "space_planet": ("우주", "행성", "지구", "space", "planet", "earth"),
    "sunset_sea": ("노을", "해변", "바다", "바닷가", "일몰", "sunset",
                   "beach", "sea", "ocean"),
    "marble_room": ("대리석", "실내", "marble", "room", "interior"),
    "metal_spheres": ("금속", "구체", "크롬", "metal", "chrome", "gold",
                      "copper", "sphere"),
    "nebula": ("성운", "은하", "nebula", "galaxy"),
}

_RENDER_INTENT = ("사실", "실사", "포토", "렌더", "사진", "photoreal",
                  "realistic", "render", "photo", "ray", "그려", "그림",
                  "draw")


def parse_render_request(text: str) -> Optional[str]:
    """The physically renderable scene a request names, or None.

    Fires only when the request carries an explicit image intent
    (그려/렌더/실사/사실적/draw/render/...) AND names a scene the
    renderer stages — an incidental "우주" in a different request
    ("우주 시뮬레이션 앱 만들어줘") never triggers a render.
    """
    lower = (text or "").lower()
    if not lower:
        return None
    if not any(word in lower for word in _RENDER_INTENT):
        return None
    for scene, words in _SCENE_WORDS.items():
        if any(word in lower for word in words):
            return scene
    return None


def default_render_path(scene_name: str) -> str:
    """Artifact path for a rendered scene under the brain data dir."""
    directory = os.path.join(os.path.expanduser("~"), ".rena_brain",
                             "images")
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"render_{scene_name}_{stamp}.png")
