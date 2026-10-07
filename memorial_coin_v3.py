"""
Etsy AI V3 - Jewelry Memorial Coin Portrait Relief
Optional face detection and coin-style relief shaping layered on V2.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

from relief_generator import (
    ReliefConfig,
    _shape_mask,
    _apply_hole,
    _mesh_from_heightmap,
    _normalize,
    make_depth_map,
    inspect_stl,
)


CoinStyle = Literal["Classic Coin", "Deep Relief", "Soft Relief"]


@dataclass
class MemorialCoinConfig:
    relief: ReliefConfig
    coin_style: CoinStyle = "Classic Coin"
    face_focus: bool = True
    background_strength: float = 0.18
    face_boost: float = 1.35
    border_width_mm: float = 1.2
    border_height_mm: float = 0.25
    inner_ring_width_mm: float = 0.55
    text: str = ""
    text_height_mm: float = 0.22
    text_mode: str = "Raised"
    text_position: str = "Bottom"
    text_size: float = 0.16


def _face_focus_mask(image: Image.Image, size: int) -> np.ndarray:
    """Best-effort face localization.

    OpenCV is optional. If unavailable or no face is found, returns a
    conservative oval centered in the portrait rather than pretending
    detection succeeded.
    """
    try:
        import cv2
    except ImportError:
        return np.ones((size, size), dtype=np.float32)

    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return np.ones((size, size), dtype=np.float32)

    x, y, w, h = max(faces, key=lambda f: int(f[2] * f[3]))
    cx = (x + w / 2) / max(1, rgb.shape[1]) * size
    cy = (y + h / 2) / max(1, rgb.shape[0]) * size
    rx = max(12, w / max(1, rgb.shape[1]) * size * 0.95)
    ry = max(16, h / max(1, rgb.shape[0]) * size * 1.35)

    yy, xx = np.mgrid[0:size, 0:size]
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    focus = np.exp(-1.8 * d).astype(np.float32)
    return np.clip(focus, 0, 1)


def _ring_masks(size: int, shape: str, border_width_px: int, inner_width_px: int):
    outer = _shape_mask(size, shape)
    inner = _shape_mask(size, shape)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if shape == "Circle":
        boundary = size * 0.47
        inner_boundary = boundary - border_width_px
        inner_mask = dist <= inner_boundary
        inner2 = dist <= max(1, inner_boundary - inner_width_px)
    elif shape == "Oval":
        rx, ry = size * 0.47, size * 0.43
        q = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        inner_mask = q <= max(0.0, 1 - border_width_px / max(rx, ry))
        inner2 = q <= max(0.0, 1 - (border_width_px + inner_width_px) / max(rx, ry))
    else:
        # For heart/dog-tag, use erosion from scipy if available; otherwise
        # use a soft radial approximation that keeps the outer border visible.
        try:
            from scipy.ndimage import binary_erosion
            inner_mask = binary_erosion(outer, iterations=max(1, border_width_px))
            inner2 = binary_erosion(outer, iterations=max(1, border_width_px + inner_width_px))
        except ImportError:
            inner_mask = outer
            inner2 = outer

    return outer, inner_mask, inner2


def _add_text_height(
    depth: np.ndarray,
    text: str,
    position: str,
    size_ratio: float,
    amount: float,
    mode: str,
) -> np.ndarray:
    if not text.strip():
        return depth

    n = depth.shape[0]
    canvas = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(canvas)
    font_size = max(10, int(n * size_ratio))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(2, (n - tw) // 2)
    if position == "Top":
        y = max(2, int(n * 0.07))
    elif position == "Center":
        y = max(2, (n - th) // 2)
    else:
        y = max(2, int(n * 0.82) - th)

    draw.text((x, y), text, fill=255, font=font)
    text_map = np.asarray(canvas, dtype=np.float32) / 255.0
    if mode == "Engraved":
        depth = np.clip(depth - text_map * amount, 0, 1)
    else:
        depth = np.clip(depth + text_map * amount, 0, 1)
    return depth


def make_memorial_coin_depth(image_bytes: bytes, cfg: MemorialCoinConfig) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    rcfg = cfg.relief
    depth = make_depth_map(
        image_bytes,
        resolution=rcfg.resolution,
        mode=rcfg.depth_model,
        invert=rcfg.invert_depth,
        gamma=rcfg.depth_gamma,
        relief_mode="Photo",
    )

    focus = _face_focus_mask(image, rcfg.resolution) if cfg.face_focus else np.ones_like(depth)
    # Keep a small amount of background relief so silhouettes remain natural,
    # but strongly prioritize facial geometry.
    depth = depth * (cfg.background_strength + (1.0 - cfg.background_strength) * focus)

    if cfg.face_focus:
        depth = np.power(np.clip(depth, 0, 1), 1.0 / max(0.65, cfg.face_boost))

    if cfg.coin_style == "Deep Relief":
        depth = np.power(np.clip(depth, 0, 1), 0.78)
    elif cfg.coin_style == "Soft Relief":
        depth = np.power(np.clip(depth, 0, 1), 1.22)

    # Compress the photo relief slightly so the border/rim remains visually dominant.
    depth = np.clip(depth * 0.88, 0, 1)

    mask = _apply_hole(
        _shape_mask(rcfg.resolution, rcfg.shape),
        rcfg.hole_diameter_mm,
        rcfg.hole_offset_mm,
        rcfg.width_mm,
        rcfg.height_mm,
    )

    px_per_mm = rcfg.resolution / max(rcfg.width_mm, rcfg.height_mm)
    border_px = max(1, int(cfg.border_width_mm * px_per_mm))
    inner_px = max(1, int(cfg.inner_ring_width_mm * px_per_mm))
    outer, inner, inner2 = _ring_masks(rcfg.resolution, rcfg.shape, border_px, inner_px)

    # Raised perimeter and inner ring.
    border = outer & ~inner
    ring = inner & ~inner2
    depth = np.clip(depth + border.astype(np.float32) * 0.18 + ring.astype(np.float32) * 0.10, 0, 1)

    depth = _add_text_height(
        depth,
        cfg.text,
        cfg.text_position,
        cfg.text_size,
        max(0.03, cfg.text_height_mm / max(0.1, rcfg.relief_height_mm)),
        cfg.text_mode,
    )

    # Remove relief from the hanging hole and outside the coin.
    depth *= mask
    return _normalize(depth)


def generate_memorial_coin_stl(image_bytes: bytes, cfg: MemorialCoinConfig) -> bytes:
    depth = make_memorial_coin_depth(image_bytes, cfg)
    rcfg = cfg.relief
    mask = _apply_hole(
        _shape_mask(rcfg.resolution, rcfg.shape),
        rcfg.hole_diameter_mm,
        rcfg.hole_offset_mm,
        rcfg.width_mm,
        rcfg.height_mm,
    )
    # Keep the edge crisp enough for casting/printing.
    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(0.35))
    depth = np.asarray(depth_img, dtype=np.float32) / 255.0
    mesh = _mesh_from_heightmap(depth, mask, rcfg)
    mesh.apply_translation(-mesh.centroid)
    return mesh.export(file_type="stl")


def preview_memorial_coin(image_bytes: bytes, cfg: MemorialCoinConfig) -> bytes:
    depth = make_memorial_coin_depth(image_bytes, cfg)
    img = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def inspect_memorial_coin(stl_bytes: bytes) -> dict:
    result = inspect_stl(stl_bytes)
    result["product_type"] = "Jewelry Memorial Portrait Coin"
    return result
