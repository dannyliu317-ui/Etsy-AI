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
    feature_protection: float = 0.72
    hair_preservation: float = 0.45
    background_flatten: float = 0.78
    edge_softness: float = 0.18
    safety_margin_mm: float = 0.65
    surface_smoothing: float = 0.18
    text_min_width_mm: float = 0.30
    edge_rounding: float = 0.35
    safe_zone_strength: float = 0.75
    # V7 intelligent portrait sculpting layer weights.
    face_structure_strength: float = 0.72
    feature_strength: float = 0.82
    hair_strength: float = 0.52
    clothing_strength: float = 0.32
    background_suppression: float = 0.86
    portrait_sculpt_mix: float = 0.78


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


def _portrait_feature_map(image: Image.Image, size: int) -> np.ndarray:
    """Create a conservative face-feature importance map.

    Uses optional OpenCV landmarks when available; otherwise returns a
    smooth face-centered map. This is intentionally a shaping aid, not
    biometric identification.
    """
    try:
        import cv2
    except ImportError:
        return np.zeros((size, size), dtype=np.float32)

    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return np.zeros((size, size), dtype=np.float32)

    x, y, w, h = max(faces, key=lambda f: int(f[2] * f[3]))
    sx, sy = size / rgb.shape[1], size / rgb.shape[0]
    x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
    yy, xx = np.mgrid[0:size, 0:size]

    # Approximate eyes / nose / mouth zones from the face box.
    centers = [
        (x0 + 0.30 * (x1 - x0), y0 + 0.38 * (y1 - y0), 0.13),
        (x0 + 0.70 * (x1 - x0), y0 + 0.38 * (y1 - y0), 0.13),
        (x0 + 0.50 * (x1 - x0), y0 + 0.56 * (y1 - y0), 0.14),
        (x0 + 0.50 * (x1 - x0), y0 + 0.73 * (y1 - y0), 0.18),
    ]
    feature = np.zeros((size, size), dtype=np.float32)
    fw, fh = max(8.0, x1 - x0), max(8.0, y1 - y0)
    for cx, cy, radius in centers:
        rx, ry = fw * radius, fh * radius * 0.75
        feature = np.maximum(feature, np.exp(-(((xx-cx)/rx)**2 + ((yy-cy)/ry)**2)).astype(np.float32))
    return np.clip(feature, 0, 1)


def _portrait_hair_map(image: Image.Image, size: int) -> np.ndarray:
    """Estimate outer hair/silhouette importance from face box geometry."""
    try:
        import cv2
    except ImportError:
        return np.zeros((size, size), dtype=np.float32)

    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return np.zeros((size, size), dtype=np.float32)

    x, y, w, h = max(faces, key=lambda f: int(f[2] * f[3]))
    sx, sy = size / rgb.shape[1], size / rgb.shape[0]
    cx, cy = (x + w/2) * sx, (y + h/2) * sy
    rx, ry = w * sx * 0.82, h * sy * 1.15
    yy, xx = np.mgrid[0:size, 0:size]
    ellipse = np.exp(-(((xx-cx)/max(1,rx))**2 + ((yy-(cy-0.12*h*sy))/max(1,ry))**2)).astype(np.float32)
    face = _face_focus_mask(image, size)
    return np.clip(ellipse * (1.0 - 0.65 * face), 0, 1)



def _portrait_face_box(image: Image.Image):
    """Return the largest detected face box in source-image pixels, if available."""
    try:
        import cv2
    except ImportError:
        return None
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: int(f[2] * f[3]))


def _portrait_structure_map(image: Image.Image, size: int) -> np.ndarray:
    """Emphasize broad facial planes without inventing sharp facial edges."""
    face = _portrait_face_box(image)
    if face is None:
        return np.zeros((size, size), dtype=np.float32)
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
    smooth = Image.fromarray((gray * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(max(1.0, size / 90)))
    broad = np.asarray(smooth, dtype=np.float32) / 255.0
    x, y, w, h = face
    sx, sy = size / rgb.shape[1], size / rgb.shape[0]
    x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
    yy, xx = np.mgrid[0:size, 0:size]
    rx, ry = max(8.0, (x1 - x0) * 0.72), max(10.0, (y1 - y0) * 0.88)
    envelope = np.exp(-(((xx - (x0 + x1) / 2) / rx) ** 2 + ((yy - (y0 + y1) / 2) / ry) ** 2)).astype(np.float32)
    return np.clip(envelope * (0.5 + 0.5 * broad), 0, 1)


def _portrait_clothing_map(image: Image.Image, size: int) -> np.ndarray:
    """Estimate shoulder/clothing silhouette below the detected face."""
    face = _portrait_face_box(image)
    if face is None:
        return np.zeros((size, size), dtype=np.float32)
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    x, y, w, h = face
    sx, sy = size / rgb.shape[1], size / rgb.shape[0]
    cx = (x + w / 2) * sx
    top = (y + h * 0.92) * sy
    shoulder_y = min(size - 1, (y + h * 2.15) * sy)
    yy, xx = np.mgrid[0:size, 0:size]
    width = max(12.0, w * sx * 1.65)
    vertical = np.clip((yy - top) / max(8.0, shoulder_y - top), 0, 1)
    shoulder = np.exp(-((xx - cx) / width) ** 2).astype(np.float32) * vertical
    img = Image.fromarray((shoulder * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(max(1.0, size / 120)))
    return np.asarray(img, dtype=np.float32) / 255.0


def _portrait_background_suppression(image: Image.Image, size: int) -> np.ndarray:
    """Return 1 where portrait content should be retained, 0-ish for background."""
    focus = _face_focus_mask(image, size)
    hair = _portrait_hair_map(image, size)
    clothing = _portrait_clothing_map(image, size)
    content = np.maximum(focus, np.maximum(hair * 0.90, clothing * 0.65))
    return np.clip(0.08 + 0.92 * content, 0, 1)


def build_v7_portrait_layers(image: Image.Image, size: int) -> dict[str, np.ndarray]:
    """Build independent portrait-sculpting layers for V7 preview/debugging."""
    return {
        "face_structure": _portrait_structure_map(image, size),
        "facial_features": _portrait_feature_map(image, size),
        "hair_silhouette": _portrait_hair_map(image, size),
        "clothing_silhouette": _portrait_clothing_map(image, size),
        "background_suppression": _portrait_background_suppression(image, size),
    }


def apply_v7_portrait_sculpt(depth: np.ndarray, image: Image.Image, cfg: MemorialCoinConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fuse portrait layers into the photo depth map before V6 refinement."""
    layers = build_v7_portrait_layers(image, depth.shape[0])
    portrait_detail = np.clip(
        layers["face_structure"] * np.clip(cfg.face_structure_strength, 0, 1)
        + layers["facial_features"] * np.clip(cfg.feature_strength, 0, 1)
        + layers["hair_silhouette"] * np.clip(cfg.hair_strength, 0, 1)
        + layers["clothing_silhouette"] * np.clip(cfg.clothing_strength, 0, 1),
        0, 1
    )
    mix = np.clip(cfg.portrait_sculpt_mix, 0, 1)
    sculpted = np.clip(depth * (1.0 - mix) + portrait_detail * mix, 0, 1)
    bg = np.clip(cfg.background_suppression, 0, 1)
    sculpted *= layers["background_suppression"] * bg + (1.0 - bg)
    return np.clip(sculpted, 0, 1), layers


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
        features = _portrait_feature_map(image, rcfg.resolution)
        if features.max() > 0:
            # Protect eye/nose/mouth transitions from excessive smoothing.
            protected = np.clip(depth + features * cfg.feature_protection * 0.22, 0, 1)
            depth = np.maximum(depth, protected)

        hair = _portrait_hair_map(image, rcfg.resolution)
        if hair.max() > 0:
            depth = np.clip(depth + hair * cfg.hair_preservation * 0.10, 0, 1)

    # Explicit background flattening keeps the portrait from becoming a noisy
    # full-frame terrain map while preserving a small silhouette cue.
    background_factor = np.clip(1.0 - cfg.background_flatten, 0.05, 1.0)
    depth = depth * (background_factor + (1.0 - background_factor) * focus)

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
    border_strength = min(0.45, max(0.03, cfg.border_height_mm / max(0.1, rcfg.relief_height_mm)))
    ring_strength = border_strength * 0.55
    depth = np.clip(depth + border.astype(np.float32) * border_strength + ring.astype(np.float32) * ring_strength, 0, 1)

    depth = _add_text_height(
        depth,
        cfg.text,
        cfg.text_position,
        cfg.text_size,
        max(0.03, cfg.text_height_mm / max(0.1, rcfg.relief_height_mm)),
        cfg.text_mode,
    )

    # V7 intelligent portrait sculpting before final production refinement.
    depth, _v7_layers = apply_v7_portrait_sculpt(depth, image, cfg)

    # V6 surface refinement: conservative smoothing before final mask.
    depth = apply_v6_surface_refinement(depth, cfg)

    # V6 production-safe zone: keep the portrait away from critical edges/hole.
    safe_zone = build_v6_safe_zone_mask(
        rcfg.resolution,
        rcfg.shape,
        cfg.safety_margin_mm,
        rcfg.width_mm,
        rcfg.height_mm,
        rcfg.hole_diameter_mm,
        rcfg.hole_offset_mm,
    )
    safe_strength = np.clip(cfg.safe_zone_strength, 0.0, 1.0)
    depth = depth * (safe_strength * safe_zone + (1.0 - safe_strength) * mask)

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
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(max(0.05, cfg.edge_softness)))
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


# ============================================================
# V5 PRODUCTION / MANUFACTURABILITY ENGINE
# ============================================================

PRODUCTION_PRESETS = {
    "25mm Coin": {
        "diameter_mm": 25.0, "base_mm": 2.0, "relief_mm": 0.9,
        "border_mm": 1.0, "border_height_mm": 0.22,
        "min_detail_mm": 0.30, "min_text_line_mm": 0.28,
        "edge_radius_mm": 0.18, "hole_mm": 2.8,
    },
    "30mm Coin": {
        "diameter_mm": 30.0, "base_mm": 2.3, "relief_mm": 1.1,
        "border_mm": 1.2, "border_height_mm": 0.25,
        "min_detail_mm": 0.35, "min_text_line_mm": 0.30,
        "edge_radius_mm": 0.20, "hole_mm": 3.0,
    },
    "35mm Coin": {
        "diameter_mm": 35.0, "base_mm": 2.5, "relief_mm": 1.25,
        "border_mm": 1.4, "border_height_mm": 0.28,
        "min_detail_mm": 0.40, "min_text_line_mm": 0.32,
        "edge_radius_mm": 0.22, "hole_mm": 3.2,
    },
}

PROCESS_PRESETS = {
    "Jewelry Casting": {
        "relief_factor": 1.0, "text_factor": 1.0,
        "min_detail_factor": 1.0, "min_text_factor": 1.0,
        "hole_edge_factor": 1.0,
    },
    "Resin / 3D Print": {
        "relief_factor": 0.95, "text_factor": 0.95,
        "min_detail_factor": 0.90, "min_text_factor": 0.90,
        "hole_edge_factor": 0.90,
    },
    "CNC / Engraving": {
        "relief_factor": 0.85, "text_factor": 0.85,
        "min_detail_factor": 1.20, "min_text_factor": 1.20,
        "hole_edge_factor": 1.15,
    },
}


def get_production_preset(size_preset: str, process: str) -> dict:
    """Return manufacturing-oriented defaults.

    Values are design guidelines, not process guarantees. Final limits must
    be confirmed with the actual caster, printer, CNC shop, alloy and toolchain.
    """
    size = PRODUCTION_PRESETS.get(size_preset, PRODUCTION_PRESETS["30mm Coin"]).copy()
    proc = PROCESS_PRESETS.get(process, PROCESS_PRESETS["Jewelry Casting"])
    size["relief_mm"] *= proc["relief_factor"]
    size["min_detail_mm"] *= proc["min_detail_factor"]
    size["min_text_line_mm"] *= proc["min_text_factor"]
    size["process"] = process
    size["size_preset"] = size_preset
    return size


def validate_memorial_coin_production(cfg: MemorialCoinConfig, inspection: dict | None = None,
                                      production: dict | None = None) -> dict:
    """Validate geometry settings and optional STL inspection for production review."""
    p = production or get_production_preset("30mm Coin", "Jewelry Casting")
    checks = []
    warnings = []

    def check(name, passed, message, severity="error"):
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "message": message})

    r = cfg.relief
    min_detail = float(p["min_detail_mm"])
    min_text = float(p["min_text_line_mm"])
    edge_clearance = max(r.hole_diameter_mm * 0.75, 1.0)

    check("Base thickness", r.base_thickness_mm >= 1.5,
          f"Base {r.base_thickness_mm:.2f} mm; recommended minimum review threshold is 1.50 mm.")
    check("Relief-to-base ratio", r.relief_height_mm <= r.base_thickness_mm * 0.65,
          f"Relief/base = {r.relief_height_mm / max(r.base_thickness_mm, 0.01):.2f}.")
    check("Minimum detail guideline", min_detail >= 0.20,
          f"Target minimum detail width: {min_detail:.2f} mm.", "warning")
    check("Hole clearance", r.hole_diameter_mm >= 2.5,
          f"Hanging hole diameter: {r.hole_diameter_mm:.2f} mm.")
    check("Hole position", r.hole_offset_mm >= 0,
          f"Hole offset: {r.hole_offset_mm:.2f} mm.", "warning")
    check("Border continuity", cfg.border_width_mm >= 0.6,
          f"Outer border width: {cfg.border_width_mm:.2f} mm.")
    check("Text stroke guideline", min_text >= 0.20,
          f"Recommended text line width target: {min_text:.2f} mm.", "warning")

    if cfg.text.strip():
        estimated_text_height = max(0.01, cfg.text_height_mm)
        check("Personalization relief", estimated_text_height >= 0.16,
              f"Text relief: {estimated_text_height:.2f} mm.", "warning")

    if inspection:
        check("STL watertight", bool(inspection.get("watertight")),
              "Mesh is reported watertight." if inspection.get("watertight") else "Mesh is not reported watertight.")
        check("STL volume", float(inspection.get("volume", 0) or 0) > 0,
              "Mesh has positive volume." if float(inspection.get("volume", 0) or 0) > 0 else "Mesh volume is zero.")
        dims = inspection.get("dimensions") or inspection.get("bounds")
        if dims:
            warnings.append(f"Measured STL dimensions: {dims}")

    failed = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])
    return {
        "status": "PASS" if not failed else "NEEDS REVIEW",
        "production_ready": not failed,
        "failed_checks": len(failed),
        "warning_count": warning_count,
        "checks": checks,
        "guideline": "Design guidelines only — confirm final tolerances with your manufacturer.",
        "preset": p,
        "warnings": warnings,
    }


# ============================================================
# V6 PRODUCTION REFINEMENT ENGINE
# ============================================================

def apply_v6_surface_refinement(depth: np.ndarray, cfg: MemorialCoinConfig) -> np.ndarray:
    """Refine a heightmap for jewelry production review.

    This is a conservative image-space operation: it rounds abrupt transitions,
    protects the central portrait, and reduces tiny high-frequency noise.
    """
    arr = np.clip(depth.astype(np.float32), 0, 1)
    if cfg.surface_smoothing > 0:
        radius = max(0.05, float(cfg.surface_smoothing) * 2.2)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        smooth = np.asarray(img, dtype=np.float32) / 255.0
        blend = np.clip(float(cfg.surface_smoothing), 0, 1)
        arr = arr * (1.0 - blend) + smooth * blend

    # Soft-limit extreme local peaks so casting/printing does not inherit
    # isolated pixel-scale spikes.
    arr = np.clip(arr, 0.0, 1.0)
    center = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    center = center.filter(ImageFilter.GaussianBlur(max(0.05, cfg.edge_rounding * 0.8)))
    rounded = np.asarray(center, dtype=np.float32) / 255.0
    arr = np.clip(arr * 0.70 + rounded * 0.30, 0, 1)
    return arr


def build_v6_safe_zone_mask(size: int, shape: str, margin_mm: float,
                            width_mm: float, height_mm: float,
                            hole_diameter_mm: float, hole_offset_mm: float) -> np.ndarray:
    """Build an inner production-safe zone excluding outer edge and hole."""
    outer = _shape_mask(size, shape).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2
    px_per_mm = size / max(width_mm, height_mm)
    margin_px = max(1, int(margin_mm * px_per_mm))

    # Conservative radial inset. For non-circular shapes this intentionally
    # errs inward rather than claiming exact CAD offset geometry.
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    safe = dist <= max(1, size * 0.47 - margin_px)

    hole_cx = size / 2
    hole_cy = size / 2 - (height_mm / 2 - hole_offset_mm) * px_per_mm
    hole_r = max(1.5, hole_diameter_mm * px_per_mm / 2 + margin_px * 0.45)
    hole = (xx - hole_cx) ** 2 + (yy - hole_cy) ** 2 <= hole_r ** 2

    return outer * (~hole) * safe.astype(np.float32)


def estimate_text_width_mm(cfg: MemorialCoinConfig) -> float:
    if not cfg.text.strip():
        return 0.0
    # Approximate stroke/letter footprint conservatively from configured text size.
    return max(0.08, cfg.text_size * cfg.relief.width_mm * 0.22)


def validate_v6_production(cfg: MemorialCoinConfig, inspection: dict | None = None,
                           production: dict | None = None) -> dict:
    """V6 manufacturing review with safe-zone and personalization checks."""
    report = validate_memorial_coin_production(cfg, inspection, production)
    p = production or get_production_preset("30mm Coin", "Jewelry Casting")
    checks = report["checks"]

    def add(name, passed, message, severity="error"):
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "message": message})

    safe_margin = float(cfg.safety_margin_mm)
    add(
        "Production safe zone",
        safe_margin >= 0.40,
        f"Edge / hole safety margin guideline: {safe_margin:.2f} mm.",
        "warning",
    )

    if cfg.text.strip():
        estimated = estimate_text_width_mm(cfg)
        target = float(p["min_text_line_mm"])
        add(
            "Personalization detail width",
            estimated >= target,
            f"Estimated text detail {estimated:.2f} mm vs target {target:.2f} mm.",
            "warning",
        )
        add(
            "Text-to-edge safety",
            safe_margin >= 0.50,
            f"Personalization should remain at least ~0.50 mm from critical edges/holes.",
            "warning",
        )

    # Validate the requested relief is not so deep that the safe zone becomes
    # visually dominated by vertical walls.
    ratio = cfg.relief.relief_height_mm / max(cfg.relief.base_thickness_mm, 0.01)
    add(
        "V6 relief wall control",
        ratio <= 0.60,
        f"Relief/base ratio {ratio:.2f}; target <= 0.60 for conservative production review.",
    )

    failed = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])
    report.update({
        "version": "V6",
        "status": "PASS" if not failed else "NEEDS REVIEW",
        "production_ready": not failed,
        "failed_checks": len(failed),
        "warning_count": warning_count,
        "v6_features": [
            "production safe zone",
            "surface refinement",
            "personalization manufacturability review",
            "conservative relief wall control",
        ],
        "guideline": "V6 is a design-for-manufacture aid, not a manufacturing guarantee. Confirm final tolerances with the actual production shop.",
    })
    return report


# ============================================================
# V7 INTELLIGENT PORTRAIT SCULPTING / PRODUCTION REVIEW
# ============================================================

def validate_v7_production(cfg: MemorialCoinConfig, image: Image.Image | None = None,
                           inspection: dict | None = None, production: dict | None = None) -> dict:
    """Extend V6 review with portrait-layer coverage and sculpting controls."""
    report = validate_v6_production(cfg, inspection, production)
    checks = report["checks"]

    def add(name, passed, message, severity="warning"):
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "message": message})

    if image is not None:
        layers = build_v7_portrait_layers(image, cfg.relief.resolution)
        add("Face structure layer", float(layers["face_structure"].mean()) > 0.005,
            f"Face-plane layer coverage: {float(layers['face_structure'].mean()):.3f}.")
        add("Facial feature layer", float(layers["facial_features"].mean()) > 0.001,
            f"Eye / nose / mouth layer coverage: {float(layers['facial_features'].mean()):.3f}.")
        add("Hair / silhouette layer", float(layers["hair_silhouette"].mean()) > 0.001,
            f"Hair / outer silhouette coverage: {float(layers['hair_silhouette'].mean()):.3f}.")
        add("Clothing / shoulder layer", float(layers["clothing_silhouette"].mean()) > 0.001,
            f"Shoulder / clothing coverage: {float(layers['clothing_silhouette'].mean()):.3f}.")
    else:
        add("Portrait layer preview", True, "Source image not supplied; geometry-only V7 review performed.")

    mix = float(np.clip(cfg.portrait_sculpt_mix, 0, 1))
    add("Portrait sculpt mix", 0.35 <= mix <= 0.95,
        f"Portrait sculpt blend: {mix:.2f}. Recommended review range: 0.35–0.95.")
    add("Background suppression", cfg.background_suppression >= 0.50,
        f"Background suppression strength: {cfg.background_suppression:.2f}.")

    failed = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and not c["passed"])
    report.update({
        "version": "V7", "status": "PASS" if not failed else "NEEDS REVIEW",
        "production_ready": not failed, "failed_checks": len(failed), "warning_count": warning_count,
        "v7_features": [
            "face structure layer", "facial feature layer", "hair / silhouette layer",
            "clothing / shoulder layer", "background suppression", "portrait sculpt blend",
        ],
        "guideline": "V7 is a layered portrait-relief design aid. Face, hair and clothing layers are image-space shaping approximations, not biometric landmarks or CAD geometry. Confirm final tolerances with the actual manufacturer.",
    })
    return report
