"""
Etsy AI V1 - Personalized Relief Pendant Generator

Pipeline:
photo -> optional AI depth map -> heightmap -> relief mesh ->
parametric pendant base -> hanging hole -> watertight STL.

The AI depth model is optional. A grayscale heightmap fallback keeps the
feature usable without downloading a large model.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps

Shape = Literal["Circle", "Oval", "Heart", "Dog Tag"]


@dataclass
class ReliefConfig:
    width_mm: float = 30.0
    height_mm: float = 30.0
    base_thickness_mm: float = 2.0
    relief_height_mm: float = 1.2
    hole_diameter_mm: float = 3.0
    hole_offset_mm: float = 4.0
    resolution: int = 180
    smoothing: float = 0.35
    contrast: float = 1.15
    shape: Shape = "Circle"
    depth_model: str = "AI Depth"
    relief_mode: str = "Portrait"
    invert_depth: bool = False
    depth_gamma: float = 0.85
    edge_fade: float = 0.08


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _resize_crop(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "black")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _grayscale_depth(image: Image.Image, size: int) -> np.ndarray:
    gray = ImageOps.grayscale(_resize_crop(image, size))
    gray = ImageOps.autocontrast(gray)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    arr = np.power(arr, 1.0)
    return arr


def _ai_depth(image: Image.Image, size: int) -> np.ndarray:
    """Optional Depth Anything V2 inference.

    Requires: torch, transformers.
    The model is loaded lazily so the normal app does not pay the startup cost.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError(
            "AI depth mode requires optional packages: torch and transformers."
        ) from exc

    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()

    source = _resize_crop(image, size)
    inputs = processor(images=source, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    depth = outputs.predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(size, size),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    arr = depth.cpu().numpy()
    # Depth models generally make closer objects larger values.
    return _normalize(arr)


def make_depth_map(
    image_bytes: bytes,
    resolution: int = 180,
    mode: str = "Grayscale",
    invert: bool = False,
    gamma: float = 1.0,
    relief_mode: str = "Portrait",
) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if mode == "AI Depth":
        depth = _ai_depth(image, resolution)
    else:
        depth = _grayscale_depth(image, resolution)

    if relief_mode == "Portrait":
        y, x = np.mgrid[0:resolution, 0:resolution]
        cx = (resolution - 1) / 2
        cy = (resolution - 1) / 2
        rx = resolution * 0.48
        ry = resolution * 0.48
        subject_weight = np.clip(
            1.0 - (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2) * 0.35,
            0.55,
            1.0,
        )
        depth = depth * subject_weight

    if invert:
        depth = 1.0 - depth

    gamma = max(0.35, min(2.5, float(gamma)))
    depth = np.power(np.clip(depth, 0, 1), gamma)
    return _normalize(depth)


def validate_relief_config(cfg: ReliefConfig) -> list[str]:
    warnings = []
    if cfg.base_thickness_mm < 1.2:
        warnings.append("Base thickness is below 1.2 mm; increase it for production.")
    if cfg.relief_height_mm > cfg.base_thickness_mm * 0.9:
        warnings.append("Relief height is high relative to the base.")
    if cfg.hole_diameter_mm < 2.0:
        warnings.append("Hanging hole is under 2 mm; verify the jump ring.")
    if cfg.resolution < 160:
        warnings.append("Low mesh resolution may soften facial details.")
    if cfg.width_mm < 20:
        warnings.append("Pendant width is under 20 mm; portrait details may be difficult to reproduce.")
    return warnings


def _shape_mask(size: int, shape: Shape) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2

    if shape == "Circle":
        r = size * 0.47
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r

    elif shape == "Oval":
        rx, ry = size * 0.47, size * 0.43
        mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1

    elif shape == "Dog Tag":
        # Rounded rectangle with a centered top hole region.
        xx = np.abs(x - cx)
        yy = np.abs(y - cy)
        w, h, radius = size * 0.43, size * 0.46, size * 0.10
        core = (xx <= w) & (yy <= h)
        corner = ((xx - (w - radius)) ** 2 + (yy - (h - radius)) ** 2 <= radius ** 2)
        mask = core & ((xx <= w - radius) | (yy <= h - radius) | corner)

    elif shape == "Heart":
        # Standard implicit heart, normalized to the square.
        xn = (x - cx) / (size * 0.43)
        yn = -(y - cy) / (size * 0.43)
        mask = (xn * xn + yn * yn - 1) ** 3 - xn * xn * yn ** 3 <= 0

    else:
        raise ValueError(f"Unsupported pendant shape: {shape}")

    return mask.astype(bool)


def _apply_hole(mask: np.ndarray, diameter_mm: float, offset_mm: float, width_mm: float, height_mm: float) -> np.ndarray:
    size = mask.shape[0]
    # Hole center near the top edge, measured from the pendant's top.
    px_per_mm = size / max(width_mm, height_mm)
    cx = size / 2
    cy = size / 2 - (height_mm / 2 - offset_mm) * px_per_mm
    r = max(1.5, diameter_mm * px_per_mm / 2)
    yy, xx = np.mgrid[0:size, 0:size]
    hole = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return mask & ~hole


def _mesh_from_heightmap(height: np.ndarray, mask: np.ndarray, cfg: ReliefConfig):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("STL export requires the 'trimesh' package.") from exc

    n = height.shape[0]
    xs = np.linspace(-cfg.width_mm / 2, cfg.width_mm / 2, n)
    ys = np.linspace(-cfg.height_mm / 2, cfg.height_mm / 2, n)
    xx, yy = np.meshgrid(xs, ys)

    z_top = cfg.base_thickness_mm + np.clip(height, 0, 1) * cfg.relief_height_mm

    # Top grid vertices. Outside the pendant remains at base level but is
    # removed by the mask when faces are generated.
    top_vertices = np.column_stack((xx.ravel(), yy.ravel(), z_top.ravel()))
    bottom_vertices = np.column_stack(
        (xx.ravel(), yy.ravel(), np.zeros(n * n, dtype=np.float32))
    )
    vertices = np.vstack((top_vertices, bottom_vertices))

    top_faces = []
    bottom_faces = []

    def idx(r, c):
        return r * n + c

    offset = n * n

    for r in range(n - 1):
        for c in range(n - 1):
            a, b = idx(r, c), idx(r, c + 1)
            d, e = idx(r + 1, c), idx(r + 1, c + 1)
            if mask[r, c] and mask[r, c + 1] and mask[r + 1, c] and mask[r + 1, c + 1]:
                top_faces.extend(((a, b, e), (a, e, d)))
                bottom_faces.extend(
                    ((offset + a, offset + e, offset + b),
                     (offset + a, offset + d, offset + e))
                )

    # Build side walls from boundary edges of the top surface.
    # An edge used by exactly one top triangle is on the perimeter
    # (including the perimeter of the hanging hole).
    edge_counts = {}
    for face in top_faces:
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((u, v)))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    side_faces = []
    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        side_faces.append((u, v, offset + v))
        side_faces.append((u, offset + v, offset + u))

    faces = np.asarray(
        top_faces + bottom_faces + side_faces,
        dtype=np.int64,
    )
    if len(faces) == 0:
        raise ValueError("The selected pendant mask produced no printable faces.")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # Repair and remove disconnected fragments.
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.process(validate=True)
    try:
        mesh.fill_holes()
    except Exception:
        pass

    return mesh


def generate_relief_stl(
    image_bytes: bytes,
    cfg: ReliefConfig,
) -> bytes:
    depth = make_depth_map(
        image_bytes,
        resolution=cfg.resolution,
        mode=cfg.depth_model,
        invert=cfg.invert_depth,
        gamma=cfg.depth_gamma,
        relief_mode=cfg.relief_mode,
    )

    # Contrast/smoothing tuned for shallow jewelry relief.
    depth = np.clip((depth - 0.5) * cfg.contrast + 0.5, 0, 1)
    depth = Image.fromarray((depth * 255).astype(np.uint8))
    if cfg.smoothing > 0:
        radius = max(0.1, cfg.smoothing * 1.8)
        depth = depth.filter(ImageFilter.GaussianBlur(radius=radius))
    depth = np.asarray(depth, dtype=np.float32) / 255.0

    mask = _shape_mask(cfg.resolution, cfg.shape)
    mask = _apply_hole(
        mask,
        cfg.hole_diameter_mm,
        cfg.hole_offset_mm,
        cfg.width_mm,
        cfg.height_mm,
    )

    # The relief should fade toward the perimeter for a cleaner castable edge.
    yy, xx = np.mgrid[0:cfg.resolution, 0:cfg.resolution]
    edge_dist = np.minimum.reduce(
        [xx, yy, cfg.resolution - 1 - xx, cfg.resolution - 1 - yy]
    )
    fade = np.clip(edge_dist / (cfg.resolution * max(0.02, cfg.edge_fade)), 0, 1)
    depth *= fade
    depth *= mask

    mesh = _mesh_from_heightmap(depth, mask, cfg)

    # Scale/center and export binary STL.
    mesh.apply_translation(-mesh.centroid)
    return mesh.export(file_type="stl")


def preview_heightmap(image_bytes: bytes, cfg: ReliefConfig) -> bytes:
    depth = make_depth_map(
        image_bytes, cfg.resolution, cfg.depth_model,
        invert=cfg.invert_depth, gamma=cfg.depth_gamma,
        relief_mode=cfg.relief_mode,
    )
    depth = np.clip((depth - 0.5) * cfg.contrast + 0.5, 0, 1)
    img = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def preview_hillshade(image_bytes: bytes, cfg: ReliefConfig) -> bytes:
    depth = make_depth_map(
        image_bytes, cfg.resolution, cfg.depth_model,
        invert=cfg.invert_depth, gamma=cfg.depth_gamma,
        relief_mode=cfg.relief_mode,
    )
    depth = np.clip((depth - 0.5) * cfg.contrast + 0.5, 0, 1)
    mask = _apply_hole(
        _shape_mask(cfg.resolution, cfg.shape),
        cfg.hole_diameter_mm, cfg.hole_offset_mm,
        cfg.width_mm, cfg.height_mm,
    )
    z = depth * cfg.relief_height_mm
    gy, gx = np.gradient(z)
    nx, ny, nz = -gx, -gy, np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    light = np.clip(nx * -0.45 + ny * -0.55 + nz * 0.72, 0, 1)
    shaded = (0.25 + 0.75 * light) * mask
    img = Image.fromarray((shaded * 255).astype(np.uint8), mode="L")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def export_relief_glb(image_bytes: bytes, cfg: ReliefConfig) -> bytes:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("3D preview requires the 'trimesh' package.") from exc
    depth = make_depth_map(
        image_bytes, cfg.resolution, cfg.depth_model,
        invert=cfg.invert_depth, gamma=cfg.depth_gamma,
        relief_mode=cfg.relief_mode,
    )
    depth = np.clip((depth - 0.5) * cfg.contrast + 0.5, 0, 1)
    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
    if cfg.smoothing > 0:
        depth_img = depth_img.filter(
            ImageFilter.GaussianBlur(max(0.1, cfg.smoothing * 1.8))
        )
    depth = np.asarray(depth_img, dtype=np.float32) / 255.0
    mask = _apply_hole(
        _shape_mask(cfg.resolution, cfg.shape),
        cfg.hole_diameter_mm, cfg.hole_offset_mm,
        cfg.width_mm, cfg.height_mm,
    )
    mesh = _mesh_from_heightmap(depth * 0.98, mask, cfg)
    mesh.apply_translation(-mesh.centroid)
    return mesh.export(file_type="glb")


def inspect_stl(stl_bytes: bytes) -> dict:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("STL inspection requires the 'trimesh' package.") from exc
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl")
    return {
        "watertight": bool(mesh.is_watertight),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "size_x_mm": round(float(mesh.extents[0]), 2),
        "size_y_mm": round(float(mesh.extents[1]), 2),
        "size_z_mm": round(float(mesh.extents[2]), 2),
        "volume_mm3": round(float(abs(mesh.volume)), 2),
    }
