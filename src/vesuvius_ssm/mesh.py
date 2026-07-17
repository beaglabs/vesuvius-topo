from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

from .geometry import CropField, normalize
from .rollout import compute_normals
from .types import SurfaceGrid


@dataclass(frozen=True)
class MeshMetrics:
    coverage: float
    anchor_fraction: float
    probability_adherence: float
    folded_quad_fraction: float
    jump_fraction: float


def extract_component(
    field: CropField,
    threshold: float = 0.45,
    seed_xyz: np.ndarray | None = None,
    minimum_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    mask = field.data >= threshold
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 2))
    if not count:
        raise ValueError(f"no surface component above threshold {threshold}")
    if seed_xyz is not None:
        seed = np.rint(field.local_zyx(seed_xyz)).astype(int)
        if np.all((seed >= 0) & (seed < np.asarray(mask.shape))):
            component_id = int(labels[tuple(seed)])
        else:
            component_id = 0
    else:
        component_id = 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if component_id == 0 or sizes[component_id] < minimum_size:
        component_id = int(np.argmax(sizes))
    component = labels == component_id
    if int(component.sum()) < minimum_size:
        raise ValueError(f"largest surface component has only {int(component.sum())} voxels")
    local_zyx = np.argwhere(component).astype(np.float32)
    physical_xyz = ((local_zyx + field.origin_zyx) * field.scale_zyx)[..., ::-1]
    probability = field.data[component]
    return physical_xyz, probability


def parameterize_points(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points_xyz.mean(axis=0)
    centered = points_xyz - center
    covariance = centered.T @ centered / max(1, len(centered))
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, ::-1]
    uv = centered @ basis[:, :2]
    return uv.astype(np.float32), center.astype(np.float32), basis.astype(np.float32)


def select_seed_layer(
    points_xyz: np.ndarray,
    probability: np.ndarray,
    seed_xyz: np.ndarray,
    half_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, center, basis = parameterize_points(points_xyz)
    normal = basis[:, 2]
    depth = (points_xyz - center) @ normal
    seed_depth = (np.asarray(seed_xyz) - center) @ normal
    selected = np.abs(depth - seed_depth) <= half_thickness
    if selected.sum() < 256:
        raise ValueError(
            f"seed-conditioned layer has only {int(selected.sum())} points; increase layer thickness"
        )
    return points_xyz[selected], probability[selected]


def rasterize_xyz(
    points_xyz: np.ndarray,
    probability: np.ndarray,
    spacing: float,
    close_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv, _, _ = parameterize_points(points_xyz)
    minimum = uv.min(axis=0)
    indices = np.rint((uv - minimum) / spacing).astype(int)
    shape = tuple((indices.max(axis=0) + 1).tolist())
    if np.prod(shape) > 16_000_000:
        raise ValueError(f"UV raster would be too large: {shape}; increase spacing")
    xyz_sum = np.zeros((*shape, 3), dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)
    best_probability = np.zeros(shape, dtype=np.float32)
    best_xyz = np.zeros((*shape, 3), dtype=np.float32)
    for point, score, (u, v) in zip(points_xyz, probability, indices):
        weight = max(float(score), 1e-3) ** 2
        xyz_sum[u, v] += point * weight
        weight_sum[u, v] += weight
        if score >= best_probability[u, v]:
            best_probability[u, v] = score
            best_xyz[u, v] = point
    anchors = weight_sum > 0
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    xyz[anchors] = (xyz_sum[anchors] / weight_sum[anchors, None]).astype(np.float32)
    # Prefer the ridge maximum when a thick prediction band maps to one UV pixel.
    xyz[anchors] = 0.5 * xyz[anchors] + 0.5 * best_xyz[anchors]
    structure = ndimage.generate_binary_structure(2, 2)
    sheet_mask = ndimage.binary_closing(anchors, structure=structure, iterations=close_radius)
    sheet_mask = ndimage.binary_fill_holes(sheet_mask)
    labels, count = ndimage.label(sheet_mask, structure=structure)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        sheet_mask = labels == int(np.argmax(sizes))
    anchors &= sheet_mask
    return xyz, anchors, sheet_mask, best_probability


def fill_enclosed_holes(xyz: np.ndarray, anchors: np.ndarray, sheet_mask: np.ndarray) -> np.ndarray:
    filled = xyz.copy()
    known = anchors & np.isfinite(xyz).all(axis=-1)
    missing = sheet_mask & ~known
    if not missing.any():
        return filled
    known_uv = np.argwhere(known)
    missing_uv = np.argwhere(missing)
    for axis in range(3):
        values = xyz[..., axis][known]
        interpolated = griddata(known_uv, values, missing_uv, method="linear")
        unresolved = ~np.isfinite(interpolated)
        if unresolved.any():
            interpolated[unresolved] = griddata(
                known_uv, values, missing_uv[unresolved], method="nearest"
            )
        filled[..., axis][missing] = interpolated.astype(np.float32)
    return filled


def optimize_surface(
    field: CropField,
    xyz: np.ndarray,
    anchors: np.ndarray,
    sheet_mask: np.ndarray,
    iterations: int = 8,
    smoothing: float = 0.25,
    projection_radius: float = 3.0,
) -> np.ndarray:
    optimized = xyz.copy()
    movable = sheet_mask & ~anchors
    anchor_values = xyz.copy()
    kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for _ in range(iterations):
        valid_float = sheet_mask.astype(np.float32)
        count = ndimage.convolve(valid_float, kernel, mode="constant", cval=0)
        neighbor_mean = np.stack(
            [ndimage.convolve(np.nan_to_num(optimized[..., axis]), kernel, mode="constant", cval=0) for axis in range(3)],
            axis=-1,
        ) / np.maximum(count[..., None], 1)
        optimized[movable] = (
            (1 - smoothing) * optimized[movable] + smoothing * neighbor_mean[movable]
        )
        normals = compute_normals(optimized, sheet_mask)
        for row, col in np.argwhere(movable):
            projected, score = field.project(
                optimized[row, col], normals[row, col], radius=projection_radius, samples=9
            )
            if score >= 0.1:
                optimized[row, col] = projected
        optimized[anchors] = anchor_values[anchors]
    optimized[~sheet_mask] = np.nan
    return optimized


def defect_vertices(xyz: np.ndarray, valid: np.ndarray, jump_factor: float = 3.0) -> np.ndarray:
    defects = np.zeros(valid.shape, dtype=bool)
    horizontal = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    vertical = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:] & valid[:-1]
    edges = np.concatenate((horizontal[horizontal_valid], vertical[vertical_valid]))
    median_edge = float(np.median(edges)) if len(edges) else 0.0
    if median_edge:
        bad_h = horizontal_valid & (horizontal > jump_factor * median_edge)
        bad_v = vertical_valid & (vertical > jump_factor * median_edge)
        defects[:, :-1] |= bad_h
        defects[:, 1:] |= bad_h
        defects[:-1] |= bad_v
        defects[1:] |= bad_v
    du = xyz[:-1, 1:] - xyz[:-1, :-1]
    dv = xyz[1:, :-1] - xyz[:-1, :-1]
    quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    area = np.cross(du, dv)
    reference = np.nanmean(area[quad_valid], axis=0) if quad_valid.any() else np.zeros(3)
    bad_quad = quad_valid & (np.einsum("...i,i->...", area, reference) <= 0)
    defects[:-1, :-1] |= bad_quad
    defects[:-1, 1:] |= bad_quad
    defects[1:, :-1] |= bad_quad
    defects[1:, 1:] |= bad_quad
    return defects


def repair_surface_defects(
    field: CropField,
    xyz: np.ndarray,
    anchors: np.ndarray,
    sheet_mask: np.ndarray,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    repaired = xyz.copy()
    repaired_anchors = anchors.copy()
    for _ in range(passes):
        defects = defect_vertices(repaired, sheet_mask)
        if not defects.any():
            break
        candidate_anchors = repaired_anchors.copy()
        candidate_anchors[ndimage.binary_dilation(defects, iterations=1)] = False
        sparse = repaired.copy()
        sparse[~candidate_anchors] = np.nan
        candidate = fill_enclosed_holes(sparse, candidate_anchors, sheet_mask)
        candidate = optimize_surface(
            field, candidate, candidate_anchors, sheet_mask, iterations=4, smoothing=0.35
        )
        if defect_vertices(candidate, sheet_mask).sum() >= defects.sum():
            break
        repaired = candidate
        repaired_anchors = candidate_anchors
    return repaired, repaired_anchors


def geometry_metrics(
    field: CropField,
    xyz: np.ndarray,
    anchors: np.ndarray,
    sheet_mask: np.ndarray,
    probability_threshold: float = 0.2,
    jump_factor: float = 3.0,
) -> MeshMetrics:
    valid = sheet_mask & np.isfinite(xyz).all(axis=-1)
    coverage = float(valid.sum() / max(1, sheet_mask.sum()))
    probability = field.sample(xyz[valid]) if valid.any() else np.empty(0)
    adherence = float((probability >= probability_threshold).mean()) if len(probability) else 0.0
    horizontal = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    vertical = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:] & valid[:-1]
    edges = np.concatenate((horizontal[horizontal_valid], vertical[vertical_valid]))
    median_edge = float(np.median(edges)) if len(edges) else 0.0
    jumps = float((edges > jump_factor * median_edge).mean()) if median_edge else 0.0
    du = xyz[:-1, 1:] - xyz[:-1, :-1]
    dv = xyz[1:, :-1] - xyz[:-1, :-1]
    quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    area = np.cross(du, dv)
    reference = np.nanmean(area[quad_valid], axis=0) if quad_valid.any() else np.zeros(3)
    folded = (np.einsum("...i,i->...", area, reference) <= 0) & quad_valid
    fold_fraction = float(folded.sum() / max(1, quad_valid.sum()))
    return MeshMetrics(coverage, float(anchors.sum() / max(1, sheet_mask.sum())), adherence, fold_fraction, jumps)


def mesh_first_unwrap(
    field: CropField,
    seed_xyz: np.ndarray | None = None,
    threshold: float = 0.45,
    spacing: float | None = None,
    optimization_iterations: int = 8,
    layer_half_thickness: float | None = None,
) -> tuple[SurfaceGrid, np.ndarray, MeshMetrics]:
    points, probability = extract_component(field, threshold, seed_xyz)
    spacing = spacing or float(np.mean(field.scale_zyx))
    if seed_xyz is not None:
        thickness = layer_half_thickness or 4.0 * spacing
        points, probability = select_seed_layer(points, probability, seed_xyz, thickness)
    xyz, anchors, sheet_mask, _ = rasterize_xyz(points, probability, spacing)
    xyz = fill_enclosed_holes(xyz, anchors, sheet_mask)
    xyz = optimize_surface(field, xyz, anchors, sheet_mask, iterations=optimization_iterations)
    xyz, anchors = repair_surface_defects(field, xyz, anchors, sheet_mask)
    valid = sheet_mask & np.isfinite(xyz).all(axis=-1)
    normals = compute_normals(xyz, valid)
    confidence = np.zeros(valid.shape, dtype=np.float32)
    confidence[valid] = field.sample(xyz[valid])
    metrics = geometry_metrics(field, xyz, anchors, sheet_mask)
    return SurfaceGrid(xyz, normals, confidence, valid), anchors, metrics
