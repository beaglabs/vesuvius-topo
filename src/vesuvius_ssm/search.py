from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage

from .geometry import CropField, normalize
from .render import export_render
from .topology import TopologyMetrics, grid_mesh_unwrap, topology_unwrap
from .volume import Volume


@dataclass
class SearchCandidate:
    candidate_id: int
    xyz: list[float]
    normal: list[float]
    surface_score: float
    ct_nonzero: float = 0.0
    ct_std: float = 0.0
    focus_std: float = 0.0
    fiber_coherence: float = 0.0
    ink_mean: float = 0.0
    ink_std: float = 0.0
    ink_p99: float = 0.0
    combined_score: float = 0.0
    status: str = "generated"
    error: str | None = None

    @property
    def point(self) -> np.ndarray:
        return np.asarray(self.xyz, dtype=np.float32)

    @property
    def unit_normal(self) -> np.ndarray:
        return normalize(np.asarray(self.normal, dtype=np.float32)[None])[0]


@dataclass
class SearchManifest:
    version: int = 1
    candidates: list[SearchCandidate] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps({"version": self.version, "candidates": [asdict(item) for item in self.candidates]}, indent=2))
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "SearchManifest":
        payload = json.loads(Path(path).read_text())
        return cls(payload["version"], [SearchCandidate(**item) for item in payload["candidates"]])


def generate_surface_candidates(
    volume: Volume,
    threshold: float = 0.45,
    maximum_candidates: int = 10_000,
    minimum_distance_voxels: int = 3,
    border: int = 3,
    support_volume: Volume | None = None,
    minimum_support: float = 1.0,
) -> list[SearchCandidate]:
    data = volume.read(tuple(slice(0, size) for size in volume.shape)).astype(np.float32)
    if np.issubdtype(volume.array.dtype, np.integer):
        data /= np.iinfo(volume.array.dtype).max
    elif data.max() > 1:
        data /= data.max()
    smoothed = ndimage.gaussian_filter(data, sigma=0.8)
    maximum = ndimage.maximum_filter(smoothed, size=2 * minimum_distance_voxels + 1)
    selected = (smoothed >= threshold) & (smoothed == maximum)
    if border:
        interior = np.zeros(selected.shape, dtype=bool)
        interior[tuple(slice(border, -border) for _ in range(3))] = True
        selected &= interior
    locations = np.argwhere(selected)
    if not len(locations):
        raise ValueError(f"no candidate maxima above threshold {threshold}")
    scores = smoothed[selected]
    order = np.argsort(scores)[::-1]
    support_values = None
    if support_volume is not None:
        support_data = support_volume.read(tuple(slice(0, size) for size in support_volume.shape))
        physical_xyz = ((locations.astype(np.float32) + volume.offset_zyx) * volume.scale_zyx)[:, ::-1]
        support_zyx = physical_xyz[:, ::-1] / support_volume.scale_zyx - support_volume.offset_zyx
        support_values = ndimage.map_coordinates(
            support_data, support_zyx.T, order=1, mode="constant", cval=0, prefilter=False
        )
        order = order[support_values[order] >= minimum_support]
    order = order[:maximum_candidates]
    gradients = np.stack(np.gradient(smoothed), axis=-1)
    candidates = []
    for candidate_id, index in enumerate(order):
        local_zyx = locations[index].astype(np.float32)
        normal_zyx = normalize(gradients[tuple(locations[index])][None])[0]
        if np.linalg.norm(normal_zyx) < 1e-5:
            neighborhood = tuple(slice(max(0, int(value) - 2), min(smoothed.shape[axis], int(value) + 3)) for axis, value in enumerate(local_zyx))
            points = np.argwhere(smoothed[neighborhood] >= threshold)
            if len(points) >= 6:
                centered = points - points.mean(axis=0)
                _, vectors = np.linalg.eigh(centered.T @ centered)
                normal_zyx = vectors[:, 0]
            else:
                normal_zyx = np.asarray([1.0, 0.0, 0.0])
        physical_xyz = ((local_zyx + volume.offset_zyx) * volume.scale_zyx)[::-1]
        candidates.append(
            SearchCandidate(
                candidate_id,
                physical_xyz.astype(float).tolist(),
                normal_zyx[::-1].astype(float).tolist(),
                float(scores[index]),
            )
        )
        if support_values is not None:
            candidates[-1].ct_nonzero = 1.0
            candidates[-1].ct_std = float(support_values[index])
    return candidates


class ChunkCacheSampler:
    def __init__(self, volume: Volume, maximum_bytes: int = 8 * 2**30):
        self.volume = volume
        self.maximum_bytes = maximum_bytes
        self.cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.bytes = 0
        chunks = getattr(volume.array, "chunks", None)
        self.chunk_shape = np.asarray(chunks or volume.shape, dtype=np.int64)

    def _chunk(self, key: tuple[int, int, int]) -> np.ndarray:
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        lower = np.asarray(key) * self.chunk_shape
        upper = np.minimum(lower + self.chunk_shape, self.volume.shape)
        value = self.volume.read(tuple(slice(int(a), int(b)) for a, b in zip(lower, upper)))
        while self.cache and self.bytes + value.nbytes > self.maximum_bytes:
            _, removed = self.cache.popitem(last=False)
            self.bytes -= removed.nbytes
        self.cache[key] = value
        self.bytes += value.nbytes
        return value

    def sample_xyz(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=np.float32)
        voxel = points[..., ::-1] / self.volume.scale_zyx - self.volume.offset_zyx
        flat = voxel.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=1)
        safe_flat = np.where(finite[:, None], flat, 0)
        base = np.floor(safe_flat).astype(np.int64)
        fraction = safe_flat - base
        result = np.zeros(len(flat), dtype=np.float32)
        shape = np.asarray(self.volume.shape)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    offset = np.asarray([dz, dy, dx])
                    indices = base + offset
                    valid = finite & np.all((indices >= 0) & (indices < shape), axis=1)
                    if not valid.any():
                        continue
                    weights = np.prod(np.where(offset, fraction, 1 - fraction), axis=1)
                    chunk_keys = indices // self.chunk_shape
                    for key_array in np.unique(chunk_keys[valid], axis=0):
                        key = tuple(int(value) for value in key_array)
                        selected = valid & np.all(chunk_keys == key_array, axis=1)
                        chunk = self._chunk(key)
                        local = indices[selected] - key_array * self.chunk_shape
                        result[selected] += chunk[tuple(local.T)].astype(np.float32) * weights[selected]
        return result.reshape(points.shape[:-1])


def tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize(np.asarray(normal, dtype=np.float32)[None])[0]
    reference = np.asarray([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
    tangent_u = normalize(np.cross(normal, reference)[None])[0]
    tangent_v = normalize(np.cross(normal, tangent_u)[None])[0]
    return tangent_u, tangent_v


def candidate_coordinates(
    candidate: SearchCandidate,
    size: int = 64,
    spacing: float = 1.0,
    depth: int = 30,
    depth_spacing: float = 1.0,
) -> np.ndarray:
    tangent_u, tangent_v = tangent_frame(candidate.unit_normal)
    coordinates = (np.arange(size, dtype=np.float32) - (size - 1) / 2) * spacing
    offsets = (np.arange(depth, dtype=np.float32) - (depth - 1) / 2) * depth_spacing
    vv, uu = np.meshgrid(coordinates, coordinates, indexing="ij")
    surface = candidate.point + uu[..., None] * tangent_u + vv[..., None] * tangent_v
    return surface[None] + offsets[:, None, None, None] * candidate.unit_normal


def render_sparse_candidate(
    sampler: ChunkCacheSampler,
    candidate: SearchCandidate,
    size: int = 64,
    spacing: float = 1.0,
    depth: int = 30,
    depth_spacing: float = 1.0,
) -> np.ndarray:
    return sampler.sample_xyz(candidate_coordinates(candidate, size, spacing, depth, depth_spacing)).astype(np.float32)


def texture_metrics(stack: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    nonzero = stack > 0
    center = stack[len(stack) // 2]
    blurred = np.stack([ndimage.gaussian_filter(layer, 1.2) for layer in stack])
    focus = np.abs(stack - blurred)
    best = np.take_along_axis(stack, np.argmax(focus, axis=0)[None], axis=0)[0]
    gx = ndimage.sobel(best, axis=1)
    gy = ndimage.sobel(best, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, 2)
    jyy = ndimage.gaussian_filter(gy * gy, 2)
    jxy = ndimage.gaussian_filter(gx * gy, 2)
    root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)
    coherence = root / (jxx + jyy + 1e-6)
    valid = best > 0
    metrics = {
        "ct_nonzero": float(nonzero.mean()),
        "ct_std": float(stack[nonzero].std()) if nonzero.any() else 0.0,
        "focus_std": float(best[valid].std()) if valid.any() else 0.0,
        "fiber_coherence": float(coherence[valid].mean()) if valid.any() else 0.0,
    }
    return best, metrics


InkScorer = Callable[[np.ndarray], np.ndarray]


def score_candidates(
    candidates: list[SearchCandidate],
    sampler: ChunkCacheSampler,
    manifest_path: str | Path,
    ink_scorer: InkScorer | None = None,
    size: int = 64,
    spacing: float = 1.0,
    depth: int = 30,
    checkpoint_every: int = 25,
) -> SearchManifest:
    manifest = SearchManifest(candidates=candidates)
    # Spatial ordering maximizes reuse of decompressed CT chunks while preserving
    # candidate IDs and final score ordering in the manifest.
    work = sorted(
        enumerate(candidates),
        key=lambda pair: tuple(
            np.floor(
                (pair[1].point[::-1] / sampler.volume.scale_zyx - sampler.volume.offset_zyx)
                / sampler.chunk_shape
            ).astype(int)
        ),
    )
    for progress, (index, candidate) in enumerate(work):
        if candidate.status == "scored":
            continue
        try:
            stack = render_sparse_candidate(sampler, candidate, size, spacing, depth)
            _, metrics = texture_metrics(stack)
            candidate.ct_nonzero = metrics["ct_nonzero"]
            candidate.ct_std = metrics["ct_std"]
            candidate.focus_std = metrics["focus_std"]
            candidate.fiber_coherence = metrics["fiber_coherence"]
            if candidate.ct_nonzero < 0.05:
                candidate.status = "rejected_masked"
                candidate.combined_score = -1.0
                if (progress + 1) % checkpoint_every == 0:
                    manifest.save(manifest_path)
                continue
            if ink_scorer is not None:
                ink = np.asarray(ink_scorer(stack), dtype=np.float32)
                candidate.ink_mean = float(ink.mean())
                candidate.ink_std = float(ink.std())
                candidate.ink_p99 = float(np.percentile(ink, 99))
            candidate.combined_score = (
                1.5 * candidate.surface_score
                + 2.0 * candidate.ct_nonzero
                + 0.02 * candidate.ct_std
                + 0.02 * candidate.focus_std
                + candidate.fiber_coherence
                + 8.0 * candidate.ink_std
                + max(0.0, candidate.ink_p99 - candidate.ink_mean)
            )
            candidate.status = "scored"
        except Exception as error:
            candidate.status = "failed"
            candidate.error = repr(error)
        if (progress + 1) % checkpoint_every == 0:
            manifest.save(manifest_path)
    manifest.save(manifest_path)
    return manifest


def non_maximum_suppression(
    candidates: Iterable[SearchCandidate],
    minimum_distance: float,
    minimum_normal_similarity: float = 0.8,
    limit: int | None = None,
) -> list[SearchCandidate]:
    ranked = sorted((item for item in candidates if item.status == "scored"), key=lambda item: item.combined_score, reverse=True)
    selected: list[SearchCandidate] = []
    for candidate in ranked:
        duplicate = False
        for accepted in selected:
            distance = np.linalg.norm(candidate.point - accepted.point)
            similarity = abs(float(candidate.unit_normal.dot(accepted.unit_normal)))
            if distance < minimum_distance and similarity >= minimum_normal_similarity:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
            if limit and len(selected) >= limit:
                break
    return selected


def unwrap_winners(
    winners: list[SearchCandidate],
    surface_volume: Volume,
    ct_volume: Volume,
    output_dir: str | Path,
    crop_radius: int = 96,
    patch_radius: float = 180.0,
    raster_spacing: float = 4.0,
    method: str = "grid",
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ct_sampler = ChunkCacheSampler(ct_volume)
    for rank, candidate in enumerate(winners):
        directory = output_dir / f"{rank:03d}-candidate-{candidate.candidate_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            center_zyx = candidate.point[::-1] / surface_volume.scale_zyx - surface_volume.offset_zyx
            lower = np.maximum(np.floor(center_zyx).astype(int) - crop_radius, 0)
            upper = np.minimum(lower + 2 * crop_radius, surface_volume.shape)
            lower = np.maximum(upper - 2 * crop_radius, 0)
            data = surface_volume.read(tuple(slice(int(a), int(b)) for a, b in zip(lower, upper)))
            field = CropField(data, lower.astype(np.float32), surface_volume.scale_zyx)
            if method == "grid":
                surface, mesh, metrics = grid_mesh_unwrap(
                    field, candidate.point, raster_spacing=raster_spacing
                )
            else:
                surface, mesh, metrics = topology_unwrap(
                    field, candidate.point, patch_radius, raster_spacing, slim_iterations=10
                )
            surface.save(directory / "surface.npz")
            np.savez_compressed(directory / "mesh.npz", vertices=mesh.vertices, faces=mesh.faces, uv=mesh.uv, boundary=mesh.boundary)
            offsets = np.arange(-15, 15, dtype=np.float32)
            export_sparse_surface(ct_sampler, surface, directory, offsets)
            result = {"candidate_id": candidate.candidate_id, "status": "complete", "metrics": asdict(metrics), "path": str(directory)}
        except Exception as error:
            result = {"candidate_id": candidate.candidate_id, "status": "failed", "error": repr(error), "path": str(directory)}
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def export_sparse_surface(
    sampler: ChunkCacheSampler,
    surface,
    output_dir: str | Path,
    offsets: np.ndarray,
) -> np.ndarray:
    output_dir = Path(output_dir)
    points = surface.xyz[..., None, :] + offsets[None, None, :, None] * surface.normals[..., None, :]
    stack = sampler.sample_xyz(points).transpose(2, 0, 1)
    stack[:, ~surface.valid] = 0
    center = stack[len(stack) // 2]
    tifffile.imwrite(output_dir / "surface_layers.tif", stack, compression="zlib")
    tifffile.imwrite(output_dir / "unwrapped_ct.tif", center)
    save_candidate_preview(output_dir / "unwrapped_ct.png", np.where(surface.valid, center, np.nan))
    return stack


class TorchscriptInkScorer:
    def __init__(self, path: str, device: str = "cuda", clip_max: float = 200.0):
        self.device = device
        self.clip_max = clip_max
        self.model = torch.jit.load(path, map_location=device).eval()

    @torch.no_grad()
    def __call__(self, stack: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(stack).float().clamp(0, self.clip_max).div(255.0)[None, None].to(self.device)
        logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 2:
            side = int(np.sqrt(logits.shape[-1]))
            logits = logits.reshape(1, 1, side, side)
        probability = torch.sigmoid(torch.nn.functional.interpolate(logits.float(), stack.shape[-2:], mode="bilinear", align_corners=False))
        return probability[0, 0].cpu().numpy()


def save_candidate_preview(path: str | Path, image: np.ndarray) -> None:
    values = image[np.isfinite(image)]
    low, high = np.percentile(values, [1, 99]) if len(values) else (0, 1)
    normalized = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    Image.fromarray((normalized * 255).astype(np.uint8)).save(path)
