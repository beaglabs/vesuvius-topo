from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy.ndimage import map_coordinates

from .types import SurfaceGrid
from .volume import Volume


def sample_surface_stack(volume: Volume, surface: SurfaceGrid, offsets: np.ndarray) -> np.ndarray:
    points = surface.xyz[..., None, :] + offsets[None, None, :, None] * surface.normals[..., None, :]
    finite_points = points[np.isfinite(points).all(-1)]
    if not len(finite_points):
        raise ValueError("surface contains no finite points")
    voxel_zyx = finite_points[:, ::-1] / volume.scale_zyx - volume.offset_zyx
    lower = np.maximum(np.floor(voxel_zyx.min(0)).astype(int) - 2, 0)
    upper = np.minimum(np.ceil(voxel_zyx.max(0)).astype(int) + 3, np.asarray(volume.shape))
    crop = volume.read(tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper)))
    all_voxels = points[..., ::-1] / volume.scale_zyx - volume.offset_zyx - lower
    coords = np.moveaxis(all_voxels, -1, 0).reshape(3, -1)
    sampled = map_coordinates(crop, coords, order=1, mode="constant", cval=0, prefilter=False)
    stack = sampled.reshape(points.shape[:-1]).transpose(2, 0, 1)
    stack[:, ~surface.valid] = 0
    return stack


@dataclass(frozen=True)
class InkManifest:
    depth: int
    input_size: int = 64
    clip_max: float = 200.0
    scale_divisor: float = 255.0
    reverse_depth: bool = False
    raw_output_size: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "InkManifest":
        return cls(**json.loads(Path(path).read_text()))


def run_torchscript_ink(stack: np.ndarray, model_path: str, manifest: InkManifest, device: str) -> np.ndarray:
    if stack.shape[0] != manifest.depth:
        raise ValueError(f"ink checkpoint requires {manifest.depth} layers, rendered {stack.shape[0]}")
    model = torch.jit.load(model_path, map_location=device).eval()
    source = torch.from_numpy(stack.astype(np.float32))
    if manifest.reverse_depth:
        source = source.flip(0)
    height, width = source.shape[-2:]
    output = torch.zeros((height, width), device=device)
    count = torch.zeros_like(output)
    size = manifest.input_size
    stride = size // 2
    with torch.no_grad():
        for y in range(0, max(1, height - size + 1), stride):
            for x in range(0, max(1, width - size + 1), stride):
                tile = source[:, y : y + size, x : x + size]
                if tile.shape[-2:] != (size, size):
                    continue
                native = tile.clamp(0, manifest.clip_max).div(manifest.scale_divisor)[None, None].to(device)
                logits = model(native)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim == 2 and manifest.raw_output_size:
                    logits = logits.reshape(1, 1, manifest.raw_output_size, manifest.raw_output_size)
                if logits.ndim == 3:
                    logits = logits[:, None]
                probability = torch.sigmoid(
                    torch.nn.functional.interpolate(logits.float(), (size, size), mode="bilinear", align_corners=False)
                )[0, 0]
                output[y : y + size, x : x + size] += probability
                count[y : y + size, x : x + size] += 1
    return (output / count.clamp_min(1)).cpu().numpy()


def save_preview(array: np.ndarray, path: Path, mask: np.ndarray | None = None) -> None:
    image = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(image)
    if mask is not None:
        finite &= mask
    if finite.any():
        low, high = np.percentile(image[finite], [1, 99])
        image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    image[~finite] = 0
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def save_ply(surface: SurfaceGrid, path: Path) -> None:
    indices = -np.ones(surface.valid.shape, dtype=int)
    vertices = surface.xyz[surface.valid]
    indices[surface.valid] = np.arange(len(vertices))
    faces = []
    for y in range(surface.valid.shape[0] - 1):
        for x in range(surface.valid.shape[1] - 1):
            quad = indices[y : y + 2, x : x + 2]
            if (quad >= 0).all():
                faces.extend(((quad[0, 0], quad[0, 1], quad[1, 0]), (quad[0, 1], quad[1, 1], quad[1, 0])))
    with path.open("w", encoding="ascii") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        file.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def export_render(
    volume: Volume,
    surface: SurfaceGrid,
    output_dir: str | Path,
    offsets: np.ndarray,
    ink_model: str | None = None,
    ink_manifest: str | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stack = sample_surface_stack(volume, surface, offsets)
    center = stack[len(stack) // 2]
    tifffile.imwrite(output_dir / "surface_layers.tif", stack, compression="zlib")
    tifffile.imwrite(output_dir / "unwrapped_ct.tif", center)
    save_preview(center, output_dir / "unwrapped_ct.png", surface.valid)
    save_preview(surface.confidence, output_dir / "confidence.png", surface.valid)
    save_ply(surface, output_dir / "surface.ply")
    outputs = {"layers": output_dir / "surface_layers.tif", "ct": output_dir / "unwrapped_ct.tif"}
    if ink_model:
        if not ink_manifest:
            raise ValueError("--ink-manifest is required with --ink-model")
        ink = run_torchscript_ink(stack, ink_model, InkManifest.load(ink_manifest), device)
        tifffile.imwrite(output_dir / "ink_probability.tif", ink.astype(np.float32))
        save_preview(ink, output_dir / "unwrapped_ink.png", surface.valid)
        outputs["ink"] = output_dir / "ink_probability.tif"
    return outputs
