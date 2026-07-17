"""Regenerate whole-scroll tile definitions from the surface prediction Zarr.

Local, standalone (no marimo, no GPU). The remote tiles.json was lost when the
Molab sandbox terminated, so we reconstruct an equivalent tile set from the
public surface prediction volume.

Strategy: process the surface volume in z-bands so we never hold the full
8398^3 grid in RAM. For each band we run marching_cubes on the padded subvolume,
offset the vertices back to global strided-grid coordinates, and accumulate the
surface point cloud (vertices + normals + prediction value). Once the cloud is
collected we partition into N_SHEETS ordered layers and sample tiles.

Coordinate convention (matches the original notebook):
  - The sampler reads CT level-0 directly and expects xyz in CT level-0 voxels.
  - The surface Zarr level-0 is 4x coarser than CT level-0 per axis, so a
    surface-voxel coordinate * 4 = CT level-0 coordinate.
  - We read the surface array with integer stride STRIDE (surface voxels), so a
    strided-grid coordinate maps to CT level-0 via coord * STRIDE * 4.
"""

from __future__ import annotations

import json
import os

import fsspec
import numpy as np
import zarr
from skimage.measure import marching_cubes

SURFACE_URL = (
    "s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/"
    "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
)

STRIDE = 8                      # surface voxels per strided-grid step
SURFACE_TO_CT0 = 4              # surface level-0 is 4x coarser than CT level-0
SCALE = STRIDE * SURFACE_TO_CT0
THRESHOLD = 0.2 * 255.0
N_SHEETS = 24
POINTS_PER_SHEET = 300
BAND = 256                      # strided-grid z-band height per marching-cubes pass
PAD = 2                         # overlap padding between bands (strided-grid voxels)
OUT_PATH = os.path.join(os.path.dirname(__file__), "tiles.json")


def unit_vector(v, eps: float = 1e-6):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + eps) if n > eps else v


def load_surface_array(url: str):
    mapper = fsspec.get_mapper(url, anon=True)
    group = zarr.open_group(mapper, mode="r")
    return group["0"]


def collect_surface_points(arr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts_ct0, normals, values) in CT level-0 coordinates."""
    shape = tuple(int(s) for s in arr.shape)
    gs = (shape[0] + STRIDE - 1) // STRIDE
    all_verts, all_norms, all_vals = [], [], []
    z0 = 0
    while z0 < gs:
        z1 = min(gs, z0 + BAND)
        # Read with padding (clamped) so marching cubes at band edges is consistent.
        rz0 = max(0, z0 - PAD)
        rz1 = min(gs, z1 + PAD)
        block = np.asarray(arr[rz0 * STRIDE:rz1 * STRIDE:STRIDE, ::STRIDE, ::STRIDE],
                           dtype=np.float32)
        try:
            verts, faces, normals, _ = marching_cubes(
                block, level=THRESHOLD, allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            z0 = z1
            continue
        # Offset verts back to global strided-grid coords (subtract padding).
        verts = verts + np.array([rz0, 0, 0], dtype=np.float32)
        # Keep only verts inside the core band (drop padded ring).
        keep = (verts[:, 0] >= z0) & (verts[:, 0] < z1)
        verts = verts[keep]
        normals = normals[keep]
        vi = np.rint(verts).astype(int)
        vi = np.clip(vi, 0, np.array(block.shape) - 1)
        # values sampled from block at padded coords (vi already includes rz0 offset)
        values = block[vi[:, 0], vi[:, 1], vi[:, 2]]
        # To CT level-0.
        all_verts.append(verts * SCALE)
        all_norms.append(normals)
        all_vals.append(values)
        z0 = z1
    if not all_verts:
        raise RuntimeError("no surface vertices extracted")
    return (np.concatenate(all_verts, axis=0),
            np.concatenate(all_norms, axis=0),
            np.concatenate(all_vals, axis=0))


def partition_sheets(verts: np.ndarray, n_sheets: int):
    center = verts.mean(axis=0)
    rel = verts - center
    spread = rel.max(axis=0) - rel.min(axis=0)
    axis = int(np.argmax(spread))
    order = np.argsort(rel[:, axis])
    return np.array_split(order, n_sheets)


def main() -> None:
    print(f"opening surface array (STRIDE={STRIDE}) ...", flush=True)
    arr = load_surface_array(SURFACE_URL)
    print("collecting surface points ...", flush=True)
    verts, normals, values = collect_surface_points(arr)
    print(f"surface vertices: {len(verts)}", flush=True)

    bands = partition_sheets(verts, N_SHEETS)
    tiles = []
    tile_id = 0
    for sheet_index, band in enumerate(bands):
        band = np.asarray(band)
        if len(band) == 0:
            continue
        sv = verts[band]
        sn = normals[band]
        if len(band) >= POINTS_PER_SHEET:
            idx = np.linspace(0, len(band) - 1, POINTS_PER_SHEET).astype(int)
        else:
            idx = np.arange(len(band))
        for vertex_index, i in enumerate(idx):
            ct0 = sv[i]
            nrm = unit_vector(sn[i])
            tiles.append({
                "tile_id": tile_id,
                "sheet_index": sheet_index,
                "vertex_index": int(vertex_index),
                "center_xyz_ct0": [float(ct0[0]), float(ct0[1]), float(ct0[2])],
                "normal_xyz": [float(nrm[0]), float(nrm[1]), float(nrm[2])],
                "surface_probability": float(values[band[i]] / 255.0),
            })
            tile_id += 1

    with open(OUT_PATH, "w") as f:
        json.dump(tiles, f)
    print(f"wrote {len(tiles)} tiles -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
