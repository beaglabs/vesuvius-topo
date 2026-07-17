"""Standalone whole-scroll texture screening (no marimo, no GPU required).

Reads tiles.json (regenerated locally from the surface Zarr), samples the CT
level-2 volume around each tile, and computes ink-screening texture metrics:
  ct_nonzero, ct_std, focus_mean, fiber_coherence, texture_score

Uses CT level-2 (4× coarser than L0) for ~64× fewer S3 chunk fetches.
Tile coordinates (in CT L0 voxels) are mapped to L2 via Volume.scale_zyx=4.0.

Progress is checkpointed to texture-screen.json every 16 tiles, so the run can be
interrupted and resumed without losing work. Dependencies come from
vesuvius_ssm (ChunkCacheSampler, Volume) plus scipy.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vesuvius_ssm.search import ChunkCacheSampler
from vesuvius_ssm.volume import Volume, CT_URL as _CT_URL

ROOT = os.path.dirname(__file__)
TILES_PATH = os.path.join(ROOT, "tiles.json")
SCREEN_PATH = os.path.join(ROOT, "texture-screen.json")
CT_URL = _CT_URL

# screening_tile_coordinates parameters (from the original notebook nXZZ cell).
# Tile coords in tiles.json are CT L0 voxels.  We sample CT L2 (4× coarser),
# but spacing stays in L0 units — the Volume's scale_zyx=4.0 converts L0→L2
# inside ChunkCacheSampler.sample_xyz (voxel = physical / scale_zyx).
TILE_SIZE = 16
TILE_SPACING = 4.0          # L0 voxels
TILE_DEPTH = 9
TILE_DEPTH_SPACING = 4.0    # L0 voxels
SAMPLER_BYTES = 4 * 2**30   # L2 chunks are smaller; 4 GiB cache is plenty
CT_LEVEL = 2


def unit_vector(v, eps: float = 1e-6):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + eps) if n > eps else v


def screening_tile_coordinates(tile, size=TILE_SIZE, spacing=TILE_SPACING,
                               depth=TILE_DEPTH, depth_spacing=TILE_DEPTH_SPACING):
    center = np.asarray(tile["center_xyz_ct0"], np.float32)
    normal = unit_vector(np.asarray(tile["normal_xyz"], np.float32))
    reference = np.asarray([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
    u = unit_vector(np.cross(normal, reference))
    v = unit_vector(np.cross(normal, u))
    axis = (np.arange(size) - (size - 1) / 2) * spacing
    vv, uu = np.meshgrid(axis, axis, indexing="ij")
    surface = center + uu[..., None] * u + vv[..., None] * v
    offsets = np.arange(depth) - (depth - 1) / 2
    return surface[None] + offsets[:, None, None, None] * normal


def screen_one(tile, sampler):
    coords = screening_tile_coordinates(tile)
    stack = sampler.sample_xyz(coords).astype(np.float32)  # (depth, H, W)
    nonzero = stack[stack > 0]
    center = stack[len(stack) // 2]
    blur = ndi.gaussian_filter(center, 1.0)
    focus = np.abs(center - blur)
    gx = ndi.sobel(center, axis=1)
    gy = ndi.sobel(center, axis=0)
    jxx = ndi.gaussian_filter(gx * gx, 1.2)
    jyy = ndi.gaussian_filter(gy * gy, 1.2)
    jxy = ndi.gaussian_filter(gx * gy, 1.2)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    record = dict(tile)
    record["ct_nonzero"] = float(len(nonzero) / stack.size)
    record["ct_std"] = float(nonzero.std()) if len(nonzero) else 0.0
    record["focus_mean"] = float(focus.mean())
    record["fiber_coherence"] = float(coherence.mean())
    record["texture_score"] = (
        2 * record["ct_nonzero"]
        + 0.025 * record["ct_std"]
        + 0.8 * record["fiber_coherence"]
        + 0.01 * record["focus_mean"]
        + 0.5 * record["surface_probability"]
    )
    return record


def load_results():
    if os.path.exists(SCREEN_PATH):
        with open(SCREEN_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    tmp = SCREEN_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, SCREEN_PATH)


def main() -> None:
    with open(TILES_PATH) as f:
        tiles = json.load(f)
    print(f"loaded {len(tiles)} tiles", flush=True)

    import fsspec
    import zarr
    mapper = fsspec.get_mapper(CT_URL, anon=True)
    ct_group = zarr.open_group(mapper, mode="r")
    ct_l2 = ct_group[str(CT_LEVEL)]       # lazy zarr array — no download yet
    # Tile coords are CT L0 voxels; scale_zyx=4.0 maps them to L2 voxels
    # inside ChunkCacheSampler.sample_xyz (voxel = physical / scale_zyx).
    volume = Volume(array=ct_l2, scale_zyx=np.ones(3) * 4.0, offset_zyx=np.zeros(3), name="ct_l2")
    sampler = ChunkCacheSampler(volume, maximum_bytes=SAMPLER_BYTES)
    print(f"CT L{CT_LEVEL} shape: {volume.shape}, chunk: {getattr(ct_l2, 'chunks', '?')}", flush=True)

    results = load_results()
    done = {int(r["tile_id"]) for r in results}
    remaining = [t for t in tiles if t["tile_id"] not in done]
    print(f"resuming: done {len(results)}/{len(tiles)}, remaining {len(remaining)}", flush=True)

    batch = []
    for tile in remaining:
        try:
            batch.append(screen_one(tile, sampler))
        except Exception as e:
            print(f"tile {tile.get('tile_id')} error: {repr(e)}", flush=True)
        if len(batch) >= 16:
            results.extend(batch)
            save_results(results)
            batch = []
    if batch:
        results.extend(batch)
        save_results(results)

    ranking = sorted(results, key=lambda r: r["texture_score"], reverse=True)
    summary = {
        "tiles": len(tiles),
        "screened": len(results),
        "failed": 0,
        "top_ids": [r["tile_id"] for r in ranking[:50]],
        "score_p99": float(np.percentile([r["texture_score"] for r in results], 99)),
    }
    print("FINAL", summary, flush=True)
    with open(os.path.join(ROOT, "screen-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Screen only N tiles then exit (for speed testing)")
    args = parser.parse_args()
    if args.benchmark:
        # Patch main to stop after N tiles
        _orig_main = main
        def _bench():
            import time
            t0 = time.time()
            with open(TILES_PATH) as f:
                tiles = json.load(f)[:args.benchmark]
            print(f"BENCHMARK: {len(tiles)} tiles", flush=True)

            import fsspec, zarr
            mapper = fsspec.get_mapper(CT_URL, anon=True)
            ct_group = zarr.open_group(mapper, mode="r")
            ct_l2 = ct_group[str(CT_LEVEL)]
            volume = Volume(array=ct_l2, scale_zyx=np.ones(3) * 4.0,
                            offset_zyx=np.zeros(3), name="ct_l2")
            sampler = ChunkCacheSampler(volume, maximum_bytes=SAMPLER_BYTES)
            print(f"CT L{CT_LEVEL} shape: {volume.shape}, chunk: {getattr(ct_l2, 'chunks', '?')}",
                  flush=True)

            for i, tile in enumerate(tiles):
                try:
                    rec = screen_one(tile, sampler)
                    elapsed = time.time() - t0
                    print(f"  tile {tile['tile_id']}: score={rec['texture_score']:.3f}  "
                          f"({elapsed:.1f}s total, {elapsed/(i+1):.1f}s/tile)", flush=True)
                except Exception as e:
                    print(f"  tile {tile.get('tile_id')} error: {repr(e)}", flush=True)
            total = time.time() - t0
            print(f"\nBENCHMARK DONE: {len(tiles)} tiles in {total:.1f}s "
                  f"({total/len(tiles):.1f}s/tile)", flush=True)
        _bench()
    else:
        main()
