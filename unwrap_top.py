"""Unwrap top-scoring tiles from texture screening.

Loads texture-screen.json, converts the top N results into SearchCandidate
objects, and runs unwrap_winners() to produce unwrapped CT surface images.

Outputs to unwrapped/ directory — one subdirectory per candidate containing:
  unwrapped_ct.png   — visual preview
  unwrapped_ct.tif   — 30-layer CT stack
  surface.npz        — parameterized surface geometry
  mesh.npz           — triangle mesh with UV
  result.json        — status + topology metrics

Usage:
  python unwrap_top.py                    # top 50 (default)
  python unwrap_top.py --top 20           # top 20
  python unwrap_top.py --min-score 4.0    # all tiles above score threshold
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vesuvius_ssm.search import SearchCandidate, unwrap_winners
from vesuvius_ssm.volume import CT_URL, SURFACE_URL, Volume, open_ome_zarr

ROOT = os.path.dirname(__file__)
SCREEN_PATH = os.path.join(ROOT, "texture-screen.json")
OUTPUT_DIR = os.path.join(ROOT, "unwrapped")


def load_candidates(top_n: int = 0, min_score: float = 0.0) -> list[SearchCandidate]:
    """Load screening results and convert to SearchCandidate objects."""
    with open(SCREEN_PATH) as f:
        results = json.load(f)

    ranked = sorted(results, key=lambda r: r["texture_score"], reverse=True)

    if min_score > 0:
        selected = [r for r in ranked if r["texture_score"] >= min_score]
    elif top_n > 0:
        selected = ranked[:top_n]
    else:
        selected = ranked[:50]

    print(f"Selected {len(selected)} candidates (from {len(results)} screened)", flush=True)

    candidates = []
    for r in selected:
        c = SearchCandidate(
            candidate_id=r["tile_id"],
            xyz=r["center_xyz_ct0"],
            normal=r["normal_xyz"],
            surface_score=r.get("surface_probability", 0.0),
            ct_nonzero=r.get("ct_nonzero", 0.0),
            ct_std=r.get("ct_std", 0.0),
            fiber_coherence=r.get("fiber_coherence", 0.0),
            combined_score=r["texture_score"],
        )
        candidates.append(c)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Unwrap top screening candidates")
    parser.add_argument("--top", type=int, default=0,
                        help="Number of top tiles to unwrap (default: 50)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Unwrap all tiles with texture_score >= this value")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory (default: unwrapped/)")
    parser.add_argument("--crop-radius", type=int, default=96,
                        help="Surface crop radius in voxels (default: 96)")
    parser.add_argument("--patch-radius", type=float, default=180.0,
                        help="Unwrap patch radius (default: 180.0)")
    parser.add_argument("--method", type=str, default="grid",
                        choices=["grid", "ridge"],
                        help="Unwrap method: 'grid' (notebook approach) or 'ridge' (marching cubes)")
    args = parser.parse_args()

    candidates = load_candidates(
        top_n=args.top if not args.min_score else 0,
        min_score=args.min_score,
    )
    if not candidates:
        print("No candidates selected.", flush=True)
        return

    print(f"Opening surface volume: {SURFACE_URL}", flush=True)
    surface_volume = open_ome_zarr(SURFACE_URL, level=0, storage_options={"anon": True})
    print(f"  shape: {surface_volume.shape}, scale: {surface_volume.scale_zyx}", flush=True)

    print(f"Opening CT volume: {CT_URL}", flush=True)
    import fsspec, zarr
    mapper = fsspec.get_mapper(CT_URL, anon=True)
    ct_group = zarr.open_group(mapper, mode="r")
    ct_l2 = ct_group["2"]
    ct_volume = Volume(array=ct_l2, scale_zyx=np.ones(3) * 4.0,
                       offset_zyx=np.zeros(3), name="ct_l2")
    print(f"  shape: {ct_volume.shape}, scale: {ct_volume.scale_zyx}", flush=True)

    print(f"\nUnwrapping {len(candidates)} candidates → {args.output} (method={args.method})", flush=True)
    results = unwrap_winners(
        candidates,
        surface_volume,
        ct_volume,
        args.output,
        crop_radius=args.crop_radius,
        patch_radius=args.patch_radius,
        method=args.method,
    )

    complete = sum(1 for r in results if r["status"] == "complete")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nDone: {complete} complete, {failed} failed", flush=True)

    if complete:
        print(f"\nPreview images in:", flush=True)
        for r in results:
            if r["status"] == "complete":
                print(f"  {r['path']}/unwrapped_ct.png", flush=True)


if __name__ == "__main__":
    main()
