from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .geometry import CropField, generate_trajectories
from .model import SurfaceTracker
from .mesh import mesh_first_unwrap
from .render import export_render
from .rollout import rollout_surface
from .training import train_tracker
from .topology import topology_unwrap
from .search import (
    ChunkCacheSampler,
    SearchManifest,
    TorchscriptInkScorer,
    generate_surface_candidates,
    non_maximum_suppression,
    score_candidates,
    unwrap_winners,
)
from .types import SurfaceGrid
from .volume import CT_URL, SURFACE_URL, open_array


def parse_bounds(value: str) -> tuple[slice, slice, slice]:
    numbers = [int(item) for item in value.split(",")]
    if len(numbers) != 6:
        raise argparse.ArgumentTypeError("bounds must be z0,z1,y0,y1,x0,x1")
    return tuple(slice(numbers[i], numbers[i + 1]) for i in range(0, 6, 2))


def parse_vector(value: str) -> np.ndarray:
    vector = np.asarray([float(item) for item in value.split(",")], dtype=np.float32)
    if vector.shape != (3,):
        raise argparse.ArgumentTypeError("vector must contain x,y,z")
    return vector


def add_volume_arguments(parser, default=SURFACE_URL):
    parser.add_argument("--volume", default=default, help="OME-Zarr URL or local .npy path")
    parser.add_argument("--level", type=int, default=2, help="OME-Zarr pyramid level")
    parser.add_argument("--bounds", type=parse_bounds, required=True, help="z0,z1,y0,y1,x0,x1 at selected level")


def load_crop(args):
    volume = open_array(args.volume, args.level)
    data = volume.read(args.bounds)
    origin = np.asarray([bound.start for bound in args.bounds], dtype=np.float32)
    return volume, data, origin


def command_prepare(args):
    volume, data, origin = load_crop(args)
    trajectories = generate_trajectories(
        data,
        origin,
        volume.scale_zyx,
        args.count,
        args.length,
        args.seed_threshold,
        np.random.default_rng(args.seed),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, trajectory in enumerate(trajectories):
        trajectory.save(output / f"trajectory-{index:06d}.npz")
    print(f"saved {len(trajectories)} trajectories to {output}")


def command_train(args):
    paths = sorted(Path(args.trajectories).glob("trajectory-*.npz"))
    train_tracker(paths, args.output, args.epochs, args.batch_size, args.learning_rate, args.sequence_length, args.device)


def command_rollout(args):
    volume, data, origin = load_crop(args)
    field = CropField(data, origin, volume.scale_zyx)
    model = SurfaceTracker.load(args.model, args.device)
    rollout_surface(
        model,
        field,
        args.seed_xyz,
        args.u_direction,
        args.height,
        args.width,
        args.output,
        args.checkpoint_rows,
        args.device,
    )
    print(f"saved XYZ surface to {args.output}")


def command_render(args):
    volume = open_array(args.ct_volume, args.level)
    surface = SurfaceGrid.load(args.surface)
    offsets = np.linspace(args.offset_min, args.offset_max, args.layers, dtype=np.float32)
    outputs = export_render(
        volume, surface, args.output, offsets, args.ink_model, args.ink_manifest, args.device
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def command_mesh_unwrap(args):
    volume, data, origin = load_crop(args)
    field = CropField(data, origin, volume.scale_zyx)
    surface, anchors, metrics = mesh_first_unwrap(
        field,
        seed_xyz=args.seed_xyz,
        threshold=args.threshold,
        spacing=args.spacing,
        optimization_iterations=args.optimization_iterations,
        layer_half_thickness=args.layer_half_thickness,
    )
    surface.save(args.output)
    np.save(Path(args.output).with_suffix(".anchors.npy"), anchors)
    print(f"saved mesh-first XYZ surface to {args.output}")
    print(
        f"coverage={metrics.coverage:.4f} anchors={metrics.anchor_fraction:.4f} "
        f"adherence={metrics.probability_adherence:.4f} "
        f"folds={metrics.folded_quad_fraction:.4f} jumps={metrics.jump_fraction:.4f}"
    )


def command_topology_unwrap(args):
    volume, data, origin = load_crop(args)
    field = CropField(data, origin, volume.scale_zyx)
    surface, mesh, metrics = topology_unwrap(
        field,
        seed_xyz=args.seed_xyz,
        patch_radius=args.patch_radius,
        raster_spacing=args.spacing,
        slim_iterations=args.slim_iterations,
    )
    surface.save(args.output)
    mesh_path = Path(args.output).with_suffix(".mesh.npz")
    np.savez_compressed(
        mesh_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        uv=mesh.uv,
        boundary=mesh.boundary,
    )
    print(f"saved topology-aware XYZ surface to {args.output}")
    print(f"saved parameterized triangle mesh to {mesh_path}")
    print(
        f"faces={metrics.faces} flips={metrics.flipped_fraction:.6f} "
        f"collapsed={metrics.collapsed_fraction:.6f} overlaps={metrics.overlap_fraction:.6f} "
        f"coverage={metrics.raster_coverage:.4f} adherence={metrics.probability_adherence:.4f}"
    )


def command_search_generate(args):
    volume = open_array(args.surface_volume, args.level)
    support = open_array(args.ct_volume, args.ct_support_level) if args.ct_volume else None
    candidates = generate_surface_candidates(
        volume,
        threshold=args.threshold,
        maximum_candidates=args.maximum_candidates,
        minimum_distance_voxels=args.minimum_distance_voxels,
        support_volume=support,
        minimum_support=args.minimum_ct_support,
    )
    SearchManifest(candidates=candidates).save(args.output)
    print(f"generated {len(candidates)} candidates in {args.output}")


def command_search_score(args):
    manifest = SearchManifest.load(args.manifest)
    volume = open_array(args.ct_volume, args.ct_level)
    sampler = ChunkCacheSampler(volume, maximum_bytes=int(args.cache_gib * 2**30))
    ink_scorer = TorchscriptInkScorer(args.ink_model, args.device) if args.ink_model else None
    score_candidates(
        manifest.candidates,
        sampler,
        args.output or args.manifest,
        ink_scorer=ink_scorer,
        size=args.size,
        spacing=args.spacing,
        depth=args.depth,
        checkpoint_every=args.checkpoint_every,
    )
    completed = sum(item.status == "scored" for item in manifest.candidates)
    print(f"scored {completed}/{len(manifest.candidates)} candidates")


def command_search_nms(args):
    manifest = SearchManifest.load(args.manifest)
    selected = non_maximum_suppression(
        manifest.candidates,
        minimum_distance=args.minimum_distance,
        minimum_normal_similarity=args.minimum_normal_similarity,
        limit=args.limit,
    )
    SearchManifest(candidates=selected).save(args.output)
    print(f"selected {len(selected)} spatially distinct candidates")


def command_search_unwrap(args):
    winners = SearchManifest.load(args.manifest).candidates[: args.limit]
    surface = open_array(args.surface_volume, args.surface_level)
    ct = open_array(args.ct_volume, args.ct_level)
    results = unwrap_winners(
        winners,
        surface,
        ct,
        args.output,
        crop_radius=args.crop_radius,
        patch_radius=args.patch_radius,
        raster_spacing=args.spacing,
    )
    completed = sum(item["status"] == "complete" for item in results)
    print(f"completed {completed}/{len(results)} topology unwrappings")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vesuvius-ssm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="trace pseudo-labelled trajectories from a probability crop")
    add_volume_arguments(prepare)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=2000)
    prepare.add_argument("--length", type=int, default=64)
    prepare.add_argument("--seed-threshold", type=float, default=0.7)
    prepare.add_argument("--seed", type=int, default=7)
    prepare.set_defaults(function=command_prepare)

    train = subparsers.add_parser("train", help="train the recurrent surface tracker")
    train.add_argument("--trajectories", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--sequence-length", type=int, default=32)
    train.add_argument("--device")
    train.set_defaults(function=command_train)

    rollout = subparsers.add_parser("rollout", help="grow a 2D XYZ surface grid")
    add_volume_arguments(rollout)
    rollout.add_argument("--model", required=True)
    rollout.add_argument("--seed-xyz", type=parse_vector, required=True)
    rollout.add_argument("--u-direction", type=parse_vector, required=True)
    rollout.add_argument("--height", type=int, default=256)
    rollout.add_argument("--width", type=int, default=256)
    rollout.add_argument("--checkpoint-rows", type=int, default=16)
    rollout.add_argument("--device", default="cuda")
    rollout.add_argument("--output", required=True)
    rollout.set_defaults(function=command_rollout)

    mesh = subparsers.add_parser("mesh-unwrap", help="build a dense XYZ map from a connected predicted surface")
    add_volume_arguments(mesh)
    mesh.add_argument("--seed-xyz", type=parse_vector, help="optional CT level-0 x,y,z point selecting a component")
    mesh.add_argument("--threshold", type=float, default=0.45)
    mesh.add_argument("--spacing", type=float, help="UV pixel spacing in CT level-0 voxel units")
    mesh.add_argument("--optimization-iterations", type=int, default=8)
    mesh.add_argument("--layer-half-thickness", type=float, help="seed-layer half thickness in CT level-0 voxels")
    mesh.add_argument("--output", required=True)
    mesh.set_defaults(function=command_mesh_unwrap)

    topology = subparsers.add_parser(
        "topology-unwrap", help="extract a seeded ridge mesh and flatten it with LSCM/SLIM"
    )
    add_volume_arguments(topology)
    topology.add_argument("--seed-xyz", type=parse_vector, required=True)
    topology.add_argument("--patch-radius", type=float, required=True, help="geodesic radius in CT level-0 voxels")
    topology.add_argument("--spacing", type=float, help="output pixel spacing in CT level-0 voxels")
    topology.add_argument("--slim-iterations", type=int, default=10)
    topology.add_argument("--output", required=True)
    topology.set_defaults(function=command_topology_unwrap)

    search_generate = subparsers.add_parser("search-generate", help="phase 1: generate coarse surface candidates")
    search_generate.add_argument("--surface-volume", default=SURFACE_URL)
    search_generate.add_argument("--level", type=int, default=5)
    search_generate.add_argument("--threshold", type=float, default=0.35)
    search_generate.add_argument("--maximum-candidates", type=int, default=10_000)
    search_generate.add_argument("--minimum-distance-voxels", type=int, default=2)
    search_generate.add_argument("--ct-volume", default=CT_URL)
    search_generate.add_argument("--ct-support-level", type=int, default=5)
    search_generate.add_argument("--minimum-ct-support", type=float, default=1.0)
    search_generate.add_argument("--output", required=True)
    search_generate.set_defaults(function=command_search_generate)

    search_score = subparsers.add_parser("search-score", help="phase 2: full-resolution sparse CT and ink scoring")
    search_score.add_argument("--manifest", required=True)
    search_score.add_argument("--output", help="defaults to updating the input manifest")
    search_score.add_argument("--ct-volume", default=CT_URL)
    search_score.add_argument("--ct-level", type=int, default=0)
    search_score.add_argument("--size", type=int, default=64)
    search_score.add_argument("--spacing", type=float, default=1.0)
    search_score.add_argument("--depth", type=int, default=30)
    search_score.add_argument("--cache-gib", type=float, default=8.0)
    search_score.add_argument("--checkpoint-every", type=int, default=25)
    search_score.add_argument("--ink-model", help="optional TorchScript ink model")
    search_score.add_argument("--device", default="cuda")
    search_score.set_defaults(function=command_search_score)

    search_nms = subparsers.add_parser("search-nms", help="phase 3: spatial and normal-aware deduplication")
    search_nms.add_argument("--manifest", required=True)
    search_nms.add_argument("--minimum-distance", type=float, default=256.0)
    search_nms.add_argument("--minimum-normal-similarity", type=float, default=0.8)
    search_nms.add_argument("--limit", type=int, default=100)
    search_nms.add_argument("--output", required=True)
    search_nms.set_defaults(function=command_search_nms)

    search_unwrap = subparsers.add_parser("search-unwrap", help="phase 4: topology unwrap ranked winners")
    search_unwrap.add_argument("--manifest", required=True)
    search_unwrap.add_argument("--surface-volume", default=SURFACE_URL)
    search_unwrap.add_argument("--surface-level", type=int, default=3)
    search_unwrap.add_argument("--ct-volume", default=CT_URL)
    search_unwrap.add_argument("--ct-level", type=int, default=0)
    search_unwrap.add_argument("--limit", type=int, default=20)
    search_unwrap.add_argument("--crop-radius", type=int, default=48)
    search_unwrap.add_argument("--patch-radius", type=float, default=1200.0)
    search_unwrap.add_argument("--spacing", type=float, default=4.0)
    search_unwrap.add_argument("--output", required=True)
    search_unwrap.set_defaults(function=command_search_unwrap)

    render = subparsers.add_parser("render", help="sample CT and export the completed unwrapping")
    render.add_argument("--ct-volume", default=CT_URL, help="OME-Zarr CT URL or local .npy")
    render.add_argument("--level", type=int, default=0)
    render.add_argument("--surface", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--layers", type=int, default=30)
    render.add_argument("--offset-min", type=float, default=-15)
    render.add_argument("--offset-max", type=float, default=14)
    render.add_argument("--ink-model", help="TorchScript ink checkpoint")
    render.add_argument("--ink-manifest", help="JSON input contract for the ink checkpoint")
    render.add_argument("--device", default="cuda")
    render.set_defaults(function=command_render)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
