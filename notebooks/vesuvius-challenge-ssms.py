# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fsspec==2026.6.0",
#     "gdown==6.1.0",
#     "libigl==2.6.2",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "s3fs==2026.6.0",
#     "scikit-image==0.26.0",
#     "scipy==1.18.0",
#     "tifffile==2026.7.14",
#     "timesformer-pytorch==0.4.1",
#     "torch==2.13.0",
#     "trimesh==4.12.2",
#     "zarr==2.18.7",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import fsspec
    import zarr
    import matplotlib.pyplot as plt

    SURFACE_URL = "s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
    CT_URL = "s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"
    import torch
    from torch import nn
    from scipy.ndimage import gaussian_filter, map_coordinates
    import tifffile
    from timesformer_pytorch import TimeSformer
    from scipy.ndimage import label as connected_components
    import scipy.ndimage as ndi
    from scipy.interpolate import griddata as scipy_griddata
    import igl
    import trimesh
    from scipy.spatial import Delaunay
    from scipy import sparse as scipy_sparse
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
    from skimage import exposure as skimage_exposure
    from collections import OrderedDict
    from skimage.measure import marching_cubes

    return (
        CT_URL,
        Delaunay,
        OrderedDict,
        SURFACE_URL,
        TimeSformer,
        connected_components,
        fsspec,
        gaussian_filter,
        igl,
        map_coordinates,
        marching_cubes,
        mo,
        ndi,
        nn,
        np,
        plt,
        scipy_dijkstra,
        scipy_griddata,
        scipy_sparse,
        skimage_exposure,
        tifffile,
        torch,
        trimesh,
        zarr,
    )


@app.cell
def _(mo):
    mo.md("""
    # PHerc0332 SSM unwrapper

    The prediction volume is about **130 billion voxels**, so we first choose a manageable box called a **crop**. The notebook then finds a high-confidence surface voxel inside that box automatically; that voxel is the **seed** where tracing starts. You do not need to know either coordinate in advance.

    The crop and seed are computational controls, not additional datasets.
    """)
    return


@app.cell
def _(SURFACE_URL, fsspec, zarr):
    surface_mapper = fsspec.get_mapper(SURFACE_URL, anon=True)
    surface_group = zarr.open_group(surface_mapper, mode="r")
    surface_shapes = {int(level_index): tuple(surface_group[str(level_index)].shape) for level_index in range(6)}
    surface_shapes
    return surface_group, surface_shapes


@app.cell
def _(mo):
    level = mo.ui.dropdown(options=[0, 1, 2, 3, 4, 5], value=3, label="Prediction pyramid level")
    crop_size = mo.ui.slider(32, 256, value=96, step=16, label="Crop edge length")
    z_fraction = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Z position")
    y_fraction = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Y position")
    x_fraction = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="X position")
    mo.vstack([level, crop_size, z_fraction, y_fraction, x_fraction])
    return crop_size, level, x_fraction, y_fraction, z_fraction


@app.cell
def _(
    crop_size,
    level,
    np,
    surface_group,
    surface_shapes,
    x_fraction,
    y_fraction,
    z_fraction,
):
    _shape = np.asarray(surface_shapes[level.value])
    _edge = np.minimum(crop_size.value, _shape)
    _center = np.asarray([z_fraction.value, y_fraction.value, x_fraction.value]) * (_shape - 1)
    _start = np.clip((_center - _edge / 2).astype(int), 0, _shape - _edge)
    _stop = _start + _edge
    crop_bounds_zyx = tuple((int(a), int(b)) for a, b in zip(_start, _stop))
    surface_array = surface_group[str(level.value)]
    surface_crop = np.asarray(surface_array[tuple(slice(a, b) for a, b in crop_bounds_zyx)], dtype=np.float32) / 255.0
    _seed_margin = max(8, min(surface_crop.shape) // 6)
    _seed_search = surface_crop[_seed_margin:-_seed_margin, _seed_margin:-_seed_margin, _seed_margin:-_seed_margin]
    _seed_interior = np.asarray(np.unravel_index(int(np.argmax(_seed_search)), _seed_search.shape))
    _seed_local = tuple((_seed_interior + _seed_margin).tolist())
    seed_local_zyx = np.asarray(_seed_local)
    seed_level_zyx = seed_local_zyx + _start
    seed_fullres_xyz = (seed_level_zyx * (4 * 2 ** level.value))[::-1].astype(float)
    selection_summary = {
        "crop_bounds_zyx": crop_bounds_zyx,
        "maximum_surface_probability": float(surface_crop[_seed_local]),
        "automatic_seed_xyz_full_resolution": seed_fullres_xyz.tolist(),
    }
    selection_summary
    return crop_bounds_zyx, seed_fullres_xyz, seed_local_zyx, surface_crop


@app.cell
def _(level, plt, seed_local_zyx, surface_crop):
    projection_image = surface_crop.max(axis=0)
    _figure, _axis = plt.subplots(figsize=(7, 7))
    _axis.imshow(projection_image, cmap="magma")
    _axis.scatter([seed_local_zyx[2]], [seed_local_zyx[1]], c="cyan", s=35, marker="+")
    _axis.set_title(f"Surface prediction crop, level {level.value}; cyan = automatic seed")
    _axis.set_axis_off()
    _figure
    return


@app.cell
def _(CT_URL, crop_bounds_zyx, level, mo, seed_fullres_xyz, surface_crop):
    mo.md(f"""
    ### Selected region

    - Level: `{level.value}` (`{4 * 2 ** level.value}x` CT level-0 spacing; prediction level 0 aligns with CT level 2)
    - Crop bounds in level coordinates `(z,y,x)`: `{crop_bounds_zyx}`
    - Automatic full-resolution seed `(x,y,z)`: `{seed_fullres_xyz.tolist()}`
    - Maximum prediction confidence: `{surface_crop.max():.3f}`

    Move the sliders until the projection contains a coherent sheet region. The cyan marker is selected automatically from the strongest predicted surface response. This seed will initialize the recurrent rollout; the source CT at `{CT_URL}` will only be sampled after the XYZ surface is complete.
    """)
    return


@app.cell
def _(CT_URL, fsspec, zarr):
    ct_mapper = fsspec.get_mapper(CT_URL, anon=True)
    ct_group = zarr.open_group(ct_mapper, mode="r")
    ct_shapes = {key: tuple(ct_group[key].shape) for key in ct_group.array_keys()}
    {
        "ct_url": CT_URL,
        "available_arrays": ct_shapes,
        "coordinate_note": "The surface and CT are registered; full-resolution XYZ coordinates index CT level 0."
    }
    return (ct_group,)


@app.cell
def _(map_coordinates, nn, np):

    def unit_vector(value, eps=1e-6):
        value = np.asarray(value, dtype=np.float32)
        return value / max(float(np.linalg.norm(value)), eps)


    def sample_field(field, points_xyz):
        points = np.asarray(points_xyz, dtype=np.float32)
        coords = np.moveaxis(points[..., ::-1], -1, 0).reshape(3, -1)
        return map_coordinates(field, coords, order=1, mode="constant", cval=0, prefilter=False).reshape(points.shape[:-1])


    def estimate_frame(field, point_xyz, radius=7):
        center = np.asarray(point_xyz)[::-1]
        lo = np.maximum(np.floor(center - radius).astype(int), 0)
        hi = np.minimum(np.ceil(center + radius + 1).astype(int), field.shape)
        block = field[tuple(slice(a, b) for a, b in zip(lo, hi))]
        points = np.argwhere(block > max(0.35, float(block.max()) * 0.55)) + lo
        points = points[np.linalg.norm(points - center, axis=1) <= radius]
        if len(points) < 12:
            raise ValueError("not enough local surface support")
        centered = points[:, ::-1] - points[:, ::-1].mean(0)
        _, eigenvectors = np.linalg.eigh(centered.T @ centered / len(centered))
        normal = unit_vector(eigenvectors[:, 0])
        tangent_u = unit_vector(eigenvectors[:, 2])
        tangent_v = unit_vector(np.cross(normal, tangent_u))
        return tangent_u, tangent_v, normal


    def project_ridge(field, point_xyz, normal, radius=2.5, samples=11):
        offsets = np.linspace(-radius, radius, samples, dtype=np.float32)
        candidates = point_xyz[None] + offsets[:, None] * normal[None]
        scores = sample_field(field, candidates)
        best = int(np.argmax(scores))
        return candidates[best].astype(np.float32), float(scores[best])


    def trace_ridge(field, seed_xyz, direction, normal, steps=48, step_size=1.0):
        points, probabilities = [], []
        point = np.asarray(seed_xyz, dtype=np.float32)
        tangent = unit_vector(direction - np.dot(direction, normal) * normal)
        for _ in range(steps):
            point, probability = project_ridge(field, point, normal)
            if probability < 0.12:
                break
            points.append(point.copy())
            probabilities.append(probability)
            candidate = point + tangent * step_size
            next_point, next_probability = project_ridge(field, candidate, normal)
            if next_probability < 0.12:
                break
            tangent = unit_vector(next_point - point)
            point = next_point
        return np.asarray(points, np.float32), np.asarray(probabilities, np.float32)


    def make_training_sequences(field, count=768, length=32, random_seed=7):
        rng = np.random.default_rng(random_seed)
        candidates = np.argwhere(field > 0.55)
        sequences = []
        for candidate in candidates[rng.permutation(len(candidates))]:
            seed_xyz = candidate[::-1].astype(np.float32)
            try:
                tangent_u, tangent_v, normal = estimate_frame(field, seed_xyz)
            except ValueError:
                continue
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.cos(angle) * tangent_u + np.sin(angle) * tangent_v
            points, probabilities = trace_ridge(field, seed_xyz, direction, normal, steps=length)
            if len(points) < length // 2:
                continue
            observations = np.zeros((length, 13), np.float32)
            targets = np.zeros((length, 3), np.float32)
            mask = np.zeros(length, np.float32)
            previous = direction.astype(np.float32)
            for index in range(min(len(points), length)):
                tangent = unit_vector(previous)
                side = unit_vector(np.cross(normal, tangent))
                observations[index] = np.r_[previous, tangent, normal, probabilities[index], 0.0, 0.0, 1.0]
                if index + 1 < len(points):
                    delta = points[index + 1] - points[index]
                    frame = np.stack((tangent, side, normal), axis=-1)
                    targets[index] = frame.T @ delta
                    previous = delta
                    mask[index] = 1
            sequences.append((observations, targets, mask))
            if len(sequences) >= count:
                break
        if len(sequences) < 8:
            raise ValueError(f"only generated {len(sequences)} trajectories")
        return sequences


    class NotebookSurfaceTracker(nn.Module):
        def __init__(self, hidden_size=96):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(13, hidden_size), nn.SiLU(), nn.LayerNorm(hidden_size))
            self.core = nn.GRU(hidden_size, hidden_size, 2, batch_first=True, dropout=0.1)
            self.delta = nn.Linear(hidden_size, 3)
            self.confidence = nn.Linear(hidden_size, 1)

        def forward(self, observations, state=None):
            belief, state = self.core(self.encoder(observations), state)
            return self.delta(belief), self.confidence(belief).squeeze(-1), state


    return (
        NotebookSurfaceTracker,
        estimate_frame,
        make_training_sequences,
        project_ridge,
        sample_field,
        trace_ridge,
        unit_vector,
    )


@app.cell
def _(
    NotebookSurfaceTracker,
    gaussian_filter,
    make_training_sequences,
    np,
    surface_crop,
    torch,
):
    training_field = gaussian_filter(surface_crop.astype(np.float32), sigma=0.8)
    training_sequences = make_training_sequences(training_field, count=512, length=32)
    training_observations = torch.from_numpy(np.stack([item[0] for item in training_sequences]))
    training_targets = torch.from_numpy(np.stack([item[1] for item in training_sequences]))
    training_mask = torch.from_numpy(np.stack([item[2] for item in training_sequences]))
    training_device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker_model = NotebookSurfaceTracker().to(training_device)
    tracker_optimizer = torch.optim.AdamW(tracker_model.parameters(), lr=8e-4, weight_decay=1e-4)
    training_history = []
    _batch_size = 64
    for _epoch in range(10):
        _permutation = torch.randperm(len(training_sequences))
        _epoch_loss = 0.0
        for _start in range(0, len(training_sequences), _batch_size):
            _indices = _permutation[_start:_start + _batch_size]
            _observations = training_observations[_indices].to(training_device)
            _targets = training_targets[_indices].to(training_device)
            _mask = training_mask[_indices].to(training_device)
            tracker_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=training_device == "cuda"):
                _predicted_delta, _predicted_confidence, _ = tracker_model(_observations)
                _delta_error = torch.nn.functional.smooth_l1_loss(_predicted_delta, _targets, reduction="none").mean(-1)
                _delta_loss = (_delta_error * _mask).sum() / _mask.sum().clamp_min(1)
                _confidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(_predicted_confidence, _mask)
                _loss = _delta_loss + 0.05 * _confidence_loss
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(tracker_model.parameters(), 1.0)
            tracker_optimizer.step()
            _epoch_loss += float(_loss.detach()) * len(_indices)
        training_history.append(_epoch_loss / len(training_sequences))
    tracker_model.eval()
    training_summary = {
        "device": training_device,
        "trajectories": len(training_sequences),
        "epochs": len(training_history),
        "initial_loss": training_history[0],
        "final_loss": training_history[-1],
    }
    training_summary
    return tracker_model, training_device, training_field, training_summary


@app.cell
def _(mo, training_summary):
    mo.callout(mo.md(f"""**Tracker trained on `{training_summary['device']}`**

    - Pseudo-trajectories: `{training_summary['trajectories']}`
    - Epochs: `{training_summary['epochs']}`
    - Loss: `{training_summary['initial_loss']:.4f}` → `{training_summary['final_loss']:.4f}`
    """), kind="success")
    return


@app.cell
def _(
    estimate_frame,
    map_coordinates,
    np,
    project_ridge,
    torch,
    training_device,
    unit_vector,
):

    def model_trace(field, model, seed_xyz, direction, reference_normal, steps, reverse=False):
        direction = unit_vector(direction) * (-1 if reverse else 1)
        point = np.asarray(seed_xyz, np.float32)
        previous = direction.copy()
        state = None
        points, confidences, valid = [], [], []
        for _ in range(steps):
            point, ridge_probability = project_ridge(field, point, reference_normal)
            if ridge_probability < 0.10:
                break
            tangent = unit_vector(previous)
            side = unit_vector(np.cross(reference_normal, tangent))
            obs = np.r_[previous, tangent, reference_normal, ridge_probability, 0.0, 0.0, 1.0].astype(np.float32)
            with torch.no_grad():
                predicted_local, confidence_logit, state = model(torch.from_numpy(obs)[None, None].to(training_device), state)
            local_delta = predicted_local[0, 0].float().cpu().numpy()
            frame = np.stack((tangent, side, reference_normal), axis=-1)
            learned_direction = unit_vector(frame @ local_delta)
            if np.dot(learned_direction, tangent) < 0:
                learned_direction *= -1
            next_direction = unit_vector(0.75 * tangent + 0.25 * learned_direction)
            points.append(point.copy())
            confidences.append(float(torch.sigmoid(confidence_logit[0, 0]).cpu()) * ridge_probability)
            valid.append(True)
            candidate, candidate_probability = project_ridge(field, point + next_direction, reference_normal)
            if candidate_probability < 0.10:
                break
            previous = candidate - point
            point = candidate
        return np.asarray(points, np.float32), np.asarray(confidences, np.float32), np.asarray(valid, bool)


    def grow_surface_grid(field, model, seed_xyz, size=96):
        tangent_u, tangent_v, normal = estimate_frame(field, seed_xyz)
        half = size // 2
        forward_v = model_trace(field, model, seed_xyz, tangent_v, normal, half + 1)
        backward_v = model_trace(field, model, seed_xyz, tangent_v, normal, half + 1, reverse=True)
        centers = np.concatenate((backward_v[0][1:][::-1], forward_v[0]), axis=0)
        grid = np.full((size, size, 3), np.nan, np.float32)
        confidence = np.zeros((size, size), np.float32)
        valid = np.zeros((size, size), bool)
        row_offset = max(0, half - (len(backward_v[0]) - 1))
        for row_index, center in enumerate(centers[:size], start=row_offset):
            if row_index >= size:
                break
            forward_u = model_trace(field, model, center, tangent_u, normal, half + 1)
            backward_u = model_trace(field, model, center, tangent_u, normal, half + 1, reverse=True)
            row_points = np.concatenate((backward_u[0][1:][::-1], forward_u[0]), axis=0)
            row_confidence = np.concatenate((backward_u[1][1:][::-1], forward_u[1]), axis=0)
            col_offset = max(0, half - (len(backward_u[0]) - 1))
            end = min(size, col_offset + len(row_points))
            count = end - col_offset
            grid[row_index, col_offset:end] = row_points[:count]
            confidence[row_index, col_offset:end] = row_confidence[:count]
            valid[row_index, col_offset:end] = True
        return grid, confidence, valid, normal


    def render_registered_ct(ct_array, local_xyz, valid, crop_start_zyx, prediction_level, offsets):
        prediction_scale_to_ct2 = 2 ** prediction_level
        global_ct2_xyz = (local_xyz + np.asarray(crop_start_zyx)[::-1]) * prediction_scale_to_ct2
        filled = np.nan_to_num(global_ct2_xyz)
        du = np.gradient(filled, axis=1)
        dv = np.gradient(filled, axis=0)
        normals = np.cross(du, dv)
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(lengths, 1e-6)
        if np.nanmean(normals[valid] @ np.asarray([0.0, 0.0, 1.0])) < 0:
            normals *= -1
        points = global_ct2_xyz[..., None, :] + np.asarray(offsets)[None, None, :, None] * normals[..., None, :]
        finite = points[np.isfinite(points).all(-1)]
        voxel_zyx = finite[:, ::-1]
        lower = np.maximum(np.floor(voxel_zyx.min(0)).astype(int) - 2, 0)
        upper = np.minimum(np.ceil(voxel_zyx.max(0)).astype(int) + 3, np.asarray(ct_array.shape))
        ct_crop = np.asarray(ct_array[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))])
        coordinates = np.moveaxis(points[..., ::-1] - lower, -1, 0).reshape(3, -1)
        sampled = map_coordinates(ct_crop, coordinates, order=1, mode="constant", cval=0, prefilter=False)
        stack = sampled.reshape(points.shape[:-1]).transpose(2, 0, 1)
        stack[:, ~valid] = 0
        return global_ct2_xyz, normals, stack, lower, upper


    return grow_surface_grid, render_registered_ct


@app.cell
def _(
    crop_bounds_zyx,
    ct_group,
    grow_surface_grid,
    level,
    np,
    render_registered_ct,
    seed_local_zyx,
    tracker_model,
    training_field,
):
    automatic_seed_local_xyz = seed_local_zyx[::-1].astype(np.float32)
    rollout_xyz_local, rollout_confidence, rollout_valid, rollout_reference_normal = grow_surface_grid(
        training_field, tracker_model, automatic_seed_local_xyz, size=96
    )
    ct_level2_array = ct_group["2"]
    render_offsets = np.arange(-15, 15, dtype=np.float32)
    rollout_xyz_ct2, rollout_normals, rendered_ct_stack, ct_crop_lower, ct_crop_upper = render_registered_ct(
        ct_level2_array,
        rollout_xyz_local,
        rollout_valid,
        np.asarray([bound[0] for bound in crop_bounds_zyx]),
        level.value,
        render_offsets,
    )
    rendered_ct_image = rendered_ct_stack[len(rendered_ct_stack) // 2]
    render_summary = {
        "valid_surface_pixels": int(rollout_valid.sum()),
        "total_surface_pixels": int(rollout_valid.size),
        "ct_level": 2,
        "ct_crop_zyx": (ct_crop_lower.tolist(), ct_crop_upper.tolist()),
        "rendered_stack_shape": rendered_ct_stack.shape,
    }
    render_summary
    return (
        rendered_ct_image,
        rendered_ct_stack,
        rollout_confidence,
        rollout_normals,
        rollout_valid,
        rollout_xyz_ct2,
    )


@app.cell
def _(
    np,
    plt,
    rendered_ct_image,
    rollout_confidence,
    rollout_valid,
    surface_crop,
):
    _valid_values = rendered_ct_image[rollout_valid]
    _low, _high = np.percentile(_valid_values, [1, 99]) if len(_valid_values) else (0, 1)
    rendered_ct_preview = np.clip((rendered_ct_image - _low) / max(_high - _low, 1e-6), 0, 1)
    _rollout_figure, _rollout_axes = plt.subplots(1, 3, figsize=(16, 5))
    _rollout_axes[0].imshow(surface_crop.max(axis=0), cmap="magma")
    _rollout_axes[0].set_title("Surface prediction")
    _rollout_axes[1].imshow(rendered_ct_preview, cmap="gray")
    _rollout_axes[1].set_title("Unwrapped CT")
    _rollout_axes[2].imshow(np.where(rollout_valid, rollout_confidence, np.nan), cmap="viridis", vmin=0, vmax=1)
    _rollout_axes[2].set_title("Tracker confidence")
    for _rollout_axis in _rollout_axes:
        _rollout_axis.set_axis_off()
    _rollout_figure.tight_layout()
    _rollout_figure
    return (rendered_ct_preview,)


@app.cell
def _(
    TimeSformer,
    np,
    rendered_ct_stack,
    rollout_valid,
    torch,
    training_device,
):
    ink_checkpoint_path = "/tmp/gp-weights/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt"
    ink_checkpoint = torch.load(ink_checkpoint_path, map_location="cpu", weights_only=False)
    ink_model = TimeSformer(
        dim=512,
        image_size=64,
        patch_size=16,
        num_frames=30,
        num_classes=16,
        channels=1,
        depth=8,
        heads=6,
        dim_head=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )
    ink_state = {key.removeprefix("backbone."): value for key, value in ink_checkpoint["state_dict"].items() if key.startswith("backbone.")}
    ink_load_result = ink_model.load_state_dict(ink_state, strict=True)
    ink_model = ink_model.to(training_device).eval()
    ink_probability_sum = torch.zeros((96, 96), device=training_device)
    ink_probability_count = torch.zeros((96, 96), device=training_device)
    ink_input_stack = torch.from_numpy(rendered_ct_stack.astype(np.float32)).clamp(0, 200).div(255.0)
    with torch.no_grad():
        for _ink_y in (0, 32):
            for _ink_x in (0, 32):
                _ink_tile = ink_input_stack[:, _ink_y:_ink_y + 64, _ink_x:_ink_x + 64]
                _ink_native = _ink_tile[None, :, None].to(training_device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=training_device == "cuda"):
                    _ink_logits = ink_model(_ink_native).reshape(1, 1, 4, 4)
                _ink_probability = torch.sigmoid(torch.nn.functional.interpolate(
                    _ink_logits.float(), size=(64, 64), mode="bilinear", align_corners=False
                ))[0, 0]
                ink_probability_sum[_ink_y:_ink_y + 64, _ink_x:_ink_x + 64] += _ink_probability
                ink_probability_count[_ink_y:_ink_y + 64, _ink_x:_ink_x + 64] += 1
    ink_probability = (ink_probability_sum / ink_probability_count.clamp_min(1)).cpu().numpy()
    ink_probability[~rollout_valid] = 0
    ink_summary = {
        "checkpoint": ink_checkpoint_path.split("/")[-1],
        "input_shape": tuple(rendered_ct_stack.shape),
        "probability_range": (float(ink_probability[rollout_valid].min()), float(ink_probability[rollout_valid].max())),
    }
    ink_summary
    return ink_model, ink_probability, ink_summary


@app.cell
def _(ink_probability, plt, rendered_ct_preview):
    _final_figure, _final_axes = plt.subplots(1, 3, figsize=(16, 5))
    _final_axes[0].imshow(rendered_ct_preview, cmap="gray")
    _final_axes[0].set_title("Unwrapped CT")
    _final_axes[1].imshow(ink_probability, cmap="gray", vmin=0, vmax=1)
    _final_axes[1].set_title("Villa ink probability")
    _final_axes[2].imshow(ink_probability > 0.5, cmap="gray")
    _final_axes[2].set_title("Ink threshold > 0.5")
    for _final_axis in _final_axes:
        _final_axis.set_axis_off()
    _final_figure.tight_layout()
    _final_figure
    return


@app.cell
def _(
    ink_probability,
    ink_summary,
    mo,
    np,
    plt,
    rendered_ct_image,
    rendered_ct_preview,
    rendered_ct_stack,
    rollout_confidence,
    rollout_normals,
    rollout_valid,
    rollout_xyz_ct2,
    tifffile,
):
    artifact_directory = "/marimo/artifacts/vesuvius-ssm-output"
    import os as _os
    _os.makedirs(artifact_directory, exist_ok=True)
    np.savez_compressed(
        f"{artifact_directory}/surface.npz",
        xyz_ct_level2=rollout_xyz_ct2,
        normals=rollout_normals,
        confidence=rollout_confidence,
        valid=rollout_valid,
    )
    tifffile.imwrite(f"{artifact_directory}/surface_layers.tif", rendered_ct_stack)
    tifffile.imwrite(f"{artifact_directory}/unwrapped_ct.tif", rendered_ct_image)
    tifffile.imwrite(f"{artifact_directory}/ink_probability.tif", ink_probability.astype(np.float32))
    plt.imsave(f"{artifact_directory}/unwrapped_ct.png", rendered_ct_preview, cmap="gray", vmin=0, vmax=1)
    plt.imsave(f"{artifact_directory}/ink_probability.png", ink_probability, cmap="gray", vmin=0, vmax=1)
    plt.imsave(f"{artifact_directory}/confidence.png", np.where(rollout_valid, rollout_confidence, 0), cmap="viridis", vmin=0, vmax=1)
    artifact_summary = {
        "directory": artifact_directory,
        "files": sorted(_os.listdir(artifact_directory)),
    }
    mo.callout(mo.md(f"""## First complete run finished

    - Surface coverage: `{rollout_valid.sum()} / {rollout_valid.size}` pixels
    - CT stack: `{rendered_ct_stack.shape}`
    - Ink checkpoint: `{ink_summary['checkpoint']}`
    - Ink range: `{ink_summary['probability_range'][0]:.3f}–{ink_summary['probability_range'][1]:.3f}` (no credible ink above 0.5 in this patch)
    - Persistent artifacts: `{artifact_summary['directory']}`
    """), kind="success")
    return


@app.cell
def _(connected_components, np, surface_group):
    scan_level = 4
    scan_array = surface_group[str(scan_level)]
    scan_shape = np.asarray(scan_array.shape)
    scan_edge = 64
    scan_rng = np.random.default_rng(20260715)
    scan_starts = []
    for _scan_index in range(20):
        _fraction = np.asarray([
            (_scan_index + 0.5) / 20,
            scan_rng.uniform(0.15, 0.85),
            scan_rng.uniform(0.15, 0.85),
        ])
        _center = (_fraction * (scan_shape - 1)).astype(int)
        _start = np.clip(_center - scan_edge // 2, 0, scan_shape - scan_edge)
        scan_starts.append(_start)

    candidate_records = []
    for _candidate_index, _start in enumerate(scan_starts):
        _stop = _start + scan_edge
        _crop = np.asarray(scan_array[tuple(slice(int(a), int(b)) for a, b in zip(_start, _stop))], dtype=np.float32) / 255.0
        _mask = _crop > 0.45
        _labels, _component_count = connected_components(_mask)
        if _component_count:
            _component_sizes = np.bincount(_labels.ravel())[1:]
            _largest = int(_component_sizes.max())
        else:
            _largest = 0
        _margin = 10
        _interior = _mask[_margin:-_margin, _margin:-_margin, _margin:-_margin]
        _boundary = _mask.copy()
        _boundary[_margin:-_margin, _margin:-_margin, _margin:-_margin] = False
        _occupancy = float(_mask.mean())
        _interior_fraction = float(_interior.mean())
        _boundary_fraction = float(_boundary.mean())
        _dynamic_range = float(np.percentile(_crop, 99) - np.percentile(_crop, 50))
        _score = (
            4.0 * min(_interior_fraction, 0.20)
            + 2.0 * min(_largest / _mask.size, 0.20)
            + _dynamic_range
            - 1.5 * _boundary_fraction
            - 2.0 * max(0.0, _occupancy - 0.35)
        )
        candidate_records.append({
            "index": _candidate_index,
            "start_level4_zyx": _start.tolist(),
            "occupancy": _occupancy,
            "largest_component": _largest,
            "interior_fraction": _interior_fraction,
            "boundary_fraction": _boundary_fraction,
            "surface_score": float(_score),
        })
    ranked_candidates = sorted(candidate_records, key=lambda item: item["surface_score"], reverse=True)
    ranked_candidates[:5]
    return candidate_records, ranked_candidates, scan_array, scan_edge


@app.cell
def _(candidate_records, np, plt, scan_array, scan_edge):
    _scan_figure, _scan_axes = plt.subplots(4, 5, figsize=(15, 12))
    for _record, _axis in zip(candidate_records, _scan_axes.ravel()):
        _start = np.asarray(_record["start_level4_zyx"])
        _crop = np.asarray(scan_array[tuple(slice(int(a), int(a + scan_edge)) for a in _start)], dtype=np.float32)
        _axis.imshow(_crop.max(axis=0), cmap="magma")
        _axis.set_title(f"#{_record['index']} score={_record['surface_score']:.2f}")
        _axis.set_axis_off()
    _scan_figure.suptitle("20 distributed candidate crops, prediction level 4")
    _scan_figure.tight_layout()
    _scan_figure
    return


@app.cell
def _(
    ct_group,
    gaussian_filter,
    grow_surface_grid,
    ink_model,
    np,
    render_registered_ct,
    surface_group,
    torch,
    tracker_model,
    training_device,
):

    def infer_ink_stack(stack, valid_mask, model, device):
        height, width = valid_mask.shape
        if height < 64 or width < 64:
            raise ValueError("ink inference requires at least 64x64")
        probability_sum = torch.zeros((height, width), device=device)
        probability_count = torch.zeros((height, width), device=device)
        source = torch.from_numpy(stack.astype(np.float32)).clamp(0, 200).div(255.0)
        y_starts = sorted(set([0, max(0, height - 64)] + list(range(0, max(1, height - 63), 32))))
        x_starts = sorted(set([0, max(0, width - 64)] + list(range(0, max(1, width - 63), 32))))
        with torch.no_grad():
            for y_start in y_starts:
                for x_start in x_starts:
                    tile = source[:, y_start:y_start + 64, x_start:x_start + 64]
                    native = tile[None, :, None].to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        logits = model(native).reshape(1, 1, 4, 4)
                    probability = torch.sigmoid(torch.nn.functional.interpolate(
                        logits.float(), size=(64, 64), mode="bilinear", align_corners=False
                    ))[0, 0]
                    probability_sum[y_start:y_start + 64, x_start:x_start + 64] += probability
                    probability_count[y_start:y_start + 64, x_start:x_start + 64] += 1
        result = (probability_sum / probability_count.clamp_min(1)).cpu().numpy()
        result[~valid_mask] = 0
        return result


    def evaluate_candidate(record, grid_size=64):
        level3_shape = np.asarray(surface_group["3"].shape)
        start4 = np.asarray(record["start_level4_zyx"])
        start3 = np.clip(start4 * 2 + 16, 0, level3_shape - 96)
        crop3 = np.asarray(surface_group["3"][tuple(slice(int(a), int(a + 96)) for a in start3)], dtype=np.float32) / 255.0
        field3 = gaussian_filter(crop3, sigma=0.8)
        margin = 16
        search = field3[margin:-margin, margin:-margin, margin:-margin]
        seed_inner = np.asarray(np.unravel_index(int(np.argmax(search)), search.shape))
        seed_zyx = seed_inner + margin
        seed_xyz = seed_zyx[::-1].astype(np.float32)
        xyz_local, confidence, valid, _ = grow_surface_grid(field3, tracker_model, seed_xyz, size=grid_size)
        xyz_ct2, normals, stack, lower, upper = render_registered_ct(
            ct_group["2"], xyz_local, valid, start3, 3, np.arange(-15, 15, dtype=np.float32)
        )
        center = stack[len(stack) // 2]
        ink = infer_ink_stack(stack, valid, ink_model, training_device)
        valid_ink = ink[valid]
        coverage = float(valid.mean())
        ink_mean = float(valid_ink.mean()) if len(valid_ink) else 0.0
        ink_std = float(valid_ink.std()) if len(valid_ink) else 0.0
        ink_p99 = float(np.percentile(valid_ink, 99)) if len(valid_ink) else 0.0
        rank_score = 2.0 * coverage + 4.0 * ink_std + max(0.0, ink_p99 - ink_mean)
        return {
            "candidate_index": record["index"],
            "surface_score": record["surface_score"],
            "start_level3_zyx": start3,
            "seed_level3_zyx": seed_zyx,
            "coverage": coverage,
            "ink_mean": ink_mean,
            "ink_std": ink_std,
            "ink_p99": ink_p99,
            "rank_score": rank_score,
            "xyz_ct2": xyz_ct2,
            "normals": normals,
            "confidence": confidence,
            "valid": valid,
            "stack": stack,
            "ct": center,
            "ink": ink,
            "ct_bounds": (lower, upper),
        }


    return evaluate_candidate, infer_ink_stack


@app.cell
def _(evaluate_candidate, ranked_candidates):
    top_candidate_results = []
    for _candidate_record in ranked_candidates[:5]:
        try:
            _candidate_result = evaluate_candidate(_candidate_record, grid_size=64)
            top_candidate_results.append(_candidate_result)
            print({key: value for key, value in _candidate_result.items() if key not in {
                "xyz_ct2", "normals", "confidence", "valid", "stack", "ct", "ink", "ct_bounds", "start_level3_zyx", "seed_level3_zyx"
            }})
        except Exception as _candidate_error:
            print("candidate failed", _candidate_record["index"], repr(_candidate_error))
    ranked_run_results = sorted(top_candidate_results, key=lambda item: item["rank_score"], reverse=True)
    candidate_metric_table = [
        {
            "candidate": item["candidate_index"],
            "coverage": round(item["coverage"], 3),
            "ink_mean": round(item["ink_mean"], 4),
            "ink_std": round(item["ink_std"], 4),
            "ink_p99": round(item["ink_p99"], 4),
            "rank_score": round(item["rank_score"], 3),
        }
        for item in ranked_run_results
    ]
    candidate_metric_table
    return candidate_metric_table, ranked_run_results


@app.cell
def _(np, plt, ranked_run_results, tifffile):
    candidate_artifact_directory = "/marimo/artifacts/vesuvius-ssm-candidates"
    import os as _candidate_os
    _candidate_os.makedirs(candidate_artifact_directory, exist_ok=True)
    _candidate_figure, _candidate_axes = plt.subplots(len(ranked_run_results), 3, figsize=(13, 4 * len(ranked_run_results)))
    for _row, _result in enumerate(ranked_run_results):
        _valid_ct = _result["ct"][_result["valid"]]
        _low, _high = np.percentile(_valid_ct, [1, 99]) if len(_valid_ct) else (0, 1)
        _preview = np.clip((_result["ct"] - _low) / max(_high - _low, 1e-6), 0, 1)
        _candidate_axes[_row, 0].imshow(_preview, cmap="gray", vmin=0, vmax=1)
        _candidate_axes[_row, 1].imshow(_result["ink"], cmap="gray", vmin=0, vmax=0.5)
        _candidate_axes[_row, 2].imshow(np.where(_result["valid"], _result["confidence"], np.nan), cmap="viridis", vmin=0, vmax=1)
        _candidate_axes[_row, 0].set_title(f"Candidate #{_result['candidate_index']} CT")
        _candidate_axes[_row, 1].set_title(f"ink p99={_result['ink_p99']:.3f}")
        _candidate_axes[_row, 2].set_title(f"coverage={_result['coverage']:.1%}")
        for _column in range(3):
            _candidate_axes[_row, _column].set_axis_off()
        _prefix = f"candidate-{_result['candidate_index']:02d}"
        np.savez_compressed(
            f"{candidate_artifact_directory}/{_prefix}-surface.npz",
            xyz_ct_level2=_result["xyz_ct2"], normals=_result["normals"],
            confidence=_result["confidence"], valid=_result["valid"],
        )
        tifffile.imwrite(f"{candidate_artifact_directory}/{_prefix}-layers.tif", _result["stack"])
        plt.imsave(f"{candidate_artifact_directory}/{_prefix}-ct.png", _preview, cmap="gray", vmin=0, vmax=1)
        plt.imsave(f"{candidate_artifact_directory}/{_prefix}-ink.png", _result["ink"], cmap="gray", vmin=0, vmax=0.5)
    _candidate_figure.tight_layout()
    _candidate_figure.savefig(f"{candidate_artifact_directory}/ranked-gallery.png", dpi=160, bbox_inches="tight")
    _candidate_figure
    return (candidate_artifact_directory,)


@app.cell
def _(estimate_frame, np, torch, trace_ridge, training_device, unit_vector):

    def predict_gap_rollout(model, ground_truth, probabilities, normal, gap_start=8, gap_length=8):
        state = None
        previous = ground_truth[1] - ground_truth[0]
        for index in range(gap_start):
            if index:
                previous = ground_truth[index] - ground_truth[index - 1]
            tangent = unit_vector(previous)
            obs = np.r_[previous, tangent, normal, probabilities[index], 0.0, 0.0, 1.0].astype(np.float32)
            with torch.no_grad():
                _, _, state = model(torch.from_numpy(obs)[None, None].to(training_device), state)
        point = ground_truth[gap_start].copy()
        predicted = []
        for _ in range(gap_length):
            tangent = unit_vector(previous)
            side = unit_vector(np.cross(normal, tangent))
            obs = np.r_[previous, tangent, normal, 0.0, 0.0, 1.0, 1.0].astype(np.float32)
            with torch.no_grad():
                local_delta, _, state = model(torch.from_numpy(obs)[None, None].to(training_device), state)
            frame = np.stack((tangent, side, normal), axis=-1)
            delta = frame @ local_delta[0, 0].float().cpu().numpy()
            if np.dot(delta, tangent) < 0:
                delta *= -1
            delta = unit_vector(delta) * max(0.5, min(1.5, float(np.linalg.norm(delta))))
            point = point + delta
            previous = delta
            predicted.append(point.copy())
        return np.asarray(predicted)


    def run_gap_benchmark(field, model, examples=64, gap_start=8, gap_length=8, random_seed=23):
        rng = np.random.default_rng(random_seed)
        candidates = np.argwhere(field > 0.55)
        records = []
        for candidate in candidates[rng.permutation(len(candidates))]:
            seed_xyz = candidate[::-1].astype(np.float32)
            try:
                tangent_u, tangent_v, normal = estimate_frame(field, seed_xyz)
            except ValueError:
                continue
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.cos(angle) * tangent_u + np.sin(angle) * tangent_v
            truth, probabilities = trace_ridge(field, seed_xyz, direction, normal, steps=gap_start + gap_length + 2)
            if len(truth) < gap_start + gap_length + 1:
                continue
            model_points = predict_gap_rollout(model, truth, probabilities, normal, gap_start, gap_length)
            previous = truth[gap_start] - truth[gap_start - 1]
            straight_points = truth[gap_start][None] + np.arange(1, gap_length + 1)[:, None] * previous[None]
            target = truth[gap_start + 1:gap_start + gap_length + 1]
            model_error = np.linalg.norm(model_points - target, axis=1)
            straight_error = np.linalg.norm(straight_points - target, axis=1)
            records.append({
                "model_mean_error": float(model_error.mean()),
                "straight_mean_error": float(straight_error.mean()),
                "model_endpoint_error": float(model_error[-1]),
                "straight_endpoint_error": float(straight_error[-1]),
            })
            if len(records) >= examples:
                break
        if not records:
            raise ValueError("no benchmark trajectories generated")
        return records


    return (run_gap_benchmark,)


@app.cell
def _(
    gaussian_filter,
    np,
    ranked_run_results,
    run_gap_benchmark,
    surface_group,
    tracker_model,
):
    best_candidate = ranked_run_results[0]
    _best_start3 = best_candidate["start_level3_zyx"]
    _best_crop = np.asarray(surface_group["3"][tuple(slice(int(a), int(a + 96)) for a in _best_start3)], dtype=np.float32) / 255.0
    benchmark_field = gaussian_filter(_best_crop, sigma=0.8)
    gap_records = run_gap_benchmark(benchmark_field, tracker_model, examples=64, gap_start=8, gap_length=8)
    _model_mean = np.asarray([record["model_mean_error"] for record in gap_records])
    _straight_mean = np.asarray([record["straight_mean_error"] for record in gap_records])
    _model_endpoint = np.asarray([record["model_endpoint_error"] for record in gap_records])
    _straight_endpoint = np.asarray([record["straight_endpoint_error"] for record in gap_records])
    gap_benchmark_summary = {
        "candidate": best_candidate["candidate_index"],
        "trajectories": len(gap_records),
        "gap_length_voxels": 8,
        "model_mean_error": float(_model_mean.mean()),
        "straight_mean_error": float(_straight_mean.mean()),
        "model_endpoint_error": float(_model_endpoint.mean()),
        "straight_endpoint_error": float(_straight_endpoint.mean()),
        "model_recovery_rate_under_3_voxels": float((_model_endpoint < 3).mean()),
        "straight_recovery_rate_under_3_voxels": float((_straight_endpoint < 3).mean()),
        "model_wins_fraction": float((_model_endpoint < _straight_endpoint).mean()),
    }
    gap_benchmark_summary
    return gap_benchmark_summary, gap_records


@app.cell
def _(gap_benchmark_summary, gap_records, np, plt):
    _gap_straight = np.asarray([record["straight_endpoint_error"] for record in gap_records])
    _gap_model = np.asarray([record["model_endpoint_error"] for record in gap_records])
    _gap_figure, _gap_axis = plt.subplots(figsize=(8, 5))
    _gap_axis.scatter(_gap_straight, _gap_model, alpha=0.65)
    _gap_limit = max(float(_gap_straight.max()), float(_gap_model.max()), 1.0)
    _gap_axis.plot([0, _gap_limit], [0, _gap_limit], "--", color="black", label="equal error")
    _gap_axis.set_xlabel("Straight continuation endpoint error (voxels)")
    _gap_axis.set_ylabel("SSM endpoint error (voxels)")
    _gap_axis.set_title(f"Artificial 8-voxel gaps on candidate #{gap_benchmark_summary['candidate']}")
    _gap_axis.legend()
    _gap_axis.grid(alpha=0.2)
    _gap_figure
    return


@app.cell
def _(
    candidate_artifact_directory,
    candidate_metric_table,
    candidate_records,
    gap_benchmark_summary,
    gap_records,
    mo,
    np,
    plt,
):
    import json as _json
    candidate_report = {
        "scan_count": len(candidate_records),
        "top_five": candidate_metric_table,
        "gap_benchmark": gap_benchmark_summary,
        "decision": "Do not scale the current GRU; it underperforms straight continuation on hidden gaps.",
    }
    with open(f"{candidate_artifact_directory}/metrics.json", "w") as _report_file:
        _json.dump(candidate_report, _report_file, indent=2)
    _report_straight = np.asarray([record["straight_endpoint_error"] for record in gap_records])
    _report_model = np.asarray([record["model_endpoint_error"] for record in gap_records])
    _report_figure, _report_axis = plt.subplots(figsize=(8, 5))
    _report_axis.scatter(_report_straight, _report_model, alpha=0.65)
    _report_limit = max(float(_report_straight.max()), float(_report_model.max()), 1.0)
    _report_axis.plot([0, _report_limit], [0, _report_limit], "--", color="black")
    _report_axis.set_xlabel("Straight endpoint error")
    _report_axis.set_ylabel("SSM endpoint error")
    _report_axis.set_title("Artificial 8-voxel gap benchmark")
    _report_figure.savefig(f"{candidate_artifact_directory}/gap-benchmark.png", dpi=160, bbox_inches="tight")
    mo.callout(mo.md(f"""## Ranked run complete

    **Best geometry candidate:** `#{candidate_metric_table[0]['candidate']}` with `{candidate_metric_table[0]['coverage']:.1%}` coverage.

    **Ink:** all five candidates remain below credible detection levels; best p99 is `{max(row['ink_p99'] for row in candidate_metric_table):.3f}`.

    **Gap benchmark:** the SSM endpoint error is `{gap_benchmark_summary['model_endpoint_error']:.2f}` voxels versus `{gap_benchmark_summary['straight_endpoint_error']:.2f}` for straight continuation. The SSM wins only `{gap_benchmark_summary['model_wins_fraction']:.1%}` of trajectories.

    **Decision:** do not scale this GRU. Add voxel-patch observations and train explicitly on masked gaps before another large run.

    Artifacts: `{candidate_artifact_directory}`
    """), kind="warn")
    return


@app.cell
def _(ndi, np, project_ridge, sample_field, scipy_griddata):

    def mesh_component(field, seed_zyx, threshold=0.4):
        mask = field >= threshold
        labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
        component_id = int(labels[tuple(seed_zyx)])
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if component_id == 0 or sizes[component_id] < 256:
            component_id = int(np.argmax(sizes))
        component = labels == component_id
        points_xyz = np.argwhere(component)[:, ::-1].astype(np.float32)
        scores = field[component].astype(np.float32)
        return points_xyz, scores


    def mesh_parameterize(points_xyz):
        center = points_xyz.mean(0)
        centered = points_xyz - center
        _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
        basis = vectors[:, ::-1]
        return (centered @ basis[:, :2]).astype(np.float32)


    def mesh_rasterize(points_xyz, scores, spacing=1.0):
        uv = mesh_parameterize(points_xyz)
        indices = np.rint((uv - uv.min(0)) / spacing).astype(int)
        shape = tuple((indices.max(0) + 1).tolist())
        xyz_sum = np.zeros((*shape, 3), np.float64)
        weights = np.zeros(shape, np.float64)
        best_score = np.zeros(shape, np.float32)
        best_xyz = np.zeros((*shape, 3), np.float32)
        for point, score, index in zip(points_xyz, scores, indices):
            u, v = index
            weight = max(float(score), 1e-3) ** 2
            xyz_sum[u, v] += point * weight
            weights[u, v] += weight
            if score >= best_score[u, v]:
                best_score[u, v] = score
                best_xyz[u, v] = point
        anchors = weights > 0
        xyz = np.full((*shape, 3), np.nan, np.float32)
        xyz[anchors] = (xyz_sum[anchors] / weights[anchors, None]).astype(np.float32)
        xyz[anchors] = 0.5 * xyz[anchors] + 0.5 * best_xyz[anchors]
        sheet_mask = ndi.binary_closing(anchors, structure=ndi.generate_binary_structure(2, 2), iterations=2)
        sheet_mask = ndi.binary_fill_holes(sheet_mask)
        mask_labels, mask_count = ndi.label(sheet_mask)
        if mask_count > 1:
            mask_sizes = np.bincount(mask_labels.ravel())
            mask_sizes[0] = 0
            sheet_mask = mask_labels == int(np.argmax(mask_sizes))
        anchors &= sheet_mask
        return xyz, anchors, sheet_mask, best_score


    def mesh_fill_holes(xyz, anchors, sheet_mask):
        result = xyz.copy()
        missing = sheet_mask & ~anchors
        known_uv = np.argwhere(anchors)
        missing_uv = np.argwhere(missing)
        if not len(missing_uv):
            return result
        for axis in range(3):
            values = xyz[..., axis][anchors]
            interpolated = scipy_griddata(known_uv, values, missing_uv, method="linear")
            unresolved = ~np.isfinite(interpolated)
            if unresolved.any():
                interpolated[unresolved] = scipy_griddata(known_uv, values, missing_uv[unresolved], method="nearest")
            result[..., axis][missing] = interpolated
        return result


    def mesh_optimize(field, xyz, anchors, sheet_mask, iterations=4):
        result = xyz.copy()
        anchor_values = xyz.copy()
        movable = sheet_mask & ~anchors
        kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
        for _ in range(iterations):
            count = ndi.convolve(sheet_mask.astype(np.float32), kernel, mode="constant")
            neighbor = np.stack([
                ndi.convolve(np.nan_to_num(result[..., axis]), kernel, mode="constant") for axis in range(3)
            ], axis=-1) / np.maximum(count[..., None], 1)
            result[movable] = 0.75 * result[movable] + 0.25 * neighbor[movable]
            du = np.gradient(np.nan_to_num(result), axis=1)
            dv = np.gradient(np.nan_to_num(result), axis=0)
            normals = np.cross(du, dv)
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
            for row, col in np.argwhere(movable):
                projected, probability = project_ridge(field, result[row, col], normals[row, col], radius=3.0)
                if probability >= 0.1:
                    result[row, col] = projected
            result[anchors] = anchor_values[anchors]
        result[~sheet_mask] = np.nan
        return result


    def mesh_quality(field, xyz, anchors, sheet_mask):
        valid = sheet_mask & np.isfinite(xyz).all(-1)
        probabilities = sample_field(field, xyz[valid])
        horizontal = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
        vertical = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
        edges = np.r_[horizontal[(valid[:, 1:] & valid[:, :-1])], vertical[(valid[1:] & valid[:-1])]]
        median_edge = float(np.median(edges)) if len(edges) else 0.0
        du = xyz[:-1, 1:] - xyz[:-1, :-1]
        dv = xyz[1:, :-1] - xyz[:-1, :-1]
        quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        area = np.cross(du, dv)
        reference = np.nanmean(area[quad_valid], axis=0)
        folded = (np.einsum("...i,i->...", area, reference) <= 0) & quad_valid
        return {
            "coverage": float(valid.sum() / max(1, sheet_mask.sum())),
            "anchor_fraction": float(anchors.sum() / max(1, sheet_mask.sum())),
            "probability_adherence": float((probabilities >= 0.2).mean()),
            "folded_quad_fraction": float(folded.sum() / max(1, quad_valid.sum())),
            "jump_fraction": float((edges > 3 * median_edge).mean()) if median_edge else 0.0,
        }


    return (
        mesh_component,
        mesh_fill_holes,
        mesh_optimize,
        mesh_quality,
        mesh_rasterize,
    )


@app.cell
def _(
    ct_group,
    gaussian_filter,
    infer_ink_stack,
    ink_model,
    mesh_component,
    mesh_fill_holes,
    mesh_optimize,
    mesh_quality,
    mesh_rasterize,
    mesh_repair_defects,
    np,
    ranked_run_results,
    render_registered_ct,
    surface_group,
    training_device,
):
    mesh_candidate = ranked_run_results[0]
    mesh_start3 = mesh_candidate["start_level3_zyx"]
    mesh_crop = np.asarray(surface_group["3"][tuple(slice(int(a), int(a + 96)) for a in mesh_start3)], dtype=np.float32) / 255.0
    mesh_field = gaussian_filter(mesh_crop, sigma=0.8)
    mesh_seed_zyx = mesh_candidate["seed_level3_zyx"]
    mesh_points_xyz, mesh_point_scores = mesh_component(mesh_field, mesh_seed_zyx, threshold=0.4)
    _mesh_center = mesh_points_xyz.mean(0)
    _mesh_centered = mesh_points_xyz - _mesh_center
    _, _mesh_vectors = np.linalg.eigh(_mesh_centered.T @ _mesh_centered / len(mesh_points_xyz))
    _mesh_normal = _mesh_vectors[:, 0]
    _mesh_depth = _mesh_centered @ _mesh_normal
    _mesh_seed_depth = (mesh_seed_zyx[::-1] - _mesh_center) @ _mesh_normal
    _mesh_layer_selection = np.abs(_mesh_depth - _mesh_seed_depth) <= 2.0
    mesh_points_xyz = mesh_points_xyz[_mesh_layer_selection]
    mesh_point_scores = mesh_point_scores[_mesh_layer_selection]
    mesh_xyz_initial, mesh_anchors, mesh_sheet_mask, mesh_anchor_scores = mesh_rasterize(mesh_points_xyz, mesh_point_scores)
    mesh_xyz_filled = mesh_fill_holes(mesh_xyz_initial, mesh_anchors, mesh_sheet_mask)
    mesh_xyz_optimized = mesh_optimize(mesh_field, mesh_xyz_filled, mesh_anchors, mesh_sheet_mask, iterations=4)
    mesh_xyz_optimized, mesh_anchors, mesh_repair_counts = mesh_repair_defects(
        mesh_field, mesh_xyz_optimized, mesh_anchors, mesh_sheet_mask, passes=2
    )
    mesh_metrics = mesh_quality(mesh_field, mesh_xyz_optimized, mesh_anchors, mesh_sheet_mask)
    mesh_valid = mesh_sheet_mask & np.isfinite(mesh_xyz_optimized).all(-1)
    mesh_xyz_ct2, mesh_normals, mesh_ct_stack, mesh_ct_lower, mesh_ct_upper = render_registered_ct(
        ct_group["2"], mesh_xyz_optimized, mesh_valid, mesh_start3, 3, np.arange(-15, 15, dtype=np.float32)
    )
    mesh_ct_image = mesh_ct_stack[len(mesh_ct_stack) // 2]
    _mesh_pad_h = max(0, 64 - mesh_valid.shape[0])
    _mesh_pad_w = max(0, 64 - mesh_valid.shape[1])
    _mesh_padded_stack = np.pad(mesh_ct_stack, ((0, 0), (0, _mesh_pad_h), (0, _mesh_pad_w)))
    _mesh_padded_valid = np.pad(mesh_valid, ((0, _mesh_pad_h), (0, _mesh_pad_w)))
    _mesh_padded_ink = infer_ink_stack(_mesh_padded_stack, _mesh_padded_valid, ink_model, training_device)
    mesh_ink_probability = _mesh_padded_ink[:mesh_valid.shape[0], :mesh_valid.shape[1]]
    mesh_ink_values = mesh_ink_probability[mesh_valid]
    mesh_run_summary = {
        **mesh_metrics,
        "candidate": mesh_candidate["candidate_index"],
        "grid_shape": mesh_valid.shape,
        "component_points": len(mesh_points_xyz),
        "valid_pixels": int(mesh_valid.sum()),
        "ink_mean": float(mesh_ink_values.mean()),
        "ink_p99": float(np.percentile(mesh_ink_values, 99)),
    }
    mesh_run_summary
    return (
        mesh_anchors,
        mesh_ct_image,
        mesh_ct_stack,
        mesh_field,
        mesh_ink_probability,
        mesh_metrics,
        mesh_normals,
        mesh_run_summary,
        mesh_seed_zyx,
        mesh_sheet_mask,
        mesh_start3,
        mesh_valid,
        mesh_xyz_ct2,
        mesh_xyz_optimized,
    )


@app.cell
def _(
    mesh_anchors,
    mesh_ct_image,
    mesh_ink_probability,
    mesh_metrics,
    mesh_run_summary,
    mesh_sheet_mask,
    mesh_valid,
    np,
    plt,
):
    _mesh_values = mesh_ct_image[mesh_valid]
    _mesh_low, _mesh_high = np.percentile(_mesh_values, [1, 99])
    mesh_ct_preview = np.clip((mesh_ct_image - _mesh_low) / max(_mesh_high - _mesh_low, 1e-6), 0, 1)
    _mesh_figure, _mesh_axes = plt.subplots(1, 4, figsize=(19, 5))
    _mesh_axes[0].imshow(mesh_anchors, cmap="gray")
    _mesh_axes[0].set_title(f"Anchors ({mesh_metrics['anchor_fraction']:.1%})")
    _mesh_axes[1].imshow(mesh_sheet_mask, cmap="gray")
    _mesh_axes[1].set_title(f"Sheet mask ({mesh_metrics['coverage']:.1%} filled)")
    _mesh_axes[2].imshow(mesh_ct_preview, cmap="gray")
    _mesh_axes[2].set_title("Mesh-first unwrapped CT")
    _mesh_axes[3].imshow(mesh_ink_probability, cmap="gray", vmin=0, vmax=0.5)
    _mesh_axes[3].set_title(f"Ink p99={mesh_run_summary['ink_p99']:.3f}")
    for _mesh_axis in _mesh_axes:
        _mesh_axis.set_axis_off()
    _mesh_figure.tight_layout()
    _mesh_figure
    return (mesh_ct_preview,)


@app.cell
def _(mesh_fill_holes, mesh_optimize, ndi, np):

    def mesh_defect_vertices(xyz, valid, jump_factor=3.0):
        defects = np.zeros(valid.shape, bool)
        horizontal = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
        vertical = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
        horizontal_valid = valid[:, 1:] & valid[:, :-1]
        vertical_valid = valid[1:] & valid[:-1]
        edges = np.r_[horizontal[horizontal_valid], vertical[vertical_valid]]
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
        reference = np.nanmean(area[quad_valid], axis=0)
        bad_quad = quad_valid & (np.einsum("...i,i->...", area, reference) <= 0)
        defects[:-1, :-1] |= bad_quad
        defects[:-1, 1:] |= bad_quad
        defects[1:, :-1] |= bad_quad
        defects[1:, 1:] |= bad_quad
        return defects


    def mesh_repair_defects(field, xyz, anchors, sheet_mask, passes=2):
        repaired = xyz.copy()
        repaired_anchors = anchors.copy()
        repair_counts = []
        for _ in range(passes):
            defects = mesh_defect_vertices(repaired, sheet_mask)
            repair_counts.append(int(defects.sum()))
            if not defects.any():
                break
            candidate_anchors = repaired_anchors.copy()
            candidate_anchors[ndi.binary_dilation(defects, iterations=1)] = False
            sparse = repaired.copy()
            sparse[~candidate_anchors] = np.nan
            candidate = mesh_fill_holes(sparse, candidate_anchors, sheet_mask)
            candidate = mesh_optimize(field, candidate, candidate_anchors, sheet_mask, iterations=4)
            candidate_count = int(mesh_defect_vertices(candidate, sheet_mask).sum())
            repair_counts.append(candidate_count)
            if candidate_count >= int(defects.sum()):
                break
            repaired = candidate
            repaired_anchors = candidate_anchors
        return repaired, repaired_anchors, repair_counts


    return (mesh_repair_defects,)


@app.cell
def _(
    mesh_component,
    mesh_field,
    mesh_fill_holes,
    mesh_optimize,
    mesh_quality,
    mesh_rasterize,
    mesh_seed_zyx,
    np,
):
    mesh_thickness_sweep = []
    for _half_thickness in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        # Re-extract because mesh_points_xyz currently contains the width-4 selection.
        _all_points, _all_scores = mesh_component(mesh_field, mesh_seed_zyx, threshold=0.4)
        _all_center = _all_points.mean(0)
        _all_centered = _all_points - _all_center
        _, _all_vectors = np.linalg.eigh(_all_centered.T @ _all_centered / len(_all_points))
        _all_normal = _all_vectors[:, 0]
        _all_depth = _all_centered @ _all_normal
        _all_seed_depth = (mesh_seed_zyx[::-1] - _all_center) @ _all_normal
        _selection = np.abs(_all_depth - _all_seed_depth) <= _half_thickness
        _xyz0, _anchors0, _mask0, _ = mesh_rasterize(_all_points[_selection], _all_scores[_selection])
        _filled0 = mesh_fill_holes(_xyz0, _anchors0, _mask0)
        _optimized0 = mesh_optimize(mesh_field, _filled0, _anchors0, _mask0, iterations=2)
        _metrics0 = mesh_quality(mesh_field, _optimized0, _anchors0, _mask0)
        mesh_thickness_sweep.append({
            "half_thickness": _half_thickness,
            "points": int(_selection.sum()),
            "shape": _mask0.shape,
            **_metrics0,
            "quality_score": _metrics0["folded_quad_fraction"] + _metrics0["jump_fraction"],
        })
    mesh_thickness_sweep
    return


@app.cell
def _(
    mesh_anchors,
    mesh_ct_image,
    mesh_ct_preview,
    mesh_ct_stack,
    mesh_ink_probability,
    mesh_normals,
    mesh_run_summary,
    mesh_sheet_mask,
    mesh_valid,
    mesh_xyz_ct2,
    mo,
    np,
    plt,
    tifffile,
):
    mesh_artifact_directory = "/marimo/artifacts/vesuvius-mesh-first"
    import os as _mesh_os
    import json as _mesh_json
    _mesh_os.makedirs(mesh_artifact_directory, exist_ok=True)
    np.savez_compressed(
        f"{mesh_artifact_directory}/surface.npz",
        xyz_ct_level2=mesh_xyz_ct2,
        normals=mesh_normals,
        anchors=mesh_anchors,
        sheet_mask=mesh_sheet_mask,
        valid=mesh_valid,
    )
    tifffile.imwrite(f"{mesh_artifact_directory}/surface-layers.tif", mesh_ct_stack)
    tifffile.imwrite(f"{mesh_artifact_directory}/unwrapped-ct.tif", mesh_ct_image)
    tifffile.imwrite(f"{mesh_artifact_directory}/ink-probability.tif", mesh_ink_probability.astype(np.float32))
    plt.imsave(f"{mesh_artifact_directory}/unwrapped-ct.png", mesh_ct_preview, cmap="gray", vmin=0, vmax=1)
    plt.imsave(f"{mesh_artifact_directory}/ink-probability.png", mesh_ink_probability, cmap="gray", vmin=0, vmax=0.5)
    with open(f"{mesh_artifact_directory}/metrics.json", "w") as _mesh_file:
        _mesh_json.dump({
            **{key: value for key, value in mesh_run_summary.items() if key != "grid_shape"},
            "grid_shape": list(mesh_run_summary["grid_shape"]),
            "quality_gate_pass": bool(
                mesh_run_summary["coverage"] >= 0.99
                and mesh_run_summary["probability_adherence"] >= 0.95
                and mesh_run_summary["folded_quad_fraction"] < 0.001
                and mesh_run_summary["jump_fraction"] < 0.01
            ),
        }, _mesh_file, indent=2)
    mesh_artifact_summary = sorted(_mesh_os.listdir(mesh_artifact_directory))
    mo.callout(mo.md(f"""## Mesh-first run

    - Coverage inside detected sheet mask: **{mesh_run_summary['coverage']:.2%}**
    - Surface adherence: **{mesh_run_summary['probability_adherence']:.2%}**
    - Folded quads: **{mesh_run_summary['folded_quad_fraction']:.2%}**
    - Long-edge jumps: **{mesh_run_summary['jump_fraction']:.2%}**

    Coverage and adherence pass. Topology gates do not yet pass, so this is not labeled a production-quality 100% unwrap.

    Artifacts: `{mesh_artifact_directory}`
    """), kind="warn")
    return


@app.cell
def _(Delaunay, igl, np, scipy_dijkstra, scipy_sparse, trimesh):

    def topo_boundary_loops(faces):
        edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = unique[counts == 1]
        adjacency = {}
        for left, right in boundary:
            adjacency.setdefault(int(left), []).append(int(right))
            adjacency.setdefault(int(right), []).append(int(left))
        if not adjacency or any(len(value) != 2 for value in adjacency.values()):
            return []
        unused = {tuple(edge) for edge in map(tuple, boundary)}
        loops = []
        while unused:
            start, current = next(iter(unused))
            previous = start
            loop = [start]
            while current != start:
                loop.append(current)
                unused.discard(tuple(sorted((previous, current))))
                following = [value for value in adjacency[current] if value != previous]
                if not following:
                    return []
                previous, current = current, following[0]
                if len(loop) > len(adjacency) + 1:
                    return []
            unused.discard(tuple(sorted((previous, current))))
            loops.append(np.asarray(loop, np.int64))
        return loops


    def topo_remove_unreferenced(vertices, faces):
        used, inverse = np.unique(faces.ravel(), return_inverse=True)
        return vertices[used], inverse.reshape(-1, 3).astype(np.int64)


    def topo_euler(vertices, faces):
        edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        return int(len(vertices) - len(np.unique(edges, axis=0)) + len(faces))


    def topo_signed_area(uv, faces):
        a, b, c = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
        return 0.5 * ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))


    def topo_seed_mesh(points_xyz, seed_xyz):
        centered = points_xyz - points_xyz.mean(0)
        _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
        basis = vectors[:, ::-1]
        initial_uv = centered @ basis[:, :2]
        faces = Delaunay(initial_uv).simplices.astype(np.int64)
        edge_index = np.stack((faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]), axis=1)
        edge_vectors = points_xyz[edge_index[..., 1]] - points_xyz[edge_index[..., 0]]
        lengths = np.linalg.norm(edge_vectors, axis=-1)
        median = float(np.median(lengths[lengths > 0]))
        normal = basis[:, 2]
        normal_step = np.abs(np.einsum("...i,i->...", edge_vectors, normal)) / np.maximum(lengths, 1e-6)
        keep = ((lengths < 2.5 * median) & (normal_step < 0.55)).all(axis=1)
        vertices, faces = topo_remove_unreferenced(points_xyz, faces[keep])
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        components = trimesh_mesh.split(only_watertight=False)
        component = min(components, key=lambda item: np.linalg.norm(item.vertices - seed_xyz, axis=1).min())
        vertices = np.asarray(component.vertices)
        faces = np.asarray(component.faces)
        seed_vertex = int(np.argmin(np.linalg.norm(vertices - seed_xyz, axis=1)))
        return vertices, faces, seed_vertex


    def topo_geodesic_disk(vertices, faces, seed_vertex, radius, retries=7):
        pairs = np.unique(np.sort(np.vstack((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]])), axis=1), axis=0)
        weights = np.linalg.norm(vertices[pairs[:,0]] - vertices[pairs[:,1]], axis=1)
        graph = scipy_sparse.coo_matrix((np.r_[weights,weights], (np.r_[pairs[:,0],pairs[:,1]], np.r_[pairs[:,1],pairs[:,0]])), shape=(len(vertices),len(vertices))).tocsr()
        distances = scipy_dijkstra(graph, indices=seed_vertex)
        current = radius
        for _ in range(retries):
            inside = distances <= current
            selected = faces[inside[faces].all(axis=1)]
            if len(selected):
                patch_vertices, patch_faces = topo_remove_unreferenced(vertices, selected)
                loops = topo_boundary_loops(patch_faces)
                if len(loops) == 1 and topo_euler(patch_vertices, patch_faces) == 1:
                    return patch_vertices, patch_faces, loops[0]
            current *= 0.8
        raise ValueError("no disk-like geodesic patch")


    def topo_parameterize(vertices, faces, boundary, iterations=10):
        vertices = np.ascontiguousarray(vertices, np.float64)
        faces64 = np.ascontiguousarray(faces, np.int64)
        boundary = np.ascontiguousarray(boundary, np.int64)
        circle = igl.map_vertices_to_circle(vertices, boundary)
        uv = np.asarray(igl.harmonic(vertices, faces64, boundary, circle, 1))
        area = topo_signed_area(uv, faces64)
        if np.median(area) < 0:
            uv[:,0] *= -1
            area = topo_signed_area(uv, faces64)
        if (area <= 1e-12).any():
            raise ValueError("harmonic map contains flips")
        slim = igl.slim_precompute(np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)), np.asfortranarray(uv), igl.MappingEnergyType.SYMMETRIC_DIRICHLET, boundary.astype(np.int32), np.asfortranarray(uv[boundary]), 1e5)
        candidate = np.asarray(igl.slim_solve(slim, iterations))[:,:2]
        candidate_area = topo_signed_area(candidate, faces64)
        if np.median(candidate_area) < 0:
            candidate[:,0] *= -1
            candidate_area = topo_signed_area(candidate, faces64)
        if (candidate_area > 1e-12).all():
            uv = candidate
        uv = (uv - uv.min(0)) / np.maximum(uv.max(0)-uv.min(0), 1e-9)
        return uv


    def topo_rasterize(vertices, faces, uv, spacing=1.0):
        area3 = 0.5*np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1).sum()
        area2 = np.abs(topo_signed_area(uv, faces)).sum()
        scale = np.sqrt(area3/max(area2,1e-12))/spacing
        pixel = (uv-uv.min(0))*scale
        width,height = np.maximum(np.ceil(pixel.max(0)).astype(int)+1,2)
        xyz=np.full((height,width,3),np.nan,np.float32); coverage=np.zeros((height,width),np.uint16)
        for face in faces:
            tri=pixel[face]; lower=np.maximum(np.floor(tri.min(0)).astype(int),0); upper=np.minimum(np.ceil(tri.max(0)).astype(int),[width-1,height-1])
            xs,ys=np.meshgrid(np.arange(lower[0],upper[0]+1),np.arange(lower[1],upper[1]+1)); p=np.stack((xs+0.5,ys+0.5),-1)
            a,b,c=tri; denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator)<1e-12: continue
            w0=((b[1]-c[1])*(p[...,0]-c[0])+(c[0]-b[0])*(p[...,1]-c[1]))/denominator
            w1=((c[1]-a[1])*(p[...,0]-c[0])+(a[0]-c[0])*(p[...,1]-c[1]))/denominator; w2=1-w0-w1
            inside=(w0>=-1e-7)&(w1>=-1e-7)&(w2>=-1e-7); yy,xx=ys[inside],xs[inside]
            values=(w0[...,None]*vertices[face[0]]+w1[...,None]*vertices[face[1]]+w2[...,None]*vertices[face[2]])[inside]
            empty=coverage[yy,xx]==0; xyz[yy[empty],xx[empty]]=values[empty]; coverage[yy,xx]+=1
        return xyz, coverage>0, coverage


    return (
        topo_boundary_loops,
        topo_euler,
        topo_geodesic_disk,
        topo_parameterize,
        topo_rasterize,
        topo_remove_unreferenced,
        topo_signed_area,
    )


@app.cell
def _(
    ct_group,
    mesh_field,
    mesh_seed_zyx,
    mesh_start3,
    mesh_valid,
    mesh_xyz_optimized,
    ndi,
    np,
    render_registered_ct,
    sample_field,
    topo_boundary_loops,
    topo_euler,
    topo_geodesic_fill_holes,
    topo_grid_mesh,
    topo_parameterize,
    topo_rasterize,
    topo_signed_area,
):
    topology_vertices0, topology_faces0, topology_seed_vertex0 = topo_grid_mesh(
        mesh_xyz_optimized, mesh_valid, mesh_seed_zyx[::-1].astype(np.float32), edge_factor=5.0
    )
    topology_vertices, topology_faces, topology_boundary = topo_geodesic_fill_holes(
        topology_vertices0, topology_faces0, topology_seed_vertex0, radius=45.0
    )
    topology_uv = topo_parameterize(topology_vertices, topology_faces, topology_boundary, iterations=10)
    topology_xyz_local, topology_valid, topology_coverage_count = topo_rasterize(
        topology_vertices, topology_faces, topology_uv, spacing=1.0
    )
    topology_area = topo_signed_area(topology_uv, topology_faces)
    topology_filled_mask = ndi.binary_fill_holes(topology_valid)
    topology_metrics = {
        "vertices": len(topology_vertices),
        "faces": len(topology_faces),
        "euler_characteristic": topo_euler(topology_vertices, topology_faces),
        "boundary_loops": len(topo_boundary_loops(topology_faces)),
        "flipped_fraction": float((topology_area <= 1e-12).mean()),
        "overlap_fraction": float((topology_coverage_count[topology_valid] > 1).mean()),
        "raster_coverage": float(topology_valid.sum()/topology_filled_mask.sum()),
    }
    topology_xyz_ct2, topology_normals, topology_ct_stack, topology_ct_lower, topology_ct_upper = render_registered_ct(
        ct_group["2"], topology_xyz_local, topology_valid, mesh_start3, 3, np.arange(-15,15,dtype=np.float32)
    )
    topology_confidence = sample_field(mesh_field, topology_xyz_local[topology_valid])
    topology_metrics["probability_adherence"] = float((topology_confidence >= 0.2).mean())
    topology_ct_image = topology_ct_stack[len(topology_ct_stack)//2]
    topology_metrics
    return (
        topology_boundary,
        topology_coverage_count,
        topology_ct_image,
        topology_ct_stack,
        topology_faces,
        topology_metrics,
        topology_normals,
        topology_uv,
        topology_valid,
        topology_vertices,
        topology_xyz_ct2,
    )


@app.cell
def _(
    np,
    plt,
    topology_coverage_count,
    topology_ct_image,
    topology_faces,
    topology_uv,
    topology_valid,
):
    _topology_values=topology_ct_image[topology_valid]
    _topology_low,_topology_high=np.percentile(_topology_values,[1,99])
    topology_ct_preview=np.clip((topology_ct_image-_topology_low)/max(_topology_high-_topology_low,1e-6),0,1)
    _topology_figure,_topology_axes=plt.subplots(1,3,figsize=(15,5))
    _topology_axes[0].triplot(topology_uv[:,0],topology_uv[:,1],topology_faces,linewidth=0.15)
    _topology_axes[0].set_title("LSCM + SLIM UV mesh")
    _topology_axes[1].imshow(topology_ct_preview,cmap="gray")
    _topology_axes[1].set_title("Topology-aware unwrapped CT")
    _topology_axes[2].imshow(topology_coverage_count,cmap="viridis")
    _topology_axes[2].set_title("Raster triangle coverage count")
    for _topology_axis in _topology_axes: _topology_axis.set_axis_off()
    _topology_figure.tight_layout(); _topology_figure
    return (topology_ct_preview,)


@app.cell
def _(np, topo_remove_unreferenced, trimesh):
    def topo_grid_mesh(xyz, valid, seed_xyz, edge_factor=3.0):
        indices=-np.ones(valid.shape,np.int64); indices[valid]=np.arange(valid.sum()); vertices=xyz[valid]
        faces=[]
        for row in range(valid.shape[0]-1):
            for col in range(valid.shape[1]-1):
                quad=indices[row:row+2,col:col+2]
                if (quad>=0).all():
                    faces.extend(((quad[0,0],quad[0,1],quad[1,0]),(quad[0,1],quad[1,1],quad[1,0])))
        faces=np.asarray(faces,np.int64)
        edges=np.stack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=1)
        lengths=np.linalg.norm(vertices[edges[...,1]]-vertices[edges[...,0]],axis=-1)
        median=float(np.median(lengths[lengths>0])); keep=(lengths<edge_factor*median).all(axis=1)
        vertices,faces=topo_remove_unreferenced(vertices,faces[keep])
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        components=mesh.split(only_watertight=False)
        component=min(components,key=lambda item:np.linalg.norm(item.vertices-seed_xyz,axis=1).min())
        vertices=np.asarray(component.vertices);faces=np.asarray(component.faces)
        seed_vertex=int(np.argmin(np.linalg.norm(vertices-seed_xyz,axis=1)))
        return vertices,faces,seed_vertex


    return (topo_grid_mesh,)


@app.cell
def _(
    np,
    scipy_dijkstra,
    scipy_sparse,
    topo_boundary_loops,
    topo_euler,
    topo_remove_unreferenced,
):
    def topo_geodesic_fill_holes(vertices,faces,seed_vertex,radius):
        pairs=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
        weights=np.linalg.norm(vertices[pairs[:,0]]-vertices[pairs[:,1]],axis=1)
        graph=scipy_sparse.coo_matrix((np.r_[weights,weights],(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
        distances=scipy_dijkstra(graph,indices=seed_vertex); selected=faces[(distances<=radius)[faces].all(axis=1)]
        patch_vertices,patch_faces=topo_remove_unreferenced(vertices,selected); loops=topo_boundary_loops(patch_faces)
        if not loops: raise ValueError("patch has no manifold boundaries")
        perimeters=[np.linalg.norm(np.roll(patch_vertices[loop],-1,axis=0)-patch_vertices[loop],axis=1).sum() for loop in loops]
        outer=int(np.argmax(perimeters)); face_normal=np.cross(patch_vertices[patch_faces[:,1]]-patch_vertices[patch_faces[:,0]],patch_vertices[patch_faces[:,2]]-patch_vertices[patch_faces[:,0]]).mean(0)
        for index,loop in enumerate(loops):
            if index==outer: continue
            center=patch_vertices[loop].mean(0); center_index=len(patch_vertices); patch_vertices=np.vstack((patch_vertices,center))
            new_faces=[]
            for left,right in zip(loop,np.roll(loop,-1)):
                face=np.asarray([left,right,center_index],np.int64)
                normal=np.cross(patch_vertices[right]-patch_vertices[left],center-patch_vertices[left])
                if np.dot(normal,face_normal)<0: face[[0,1]]=face[[1,0]]
                new_faces.append(face)
            patch_faces=np.vstack((patch_faces,np.asarray(new_faces)))
        final_loops=topo_boundary_loops(patch_faces)
        if len(final_loops)!=1 or topo_euler(patch_vertices,patch_faces)!=1: raise ValueError("hole filling did not produce disk")
        return patch_vertices,patch_faces,final_loops[0]


    return (topo_geodesic_fill_holes,)


@app.cell
def _(
    infer_ink_stack,
    ink_model,
    mo,
    np,
    plt,
    tifffile,
    topology_boundary,
    topology_ct_image,
    topology_ct_preview,
    topology_ct_stack,
    topology_faces,
    topology_metrics,
    topology_normals,
    topology_uv,
    topology_valid,
    topology_vertices,
    topology_xyz_ct2,
    training_device,
):
    _topology_pad_h=max(0,64-topology_valid.shape[0]);_topology_pad_w=max(0,64-topology_valid.shape[1])
    _topology_stack_padded=np.pad(topology_ct_stack,((0,0),(0,_topology_pad_h),(0,_topology_pad_w)))
    _topology_valid_padded=np.pad(topology_valid,((0,_topology_pad_h),(0,_topology_pad_w)))
    _topology_ink_padded=infer_ink_stack(_topology_stack_padded,_topology_valid_padded,ink_model,training_device)
    topology_ink_probability=_topology_ink_padded[:topology_valid.shape[0],:topology_valid.shape[1]]
    topology_ink_values=topology_ink_probability[topology_valid]
    topology_metrics["ink_mean"]=float(topology_ink_values.mean());topology_metrics["ink_p99"]=float(np.percentile(topology_ink_values,99))
    topology_artifact_directory="/marimo/artifacts/vesuvius-topology-unwrap"
    import os as _topology_os
    import json as _topology_json
    _topology_os.makedirs(topology_artifact_directory,exist_ok=True)
    np.savez_compressed(f"{topology_artifact_directory}/surface.npz",xyz_ct_level2=topology_xyz_ct2,normals=topology_normals,valid=topology_valid)
    np.savez_compressed(f"{topology_artifact_directory}/parameterized-mesh.npz",vertices=topology_vertices,faces=topology_faces,uv=topology_uv,boundary=topology_boundary)
    tifffile.imwrite(f"{topology_artifact_directory}/surface-layers.tif",topology_ct_stack)
    tifffile.imwrite(f"{topology_artifact_directory}/unwrapped-ct.tif",topology_ct_image)
    tifffile.imwrite(f"{topology_artifact_directory}/ink-probability.tif",topology_ink_probability.astype(np.float32))
    plt.imsave(f"{topology_artifact_directory}/unwrapped-ct.png",topology_ct_preview,cmap="gray",vmin=0,vmax=1)
    plt.imsave(f"{topology_artifact_directory}/ink-probability.png",topology_ink_probability,cmap="gray",vmin=0,vmax=0.5)
    with open(f"{topology_artifact_directory}/metrics.json","w") as _topology_file:_topology_json.dump(topology_metrics,_topology_file,indent=2)
    topology_artifact_summary=sorted(_topology_os.listdir(topology_artifact_directory))
    mo.callout(mo.md(f"""## Topology-aware unwrap complete

    - Mesh: **{topology_metrics['vertices']:,} vertices / {topology_metrics['faces']:,} faces**
    - UV flips: **{topology_metrics['flipped_fraction']:.2%}**
    - UV overlaps: **{topology_metrics['overlap_fraction']:.2%}**
    - Raster coverage: **{topology_metrics['raster_coverage']:.2%}**
    - Surface adherence: **{topology_metrics['probability_adherence']:.2%}**
    - Ink p99: **{topology_metrics['ink_p99']:.3f}**

    Artifacts: `{topology_artifact_directory}`
    """),kind="success")
    return


@app.cell
def _(
    ct_group,
    map_coordinates,
    mesh_start3,
    np,
    topo_rasterize,
    topology_faces,
    topology_uv,
    topology_vertices,
):
    fullres_xyz_local, fullres_valid, fullres_coverage_count = topo_rasterize(
        topology_vertices, topology_faces, topology_uv, spacing=0.25
    )
    _fullres_filled = np.nan_to_num(fullres_xyz_local)
    _fullres_du = np.gradient(_fullres_filled, axis=1)
    _fullres_dv = np.gradient(_fullres_filled, axis=0)
    fullres_normals_local = np.cross(_fullres_du, _fullres_dv)
    fullres_normals_local /= np.maximum(np.linalg.norm(fullres_normals_local, axis=-1, keepdims=True), 1e-6)
    fullres_xyz_ct0 = (fullres_xyz_local + mesh_start3[::-1]) * 32.0
    fullres_offsets = np.arange(-30, 31, dtype=np.float32)
    fullres_sample_points = fullres_xyz_ct0[..., None, :] + fullres_offsets[None, None, :, None] * fullres_normals_local[..., None, :]
    _fullres_finite = fullres_sample_points[np.isfinite(fullres_sample_points).all(-1)]
    _fullres_voxels_zyx = _fullres_finite[:, ::-1]
    fullres_crop_lower_zyx = np.maximum(np.floor(_fullres_voxels_zyx.min(0)).astype(int)-2, 0)
    fullres_crop_upper_zyx = np.minimum(np.ceil(_fullres_voxels_zyx.max(0)).astype(int)+3, np.asarray(ct_group["0"].shape))
    fullres_ct_crop = np.asarray(ct_group["0"][tuple(slice(int(a),int(b)) for a,b in zip(fullres_crop_lower_zyx,fullres_crop_upper_zyx))])
    _fullres_coordinates = np.moveaxis(fullres_sample_points[..., ::-1] - fullres_crop_lower_zyx, -1, 0).reshape(3,-1)
    _fullres_sampled = map_coordinates(fullres_ct_crop, _fullres_coordinates, order=1, mode="constant", cval=0, prefilter=False)
    fullres_ct_stack = _fullres_sampled.reshape(fullres_sample_points.shape[:-1]).transpose(2,0,1)
    fullres_ct_stack[:,~fullres_valid] = 0
    fullres_render_summary = {
        "raster_shape": fullres_valid.shape,
        "valid_pixels": int(fullres_valid.sum()),
        "depth_layers": len(fullres_offsets),
        "ct_crop_shape": fullres_ct_crop.shape,
        "ct_crop_gib": float(fullres_ct_crop.nbytes/2**30),
    }
    fullres_render_summary
    return


@app.cell
def _(candidate_records, ct_group, np):
    ct_candidate_records=[]
    _ct5=ct_group["5"];_ct5_shape=np.asarray(_ct5.shape)
    for _record in candidate_records:
        _surface_start4=np.asarray(_record["start_level4_zyx"])
        _ct5_start=np.clip(_surface_start4*2,0,_ct5_shape-128)
        _ct5_crop=np.asarray(_ct5[tuple(slice(int(a),int(a+128)) for a in _ct5_start)])
        _nonzero=_ct5_crop[_ct5_crop>0]
        _nonzero_fraction=float(len(_nonzero)/_ct5_crop.size)
        _ct_std=float(_nonzero.std()) if len(_nonzero) else 0.0
        _ct_range=float(np.percentile(_nonzero,99)-np.percentile(_nonzero,1)) if len(_nonzero) else 0.0
        _combined=_record["surface_score"]+3*_nonzero_fraction+0.01*_ct_std+0.002*_ct_range
        ct_candidate_records.append({**_record,"ct5_start_zyx":_ct5_start.tolist(),"ct_nonzero_fraction":_nonzero_fraction,"ct_std":_ct_std,"ct_range":_ct_range,"combined_score":_combined})
    ct_ranked_candidates=sorted(ct_candidate_records,key=lambda item:item["combined_score"],reverse=True)
    ct_ranked_candidates[:10]
    return (ct_ranked_candidates,)


@app.cell
def _(
    ct_ranked_candidates,
    gaussian_filter,
    igl,
    mesh_component,
    mesh_fill_holes,
    mesh_optimize,
    mesh_rasterize,
    np,
    surface_group,
    topo_geodesic_disk,
    topo_grid_mesh,
    topo_parameterize_robust,
    topo_rasterize,
    topo_signed_area,
):
    text_candidate=ct_ranked_candidates[2]
    text_start4=np.asarray(text_candidate["start_level4_zyx"])
    text_start3=np.clip(text_start4*2+16,0,np.asarray(surface_group["3"].shape)-96)
    text_crop=np.asarray(surface_group["3"][tuple(slice(int(a),int(a+96)) for a in text_start3)],dtype=np.float32)/255.0
    text_field=gaussian_filter(text_crop,sigma=0.8)
    _text_margin=16;_text_search=text_field[_text_margin:-_text_margin,_text_margin:-_text_margin,_text_margin:-_text_margin]
    text_seed_zyx=np.asarray(np.unravel_index(int(np.argmax(_text_search)),_text_search.shape))+_text_margin
    text_all_points,text_all_scores=mesh_component(text_field,text_seed_zyx,threshold=0.4)
    _text_center=text_all_points.mean(0);_text_centered=text_all_points-_text_center
    _,_text_vectors=np.linalg.eigh(_text_centered.T@_text_centered/len(text_all_points));_text_normal=_text_vectors[:,0]
    _text_depth=_text_centered@_text_normal;_text_seed_depth=(text_seed_zyx[::-1]-_text_center)@_text_normal
    _text_selection=np.abs(_text_depth-_text_seed_depth)<=2.0
    text_points=text_all_points[_text_selection];text_scores=text_all_scores[_text_selection]
    text_xyz0,text_anchors,text_sheet_mask,_=mesh_rasterize(text_points,text_scores)
    text_xyz_filled=mesh_fill_holes(text_xyz0,text_anchors,text_sheet_mask)
    text_xyz_optimized=mesh_optimize(text_field,text_xyz_filled,text_anchors,text_sheet_mask,iterations=4)
    text_valid=text_sheet_mask&np.isfinite(text_xyz_optimized).all(-1)
    text_vertices0,text_faces0,text_seed_vertex0=topo_grid_mesh(text_xyz_optimized,text_valid,text_seed_zyx[::-1].astype(np.float32),edge_factor=5.0)
    text_vertices,text_faces,text_boundary=topo_geodesic_disk(
     text_vertices0,text_faces0,text_seed_vertex0,radius=45.0
    )
    text_faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(text_faces,np.int64))[0])
    text_uv=topo_parameterize_robust(text_vertices,text_faces,text_boundary,iterations=10)
    text_xyz_local,text_valid_uv,text_coverage_count=topo_rasterize(text_vertices,text_faces,text_uv,spacing=0.25)
    text_area=topo_signed_area(text_uv,text_faces)
    text_bbox_ct0_xyz=np.vstack(((text_xyz_local[text_valid_uv]+text_start3[::-1])*32)).reshape(-1,3)
    text_geometry_summary={
     "candidate":text_candidate["index"],"vertices":len(text_vertices),"faces":len(text_faces),"raster_shape":text_valid_uv.shape,
     "flips":float((text_area<=1e-12).mean()),"overlaps":float((text_coverage_count[text_valid_uv]>1).mean()),
     "bbox_extent_ct0_xyz":(np.ceil(text_bbox_ct0_xyz.max(0)+35)-np.floor(text_bbox_ct0_xyz.min(0)-35)).astype(int).tolist(),
     "estimated_crop_gib":float(np.prod(np.ceil(text_bbox_ct0_xyz.max(0)+35)-np.floor(text_bbox_ct0_xyz.min(0)-35))/2**30),
    }
    text_geometry_summary
    return (
        text_faces,
        text_start3,
        text_uv,
        text_valid_uv,
        text_vertices,
        text_xyz_local,
    )


@app.cell
def _(igl, np, scipy_sparse, topo_signed_area):
    def topo_parameterize_robust(vertices,faces,boundary,iterations=10):
     vertices=np.ascontiguousarray(vertices,np.float64);faces64=np.ascontiguousarray(faces,np.int64);boundary=np.ascontiguousarray(boundary,np.int64)
     circle=igl.map_vertices_to_circle(vertices,boundary);uv=np.asarray(igl.harmonic(vertices,faces64,boundary,circle,1));area=topo_signed_area(uv,faces64)
     if np.median(area)<0:uv[:,0]*=-1;area=topo_signed_area(uv,faces64)
     if (area<=1e-12).any():
      pins=np.asarray([boundary[0],boundary[len(boundary)//2]],np.int64);pin_uv=np.asarray([[0.,0.],[1.,0.]])
      uv=np.asarray(igl.lscm(vertices,faces64,pins,pin_uv)[0]);area=topo_signed_area(uv,faces64)
      if np.median(area)<0:uv[:,0]*=-1;area=topo_signed_area(uv,faces64)
     if (area<=1e-12).any():
      edges=np.unique(np.sort(np.vstack((faces64[:,[0,1]],faces64[:,[1,2]],faces64[:,[2,0]])),axis=1),axis=0)
      adjacency=scipy_sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
      laplacian=scipy_sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
      interior=np.setdiff1d(np.arange(len(vertices)),boundary)
      uv=np.zeros((len(vertices),2),np.float64);uv[boundary]=circle
      uv[interior]=scipy_sparse.linalg.spsolve(laplacian[interior][:,interior],-laplacian[interior][:,boundary]@circle)
      area=topo_signed_area(uv,faces64)
      if np.median(area)<0:uv[:,0]*=-1;area=topo_signed_area(uv,faces64)
     if (area<=1e-12).any():raise ValueError(f"Tutte map still has {int((area<=1e-12).sum())} flips")
     slim=igl.slim_precompute(np.asfortranarray(vertices),np.asfortranarray(faces.astype(np.int32)),np.asfortranarray(uv),igl.MappingEnergyType.SYMMETRIC_DIRICHLET,boundary.astype(np.int32),np.asfortranarray(uv[boundary]),1e5)
     candidate=np.asarray(igl.slim_solve(slim,iterations))[:,:2];candidate_area=topo_signed_area(candidate,faces64)
     if np.median(candidate_area)<0:candidate[:,0]*=-1;candidate_area=topo_signed_area(candidate,faces64)
     if (candidate_area>1e-12).all():uv=candidate
     return (uv-uv.min(0))/np.maximum(uv.max(0)-uv.min(0),1e-9)


    return (topo_parameterize_robust,)


@app.cell
def _(
    ct_group,
    infer_ink_stack,
    ink_model,
    map_coordinates,
    ndi,
    np,
    skimage_exposure,
    text_start3,
    text_valid_uv,
    text_xyz_local,
    training_device,
):
    _text_filled=np.nan_to_num(text_xyz_local);_text_du=np.gradient(_text_filled,axis=1);_text_dv=np.gradient(_text_filled,axis=0)
    text_normals=np.cross(_text_du,_text_dv);text_normals/=np.maximum(np.linalg.norm(text_normals,axis=-1,keepdims=True),1e-6)
    text_xyz_ct0=(text_xyz_local+text_start3[::-1])*32.0
    text_offsets=np.arange(-30,31,dtype=np.float32)
    text_sample_points=text_xyz_ct0[...,None,:]+text_offsets[None,None,:,None]*text_normals[...,None,:]
    _text_finite=text_sample_points[np.isfinite(text_sample_points).all(-1)];_text_voxels=_text_finite[:,::-1]
    text_ct_lower=np.maximum(np.floor(_text_voxels.min(0)).astype(int)-2,0);text_ct_upper=np.minimum(np.ceil(_text_voxels.max(0)).astype(int)+3,np.asarray(ct_group["0"].shape))
    text_ct_crop=np.asarray(ct_group["0"][tuple(slice(int(a),int(b)) for a,b in zip(text_ct_lower,text_ct_upper))])
    _text_coordinates=np.moveaxis(text_sample_points[...,::-1]-text_ct_lower,-1,0).reshape(3,-1)
    _text_sampled=map_coordinates(text_ct_crop,_text_coordinates,order=1,mode="constant",cval=0,prefilter=False)
    text_ct_stack=_text_sampled.reshape(text_sample_points.shape[:-1]).transpose(2,0,1);text_ct_stack[:,~text_valid_uv]=0
    text_center=text_ct_stack[30].astype(np.float32);text_mean=text_ct_stack.mean(0);text_max=text_ct_stack.max(0);text_std=text_ct_stack.std(0)
    _text_stack_float=text_ct_stack.astype(np.float32);_text_blur=np.stack([ndi.gaussian_filter(layer,1.2) for layer in _text_stack_float])
    _text_focus=np.abs(_text_stack_float-_text_blur);text_best_index=np.argmax(_text_focus,axis=0)
    text_best=np.take_along_axis(_text_stack_float,text_best_index[None],axis=0)[0]
    def _normalize_render(image):
     values=image[text_valid_uv];low,high=np.percentile(values,[1,99]);return np.clip((image-low)/max(high-low,1e-6),0,1)
    text_previews={"center":_normalize_render(text_center),"mean":_normalize_render(text_mean),"max":_normalize_render(text_max),"std":_normalize_render(text_std),"best-focus":_normalize_render(text_best)}
    text_enhanced=skimage_exposure.equalize_adapthist(text_previews["best-focus"],clip_limit=0.02);text_enhanced[~text_valid_uv]=0
    _text_ink_stack=text_ct_stack[15:45]
    text_ink_probability=infer_ink_stack(_text_ink_stack,text_valid_uv,ink_model,training_device)
    text_ink_values=text_ink_probability[text_valid_uv]
    text_render_metrics={"ct_nonzero_fraction":float((text_ct_stack[:,text_valid_uv]>0).mean()),"ct_std":float(text_ct_stack[:,text_valid_uv].std()),"spatial_std":float(text_best[text_valid_uv].std()),"ink_mean":float(text_ink_values.mean()),"ink_std":float(text_ink_values.std()),"ink_p99":float(np.percentile(text_ink_values,99))}
    text_render_metrics
    return (
        text_ct_stack,
        text_enhanced,
        text_ink_probability,
        text_normals,
        text_previews,
        text_render_metrics,
        text_xyz_ct0,
    )


@app.cell
def _(plt, text_enhanced, text_previews):
    _text_figure,_text_axes=plt.subplots(2,3,figsize=(15,10))
    for _axis,(_name,_image) in zip(_text_axes.ravel(),text_previews.items()):_axis.imshow(_image,cmap="gray");_axis.set_title(_name);_axis.set_axis_off()
    _text_axes[1,2].imshow(text_enhanced,cmap="gray");_text_axes[1,2].set_title("best-focus + CLAHE");_text_axes[1,2].set_axis_off();_text_figure.tight_layout();_text_figure
    return


@app.cell
def _(ct_ranked_candidates, evaluate_candidate):
    ct_rich_screen_results=[]
    for _ct_record in ct_ranked_candidates[:5]:
     try:
      _result=evaluate_candidate(_ct_record,grid_size=64)
      ct_rich_screen_results.append({"candidate":_ct_record["index"],"coverage":_result["coverage"],"ink_mean":_result["ink_mean"],"ink_std":_result["ink_std"],"ink_p99":_result["ink_p99"],"ct":_result["ct"],"ink":_result["ink"],"valid":_result["valid"]})
     except Exception as _screen_error:print("screen failed",_ct_record["index"],repr(_screen_error))
    ct_rich_screen_ranking=sorted(ct_rich_screen_results,key=lambda item:(item["ink_std"],item["ink_p99"]),reverse=True)
    [{key:value for key,value in item.items() if key not in {"ct","ink","valid"}} for item in ct_rich_screen_ranking]
    return (ct_rich_screen_ranking,)


@app.cell
def _(ct_rich_screen_ranking, np, plt):
    _screen_figure,_screen_axes=plt.subplots(len(ct_rich_screen_ranking),2,figsize=(9,4*len(ct_rich_screen_ranking)))
    for _row,_result in enumerate(ct_rich_screen_ranking):
     _values=_result["ct"][_result["valid"]];_low,_high=np.percentile(_values,[1,99]);_preview=np.clip((_result["ct"]-_low)/max(_high-_low,1e-6),0,1)
     _screen_axes[_row,0].imshow(_preview,cmap="gray");_screen_axes[_row,0].set_title(f"candidate {_result['candidate']} CT")
     _screen_axes[_row,1].imshow(_result["ink"],cmap="gray",vmin=0,vmax=0.5);_screen_axes[_row,1].set_title(f"ink std={_result['ink_std']:.4f}, p99={_result['ink_p99']:.3f}")
     for _axis in _screen_axes[_row]:_axis.set_axis_off()
    _screen_figure.tight_layout();_screen_figure
    return


@app.cell
def _(
    np,
    plt,
    text_ct_stack,
    text_enhanced,
    text_faces,
    text_ink_probability,
    text_normals,
    text_previews,
    text_render_metrics,
    text_uv,
    text_valid_uv,
    text_vertices,
    text_xyz_ct0,
    tifffile,
):
    text_search_directory="/marimo/artifacts/vesuvius-text-search"
    import os as _text_os
    import json as _text_json
    _text_os.makedirs(text_search_directory,exist_ok=True)
    def _save_text_candidate(candidate_id):
     prefix=f"candidate-{candidate_id:02d}"
     np.savez_compressed(f"{text_search_directory}/{prefix}-surface.npz",xyz_ct0=text_xyz_ct0,normals=text_normals,valid=text_valid_uv,uv=text_uv,vertices=text_vertices,faces=text_faces)
     tifffile.imwrite(f"{text_search_directory}/{prefix}-layers.tif",text_ct_stack)
     tifffile.imwrite(f"{text_search_directory}/{prefix}-ink.tif",text_ink_probability.astype(np.float32))
     for name,image in text_previews.items():plt.imsave(f"{text_search_directory}/{prefix}-{name}.png",image,cmap="gray",vmin=0,vmax=1)
     plt.imsave(f"{text_search_directory}/{prefix}-enhanced.png",text_enhanced,cmap="gray",vmin=0,vmax=1)
     plt.imsave(f"{text_search_directory}/{prefix}-ink.png",text_ink_probability,cmap="gray",vmin=0,vmax=0.5)
     with open(f"{text_search_directory}/{prefix}-metrics.json","w") as file:_text_json.dump(text_render_metrics,file,indent=2)
    _save_text_candidate(15)
    text_search_files=sorted(_text_os.listdir(text_search_directory));text_search_files
    return (text_search_directory,)


@app.cell
def _(
    mo,
    np,
    plt,
    text_ct_stack,
    text_enhanced,
    text_faces,
    text_ink_probability,
    text_normals,
    text_previews,
    text_render_metrics,
    text_search_directory,
    text_uv,
    text_valid_uv,
    text_vertices,
    text_xyz_ct0,
    tifffile,
):
    import json as _text_json
    _candidate_id=7;_prefix=f"candidate-{_candidate_id:02d}"
    np.savez_compressed(f"{text_search_directory}/{_prefix}-surface.npz",xyz_ct0=text_xyz_ct0,normals=text_normals,valid=text_valid_uv,uv=text_uv,vertices=text_vertices,faces=text_faces)
    tifffile.imwrite(f"{text_search_directory}/{_prefix}-layers.tif",text_ct_stack)
    tifffile.imwrite(f"{text_search_directory}/{_prefix}-ink.tif",text_ink_probability.astype(np.float32))
    for _name,_image in text_previews.items():plt.imsave(f"{text_search_directory}/{_prefix}-{_name}.png",_image,cmap="gray",vmin=0,vmax=1)
    plt.imsave(f"{text_search_directory}/{_prefix}-enhanced.png",text_enhanced,cmap="gray",vmin=0,vmax=1)
    plt.imsave(f"{text_search_directory}/{_prefix}-ink.png",text_ink_probability,cmap="gray",vmin=0,vmax=0.5)
    with open(f"{text_search_directory}/{_prefix}-metrics.json","w") as _candidate_file:_text_json.dump(text_render_metrics,_candidate_file,indent=2)
    text_search_conclusion={"candidates_rendered_full_resolution":[15,7],"papyrus_texture_found":True,"credible_ink_found":False,"best_full_resolution_ink_p99":max(0.15432238578796387,text_render_metrics["ink_p99"]),"decision":"Do not interpret low model background as text; expand search to more CT-rich surface regions."}
    with open(f"{text_search_directory}/conclusion.json","w") as _conclusion_file:_text_json.dump(text_search_conclusion,_conclusion_file,indent=2)
    mo.callout(mo.md(f"""## Full-resolution text search

    Papyrus-scale CT texture was rendered successfully for candidates 15 and 7.

    - Candidate 15 ink p99: `0.154`
    - Candidate 7 ink p99: `{text_render_metrics['ink_p99']:.3f}`
    - Credible threshold: approximately `0.5`

    **No credible text was found in these patches.** The enhanced CT views are real surface texture; the low ink maps should not be read as letters.

    Artifacts: `{text_search_directory}`
    """),kind="warn")
    return


@app.cell
def _(ndi, np, surface_group):
    search_surface5=np.asarray(surface_group["5"],dtype=np.float32)/255.0
    search_surface5_smooth=ndi.gaussian_filter(search_surface5,0.8)
    _search_max=ndi.maximum_filter(search_surface5_smooth,size=5)
    _search_mask=(search_surface5_smooth>=0.35)&(search_surface5_smooth==_search_max)
    _search_mask[:3]=False;_search_mask[-3:]=False;_search_mask[:,:3]=False;_search_mask[:,-3:]=False;_search_mask[:,:,:3]=False;_search_mask[:,:,-3:]=False
    _search_locations=np.argwhere(_search_mask);_search_scores=search_surface5_smooth[_search_mask];_search_order=np.argsort(_search_scores)[::-1][:5000]
    large_search_candidates=[]
    for _candidate_id,_location_index in enumerate(_search_order):
     _zyx=_search_locations[_location_index];_lo=np.maximum(_zyx-3,0);_hi=np.minimum(_zyx+4,search_surface5_smooth.shape)
     _block=search_surface5_smooth[tuple(slice(int(a),int(b)) for a,b in zip(_lo,_hi))];_points=np.argwhere(_block>=0.25)
     if len(_points)>=6:
      _centered=_points-_points.mean(0);_,_vectors=np.linalg.eigh(_centered.T@_centered);_normal_zyx=_vectors[:,0]
     else:_normal_zyx=np.asarray([1.,0.,0.])
     large_search_candidates.append({"candidate_id":_candidate_id,"zyx5":_zyx.tolist(),"xyz_ct0":(_zyx[::-1]*128).astype(float).tolist(),"normal_xyz":_normal_zyx[::-1].astype(float).tolist(),"surface_score":float(_search_scores[_location_index])})
    large_search_phase1_summary={"level5_shape":search_surface5.shape,"raw_maxima":len(_search_locations),"retained":len(large_search_candidates),"threshold":0.35}
    large_search_phase1_summary
    return (search_surface5_smooth,)


@app.cell
def _(OrderedDict, np, unit_vector):
    class NotebookChunkSampler:
     def __init__(self,array,max_bytes=4*2**30):
      self.array=array;self.chunk_shape=np.asarray(array.chunks);self.max_bytes=max_bytes;self.cache=OrderedDict();self.bytes=0
     def chunk(self,key):
      if key in self.cache:
       value=self.cache.pop(key);self.cache[key]=value;return value
      lower=np.asarray(key)*self.chunk_shape;upper=np.minimum(lower+self.chunk_shape,self.array.shape)
      value=np.asarray(self.array[tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))])
      while self.cache and self.bytes+value.nbytes>self.max_bytes:
       _,removed=self.cache.popitem(last=False);self.bytes-=removed.nbytes
      self.cache[key]=value;self.bytes+=value.nbytes;return value
     def sample(self,xyz):
      points=np.asarray(xyz,np.float32);flat=points[...,::-1].reshape(-1,3);base=np.floor(flat).astype(np.int64);fraction=flat-base;result=np.zeros(len(flat),np.float32);shape=np.asarray(self.array.shape)
      for dz in (0,1):
       for dy in (0,1):
        for dx in (0,1):
         offset=np.asarray([dz,dy,dx]);indices=base+offset;valid=np.all((indices>=0)&(indices<shape),axis=1);weights=np.prod(np.where(offset,fraction,1-fraction),axis=1);keys=indices//self.chunk_shape
         for key_array in np.unique(keys[valid],axis=0):
          selected=valid&np.all(keys==key_array,axis=1);chunk=self.chunk(tuple(int(v) for v in key_array));local=indices[selected]-key_array*self.chunk_shape;result[selected]+=chunk[tuple(local.T)].astype(np.float32)*weights[selected]
      return result.reshape(points.shape[:-1])

    def notebook_candidate_coords(candidate,size=24,spacing=2.0,depth=16):
     center=np.asarray(candidate["xyz_ct0"],np.float32);normal=unit_vector(np.asarray(candidate["normal_xyz"],np.float32));reference=np.asarray([1.,0.,0.]) if abs(normal[0])<0.8 else np.asarray([0.,1.,0.]);u=unit_vector(np.cross(normal,reference));v=unit_vector(np.cross(normal,u));axis=(np.arange(size)-(size-1)/2)*spacing;vv,uu=np.meshgrid(axis,axis,indexing="ij");surface=center+uu[...,None]*u+vv[...,None]*v;offsets=np.arange(depth)-(depth-1)/2;return surface[None]+offsets[:,None,None,None]*normal


    return NotebookChunkSampler, notebook_candidate_coords


@app.cell
def _(
    NotebookChunkSampler,
    ct_group,
    large_search_supported,
    ndi,
    notebook_candidate_coords,
    np,
):
    large_search_sampler=NotebookChunkSampler(ct_group["0"])
    _live_candidates=sorted(large_search_supported[:32],key=lambda item:tuple((np.asarray(item["xyz_ct0"])[::-1]//192).astype(int)))
    large_search_scored=[]
    for _progress,_candidate in enumerate(_live_candidates):
     _stack=large_search_sampler.sample(notebook_candidate_coords(_candidate,size=24,spacing=2.0,depth=16))
     _nonzero=_stack[_stack>0];_center=_stack[len(_stack)//2];_blur=ndi.gaussian_filter(_center,1.2);_focus=np.abs(_center-_blur)
     _gx=ndi.sobel(_center,axis=1);_gy=ndi.sobel(_center,axis=0);_jxx=ndi.gaussian_filter(_gx*_gx,2);_jyy=ndi.gaussian_filter(_gy*_gy,2);_jxy=ndi.gaussian_filter(_gx*_gy,2);_coherence=np.sqrt((_jxx-_jyy)**2+4*_jxy**2)/(_jxx+_jyy+1e-6)
     _record={**_candidate,"ct_nonzero":float(len(_nonzero)/_stack.size),"ct_std":float(_nonzero.std()) if len(_nonzero) else 0.0,"focus_mean":float(_focus.mean()),"fiber_coherence":float(_coherence.mean())}
     _record["combined_score"]=1.5*_record["surface_score"]+2*_record["ct_nonzero"]+0.02*_record["ct_std"]+0.5*_record["fiber_coherence"]
     large_search_scored.append(_record)
    large_search_phase2_ranking=sorted(large_search_scored,key=lambda item:item["combined_score"],reverse=True)
    large_search_phase2_summary={"scored":len(large_search_scored),"cache_chunks":len(large_search_sampler.cache),"cache_gib":large_search_sampler.bytes/2**30,"top":[{key:item[key] for key in ["candidate_id","ct_nonzero","ct_std","fiber_coherence","combined_score"]} for item in large_search_phase2_ranking[:10]]}
    large_search_phase2_summary
    return (
        large_search_phase2_ranking,
        large_search_phase2_summary,
        large_search_sampler,
        large_search_scored,
    )


@app.cell
def _(ct_group, ndi, np, search_surface5_smooth):
    search_ct5=np.asarray(ct_group["5"])
    _all_max=ndi.maximum_filter(search_surface5_smooth,size=5);_all_mask=(search_surface5_smooth>=0.35)&(search_surface5_smooth==_all_max)
    _all_mask[:3]=False;_all_mask[-3:]=False;_all_mask[:,:3]=False;_all_mask[:,-3:]=False;_all_mask[:,:,:3]=False;_all_mask[:,:,-3:]=False
    _all_locations=np.argwhere(_all_mask);_all_scores=search_surface5_smooth[_all_mask]
    _ct_indices=_all_locations*4;_inside=np.all((_ct_indices>=0)&(_ct_indices<np.asarray(search_ct5.shape)),axis=1);_support=np.zeros(len(_all_locations),bool)
    _support[_inside]=search_ct5[tuple(_ct_indices[_inside].T)]>0
    _supported_order=np.argsort(_all_scores)[::-1];_supported_order=_supported_order[_support[_supported_order]][:5000]
    large_search_supported=[]
    for _candidate_id,_location_index in enumerate(_supported_order):
     _zyx=_all_locations[_location_index];_lo=np.maximum(_zyx-3,0);_hi=np.minimum(_zyx+4,search_surface5_smooth.shape);_block=search_surface5_smooth[tuple(slice(int(a),int(b)) for a,b in zip(_lo,_hi))];_points=np.argwhere(_block>=0.25)
     if len(_points)>=6:_centered=_points-_points.mean(0);_,_vectors=np.linalg.eigh(_centered.T@_centered);_normal_zyx=_vectors[:,0]
     else:_normal_zyx=np.asarray([1.,0.,0.])
     large_search_supported.append({"candidate_id":_candidate_id,"zyx5":_zyx.tolist(),"xyz_ct0":(_zyx[::-1]*128).astype(float).tolist(),"normal_xyz":_normal_zyx[::-1].astype(float).tolist(),"surface_score":float(_all_scores[_location_index])})
    large_search_support_summary={"raw_maxima":len(_all_locations),"ct_supported_maxima":int(_support.sum()),"retained":len(large_search_supported)}
    large_search_support_summary
    return large_search_support_summary, large_search_supported


@app.cell
def _(large_search_phase2_ranking, np, unit_vector):
    large_search_winners=[]
    for _candidate in large_search_phase2_ranking:
     _point=np.asarray(_candidate["xyz_ct0"]);_normal=unit_vector(np.asarray(_candidate["normal_xyz"]))
     _duplicate=False
     for _accepted in large_search_winners:
      _distance=np.linalg.norm(_point-np.asarray(_accepted["xyz_ct0"]));_similarity=abs(float(_normal@unit_vector(np.asarray(_accepted["normal_xyz"]))))
      if _distance<256 and _similarity>=0.8:_duplicate=True;break
     if not _duplicate:large_search_winners.append(_candidate)
     if len(large_search_winners)>=10:break
    large_search_phase3_summary={"input":len(large_search_phase2_ranking),"selected":len(large_search_winners),"candidate_ids":[item["candidate_id"] for item in large_search_winners]}
    large_search_phase3_summary
    return large_search_phase3_summary, large_search_winners


@app.cell
def _(
    large_search_phase2_summary,
    large_search_phase3_summary,
    large_search_scored,
    large_search_support_summary,
    large_search_winners,
    mo,
):
    large_search_directory="/marimo/artifacts/vesuvius-large-search"
    import os as _large_os
    import json as _large_json
    _large_os.makedirs(large_search_directory,exist_ok=True)
    with open(f"{large_search_directory}/phase1-summary.json","w") as file:_large_json.dump(large_search_support_summary,file,indent=2)
    with open(f"{large_search_directory}/phase2-scored.json","w") as file:_large_json.dump(large_search_scored,file,indent=2)
    with open(f"{large_search_directory}/phase3-winners.json","w") as file:_large_json.dump(large_search_winners,file,indent=2)
    mo.callout(mo.md(f"""## Four-phase search live proof

    - Coarse maxima: `{large_search_support_summary['raw_maxima']:,}`
    - CT-supported candidates: `{large_search_support_summary['ct_supported_maxima']:,}`
    - Full-resolution candidates scored in bounded proof: `{large_search_phase2_summary['scored']}`
    - Distinct winners after NMS: `{large_search_phase3_summary['selected']}`

    Artifacts: `{large_search_directory}`
    """),kind="success")
    return (large_search_directory,)


@app.cell
def _(
    gaussian_filter,
    igl,
    infer_ink_stack,
    ink_model,
    large_search_sampler,
    large_search_winners,
    mesh_component,
    mesh_fill_holes,
    mesh_optimize,
    mesh_rasterize,
    ndi,
    np,
    surface_group,
    topo_geodesic_disk,
    topo_geodesic_fill_holes,
    topo_grid_mesh,
    topo_parameterize_robust,
    topo_rasterize,
    topo_signed_area,
    training_device,
):
    winner_candidate=large_search_winners[0]
    winner_center_ct0=np.asarray(winner_candidate["xyz_ct0"],np.float32);winner_center3=winner_center_ct0[::-1]/32.0
    winner_start3=np.clip(np.floor(winner_center3).astype(int)-48,0,np.asarray(surface_group["3"].shape)-96)
    winner_crop=np.asarray(surface_group["3"][tuple(slice(int(a),int(a+96)) for a in winner_start3)],dtype=np.float32)/255.0
    winner_field=gaussian_filter(winner_crop,0.8);winner_seed_zyx=winner_center3-winner_start3
    winner_all_points,winner_all_scores=mesh_component(winner_field,np.rint(winner_seed_zyx).astype(int),threshold=0.4)
    _winner_center=winner_all_points.mean(0);_winner_centered=winner_all_points-_winner_center;_,_winner_vectors=np.linalg.eigh(_winner_centered.T@_winner_centered/len(winner_all_points));_winner_normal=_winner_vectors[:,0];_winner_depth=_winner_centered@_winner_normal;_winner_seed_depth=(winner_seed_zyx[::-1]-_winner_center)@_winner_normal;_winner_selection=np.abs(_winner_depth-_winner_seed_depth)<=2.0
    winner_points=winner_all_points[_winner_selection];winner_scores=winner_all_scores[_winner_selection]
    winner_xyz0,winner_anchors,winner_sheet_mask,_=mesh_rasterize(winner_points,winner_scores);winner_xyz_filled=mesh_fill_holes(winner_xyz0,winner_anchors,winner_sheet_mask);winner_xyz_optimized=mesh_optimize(winner_field,winner_xyz_filled,winner_anchors,winner_sheet_mask,iterations=4);winner_valid0=winner_sheet_mask&np.isfinite(winner_xyz_optimized).all(-1)
    winner_vertices0,winner_faces0,winner_seed_vertex0=topo_grid_mesh(winner_xyz_optimized,winner_valid0,winner_seed_zyx[::-1].astype(np.float32),edge_factor=5.0)
    try:winner_vertices,winner_faces,winner_boundary=topo_geodesic_fill_holes(winner_vertices0,winner_faces0,winner_seed_vertex0,radius=45.0)
    except ValueError:winner_vertices,winner_faces,winner_boundary=topo_geodesic_disk(winner_vertices0,winner_faces0,winner_seed_vertex0,radius=45.0)
    winner_faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(winner_faces,np.int64))[0]);winner_uv=topo_parameterize_robust(winner_vertices,winner_faces,winner_boundary,iterations=10)
    winner_xyz_local,winner_valid,winner_coverage_count=topo_rasterize(winner_vertices,winner_faces,winner_uv,spacing=0.25)
    _winner_filled=np.nan_to_num(winner_xyz_local);_winner_du=np.gradient(_winner_filled,axis=1);_winner_dv=np.gradient(_winner_filled,axis=0);winner_normals=np.cross(_winner_du,_winner_dv);winner_normals/=np.maximum(np.linalg.norm(winner_normals,axis=-1,keepdims=True),1e-6)
    winner_xyz_ct0=(winner_xyz_local+winner_start3[::-1])*32.0;winner_offsets=np.arange(-15,15,dtype=np.float32);winner_sample_points=winner_xyz_ct0[...,None,:]+winner_offsets[None,None,:,None]*winner_normals[...,None,:]
    winner_ct_stack=large_search_sampler.sample(winner_sample_points).transpose(2,0,1);winner_ct_stack[:,~winner_valid]=0
    _winner_pad_h=max(0,64-winner_valid.shape[0]);_winner_pad_w=max(0,64-winner_valid.shape[1]);_winner_stack_pad=np.pad(winner_ct_stack,((0,0),(0,_winner_pad_h),(0,_winner_pad_w)));_winner_valid_pad=np.pad(winner_valid,((0,_winner_pad_h),(0,_winner_pad_w)));_winner_ink_pad=infer_ink_stack(_winner_stack_pad,_winner_valid_pad,ink_model,training_device);winner_ink=_winner_ink_pad[:winner_valid.shape[0],:winner_valid.shape[1]]
    winner_area=topo_signed_area(winner_uv,winner_faces);winner_phase4_summary={"candidate_id":winner_candidate["candidate_id"],"vertices":len(winner_vertices),"faces":len(winner_faces),"raster_shape":winner_valid.shape,"flips":float((winner_area<=1e-12).mean()),"overlaps":float((winner_coverage_count[winner_valid]>1).mean()),"coverage":float(winner_valid.sum()/ndi.binary_fill_holes(winner_valid).sum()),"ct_nonzero":float((winner_ct_stack[:,winner_valid]>0).mean()),"ink_p99":float(np.percentile(winner_ink[winner_valid],99))}
    winner_phase4_summary
    return (
        winner_boundary,
        winner_candidate,
        winner_ct_stack,
        winner_faces,
        winner_ink,
        winner_normals,
        winner_phase4_summary,
        winner_uv,
        winner_valid,
        winner_vertices,
        winner_xyz_ct0,
    )


@app.cell
def _(
    large_search_directory,
    large_search_phase2_summary,
    large_search_phase3_summary,
    large_search_support_summary,
    mo,
    np,
    plt,
    tifffile,
    winner_boundary,
    winner_candidate,
    winner_ct_stack,
    winner_faces,
    winner_ink,
    winner_normals,
    winner_phase4_summary,
    winner_uv,
    winner_valid,
    winner_vertices,
    winner_xyz_ct0,
):
    import os as _large_os
    import json as _large_json
    winner_directory=f"{large_search_directory}/phase4-candidate-{winner_candidate['candidate_id']:06d}"
    _large_os.makedirs(winner_directory,exist_ok=True)
    np.savez_compressed(f"{winner_directory}/surface.npz",xyz_ct0=winner_xyz_ct0,normals=winner_normals,valid=winner_valid)
    np.savez_compressed(f"{winner_directory}/mesh.npz",vertices=winner_vertices,faces=winner_faces,uv=winner_uv,boundary=winner_boundary)
    tifffile.imwrite(f"{winner_directory}/surface-layers.tif",winner_ct_stack)
    tifffile.imwrite(f"{winner_directory}/ink-probability.tif",winner_ink.astype(np.float32))
    _winner_center=winner_ct_stack[len(winner_ct_stack)//2].astype(np.float32);_winner_values=_winner_center[winner_valid];_winner_low,_winner_high=np.percentile(_winner_values,[1,99]);winner_preview=np.clip((_winner_center-_winner_low)/max(_winner_high-_winner_low,1e-6),0,1);winner_preview[~winner_valid]=0
    plt.imsave(f"{winner_directory}/unwrapped-ct.png",winner_preview,cmap="gray",vmin=0,vmax=1);plt.imsave(f"{winner_directory}/ink.png",winner_ink,cmap="gray",vmin=0,vmax=0.5)
    with open(f"{winner_directory}/result.json","w") as _winner_file:_large_json.dump(winner_phase4_summary,_winner_file,indent=2)
    large_search_live_summary={"phase1":large_search_support_summary,"phase2":large_search_phase2_summary,"phase3":large_search_phase3_summary,"phase4":winner_phase4_summary,"production_cli_status":"implemented and locally tested","live_proof_scope":"32 full-resolution candidates from 3865 supported candidates"}
    with open(f"{large_search_directory}/live-summary.json","w") as _summary_file:_large_json.dump(large_search_live_summary,_summary_file,indent=2)
    mo.callout(mo.md(f"""## Phases 1–4 complete

    - CT-supported coarse candidates: **{large_search_support_summary['ct_supported_maxima']:,}**
    - Bounded full-resolution proof: **{large_search_phase2_summary['scored']}** candidates
    - NMS winners: **{large_search_phase3_summary['selected']}**
    - Topology winner coverage: **{winner_phase4_summary['coverage']:.2%}**
    - UV flips: **{winner_phase4_summary['flips']:.2%}**
    - Ink p99: **{winner_phase4_summary['ink_p99']:.3f}**

    Artifacts: `{large_search_directory}`
    """),kind="success")
    return


@app.cell
def _(
    NotebookChunkSampler,
    ct_group,
    large_search_supported,
    ndi,
    notebook_candidate_coords,
    np,
):
    import json as _production_json
    import os as _production_os
    production_directory="/marimo/artifacts/vesuvius-production-search"
    _production_os.makedirs(production_directory,exist_ok=True)
    production_checkpoint_path=f"{production_directory}/phase2a-fullres-checkpoint.json"
    if _production_os.path.exists(production_checkpoint_path):
     with open(production_checkpoint_path) as _checkpoint_file: production_results=_production_json.load(_checkpoint_file)
    else: production_results=[]
    _completed_ids={item["candidate_id"] for item in production_results}
    production_sampler=NotebookChunkSampler(ct_group["0"],max_bytes=8*2**30)
    _production_work=sorted(large_search_supported,key=lambda item:tuple((np.asarray(item["xyz_ct0"])[::-1]//192).astype(int)))
    for _production_index,_candidate in enumerate(_production_work):
     if _candidate["candidate_id"] in _completed_ids: continue
     try:
      _stack=production_sampler.sample(notebook_candidate_coords(_candidate,size=16,spacing=3.0,depth=8))
      _nonzero=_stack[_stack>0];_center=_stack[len(_stack)//2];_blur=ndi.gaussian_filter(_center,1.0);_focus=np.abs(_center-_blur)
      _gx=ndi.sobel(_center,axis=1);_gy=ndi.sobel(_center,axis=0);_jxx=ndi.gaussian_filter(_gx*_gx,1.5);_jyy=ndi.gaussian_filter(_gy*_gy,1.5);_jxy=ndi.gaussian_filter(_gx*_gy,1.5);_coherence=np.sqrt((_jxx-_jyy)**2+4*_jxy**2)/(_jxx+_jyy+1e-6)
      _result={**_candidate,"ct_nonzero":float(len(_nonzero)/_stack.size),"ct_std":float(_nonzero.std()) if len(_nonzero) else 0.0,"focus_mean":float(_focus.mean()),"fiber_coherence":float(_coherence.mean()),"status":"scored"}
      _result["combined_score"]=1.5*_result["surface_score"]+2*_result["ct_nonzero"]+0.02*_result["ct_std"]+0.5*_result["fiber_coherence"]
     except Exception as _production_error:
      _result={**_candidate,"status":"failed","error":repr(_production_error),"combined_score":-1.0}
     production_results.append(_result)
     if len(production_results)%25==0:
      with open(production_checkpoint_path+".tmp","w") as _checkpoint_file:_production_json.dump(production_results,_checkpoint_file)
      _production_os.replace(production_checkpoint_path+".tmp",production_checkpoint_path)
      print(f"phase2a {len(production_results)}/{len(_production_work)} cache={production_sampler.bytes/2**30:.2f}GiB")
    with open(production_checkpoint_path+".tmp","w") as _checkpoint_file:_production_json.dump(production_results,_checkpoint_file)
    _production_os.replace(production_checkpoint_path+".tmp",production_checkpoint_path)
    production_phase2a_ranking=sorted((item for item in production_results if item["status"]=="scored"),key=lambda item:item["combined_score"],reverse=True)
    production_phase2a_summary={"total_supported":len(_production_work),"completed":len(production_results),"scored":len(production_phase2a_ranking),"failed":sum(item["status"]=="failed" for item in production_results),"checkpoint":production_checkpoint_path,"top_ids":[item["candidate_id"] for item in production_phase2a_ranking[:20]]}
    production_phase2a_summary
    return production_directory, production_phase2a_ranking


@app.cell
def _(
    NotebookChunkSampler,
    ct_group,
    infer_ink_stack,
    ink_model,
    notebook_candidate_coords,
    np,
    production_directory,
    production_phase2a_ranking,
    training_device,
    unit_vector,
):
    import json as _ink_json
    import os as _ink_os
    production_finalists=[]
    for _candidate in production_phase2a_ranking:
     _point=np.asarray(_candidate["xyz_ct0"],np.float32);_normal=unit_vector(np.asarray(_candidate["normal_xyz"],np.float32));_duplicate=False
     for _accepted in production_finalists:
      _distance=np.linalg.norm(_point-np.asarray(_accepted["xyz_ct0"]));_similarity=abs(float(_normal@unit_vector(np.asarray(_accepted["normal_xyz"],np.float32))))
      if _distance<256 and _similarity>=0.8:_duplicate=True;break
     if not _duplicate:production_finalists.append(_candidate)
     if len(production_finalists)>=100:break
    production_ink_checkpoint=f"{production_directory}/phase2b-ink-checkpoint.json"
    if _ink_os.path.exists(production_ink_checkpoint):
     with open(production_ink_checkpoint) as _ink_file:production_ink_results=_ink_json.load(_ink_file)
    else:production_ink_results=[]
    _ink_completed={item["candidate_id"] for item in production_ink_results}
    production_ink_sampler=NotebookChunkSampler(ct_group["0"],max_bytes=8*2**30)
    _ink_work=sorted(production_finalists,key=lambda item:tuple((np.asarray(item["xyz_ct0"])[::-1]//192).astype(int)))
    for _ink_index,_candidate in enumerate(_ink_work):
     if _candidate["candidate_id"] in _ink_completed:continue
     try:
      _stack=production_ink_sampler.sample(notebook_candidate_coords(_candidate,size=64,spacing=1.0,depth=30))
      _ink=infer_ink_stack(_stack,np.ones((64,64),bool),ink_model,training_device);_values=_ink.ravel()
      _result={**_candidate,"ink_mean":float(_values.mean()),"ink_std":float(_values.std()),"ink_p99":float(np.percentile(_values,99)),"ink_max":float(_values.max()),"status":"scored"}
      _result["final_score"]=_candidate["combined_score"]+8*_result["ink_std"]+max(0.,_result["ink_p99"]-_result["ink_mean"])
     except Exception as _ink_error:_result={**_candidate,"status":"failed","error":repr(_ink_error),"final_score":-1.0}
     production_ink_results.append(_result)
     with open(production_ink_checkpoint+".tmp","w") as _ink_file:_ink_json.dump(production_ink_results,_ink_file)
     _ink_os.replace(production_ink_checkpoint+".tmp",production_ink_checkpoint)
     print(f"phase2b {len(production_ink_results)}/{len(_ink_work)} ink_p99={_result.get('ink_p99',0):.3f}")
    production_ink_ranking=sorted((item for item in production_ink_results if item["status"]=="scored"),key=lambda item:item["final_score"],reverse=True)
    production_phase2b_summary={"finalists":len(production_finalists),"completed":len(production_ink_results),"failed":sum(item["status"]=="failed" for item in production_ink_results),"checkpoint":production_ink_checkpoint,"top":[{key:item[key] for key in ["candidate_id","ink_mean","ink_std","ink_p99","ink_max","final_score"]} for item in production_ink_ranking[:20]]}
    production_phase2b_summary
    return (production_ink_ranking,)


@app.cell
def _(
    gaussian_filter,
    igl,
    infer_ink_stack,
    ink_model,
    mesh_component,
    mesh_fill_holes,
    mesh_optimize,
    mesh_rasterize,
    ndi,
    np,
    plt,
    surface_group,
    tifffile,
    topo_geodesic_disk,
    topo_geodesic_fill_holes,
    topo_grid_mesh,
    topo_parameterize_robust,
    topo_rasterize,
    topo_signed_area,
    training_device,
):
    def production_topology_unwrap(candidate, output_root, sampler):
        import json as _json
        import os as _os
        candidate_id = int(candidate["candidate_id"])
        directory = f"{output_root}/candidate-{candidate_id:06d}"
        _os.makedirs(directory, exist_ok=True)
        center_ct0 = np.asarray(candidate["xyz_ct0"], np.float32)
        center3 = center_ct0[::-1] / 32.0
        start3 = np.clip(np.floor(center3).astype(int) - 48, 0, np.asarray(surface_group["3"].shape) - 96)
        crop = np.asarray(surface_group["3"][tuple(slice(int(a), int(a + 96)) for a in start3)], dtype=np.float32) / 255.0
        field = gaussian_filter(crop, 0.8)
        seed_zyx = center3 - start3
        all_points, all_scores = mesh_component(field, np.rint(seed_zyx).astype(int), threshold=0.4)
        point_center = all_points.mean(0)
        centered = all_points - point_center
        _, vectors = np.linalg.eigh(centered.T @ centered / len(all_points))
        layer_normal = vectors[:, 0]
        depth = centered @ layer_normal
        nearest_seed_point = int(np.argmin(np.linalg.norm(all_points - seed_zyx[::-1], axis=1)))
        seed_depth = depth[nearest_seed_point]
        selection = np.abs(depth - seed_depth) <= 2.0
        points, scores = all_points[selection], all_scores[selection]
        if len(points) < 256:
            raise ValueError(f"seed layer has only {len(points)} points")
        xyz0, anchors, sheet_mask, _ = mesh_rasterize(points, scores)
        xyz_filled = mesh_fill_holes(xyz0, anchors, sheet_mask)
        xyz_optimized = mesh_optimize(field, xyz_filled, anchors, sheet_mask, iterations=4)
        valid0 = sheet_mask & np.isfinite(xyz_optimized).all(-1)
        vertices0, faces0, seed_vertex0 = topo_grid_mesh(xyz_optimized, valid0, seed_zyx[::-1].astype(np.float32), edge_factor=5.0)
        try:
            vertices, faces, boundary = topo_geodesic_fill_holes(vertices0, faces0, seed_vertex0, radius=45.0)
        except ValueError:
            vertices, faces, boundary = topo_geodesic_disk(vertices0, faces0, seed_vertex0, radius=45.0)
        faces = np.asarray(igl.bfs_orient(np.ascontiguousarray(faces, np.int64))[0])
        uv = topo_parameterize_robust(vertices, faces, boundary, iterations=10)
        xyz_local, valid, coverage_count = topo_rasterize(vertices, faces, uv, spacing=0.25)
        filled = np.nan_to_num(xyz_local)
        du, dv = np.gradient(filled, axis=1), np.gradient(filled, axis=0)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
        xyz_ct0 = (xyz_local + start3[::-1]) * 32.0
        offsets = np.arange(-15, 15, dtype=np.float32)
        sample_points = xyz_ct0[..., None, :] + offsets[None, None, :, None] * normals[..., None, :]
        stack = sampler.sample(sample_points).transpose(2, 0, 1)
        stack[:, ~valid] = 0
        pad_h, pad_w = max(0, 64 - valid.shape[0]), max(0, 64 - valid.shape[1])
        padded_stack = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)))
        padded_valid = np.pad(valid, ((0, pad_h), (0, pad_w)))
        padded_ink = infer_ink_stack(padded_stack, padded_valid, ink_model, training_device)
        ink = padded_ink[: valid.shape[0], : valid.shape[1]]
        area = topo_signed_area(uv, faces)
        center = stack[len(stack) // 2].astype(np.float32)
        values = center[valid]
        low, high = np.percentile(values, [1, 99])
        preview = np.clip((center - low) / max(high - low, 1e-6), 0, 1)
        preview[~valid] = 0
        metrics = {"candidate_id": candidate_id, "vertices": len(vertices), "faces": len(faces), "raster_shape": list(valid.shape), "flips": float((area <= 1e-12).mean()), "overlaps": float((coverage_count[valid] > 1).mean()), "coverage": float(valid.sum() / max(1, ndi.binary_fill_holes(valid).sum())), "ct_nonzero": float((stack[:, valid] > 0).mean()), "ink_mean": float(ink[valid].mean()), "ink_std": float(ink[valid].std()), "ink_p99": float(np.percentile(ink[valid], 99)), "screen_ink_p99": float(candidate["ink_p99"])}
        np.savez_compressed(f"{directory}/surface.npz", xyz_ct0=xyz_ct0, normals=normals, valid=valid)
        np.savez_compressed(f"{directory}/mesh.npz", vertices=vertices, faces=faces, uv=uv, boundary=boundary)
        tifffile.imwrite(f"{directory}/surface-layers.tif", stack)
        tifffile.imwrite(f"{directory}/ink-probability.tif", ink.astype(np.float32))
        plt.imsave(f"{directory}/unwrapped-ct.png", preview, cmap="gray", vmin=0, vmax=1)
        plt.imsave(f"{directory}/ink.png", ink, cmap="gray", vmin=0, vmax=1)
        with open(f"{directory}/result.json", "w") as _result_file:
            _json.dump(metrics, _result_file, indent=2)
        return metrics


    return (production_topology_unwrap,)


@app.cell
def _(
    NotebookChunkSampler,
    ct_group,
    production_directory,
    production_ink_ranking,
    production_topology_unwrap,
):
    import json as _phase4_json
    import os as _phase4_os
    production_phase4_directory = f"{production_directory}/phase4-topology"
    _phase4_os.makedirs(production_phase4_directory, exist_ok=True)
    production_phase4_checkpoint = f"{production_phase4_directory}/results.json"
    if _phase4_os.path.exists(production_phase4_checkpoint):
        with open(production_phase4_checkpoint) as _checkpoint_file:
            production_phase4_results = _phase4_json.load(_checkpoint_file)
    else:
        production_phase4_results = []
    _attempted = {item["candidate_id"] for item in production_phase4_results}
    _completed = sum(item["status"] == "complete" for item in production_phase4_results)
    production_phase4_sampler = NotebookChunkSampler(ct_group["0"], max_bytes=8 * 2**30)
    for _candidate in production_ink_ranking:
        if _completed >= 5:
            break
        if _candidate["candidate_id"] in _attempted:
            continue
        try:
            _metrics = production_topology_unwrap(_candidate, production_phase4_directory, production_phase4_sampler)
            _record = {"candidate_id": _candidate["candidate_id"], "status": "complete", "metrics": _metrics}
            _completed += 1
            print(f"phase4 complete candidate={_candidate['candidate_id']} coverage={_metrics['coverage']:.3f} ink_p99={_metrics['ink_p99']:.3f}")
        except Exception as _phase4_error:
            _record = {"candidate_id": _candidate["candidate_id"], "status": "failed", "error": repr(_phase4_error)}
            print(f"phase4 failed candidate={_candidate['candidate_id']} error={_phase4_error!r}")
        production_phase4_results.append(_record)
        with open(production_phase4_checkpoint + ".tmp", "w") as _checkpoint_file:
            _phase4_json.dump(production_phase4_results, _checkpoint_file, indent=2)
        _phase4_os.replace(production_phase4_checkpoint + ".tmp", production_phase4_checkpoint)
    production_phase4_summary = {"attempted": len(production_phase4_results), "completed": sum(item["status"] == "complete" for item in production_phase4_results), "failed": sum(item["status"] == "failed" for item in production_phase4_results), "checkpoint": production_phase4_checkpoint, "results": production_phase4_results}
    production_phase4_summary
    return


@app.cell
def _(map_coordinates, marching_cubes, ndi, np, trimesh):
    def ridge_sample_local(array, points_zyx):
        coordinates = np.asarray(points_zyx, np.float32).reshape(-1, 3).T
        return map_coordinates(array, coordinates, order=1, mode="constant", cval=0, prefilter=False).reshape(np.asarray(points_zyx).shape[:-1])


    def ridge_extract_seed_mesh(field, seed_zyx, probability_floor=0.15, anisotropy_floor=0.25):
        probability = np.asarray(field, np.float32)
        gradient = np.stack(np.gradient(probability), axis=-1)
        hessian = np.empty(probability.shape + (3, 3), np.float32)
        derivatives = [np.gradient(gradient[..., axis]) for axis in range(3)]
        for row in range(3):
            for col in range(3):
                hessian[..., row, col] = 0.5 * (derivatives[row][col] + derivatives[col][row])
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        normal_zyx = eigenvectors[..., :, 0]
        normal_curvature = np.maximum(-eigenvalues[..., 0], 0)
        anisotropy = normal_curvature / (np.abs(eigenvalues[..., 1]) + np.abs(eigenvalues[..., 2]) + normal_curvature + 1e-6)
        seed = np.clip(np.rint(seed_zyx).astype(int), 0, np.asarray(probability.shape) - 1)
        lower, upper = np.maximum(seed - 4, 0), np.minimum(seed + 5, probability.shape)
        neighborhood = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        local_confidence = probability[neighborhood] * anisotropy[neighborhood]
        best = lower + np.asarray(np.unravel_index(int(np.argmax(local_confidence)), local_confidence.shape))
        reference = normal_zyx[tuple(best)]
        orientation = np.sign(np.einsum("...i,i->...", normal_zyx, reference)); orientation[orientation == 0] = 1
        normal_zyx = normal_zyx * orientation[..., None]
        ridge = np.einsum("...i,...i->...", gradient, normal_zyx)
        positive_curvature = normal_curvature[normal_curvature > 0]
        curvature_floor = float(np.percentile(positive_curvature, 15)) if len(positive_curvature) else 0.0
        candidate_mask = (probability >= probability_floor) & (anisotropy >= anisotropy_floor) & (normal_curvature >= curvature_floor)
        candidate_mask = ndi.binary_dilation(candidate_mask, iterations=1)
        if not candidate_mask.any() or ridge[candidate_mask].min() > 0 or ridge[candidate_mask].max() < 0:
            raise ValueError("no ridge zero crossing near candidate")
        vertices_zyx, faces, _, _ = marching_cubes(ridge, level=0.0, mask=candidate_mask, allow_degenerate=False)
        vertex_probability = ridge_sample_local(probability, vertices_zyx)
        sampled_normals = np.stack([ridge_sample_local(normal_zyx[..., axis], vertices_zyx) for axis in range(3)], axis=-1)
        sampled_normals /= np.maximum(np.linalg.norm(sampled_normals, axis=-1, keepdims=True), 1e-6)
        keep_faces = (vertex_probability[faces] >= probability_floor).all(axis=1)
        faces = faces[keep_faces]
        vertices_xyz = vertices_zyx[:, ::-1]
        normals_xyz = sampled_normals[:, ::-1]
        used, inverse = np.unique(faces.ravel(), return_inverse=True)
        vertices_xyz, normals_xyz, vertex_probability = vertices_xyz[used], normals_xyz[used], vertex_probability[used]
        faces = inverse.reshape(-1, 3).astype(np.int64)
        edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
        vectors = vertices_xyz[edges[..., 1]] - vertices_xyz[edges[..., 0]]
        lengths = np.linalg.norm(vectors, axis=-1)
        nominal = float(np.median(lengths[lengths > 1e-6]))
        directions = vectors / np.maximum(lengths[..., None], 1e-6)
        normal_a, normal_b = normals_xyz[edges[..., 0]], normals_xyz[edges[..., 1]]
        dots = np.einsum("...i,...i->...", normal_a, normal_b)
        agreement = np.abs(dots)
        mean_normal = normal_a + np.where(dots[..., None] < 0, -normal_b, normal_b)
        mean_normal /= np.maximum(np.linalg.norm(mean_normal, axis=-1, keepdims=True), 1e-6)
        normal_step = np.abs(np.einsum("...i,...i->...", directions, mean_normal))
        keep = ((lengths <= 2.5 * nominal) & (agreement >= np.cos(np.deg2rad(50))) & (normal_step <= 0.55)).all(axis=1)
        faces = faces[keep]
        used, inverse = np.unique(faces.ravel(), return_inverse=True)
        vertices_xyz, normals_xyz, vertex_probability = vertices_xyz[used], normals_xyz[used], vertex_probability[used]
        faces = inverse.reshape(-1, 3).astype(np.int64)
        mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
        if not components:
            raise ValueError("ridge mesh has no components after bridge pruning")
        seed_xyz = np.asarray(seed_zyx)[::-1]
        component_distances = [float(np.linalg.norm(item.vertices - seed_xyz, axis=1).min()) for item in components]
        eligible_components = [item for item, distance in zip(components, component_distances) if distance <= 5.0]
        if not eligible_components:
            raise ValueError(f"no ridge component within 5 voxels; nearest={min(component_distances):.2f}")
        component = max(eligible_components, key=lambda item: len(item.faces))
        vertices = np.asarray(component.vertices, np.float64)
        faces = np.asarray(component.faces, np.int64)
        seed_vertex = int(np.argmin(np.linalg.norm(vertices - seed_xyz, axis=1)))
        face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=-1, keepdims=True), 1e-6)
        adjacency = component.face_adjacency
        continuity = np.abs(np.einsum("ij,ij->i", face_normals[adjacency[:, 0]], face_normals[adjacency[:, 1]])) if len(adjacency) else np.ones(1)
        edge_pairs = np.unique(np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        edge_lengths = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
        metrics = {
            "seed_distance": float(np.linalg.norm(vertices[seed_vertex] - seed_xyz)),
            "vertices": len(vertices),
            "faces": len(faces),
            "normal_continuity_p01": float(np.percentile(continuity, 1)),
            "edge_outlier_fraction": float((edge_lengths > 2.5 * np.median(edge_lengths)).mean()),
            "median_edge": float(np.median(edge_lengths)),
            "component_extent": (vertices.max(0) - vertices.min(0)).tolist(),
        }
        return vertices, faces, seed_vertex, metrics


    return (ridge_extract_seed_mesh,)


@app.cell
def _(gaussian_filter, np, ridge_extract_seed_mesh, surface_group):
    import json as _ridge_json
    with open("/marimo/artifacts/vesuvius-production-search/phase2b-ink-checkpoint.json") as _ridge_file:
        _ridge_records = _ridge_json.load(_ridge_file)
    ridge_candidate_153 = next(item for item in _ridge_records if item["candidate_id"] == 153)
    ridge_center_ct0_153 = np.asarray(ridge_candidate_153["xyz_ct0"], np.float32)
    ridge_center3_153 = ridge_center_ct0_153[::-1] / 32.0
    ridge_start3_153 = np.clip(np.floor(ridge_center3_153).astype(int) - 48, 0, np.asarray(surface_group["3"].shape) - 96)
    ridge_crop_153 = np.asarray(surface_group["3"][tuple(slice(int(a), int(a + 96)) for a in ridge_start3_153)], dtype=np.float32) / 255.0
    ridge_field_153 = gaussian_filter(ridge_crop_153, 0.8)
    ridge_seed_zyx_153 = ridge_center3_153 - ridge_start3_153
    ridge_vertices_153, ridge_faces_153, ridge_seed_vertex_153, ridge_metrics_153 = ridge_extract_seed_mesh(ridge_field_153, ridge_seed_zyx_153)
    ridge_gate_153 = {
        "seed_distance_pass": ridge_metrics_153["seed_distance"] <= 5.0,
        "normal_continuity_pass": ridge_metrics_153["normal_continuity_p01"] >= np.cos(np.deg2rad(50)),
        "edge_outlier_pass": ridge_metrics_153["edge_outlier_fraction"] <= 0.01,
        "size_pass": ridge_metrics_153["vertices"] >= 300,
    }
    ridge_gate_153["accepted"] = all(ridge_gate_153.values())
    {"metrics": ridge_metrics_153, "gates": ridge_gate_153}
    return (
        ridge_center_ct0_153,
        ridge_faces_153,
        ridge_seed_zyx_153,
        ridge_start3_153,
        ridge_vertices_153,
    )


@app.cell
def _(
    ct_group,
    np,
    plt,
    ridge_center_ct0_153,
    ridge_start3_153,
    ridge_vertices_153,
):
    _ct2 = ct_group["2"]
    _center2_zyx = np.rint(ridge_center_ct0_153[::-1] / 4).astype(int)
    _radius2 = 320
    _z0, _y0, _x0 = _center2_zyx
    _xy = np.asarray(_ct2[_z0, max(0,_y0-_radius2):min(_ct2.shape[1],_y0+_radius2), max(0,_x0-_radius2):min(_ct2.shape[2],_x0+_radius2)])
    _xz = np.asarray(_ct2[max(0,_z0-_radius2):min(_ct2.shape[0],_z0+_radius2), _y0, max(0,_x0-_radius2):min(_ct2.shape[2],_x0+_radius2)])
    _yz = np.asarray(_ct2[max(0,_z0-_radius2):min(_ct2.shape[0],_z0+_radius2), max(0,_y0-_radius2):min(_ct2.shape[1],_y0+_radius2), _x0])
    _vertices2_xyz = (ridge_vertices_153 + ridge_start3_153[::-1]) * 8.0
    _diagnostic_figure, _diagnostic_axes = plt.subplots(1,3,figsize=(18,6))
    _diagnostic_axes[0].imshow(_xy,cmap="gray"); _near = np.abs(_vertices2_xyz[:,2]-_z0)<=2; _diagnostic_axes[0].scatter(_vertices2_xyz[_near,0]-max(0,_x0-_radius2),_vertices2_xyz[_near,1]-max(0,_y0-_radius2),s=1,c="red"); _diagnostic_axes[0].set_title("XY CT + ridge")
    _diagnostic_axes[1].imshow(_xz,cmap="gray"); _near = np.abs(_vertices2_xyz[:,1]-_y0)<=2; _diagnostic_axes[1].scatter(_vertices2_xyz[_near,0]-max(0,_x0-_radius2),_vertices2_xyz[_near,2]-max(0,_z0-_radius2),s=1,c="red"); _diagnostic_axes[1].set_title("XZ CT + ridge")
    _diagnostic_axes[2].imshow(_yz,cmap="gray"); _near = np.abs(_vertices2_xyz[:,0]-_x0)<=2; _diagnostic_axes[2].scatter(_vertices2_xyz[_near,1]-max(0,_y0-_radius2),_vertices2_xyz[_near,2]-max(0,_z0-_radius2),s=1,c="red"); _diagnostic_axes[2].set_title("YZ CT + ridge")
    for _axis in _diagnostic_axes: _axis.set_axis_off()
    _diagnostic_figure.tight_layout(); _diagnostic_figure
    return


@app.cell
def _(
    np,
    ridge_faces_153,
    ridge_seed_zyx_153,
    ridge_vertices_153,
    topo_remove_unreferenced,
    trimesh,
):
    ridge_clean_vertices_153=ridge_vertices_153.copy();ridge_clean_faces_153=ridge_faces_153.copy()
    for _clean_iteration in range(3):
     _face_edges=np.stack((ridge_clean_faces_153[:,[0,1]],ridge_clean_faces_153[:,[1,2]],ridge_clean_faces_153[:,[2,0]]),axis=1);_sorted_edges=np.sort(_face_edges.reshape(-1,2),axis=1);_unique_edges,_inverse_edges,_edge_counts=np.unique(_sorted_edges,axis=0,return_inverse=True,return_counts=True);_face_counts=_edge_counts[_inverse_edges].reshape(-1,3);_keep_faces=(_face_counts<=2).all(axis=1)
     if _keep_faces.all():break
     ridge_clean_faces_153=ridge_clean_faces_153[_keep_faces];ridge_clean_vertices_153,ridge_clean_faces_153=topo_remove_unreferenced(ridge_clean_vertices_153,ridge_clean_faces_153)
    _clean_mesh=trimesh.Trimesh(vertices=ridge_clean_vertices_153,faces=ridge_clean_faces_153,process=False);_clean_components=_clean_mesh.split(only_watertight=False);_seed_xyz=ridge_seed_zyx_153[::-1];_eligible=[component for component in _clean_components if np.linalg.norm(component.vertices-_seed_xyz,axis=1).min()<=5.0]
    if not _eligible:raise ValueError("cleanup removed all seed-local ridge components")
    _clean_component=max(_eligible,key=lambda component:len(component.faces));ridge_clean_vertices_153=np.asarray(_clean_component.vertices);ridge_clean_faces_153=np.asarray(_clean_component.faces);ridge_clean_seed_vertex_153=int(np.argmin(np.linalg.norm(ridge_clean_vertices_153-_seed_xyz,axis=1)))
    ridge_cleanup_summary_153={"vertices":len(ridge_clean_vertices_153),"faces":len(ridge_clean_faces_153),"seed_distance":float(np.linalg.norm(ridge_clean_vertices_153[ridge_clean_seed_vertex_153]-_seed_xyz))}
    ridge_cleanup_summary_153
    return


@app.cell
def _(
    gaussian_filter,
    igl,
    np,
    ridge_extract_seed_mesh,
    surface_group,
    topo_boundary_loops,
    topo_euler,
    topo_geodesic_disk,
    topo_geodesic_fill_holes,
    topo_remove_unreferenced,
    trimesh,
):
    def validate_ridge_candidate(candidate):
     center_ct0=np.asarray(candidate["xyz_ct0"],np.float32);center3=center_ct0[::-1]/32.0;start3=np.clip(np.floor(center3).astype(int)-48,0,np.asarray(surface_group["3"].shape)-96);crop=np.asarray(surface_group["3"][tuple(slice(int(a),int(a+96)) for a in start3)],dtype=np.float32)/255.0;field=gaussian_filter(crop,0.8);seed_zyx=center3-start3
     vertices,faces,seed_vertex,raw_metrics=ridge_extract_seed_mesh(field,seed_zyx)
     for _ in range(3):
      face_edges=np.stack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=1);sorted_edges=np.sort(face_edges.reshape(-1,2),axis=1);_,inverse,counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True);keep=(counts[inverse].reshape(-1,3)<=2).all(axis=1)
      if keep.all():break
      faces=faces[keep];vertices,faces=topo_remove_unreferenced(vertices,faces)
     mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False);components=mesh.split(only_watertight=False);seed_xyz=seed_zyx[::-1];eligible=[component for component in components if np.linalg.norm(component.vertices-seed_xyz,axis=1).min()<=5]
     if not eligible:raise ValueError("no manifold seed-local component")
     component=max(eligible,key=lambda item:len(item.faces));vertices=np.asarray(component.vertices);faces=np.asarray(component.faces);seed_vertex=int(np.argmin(np.linalg.norm(vertices-seed_xyz,axis=1)))
     last_error=None
     for radius in (30.,20.,15.,10.,8.):
      try:
       try:patch_vertices,patch_faces,boundary=topo_geodesic_fill_holes(vertices,faces,seed_vertex,radius)
       except ValueError:patch_vertices,patch_faces,boundary=topo_geodesic_disk(vertices,faces,seed_vertex,radius)
       break
      except Exception as error:last_error=error
     else:raise ValueError(f"no disk-like ridge patch: {last_error}")
     patch_faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(patch_faces,np.int64))[0]);face_normals=np.cross(patch_vertices[patch_faces[:,1]]-patch_vertices[patch_faces[:,0]],patch_vertices[patch_faces[:,2]]-patch_vertices[patch_faces[:,0]]);face_normals/=np.maximum(np.linalg.norm(face_normals,axis=-1,keepdims=True),1e-6);patch_mesh=trimesh.Trimesh(vertices=patch_vertices,faces=patch_faces,process=False);adjacency=patch_mesh.face_adjacency;continuity=np.abs(np.einsum("ij,ij->i",face_normals[adjacency[:,0]],face_normals[adjacency[:,1]])) if len(adjacency) else np.ones(1);edges=np.unique(np.sort(np.vstack((patch_faces[:,[0,1]],patch_faces[:,[1,2]],patch_faces[:,[2,0]])),axis=1),axis=0);lengths=np.linalg.norm(patch_vertices[edges[:,0]]-patch_vertices[edges[:,1]],axis=1)
     metrics={"candidate_id":int(candidate["candidate_id"]),"vertices":len(patch_vertices),"faces":len(patch_faces),"seed_distance":float(np.linalg.norm(patch_vertices-seed_xyz,axis=1).min()),"normal_continuity_p01":float(np.percentile(continuity,1)),"edge_outlier_fraction":float((lengths>2.5*np.median(lengths)).mean()),"boundary_loops":len(topo_boundary_loops(patch_faces)),"euler":topo_euler(patch_vertices,patch_faces),"radius":radius}
     accepted=metrics["seed_distance"]<=5 and metrics["normal_continuity_p01"]>=np.cos(np.deg2rad(50)) and metrics["edge_outlier_fraction"]<=0.01 and metrics["boundary_loops"]==1 and metrics["euler"]==1 and metrics["vertices"]>=100
     if not accepted:raise ValueError(f"ridge gates failed: {metrics}")
     return {"candidate":candidate,"field":field,"start3":start3,"seed_zyx":seed_zyx,"vertices":patch_vertices,"faces":patch_faces,"boundary":boundary,"metrics":metrics}


    return (validate_ridge_candidate,)


@app.cell
def _(validate_ridge_candidate):
    import json as _validation_json
    with open("/marimo/artifacts/vesuvius-production-search/phase2b-ink-checkpoint.json") as _validation_file:_validation_records=_validation_json.load(_validation_file)
    _validation_ranked=sorted((item for item in _validation_records if item.get("status")=="scored"),key=lambda item:item.get("ink_p99",0),reverse=True)
    ridge_validation_results=[];ridge_accepted_patches=[]
    for _candidate in _validation_ranked[:50]:
     try:
      _patch=validate_ridge_candidate(_candidate);ridge_accepted_patches.append(_patch);ridge_validation_results.append({"candidate_id":_candidate["candidate_id"],"status":"accepted","metrics":_patch["metrics"],"screen_ink_p99":_candidate["ink_p99"]});print("ridge accepted",_candidate["candidate_id"],_patch["metrics"])
     except Exception as _validation_error:
      ridge_validation_results.append({"candidate_id":_candidate["candidate_id"],"status":"rejected","error":repr(_validation_error),"screen_ink_p99":_candidate["ink_p99"]});print("ridge rejected",_candidate["candidate_id"],repr(_validation_error))
     if len(ridge_accepted_patches)>=5:break
    ridge_validation_summary={"attempted":len(ridge_validation_results),"accepted":len(ridge_accepted_patches),"rejected":sum(item["status"]=="rejected" for item in ridge_validation_results),"results":ridge_validation_results}
    ridge_validation_summary
    return ridge_accepted_patches, ridge_validation_results


@app.cell
def _(
    ct_group,
    igl,
    infer_ink_stack,
    ink_model,
    ndi,
    np,
    plt,
    tifffile,
    topo_parameterize_robust,
    topo_rasterize,
    topo_signed_area,
    training_device,
):
    def render_validated_ridge_patch(patch, output_root, sampler):
     import json as _json
     import os as _os
     candidate=patch["candidate"];candidate_id=int(candidate["candidate_id"]);directory=f"{output_root}/candidate-{candidate_id:06d}";_os.makedirs(directory,exist_ok=True)
     vertices=patch["vertices"];faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(patch["faces"],np.int64))[0]);boundary=patch["boundary"];uv=topo_parameterize_robust(vertices,faces,boundary,iterations=10);xyz_local,valid,coverage_count=topo_rasterize(vertices,faces,uv,spacing=0.25)
     filled=np.nan_to_num(xyz_local);du,dv=np.gradient(filled,axis=1),np.gradient(filled,axis=0);normals=np.cross(du,dv);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-6);xyz_ct0=(xyz_local+patch["start3"][::-1])*32.0;offsets=np.arange(-15,15,dtype=np.float32);sample_points=xyz_ct0[...,None,:]+offsets[None,None,:,None]*normals[...,None,:];stack=sampler.sample(sample_points).transpose(2,0,1);stack[:,~valid]=0
     pad_h,pad_w=max(0,64-valid.shape[0]),max(0,64-valid.shape[1]);padded_stack=np.pad(stack,((0,0),(0,pad_h),(0,pad_w)));padded_valid=np.pad(valid,((0,pad_h),(0,pad_w)));padded_ink=infer_ink_stack(padded_stack,padded_valid,ink_model,training_device);ink=padded_ink[:valid.shape[0],:valid.shape[1]]
     center=stack[len(stack)//2].astype(np.float32);values=center[valid];low,high=np.percentile(values,[1,99]);preview=np.clip((center-low)/max(high-low,1e-6),0,1);preview[~valid]=0;area=topo_signed_area(uv,faces)
     metrics={**patch["metrics"],"raster_shape":list(valid.shape),"uv_flips":float((area<=1e-12).mean()),"uv_overlaps":float((coverage_count[valid]>1).mean()),"raster_coverage":float(valid.sum()/max(1,ndi.binary_fill_holes(valid).sum())),"ct_nonzero":float((stack[:,valid]>0).mean()),"ink_mean":float(ink[valid].mean()),"ink_std":float(ink[valid].std()),"ink_p99":float(np.percentile(ink[valid],99)),"screen_ink_p99":float(candidate["ink_p99"])}
     np.savez_compressed(f"{directory}/surface.npz",xyz_ct0=xyz_ct0,normals=normals,valid=valid);np.savez_compressed(f"{directory}/ridge-mesh.npz",vertices=vertices,faces=faces,uv=uv,boundary=boundary);tifffile.imwrite(f"{directory}/surface-layers.tif",stack);tifffile.imwrite(f"{directory}/ink-probability.tif",ink.astype(np.float32));plt.imsave(f"{directory}/unwrapped-ct.png",preview,cmap="gray",vmin=0,vmax=1);plt.imsave(f"{directory}/ink.png",ink,cmap="gray",vmin=0,vmax=1)
     center2=np.rint(np.asarray(candidate["xyz_ct0"])[::-1]/4).astype(int);radius=240;z0,y0,x0=center2;ct2=ct_group["2"];xy=np.asarray(ct2[z0,max(0,y0-radius):min(ct2.shape[1],y0+radius),max(0,x0-radius):min(ct2.shape[2],x0+radius)]);xz=np.asarray(ct2[max(0,z0-radius):min(ct2.shape[0],z0+radius),y0,max(0,x0-radius):min(ct2.shape[2],x0+radius)]);yz=np.asarray(ct2[max(0,z0-radius):min(ct2.shape[0],z0+radius),max(0,y0-radius):min(ct2.shape[1],y0+radius),x0]);vertices2=(vertices+patch["start3"][::-1])*8.0;figure,axes=plt.subplots(1,3,figsize=(15,5));axes[0].imshow(xy,cmap="gray");near=np.abs(vertices2[:,2]-z0)<=2;axes[0].scatter(vertices2[near,0]-max(0,x0-radius),vertices2[near,1]-max(0,y0-radius),s=2,c="red");axes[1].imshow(xz,cmap="gray");near=np.abs(vertices2[:,1]-y0)<=2;axes[1].scatter(vertices2[near,0]-max(0,x0-radius),vertices2[near,2]-max(0,z0-radius),s=2,c="red");axes[2].imshow(yz,cmap="gray");near=np.abs(vertices2[:,0]-x0)<=2;axes[2].scatter(vertices2[near,1]-max(0,y0-radius),vertices2[near,2]-max(0,z0-radius),s=2,c="red");[axis.set_axis_off() for axis in axes];figure.tight_layout();figure.savefig(f"{directory}/ct-ridge-overlays.png",dpi=160,bbox_inches="tight");plt.close(figure)
     with open(f"{directory}/result.json","w") as _result_file:_json.dump(metrics,_result_file,indent=2)
     return metrics


    return (render_validated_ridge_patch,)


@app.cell
def _(
    NotebookChunkSampler,
    ct_group,
    render_validated_ridge_patch,
    ridge_accepted_patches,
    ridge_validation_results,
):
    import json as _corrected_json
    import os as _corrected_os
    corrected_phase4_directory="/marimo/artifacts/vesuvius-production-search/phase4-ridge-corrected"
    _corrected_os.makedirs(corrected_phase4_directory,exist_ok=True)
    corrected_phase4_sampler=NotebookChunkSampler(ct_group["0"],max_bytes=8*2**30)
    corrected_phase4_results=[]
    for _patch in ridge_accepted_patches:
     try:
      _metrics=render_validated_ridge_patch(_patch,corrected_phase4_directory,corrected_phase4_sampler);corrected_phase4_results.append({"candidate_id":_metrics["candidate_id"],"status":"complete","metrics":_metrics});print("corrected complete",_metrics)
     except Exception as _corrected_error:
      corrected_phase4_results.append({"candidate_id":_patch["candidate"]["candidate_id"],"status":"failed","error":repr(_corrected_error)});print("corrected failed",repr(_corrected_error))
    corrected_phase4_report={"legacy_phase4_valid":False,"legacy_reason":"PCA/grid geometry flattened mixed or incorrect layers","ridge_validation":ridge_validation_results,"corrected_results":corrected_phase4_results,"accepted_count":sum(item["status"]=="complete" for item in corrected_phase4_results)}
    with open(f"{corrected_phase4_directory}/report.json","w") as _report_file:_corrected_json.dump(corrected_phase4_report,_report_file,indent=2)
    corrected_phase4_report
    return


@app.cell
def _(np):
    def subdivide_triangle_mesh(vertices,faces):
     edge_map={};new_vertices=[value.copy() for value in vertices];new_faces=[]
     def midpoint(left,right):
      key=tuple(sorted((int(left),int(right))))
      if key not in edge_map:edge_map[key]=len(new_vertices);new_vertices.append(0.5*(vertices[key[0]]+vertices[key[1]]))
      return edge_map[key]
     for a,b,c in faces:
      ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a);new_faces.extend(((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)))
     return np.asarray(new_vertices,np.float32),np.asarray(new_faces,np.int64)


    def mesh_vertex_normals(vertices,faces):
     normals=np.zeros_like(vertices,np.float32);face_normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]);face_normals/=np.maximum(np.linalg.norm(face_normals,axis=-1,keepdims=True),1e-6)
     for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
     normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-6);return normals


    def refine_mesh_on_prediction(vertices_surface0,faces,level,sampler,search_radius=4.0,samples=17):
     scale=2**level;vertices_level=vertices_surface0/scale;normals=mesh_vertex_normals(vertices_level,faces);offsets=np.linspace(-search_radius,search_radius,samples,dtype=np.float32);candidates=vertices_level[:,None,:]+offsets[None,:,None]*normals[:,None,:];scores=sampler.sample(candidates);best=np.argmax(scores,axis=1);refined_level=candidates[np.arange(len(candidates)),best];return refined_level*scale,scores[np.arange(len(scores)),best]


    return refine_mesh_on_prediction, subdivide_triangle_mesh


@app.cell
def _(
    NotebookChunkSampler,
    igl,
    np,
    refine_mesh_on_prediction,
    ridge_accepted_patches,
    subdivide_triangle_mesh,
    surface_group,
    topo_boundary_loops,
    topo_parameterize_robust,
    topo_rasterize,
):
    surface_sampler2=NotebookChunkSampler(surface_group["2"],max_bytes=2*2**30);surface_sampler1=NotebookChunkSampler(surface_group["1"],max_bytes=2*2**30);surface_sampler0=NotebookChunkSampler(surface_group["0"],max_bytes=2*2**30)
    refined_atlas_charts=[]
    for _patch in ridge_accepted_patches:
     _vertices=(_patch["vertices"]+_patch["start3"][::-1])*8.0;_faces=_patch["faces"].copy()
     _vertices,_faces=subdivide_triangle_mesh(_vertices,_faces);_vertices,_score2=refine_mesh_on_prediction(_vertices,_faces,2,surface_sampler2,search_radius=2.5)
     _vertices,_faces=subdivide_triangle_mesh(_vertices,_faces);_vertices,_score1=refine_mesh_on_prediction(_vertices,_faces,1,surface_sampler1,search_radius=2.5)
     _vertices,_faces=subdivide_triangle_mesh(_vertices,_faces);_vertices,_score0=refine_mesh_on_prediction(_vertices,_faces,0,surface_sampler0,search_radius=2.5)
     _faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(_faces,np.int64))[0]);_loops=topo_boundary_loops(_faces)
     if len(_loops)!=1:raise ValueError(f"refined candidate {_patch['candidate']['candidate_id']} has {len(_loops)} boundaries")
     _uv=topo_parameterize_robust(_vertices,_faces,_loops[0],iterations=10);_xyz_surface0,_valid,_coverage=topo_rasterize(_vertices,_faces,_uv,spacing=0.25);_xyz_ct0=_xyz_surface0*4.0;_filled=np.nan_to_num(_xyz_ct0);_du=np.gradient(_filled,axis=1);_dv=np.gradient(_filled,axis=0);_normals=np.cross(_du,_dv);_normals/=np.maximum(np.linalg.norm(_normals,axis=-1,keepdims=True),1e-6)
     refined_atlas_charts.append({"candidate_id":_patch["candidate"]["candidate_id"],"vertices_surface0":_vertices,"faces":_faces,"uv":_uv,"xyz_ct0":_xyz_ct0,"normals":_normals,"valid":_valid,"prediction_score_mean":float(_score0.mean()),"shape":_valid.shape})
     print("refined chart",_patch["candidate"]["candidate_id"],_valid.shape,len(_vertices),len(_faces))
    refined_atlas_summary=[{"candidate_id":chart["candidate_id"],"shape":list(chart["shape"]),"vertices":len(chart["vertices_surface0"]),"faces":len(chart["faces"]),"prediction_score_mean":chart["prediction_score_mean"]} for chart in refined_atlas_charts]
    refined_atlas_summary
    return


@app.cell
def _(
    igl,
    np,
    refine_mesh_on_prediction,
    ridge_accepted_patches,
    subdivide_triangle_mesh,
    topo_boundary_loops,
    topo_parameterize_robust,
    topo_rasterize,
    topo_signed_area,
    validate_ridge_candidate,
):
    def refine_and_save_patch(patch, output_root, sampler2, sampler1, sampler0, force=False):
     import json as _json
     import os as _os
     candidate_id=int(patch["candidate"]["candidate_id"]);directory=f"{output_root}/candidate-{candidate_id:06d}";_os.makedirs(directory,exist_ok=True);chart_path=f"{directory}/refined-chart.npz";result_path=f"{directory}/refinement.json"
     if not force and _os.path.exists(chart_path) and _os.path.exists(result_path):
      with open(result_path) as _file:return _json.load(_file)
     vertices=(patch["vertices"]+patch["start3"][::-1])*8.0;faces=patch["faces"].copy()
     vertices,faces=subdivide_triangle_mesh(vertices,faces);vertices,score2=refine_mesh_on_prediction(vertices,faces,2,sampler2,search_radius=2.5)
     vertices,faces=subdivide_triangle_mesh(vertices,faces);vertices,score1=refine_mesh_on_prediction(vertices,faces,1,sampler1,search_radius=2.5)
     vertices,faces=subdivide_triangle_mesh(vertices,faces);vertices,score0=refine_mesh_on_prediction(vertices,faces,0,sampler0,search_radius=2.5)
     faces=np.asarray(igl.bfs_orient(np.ascontiguousarray(faces,np.int64))[0]);loops=topo_boundary_loops(faces)
     if len(loops)!=1:raise ValueError(f"refined patch has {len(loops)} boundaries")
     uv=topo_parameterize_robust(vertices,faces,loops[0],iterations=10);xyz_surface0,valid,coverage=topo_rasterize(vertices,faces,uv,spacing=0.25);xyz_ct0=xyz_surface0*4.0;filled=np.nan_to_num(xyz_ct0);du=np.gradient(filled,axis=1);dv=np.gradient(filled,axis=0);normals=np.cross(du,dv);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-6)
     area=topo_signed_area(uv,faces);result={"candidate_id":candidate_id,"status":"refined","shape":list(valid.shape),"vertices":len(vertices),"faces":len(faces),"prediction_score_level2":float(score2.mean()),"prediction_score_level1":float(score1.mean()),"prediction_score_level0":float(score0.mean()),"uv_flips":float((area<=1e-12).mean()),"uv_overlaps":float((coverage[valid]>1).mean()),"artifact":chart_path}
     np.savez_compressed(chart_path,vertices_surface0=vertices,faces=faces,uv=uv,boundary=loops[0],xyz_ct0=xyz_ct0,normals=normals,valid=valid)
     with open(result_path+".tmp","w") as _file:_json.dump(result,_file,indent=2)
     _os.replace(result_path+".tmp",result_path);return result


    def process_patch_queue(candidate_ids, output_root, sampler2, sampler1, sampler0, force=False):
     import json as _json
     import os as _os
     _os.makedirs(output_root,exist_ok=True);manifest_path=f"{output_root}/queue.json"
     if _os.path.exists(manifest_path):
      with open(manifest_path) as _file:manifest=_json.load(_file)
     else:manifest={"candidate_ids":[],"results":[]}
     known={item["candidate_id"]:item for item in manifest["results"]};manifest["candidate_ids"]=list(dict.fromkeys(manifest["candidate_ids"]+[int(value) for value in candidate_ids]))
     phase2b_path="/marimo/artifacts/vesuvius-production-search/phase2b-ink-checkpoint.json"
     if _os.path.exists(phase2b_path):
      with open(phase2b_path) as _file:records=_json.load(_file)
     else:records=[]
     lookup={int(item["candidate_id"]):item for item in records}
     accepted_lookup={int(item["candidate"]["candidate_id"]):item for item in ridge_accepted_patches}
     for candidate_id in manifest["candidate_ids"]:
      if not force and candidate_id in known and known[candidate_id].get("status")=="refined":continue
      try:
       patch=accepted_lookup.get(candidate_id)
       if patch is None:
        if candidate_id not in lookup:raise KeyError(f"candidate {candidate_id} not in phase2b manifest")
        patch=validate_ridge_candidate(lookup[candidate_id])
       result=refine_and_save_patch(patch,output_root,sampler2,sampler1,sampler0,force=force)
      except Exception as error:result={"candidate_id":candidate_id,"status":"failed","error":repr(error)}
      known[candidate_id]=result;manifest["results"]=list(known.values())
      with open(manifest_path+".tmp","w") as _file:_json.dump(manifest,_file,indent=2)
      _os.replace(manifest_path+".tmp",manifest_path);print("patch queue",candidate_id,result["status"])
     return manifest


    return (process_patch_queue,)


@app.cell
def _(NotebookChunkSampler, process_patch_queue, surface_group):
    patch_queue_root="/marimo/artifacts/vesuvius-refined-atlas"
    patch_queue_sampler2=NotebookChunkSampler(surface_group["2"],max_bytes=2*2**30)
    patch_queue_sampler1=NotebookChunkSampler(surface_group["1"],max_bytes=2*2**30)
    patch_queue_sampler0=NotebookChunkSampler(surface_group["0"],max_bytes=2*2**30)
    patch_queue_manifest=process_patch_queue([2517,2638,1943,131,51],patch_queue_root,patch_queue_sampler2,patch_queue_sampler1,patch_queue_sampler0)
    patch_queue_manifest
    return (patch_queue_root,)


@app.cell
def _(np, patch_queue_root, zarr):
    import json as _atlas_json
    import os as _atlas_os
    with open(f"{patch_queue_root}/queue.json") as _atlas_file:_queue=_atlas_json.load(_atlas_file)
    _completed=[item for item in _queue["results"] if item.get("status")=="refined"]
    _atlas_charts=[]
    for item in _completed:
     data=np.load(item["artifact"]);_atlas_charts.append({"candidate_id":item["candidate_id"],"xyz":data["xyz_ct0"],"normals":data["normals"],"valid":data["valid"]})
    _columns=3;_padding=32;_column_widths=[]
    for column in range(_columns):_column_widths.append(max((chart["valid"].shape[1] for index,chart in enumerate(_atlas_charts) if index%_columns==column),default=0))
    _rows=int(np.ceil(len(_atlas_charts)/_columns));_row_heights=[]
    for row in range(_rows):_row_heights.append(max(chart["valid"].shape[0] for chart in _atlas_charts[row*_columns:(row+1)*_columns]))
    _atlas_width=sum(_column_widths)+_padding*(_columns+1);_atlas_height=sum(_row_heights)+_padding*(_rows+1)
    atlas_xyz=np.full((_atlas_height,_atlas_width,3),np.nan,np.float32);atlas_normals=np.zeros_like(atlas_xyz);atlas_valid=np.zeros((_atlas_height,_atlas_width),bool);atlas_chart_id=np.full((_atlas_height,_atlas_width),-1,np.int32);atlas_layout=[]
    _y=_padding
    for row in range(_rows):
     _x=_padding
     for column in range(_columns):
      index=row*_columns+column
      if index>=len(_atlas_charts):break
      chart=_atlas_charts[index];height,width=chart["valid"].shape;atlas_xyz[_y:_y+height,_x:_x+width]=chart["xyz"];atlas_normals[_y:_y+height,_x:_x+width]=chart["normals"];atlas_valid[_y:_y+height,_x:_x+width]=chart["valid"];atlas_chart_id[_y:_y+height,_x:_x+width][chart["valid"]]=chart["candidate_id"];atlas_layout.append({"candidate_id":chart["candidate_id"],"y":_y,"x":_x,"height":height,"width":width});_x+=_column_widths[column]+_padding
     _y+=_row_heights[row]+_padding
    atlas_zarr_path=f"{patch_queue_root}/atlas.zarr"
    _atlas_group=zarr.open_group(atlas_zarr_path,mode="w");_atlas_group.create_dataset("xyz_ct0",data=atlas_xyz,chunks=(256,256,3),compressor=zarr.Blosc(cname="zstd",clevel=5));_atlas_group.create_dataset("normals",data=atlas_normals,chunks=(256,256,3),compressor=zarr.Blosc(cname="zstd",clevel=5));_atlas_group.create_dataset("valid",data=atlas_valid,chunks=(256,256),compressor=zarr.Blosc(cname="zstd",clevel=5));_atlas_group.create_dataset("chart_id",data=atlas_chart_id,chunks=(256,256),compressor=zarr.Blosc(cname="zstd",clevel=5));_atlas_group.attrs["layout"]=atlas_layout
    atlas_pack_summary={"shape":list(atlas_valid.shape),"valid_pixels":int(atlas_valid.sum()),"charts":len(_atlas_charts),"layout":atlas_layout,"path":atlas_zarr_path}
    with open(f"{patch_queue_root}/atlas-layout.json","w") as _layout_file:_atlas_json.dump(atlas_pack_summary,_layout_file,indent=2)
    atlas_pack_summary
    return


@app.cell
def _(ndi, np, plt, skimage_exposure, tifffile, zarr):
    def render_refined_patch_depths(candidate_id,root,sampler,force=False):
     import json as _json
     import os as _os
     directory=f"{root}/candidate-{int(candidate_id):06d}";chart=np.load(f"{directory}/refined-chart.npz");xyz=chart["xyz_ct0"].astype(np.float32);normals=chart["normals"].astype(np.float32);valid=chart["valid"].astype(bool);offsets=np.arange(-40,41,dtype=np.float32);stack_path=f"{directory}/depth-stack.zarr";group=zarr.open_group(stack_path,mode="a");height,width=valid.shape
     if "layers" not in group:group.create_dataset("layers",shape=(len(offsets),height,width),chunks=(1,min(256,height),min(256,width)),dtype="u1",compressor=zarr.Blosc(cname="zstd",clevel=5),fill_value=0)
     completed=set(group.attrs.get("completed_offsets",[]))
     if force:completed=set();group["layers"][:]=0
     safe_xyz=np.where(valid[...,None],xyz,0);safe_normals=np.where(valid[...,None],normals,0)
     for start in range(0,len(offsets),5):
      indices=[index for index in range(start,min(start+5,len(offsets))) if index not in completed]
      if not indices:continue
      batch_offsets=offsets[indices];points=safe_xyz[None]+batch_offsets[:,None,None,None]*safe_normals[None];sampled=sampler.sample(points).astype(np.float32);sampled[:,~valid]=0;group["layers"][indices]=np.clip(np.rint(sampled),0,255).astype(np.uint8);completed.update(indices);group.attrs["completed_offsets"]=sorted(completed);print("depth render",candidate_id,len(completed),"/",len(offsets))
     layers=np.asarray(group["layers"],dtype=np.float32);center=layers[len(offsets)//2];mean=layers.mean(0);maximum=layers.max(0);minimum=layers.min(0);std=layers.std(0);focus_best=np.zeros((height,width),np.float32);focus_score=np.full((height,width),-1,np.float32);focus_index=np.zeros((height,width),np.int16)
     for index,layer in enumerate(layers):
      score=np.abs(layer-ndi.gaussian_filter(layer,1.2));better=score>focus_score;focus_score[better]=score[better];focus_best[better]=layer[better];focus_index[better]=index
     def normalized(image):
      values=image[valid];low,high=np.percentile(values,[1,99]);result=np.clip((image-low)/max(high-low,1e-6),0,1);result[~valid]=0;return result
     previews={"center":normalized(center),"mean":normalized(mean),"max":normalized(maximum),"min":normalized(minimum),"std":normalized(std),"best-focus":normalized(focus_best)};enhanced=skimage_exposure.equalize_adapthist(previews["best-focus"],clip_limit=0.02);enhanced[~valid]=0
     for name,image in previews.items():plt.imsave(f"{directory}/{name}.png",image,cmap="gray",vmin=0,vmax=1)
     plt.imsave(f"{directory}/best-focus-clahe.png",enhanced,cmap="gray",vmin=0,vmax=1);tifffile.imwrite(f"{directory}/best-focus-offset.tif",focus_index);result={"candidate_id":int(candidate_id),"status":"rendered","shape":[height,width],"offsets":offsets.tolist(),"stack":stack_path,"ct_std":float(layers[:,valid].std()),"best_focus_std":float(focus_best[valid].std())}
     with open(f"{directory}/rendering.json","w") as _file:_json.dump(result,_file,indent=2)
     return result


    def render_patch_queue(root,candidate_ids,sampler):
     import json as _json
     results=[]
     for candidate_id in candidate_ids:
      try:result=render_refined_patch_depths(candidate_id,root,sampler);print("rendered patch",candidate_id)
      except Exception as error:result={"candidate_id":int(candidate_id),"status":"failed","error":repr(error)};print("render failed",candidate_id,repr(error))
      results.append(result)
     with open(f"{root}/render-queue.json","w") as _file:_json.dump(results,_file,indent=2)
     return results


    return (render_patch_queue,)


@app.cell
def _(NotebookChunkSampler, ct_group, patch_queue_root, render_patch_queue):
    depth_render_sampler=NotebookChunkSampler(ct_group["0"],max_bytes=8*2**30)
    depth_render_results=render_patch_queue(patch_queue_root,[2517,2638,1943,131,51],depth_render_sampler)
    depth_render_results
    return


if __name__ == "__main__":
    app.run()
