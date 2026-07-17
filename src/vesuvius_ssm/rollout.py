from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .geometry import CropField, normalize
from .model import SurfaceTracker
from .types import SurfaceGrid


def observation(point, previous, tangent, normal, probability, curvature=0.0):
    return np.concatenate(
        (previous, tangent, normal, np.asarray([probability, curvature, 0.0, 1.0], dtype=np.float32))
    ).astype(np.float32)


@torch.no_grad()
def predict_step(model, obs, tangent, normal, state, device):
    tensor = torch.from_numpy(obs)[None, None].to(device)
    output = model(tensor, state)
    local = output["delta_local"][0, 0].float().cpu().numpy()
    side = normalize(np.cross(normal, tangent)[None])[0]
    frame = np.stack((tangent, side, normal), axis=-1)
    delta = frame @ local
    confidence = torch.sigmoid(output["confidence_logit"][0, 0]).item()
    uncertainty = output["log_variance"][0, 0].exp().mean().sqrt().item()
    return delta, confidence, uncertainty, output["state"]


def trace_model(
    model: SurfaceTracker,
    field: CropField,
    seed: np.ndarray,
    initial_tangent: np.ndarray,
    steps: int,
    device: str,
    reverse: bool = False,
):
    points = np.full((steps, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(steps, dtype=np.float32)
    valid = np.zeros(steps, dtype=bool)
    point = seed.astype(np.float32)
    normal = field.normal(point[None])[0]
    tangent = normalize((initial_tangent - initial_tangent.dot(normal) * normal)[None])[0]
    if reverse:
        tangent = -tangent
    previous = tangent * 1.5
    state = None
    last_normal = normal
    for index in range(steps):
        probability = float(field.sample(point[None])[0])
        obs = observation(point, previous, tangent, normal, probability, np.linalg.norm(normal - last_normal))
        delta, predicted_confidence, uncertainty, state = predict_step(model, obs, tangent, normal, state, device)
        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 0.05:
            delta = tangent * 1.5
        if delta.dot(tangent) < 0:
            delta = -delta
        candidate, ridge_probability = field.project(point + delta, normal)
        points[index] = point
        confidence[index] = predicted_confidence * ridge_probability / (1.0 + uncertainty)
        valid[index] = ridge_probability >= 0.1
        if not valid[index]:
            break
        next_normal = field.normal(candidate[None])[0]
        if next_normal.dot(normal) < 0:
            next_normal = -next_normal
        previous = candidate - point
        tangent = normalize((previous - previous.dot(next_normal) * next_normal)[None])[0]
        point, last_normal, normal = candidate, normal, next_normal
    return points, confidence, valid


def compute_normals(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.nan_to_num(xyz)
    du = np.gradient(filled, axis=1)
    dv = np.gradient(filled, axis=0)
    normals = normalize(np.cross(du, dv))
    normals[~valid] = 0
    for row in range(normals.shape[0]):
        for col in range(normals.shape[1]):
            if row and valid[row, col] and valid[row - 1, col] and normals[row, col].dot(normals[row - 1, col]) < 0:
                normals[row, col] *= -1
            elif col and valid[row, col] and valid[row, col - 1] and normals[row, col].dot(normals[row, col - 1]) < 0:
                normals[row, col] *= -1
    return normals.astype(np.float32)


def rollout_surface(
    model: SurfaceTracker,
    field: CropField,
    seed_xyz: np.ndarray,
    u_direction: np.ndarray,
    height: int,
    width: int,
    output: str | Path,
    checkpoint_rows: int = 16,
    device: str | None = None,
) -> SurfaceGrid:
    device = device or next(model.parameters()).device.type
    model.eval()
    v_direction = normalize(np.cross(field.normal(seed_xyz[None])[0], u_direction)[None])[0]
    upper_count = height // 2 + 1
    lower_count = height - upper_count
    upper, upper_conf, upper_valid = trace_model(model, field, seed_xyz, v_direction, upper_count, device)
    lower, lower_conf, lower_valid = trace_model(model, field, seed_xyz, v_direction, lower_count + 1, device, reverse=True)
    centers = np.concatenate((lower[1:][::-1], upper), axis=0)
    center_conf = np.concatenate((lower_conf[1:][::-1], upper_conf))
    center_valid = np.concatenate((lower_valid[1:][::-1], upper_valid))
    xyz = np.full((height, width, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    middle = width // 2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row, center in enumerate(centers[:height]):
        if not center_valid[row]:
            continue
        right, right_conf, right_valid = trace_model(model, field, center, u_direction, width - middle, device)
        left, left_conf, left_valid = trace_model(model, field, center, u_direction, middle + 1, device, reverse=True)
        xyz[row] = np.concatenate((left[1:][::-1], right), axis=0)[:width]
        confidence[row] = np.concatenate((left_conf[1:][::-1], right_conf))[:width] * center_conf[row]
        valid[row] = np.concatenate((left_valid[1:][::-1], right_valid))[:width]
        if checkpoint_rows and (row + 1) % checkpoint_rows == 0:
            SurfaceGrid(xyz, compute_normals(xyz, valid), confidence, valid).save(output.with_suffix(".partial.npz"))
    surface = SurfaceGrid(xyz, compute_normals(xyz, valid), confidence, valid)
    surface.save(output)
    return surface
