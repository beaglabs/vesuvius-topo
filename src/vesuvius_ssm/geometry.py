from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .types import Trajectory


def normalize(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), eps)


class CropField:
    """Continuous scalar field backed by a manageable in-memory z/y/x crop."""

    def __init__(self, data: np.ndarray, origin_zyx: np.ndarray, scale_zyx: np.ndarray, sigma: float = 1.0):
        self.data = gaussian_filter(data.astype(np.float32), sigma=sigma)
        if np.issubdtype(data.dtype, np.integer):
            self.data /= np.iinfo(data.dtype).max
        elif self.data.max() > 1:
            self.data /= self.data.max()
        self.origin_zyx = np.asarray(origin_zyx, dtype=np.float32)
        self.scale_zyx = np.asarray(scale_zyx, dtype=np.float32)
        gradients = np.gradient(self.data, *self.scale_zyx)
        self.grad_xyz = np.stack(gradients[::-1])

    def local_zyx(self, xyz: np.ndarray) -> np.ndarray:
        return np.asarray(xyz)[..., ::-1] / self.scale_zyx - self.origin_zyx

    def sample(self, xyz: np.ndarray) -> np.ndarray:
        coords = self.local_zyx(xyz).reshape(-1, 3).T
        return map_coordinates(self.data, coords, order=1, mode="constant", cval=0, prefilter=False).reshape(
            np.asarray(xyz).shape[:-1]
        )

    def normal(self, xyz: np.ndarray) -> np.ndarray:
        coords = self.local_zyx(xyz).reshape(-1, 3).T
        grad = np.stack(
            [map_coordinates(g, coords, order=1, mode="nearest", prefilter=False) for g in self.grad_xyz], axis=-1
        )
        return normalize(grad).reshape(np.asarray(xyz).shape)

    def project(self, xyz: np.ndarray, normal: np.ndarray, radius: float = 3.0, samples: int = 9):
        offsets = np.linspace(-radius, radius, samples, dtype=np.float32)
        candidates = xyz[None] + offsets[:, None] * normal[None]
        scores = self.sample(candidates)
        index = int(np.argmax(scores))
        return candidates[index], float(scores[index])


def trace_trajectory(
    field: CropField,
    seed_xyz: np.ndarray,
    tangent: np.ndarray,
    length: int = 64,
    step: float = 1.5,
    threshold: float = 0.2,
) -> Trajectory:
    xyz = np.zeros((length, 3), dtype=np.float32)
    tangents = np.zeros_like(xyz)
    normals = np.zeros_like(xyz)
    probabilities = np.zeros(length, dtype=np.float32)
    xyz[0] = seed_xyz
    normal = field.normal(seed_xyz[None])[0]
    tangent = normalize((tangent - tangent.dot(normal) * normal)[None])[0]
    valid_length = 1
    for i in range(length):
        if i:
            predicted = xyz[i - 1] + step * tangent
            point, probability = field.project(predicted, normal)
            if probability < threshold:
                break
            next_normal = field.normal(point[None])[0]
            if next_normal.dot(normal) < 0:
                next_normal = -next_normal
            tangent = normalize((tangent - tangent.dot(next_normal) * next_normal)[None])[0]
            xyz[i] = point
            normal = next_normal
            valid_length = i + 1
        probabilities[i] = field.sample(xyz[i][None])[0]
        tangents[i] = tangent
        normals[i] = normal
    return Trajectory(
        xyz[:valid_length], tangents[:valid_length], normals[:valid_length], probabilities[:valid_length]
    )


def generate_trajectories(
    data: np.ndarray,
    origin_zyx: np.ndarray,
    scale_zyx: np.ndarray,
    count: int,
    length: int,
    seed_threshold: float,
    rng: np.random.Generator,
) -> list[Trajectory]:
    field = CropField(data, origin_zyx, scale_zyx)
    seeds = np.argwhere(field.data >= seed_threshold)
    if not len(seeds):
        raise ValueError(f"crop contains no voxels above seed threshold {seed_threshold}")
    trajectories = []
    for index in rng.choice(len(seeds), size=min(count * 3, len(seeds)), replace=False):
        seed_local_zyx = seeds[index].astype(np.float32)
        seed_xyz = ((seed_local_zyx + origin_zyx) * scale_zyx)[::-1]
        direction = normalize(rng.normal(size=(1, 3)).astype(np.float32))[0]
        trajectory = trace_trajectory(field, seed_xyz, direction, length=length)
        if len(trajectory.xyz) >= max(8, length // 3):
            trajectories.append(trajectory)
        if len(trajectories) == count:
            break
    if not trajectories:
        raise ValueError("no usable trajectories could be traced in this crop")
    return trajectories
