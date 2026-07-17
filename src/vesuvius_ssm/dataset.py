from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import Trajectory


def trajectory_observations(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    xyz = trajectory.xyz.astype(np.float32)
    tangent = trajectory.tangent.astype(np.float32)
    normal = trajectory.normal.astype(np.float32)
    previous = np.zeros_like(xyz)
    previous[1:] = xyz[1:] - xyz[:-1]
    curvature = np.zeros(len(xyz), dtype=np.float32)
    if len(xyz) > 1:
        curvature[1:] = np.linalg.norm(normal[1:] - normal[:-1], axis=-1)
    observations = np.concatenate(
        (
            previous,
            tangent,
            normal,
            trajectory.probability[:, None],
            curvature[:, None],
            np.zeros((len(xyz), 1), dtype=np.float32),
            np.ones((len(xyz), 1), dtype=np.float32),
        ),
        axis=-1,
    )
    frame = np.stack(
        (tangent, np.cross(normal, tangent), normal), axis=-1
    )
    world_delta = np.zeros_like(xyz)
    world_delta[:-1] = xyz[1:] - xyz[:-1]
    delta_local = np.einsum("tji,tj->ti", frame, world_delta)
    return observations, delta_local


class TrajectoryDataset(Dataset):
    def __init__(self, paths: list[str | Path], sequence_length: int = 32):
        self.paths = [Path(path) for path in paths]
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        data = np.load(self.paths[index])
        trajectory = Trajectory(data["xyz"], data["tangent"], data["normal"], data["probability"])
        obs, delta = trajectory_observations(trajectory)
        length = min(len(obs), self.sequence_length)
        padded_obs = np.zeros((self.sequence_length, obs.shape[-1]), dtype=np.float32)
        padded_delta = np.zeros((self.sequence_length, 3), dtype=np.float32)
        mask = np.zeros(self.sequence_length, dtype=np.float32)
        confidence = np.zeros(self.sequence_length, dtype=np.float32)
        padded_obs[:length] = obs[:length]
        padded_delta[:length] = delta[:length]
        mask[: max(0, length - 1)] = 1
        confidence[:length] = trajectory.probability[:length]
        return {
            "observation": torch.from_numpy(padded_obs),
            "delta_local": torch.from_numpy(padded_delta),
            "mask": torch.from_numpy(mask),
            "confidence": torch.from_numpy(confidence),
        }
