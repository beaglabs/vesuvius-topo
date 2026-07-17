from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trajectory:
    xyz: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    probability: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            xyz=self.xyz.astype(np.float32),
            tangent=self.tangent.astype(np.float32),
            normal=self.normal.astype(np.float32),
            probability=self.probability.astype(np.float32),
        )


@dataclass
class SurfaceGrid:
    xyz: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            xyz=self.xyz.astype(np.float32),
            normals=self.normals.astype(np.float32),
            confidence=self.confidence.astype(np.float32),
            valid=self.valid.astype(bool),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceGrid":
        data = np.load(path)
        return cls(data["xyz"], data["normals"], data["confidence"], data["valid"])
