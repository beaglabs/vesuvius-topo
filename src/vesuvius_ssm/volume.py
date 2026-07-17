from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import fsspec
import numpy as np
import zarr
from scipy.ndimage import map_coordinates


SURFACE_URL = (
    "s3://vesuvius-challenge-open-data/PHerc0332/representations/predictions/surfaces/"
    "20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr"
)
CT_URL = (
    "s3://vesuvius-challenge-open-data/PHerc0332/volumes/"
    "20251211183505-2.399um-0.2m-78keV-masked.zarr"
)


@dataclass
class Volume:
    array: object
    scale_zyx: np.ndarray
    offset_zyx: np.ndarray
    name: str = "volume"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)

    def read(self, bounds_zyx: Sequence[slice]) -> np.ndarray:
        return np.asarray(self.array[tuple(bounds_zyx)])

    def sample_xyz(self, xyz: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
        points = np.asarray(xyz, dtype=np.float32)
        voxel_zyx = points[..., ::-1] / self.scale_zyx - self.offset_zyx
        finite = voxel_zyx[np.isfinite(voxel_zyx).all(-1)]
        if not len(finite):
            return np.full(points.shape[:-1], cval, dtype=np.float32)
        lower = np.maximum(np.floor(finite.min(0)).astype(int) - 2, 0)
        upper = np.minimum(np.ceil(finite.max(0)).astype(int) + 3, np.asarray(self.shape))
        crop = self.read(tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper)))
        flat = (voxel_zyx - lower).reshape(-1, 3).T
        values = map_coordinates(
            crop, flat, order=order, mode="constant", cval=cval, prefilter=False
        )
        return values.reshape(points.shape[:-1])


def open_ome_zarr(url: str, level: int = 0, storage_options: dict | None = None) -> Volume:
    storage_options = storage_options or {}
    mapper = fsspec.get_mapper(url, **storage_options)
    group = zarr.open_group(mapper, mode="r")
    attrs = dict(group.attrs)
    multiscale = attrs.get("multiscales", [{}])[0]
    datasets = multiscale.get("datasets", [])
    path = str(level)
    scale = np.ones(3, dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)
    if datasets:
        dataset = datasets[level]
        path = dataset["path"]
        for transform in dataset.get("coordinateTransformations", []):
            if transform["type"] == "scale":
                scale = np.asarray(transform["scale"], dtype=np.float32)
            elif transform["type"] == "translation":
                offset = np.asarray(transform["translation"], dtype=np.float32) / scale
    # The published PHerc0332 surface prediction level 0 is registered to CT
    # level 2. Keep all learned geometry in CT level-0 voxel coordinates.
    if url.rstrip("/") == SURFACE_URL.rstrip("/"):
        scale *= 4.0
        offset *= 4.0
    return Volume(group[path], scale, offset, url)


def open_array(url: str, level: int = 0) -> Volume:
    if url.endswith(".npy"):
        return Volume(np.load(url, mmap_mode="r"), np.ones(3), np.zeros(3), url)
    return open_ome_zarr(url, level=level, storage_options={"anon": True} if url.startswith("s3://") else {})
