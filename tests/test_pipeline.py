from pathlib import Path

import numpy as np

from vesuvius_ssm.dataset import trajectory_observations
from vesuvius_ssm.geometry import CropField, generate_trajectories
from vesuvius_ssm.render import export_render
from vesuvius_ssm.types import SurfaceGrid
from vesuvius_ssm.volume import Volume


def plane_volume(size=32):
    z, y, x = np.mgrid[:size, :size, :size]
    return (255 * np.exp(-0.5 * ((z - (12 + 0.1 * x)) / 1.25) ** 2)).astype(np.uint8)


def test_trajectory_generation_has_model_contract():
    trajectories = generate_trajectories(
        plane_volume(), np.zeros(3), np.ones(3), count=4, length=20, seed_threshold=0.7, rng=np.random.default_rng(3)
    )
    observation, delta = trajectory_observations(trajectories[0])
    assert observation.shape[1] == 13
    assert delta.shape[1] == 3
    assert len(trajectories[0].xyz) >= 8


def test_surface_render_exports(tmp_path: Path):
    data = plane_volume().astype(np.float32)
    height, width = 8, 9
    yy, xx = np.mgrid[:height, :width]
    xyz = np.stack((xx + 8, yy + 8, 12 + 0.1 * (xx + 8)), axis=-1).astype(np.float32)
    normals = np.zeros_like(xyz)
    normals[..., 2] = 1
    surface = SurfaceGrid(xyz, normals, np.ones((height, width), np.float32), np.ones((height, width), bool))
    outputs = export_render(
        Volume(data, np.ones(3), np.zeros(3)), surface, tmp_path, np.arange(-2, 3, dtype=np.float32)
    )
    assert outputs["layers"].exists()
    assert (tmp_path / "unwrapped_ct.png").exists()
    assert (tmp_path / "surface.ply").exists()


def test_crop_projection_finds_plane():
    field = CropField(plane_volume(), np.zeros(3), np.ones(3))
    point, probability = field.project(np.array([16, 16, 8], np.float32), np.array([0, 0, 1], np.float32), radius=8)
    assert probability > 0.5
    assert abs(point[2] - 13.6) < 2
