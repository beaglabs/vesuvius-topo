import numpy as np

from vesuvius_ssm.geometry import CropField
from vesuvius_ssm.mesh import mesh_first_unwrap


def folded_sheet(size=40):
    z, y, x = np.mgrid[:size, :size, :size]
    surface = 17 + 2.5 * np.sin(x / 8) + 1.5 * np.cos(y / 10)
    probability = np.exp(-0.5 * ((z - surface) / 1.0) ** 2)
    # Enclosed prediction hole that should be interpolated and projected.
    probability[:, 17:22, 17:22] = 0
    return (probability * 255).astype(np.uint8)


def test_mesh_first_fills_enclosed_hole_without_folds():
    field = CropField(folded_sheet(), np.zeros(3), np.ones(3), sigma=0.5)
    surface, anchors, metrics = mesh_first_unwrap(
        field,
        threshold=0.35,
        spacing=1.0,
        optimization_iterations=2,
    )
    assert metrics.coverage >= 0.99
    assert metrics.anchor_fraction < metrics.coverage
    assert metrics.folded_quad_fraction < 0.02
    assert metrics.jump_fraction < 0.02
    assert surface.valid.shape == anchors.shape


def test_mesh_metrics_do_not_count_outside_sheet():
    data = folded_sheet()
    data[:, :8, :] = 0
    field = CropField(data, np.zeros(3), np.ones(3), sigma=0.5)
    surface, _, metrics = mesh_first_unwrap(field, threshold=0.35, optimization_iterations=1)
    assert metrics.coverage >= 0.99
    assert surface.valid.sum() < surface.valid.size
