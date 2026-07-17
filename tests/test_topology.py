import numpy as np

from vesuvius_ssm.geometry import CropField
from vesuvius_ssm.topology import topology_unwrap


def probability_sheet(size=36, second_layer=False):
    z, y, x = np.mgrid[:size, :size, :size]
    first = np.exp(-0.5 * ((z - (16 + 1.5 * np.sin(x / 9))) / 1.2) ** 2)
    if second_layer:
        second = np.exp(-0.5 * ((z - (23 + 1.0 * np.cos(y / 8))) / 1.2) ** 2)
        first = np.maximum(first, second)
    return (first * 255).astype(np.uint8)


def test_topology_unwrap_produces_unflipped_disk():
    field = CropField(probability_sheet(), np.zeros(3), np.ones(3), sigma=0.6)
    surface, mesh, metrics = topology_unwrap(
        field,
        seed_xyz=np.asarray([18, 18, 16], dtype=np.float32),
        patch_radius=11,
        slim_iterations=2,
    )
    assert metrics.euler_characteristic == 1
    assert metrics.boundary_loops == 1
    assert metrics.flipped_fraction == 0
    assert metrics.collapsed_fraction == 0
    assert metrics.probability_adherence > 0.95
    assert surface.valid.any()
    assert mesh.uv.shape == (len(mesh.vertices), 2)


def test_seed_selects_near_layer_from_parallel_sheets():
    field = CropField(probability_sheet(second_layer=True), np.zeros(3), np.ones(3), sigma=0.6)
    surface, _, metrics = topology_unwrap(
        field,
        seed_xyz=np.asarray([18, 18, 16], dtype=np.float32),
        patch_radius=9,
        slim_iterations=0,
    )
    median_z = float(np.nanmedian(surface.xyz[..., 2]))
    assert median_z < 20
    assert metrics.flipped_fraction == 0
