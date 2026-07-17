from pathlib import Path

import numpy as np
import zarr

from vesuvius_ssm.search import (
    ChunkCacheSampler,
    SearchCandidate,
    SearchManifest,
    generate_surface_candidates,
    non_maximum_suppression,
    score_candidates,
)
from vesuvius_ssm.volume import Volume


def test_chunk_sampler_matches_linear_volume():
    z, y, x = np.mgrid[:18, :20, :22]
    data = (3 * z + 2 * y + x).astype(np.float32)
    array = zarr.array(data, chunks=(6, 7, 8))
    sampler = ChunkCacheSampler(Volume(array, np.ones(3), np.zeros(3)), maximum_bytes=100_000)
    xyz = np.asarray([[4.25, 5.5, 6.75], [15.1, 12.2, 9.3]], dtype=np.float32)
    values = sampler.sample_xyz(xyz)
    expected = 3 * xyz[:, 2] + 2 * xyz[:, 1] + xyz[:, 0]
    np.testing.assert_allclose(values, expected, atol=1e-4)
    assert sampler.sample_xyz(np.asarray([[np.nan, np.nan, np.nan]], np.float32))[0] == 0


def test_generate_score_resume_and_nms(tmp_path: Path):
    z, y, x = np.mgrid[:24, :24, :24]
    probability = np.exp(-0.5 * ((z - 12) / 1.2) ** 2)
    surface = Volume(zarr.array((probability * 255).astype(np.uint8), chunks=(8, 8, 8)), np.ones(3), np.zeros(3))
    candidates = generate_surface_candidates(surface, threshold=0.4, maximum_candidates=20, minimum_distance_voxels=1)
    assert candidates

    support_data = np.zeros((24, 24, 24), dtype=np.uint8)
    support_data[:, :, 8:] = 1
    support = Volume(zarr.array(support_data, chunks=(8, 8, 8)), np.ones(3), np.zeros(3))
    supported = generate_surface_candidates(
        surface,
        threshold=0.4,
        maximum_candidates=20,
        minimum_distance_voxels=1,
        support_volume=support,
    )
    assert supported
    assert all(candidate.xyz[0] >= 8 for candidate in supported)

    ct = (x + y + z).astype(np.float32)
    sampler = ChunkCacheSampler(Volume(zarr.array(ct, chunks=(8, 8, 8)), np.ones(3), np.zeros(3)), maximum_bytes=100_000)
    manifest_path = tmp_path / "manifest.json"
    manifest = score_candidates(candidates[:5], sampler, manifest_path, size=8, spacing=1, depth=4, checkpoint_every=2)
    assert all(item.status == "scored" for item in manifest.candidates)
    loaded = SearchManifest.load(manifest_path)
    assert len(loaded.candidates) == 5
    selected = non_maximum_suppression(loaded.candidates, minimum_distance=3, limit=3)
    assert 1 <= len(selected) <= 3


def test_normal_aware_nms_keeps_crossing_sheets():
    candidates = [
        SearchCandidate(0, [0, 0, 0], [0, 0, 1], 1, combined_score=2, status="scored"),
        SearchCandidate(1, [1, 0, 0], [0, 0, 1], 1, combined_score=1, status="scored"),
        SearchCandidate(2, [1, 0, 0], [0, 1, 0], 1, combined_score=1.5, status="scored"),
    ]
    selected = non_maximum_suppression(candidates, minimum_distance=4, minimum_normal_similarity=0.8)
    assert [item.candidate_id for item in selected] == [0, 2]
