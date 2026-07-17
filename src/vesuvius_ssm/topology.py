from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
import trimesh
from scipy import ndimage, sparse
from scipy.interpolate import griddata as scipy_griddata
from scipy.sparse.csgraph import dijkstra
from skimage.measure import marching_cubes

from .geometry import CropField, normalize
from .types import SurfaceGrid


@dataclass(frozen=True)
class TopologyMetrics:
    vertices: int
    faces: int
    boundary_loops: int
    euler_characteristic: int
    flipped_fraction: float
    collapsed_fraction: float
    overlap_fraction: float
    raster_coverage: float
    probability_adherence: float


@dataclass
class ParameterizedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    boundary: np.ndarray


def _sample_local(array: np.ndarray, local_zyx: np.ndarray, cval: float = 0.0) -> np.ndarray:
    coordinates = np.asarray(local_zyx, dtype=np.float32).reshape(-1, 3).T
    return ndimage.map_coordinates(
        array, coordinates, order=1, mode="constant", cval=cval, prefilter=False
    ).reshape(np.asarray(local_zyx).shape[:-1])


def ridge_features(
    field: CropField,
    seed_xyz: np.ndarray,
    probability_floor: float = 0.15,
    anisotropy_floor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = field.data.astype(np.float32)
    spacing = field.scale_zyx.astype(np.float32)
    gradient = np.stack(np.gradient(probability, *spacing), axis=-1)
    hessian = np.empty(probability.shape + (3, 3), dtype=np.float32)
    derivatives = [np.gradient(gradient[..., axis], *spacing) for axis in range(3)]
    for row in range(3):
        for col in range(3):
            hessian[..., row, col] = 0.5 * (derivatives[row][col] + derivatives[col][row])
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    normal_zyx = eigenvectors[..., :, 0]
    normal_curvature = np.maximum(-eigenvalues[..., 0], 0)
    anisotropy = normal_curvature / (
        np.abs(eigenvalues[..., 1]) + np.abs(eigenvalues[..., 2]) + normal_curvature + 1e-6
    )
    seed_local = np.rint(field.local_zyx(seed_xyz)).astype(int)
    seed_local = np.clip(seed_local, 0, np.asarray(probability.shape) - 1)
    radius = 4
    lower = np.maximum(seed_local - radius, 0)
    upper = np.minimum(seed_local + radius + 1, np.asarray(probability.shape))
    neighborhood = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    local_confidence = probability[neighborhood] * anisotropy[neighborhood]
    best_local = np.asarray(np.unravel_index(int(np.argmax(local_confidence)), local_confidence.shape))
    best = lower + best_local
    reference = normal_zyx[tuple(best)]
    orientation = np.sign(np.einsum("...i,i->...", normal_zyx, reference))
    orientation[orientation == 0] = 1
    normal_zyx *= orientation[..., None]
    ridge = np.einsum("...i,...i->...", gradient, normal_zyx)
    curvature_values = normal_curvature[normal_curvature > 0]
    curvature_floor = float(np.percentile(curvature_values, 15)) if len(curvature_values) else 0.0
    candidate = (
        (probability >= probability_floor)
        & (anisotropy >= anisotropy_floor)
        & (normal_curvature >= curvature_floor)
    )
    candidate = ndimage.binary_dilation(candidate, iterations=1)
    confidence = probability * anisotropy * normal_curvature
    return ridge.astype(np.float32), normal_zyx.astype(np.float32), candidate, confidence


def extract_ridge_mesh(
    field: CropField,
    seed_xyz: np.ndarray,
    probability_floor: float = 0.15,
    anisotropy_floor: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ridge, normal_zyx, candidate, confidence = ridge_features(
        field, seed_xyz, probability_floor, anisotropy_floor
    )
    if not (float(ridge[candidate].min()) <= 0 <= float(ridge[candidate].max())):
        raise ValueError("candidate region does not contain a probability ridge zero crossing")
    vertices_zyx, faces, _, _ = marching_cubes(
        ridge,
        level=0.0,
        spacing=tuple(field.scale_zyx.tolist()),
        mask=candidate,
        allow_degenerate=False,
    )
    local_zyx = vertices_zyx / field.scale_zyx
    scores = _sample_local(confidence, local_zyx)
    probability = _sample_local(field.data, local_zyx)
    sampled_normals = np.stack(
        [_sample_local(normal_zyx[..., axis], local_zyx) for axis in range(3)], axis=-1
    )
    sampled_normals = normalize(sampled_normals)
    valid_vertex = probability >= probability_floor
    keep_face = valid_vertex[faces].all(axis=1)
    faces = faces[keep_face]
    vertices_xyz = (vertices_zyx + field.origin_zyx * field.scale_zyx)[:, ::-1]
    normals_xyz = sampled_normals[:, ::-1]
    vertices_xyz, faces, normals_xyz, scores = _remove_unreferenced(
        vertices_xyz, faces, normals_xyz, scores
    )
    return vertices_xyz, faces, normals_xyz


def _remove_unreferenced(vertices, faces, *attributes):
    if not len(faces):
        raise ValueError("mesh contains no faces")
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    new_faces = inverse.reshape(-1, 3).astype(np.int64)
    return (vertices[used], new_faces, *(attribute[used] for attribute in attributes))


def prune_mesh_bridges(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    maximum_edge_factor: float = 2.5,
    maximum_normal_angle: float = 50.0,
    maximum_normal_step: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    vectors = vertices[edges[..., 1]] - vertices[edges[..., 0]]
    lengths = np.linalg.norm(vectors, axis=-1)
    positive = lengths[lengths > 1e-6]
    nominal = float(np.median(positive)) if len(positive) else 0.0
    directions = vectors / np.maximum(lengths[..., None], 1e-6)
    normal_a = normals[edges[..., 0]]
    normal_b = normals[edges[..., 1]]
    agreement = np.abs(np.einsum("...i,...i->...", normal_a, normal_b))
    mean_normal = normalize(normal_a + np.sign(np.einsum("...i,...i->...", normal_a, normal_b))[..., None] * normal_b)
    normal_step = np.abs(np.einsum("...i,...i->...", directions, mean_normal))
    keep = (
        (lengths <= maximum_edge_factor * nominal)
        & (agreement >= np.cos(np.deg2rad(maximum_normal_angle)))
        & (normal_step <= maximum_normal_step)
    ).all(axis=1)
    vertices, faces, normals = _remove_unreferenced(vertices, faces[keep], normals)
    return vertices, faces, normals


def remove_nonmanifold_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove faces that reference non-manifold edges (shared by >2 faces)."""
    for _ in range(iterations):
        edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
        sorted_edges = np.sort(edges.reshape(-1, 2), axis=1)
        _, inverse, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
        keep = (counts[inverse].reshape(-1, 3) <= 2).all(axis=1)
        if keep.all():
            break
        faces = faces[keep]
        vertices, faces, normals = _remove_unreferenced(vertices, faces, normals)
    return vertices, faces, normals


def select_seed_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    seed_xyz: np.ndarray,
    maximum_seed_distance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("ridge mesh has no connected components")
    # Pick the largest component (by face count) among those within
    # maximum_seed_distance of the seed.  The notebook uses this strategy
    # because the closest component is often a tiny degenerate fragment.
    component_distances = [
        float(np.linalg.norm(item.vertices - seed_xyz, axis=1).min())
        for item in components
    ]
    eligible = [
        (item, dist)
        for item, dist in zip(components, component_distances)
        if dist <= maximum_seed_distance
    ]
    if eligible:
        component = max(eligible, key=lambda pair: len(pair[0].faces))[0]
    else:
        # Fallback: closest component (original behaviour)
        component = min(components, key=lambda item: component_distances[components.index(item)])
    original_index = np.asarray(component.metadata.get("vertex_index", []))
    if len(original_index) != len(component.vertices):
        # Match marching-cubes vertices exactly; trimesh split preserves their coordinates.
        lookup = {tuple(value): index for index, value in enumerate(vertices)}
        original_index = np.asarray([lookup[tuple(value)] for value in component.vertices])
    selected_normals = normals[original_index]
    return (
        np.asarray(component.vertices, dtype=np.float64),
        np.asarray(component.faces, dtype=np.int64),
        selected_normals,
        int(np.argmin(np.linalg.norm(component.vertices - seed_xyz, axis=1))),
    )


def _build_geodesic_graph(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a weighted edge graph for geodesic distance computation."""
    edge_pairs = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    weights = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
    graph = sparse.coo_matrix(
        (np.r_[weights, weights], (np.r_[edge_pairs[:, 0], edge_pairs[:, 1]], np.r_[edge_pairs[:, 1], edge_pairs[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    return graph, edge_pairs


def _extract_geodesic_patch(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    distances: np.ndarray,
    radius: float,
    minimum_vertices: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract a disk-like patch at a given geodesic radius.

    Tries fill-holes-first (accept multiple inner boundaries, fill them,
    require 1 outer loop + χ=1), then falls back to requiring exactly 1
    boundary loop + χ=1 outright.  Returns None if no valid patch is found.
    """
    inside = distances <= radius
    selected_faces = faces[inside[faces].all(axis=1)]
    if not len(selected_faces):
        return None
    patch_vertices, patch_faces, patch_normals = _remove_unreferenced(
        vertices, selected_faces, normals
    )
    if len(patch_vertices) < minimum_vertices:
        return None
    if manifold_edge_violations(patch_faces) != 0:
        return None
    loops = boundary_loops(patch_faces)
    if not loops:
        return None
    # Strategy 1: already a single disk
    chi = euler_characteristic(len(patch_vertices), patch_faces)
    if len(loops) == 1 and chi == 1:
        return patch_vertices, patch_faces, patch_normals, loops[0]
    # Strategy 2: fill inner boundary loops (notebook's topo_geodesic_fill_holes)
    if len(loops) > 1:
        filled = fill_inner_boundary_loops(patch_vertices, patch_faces, loops)
        if filled is not None:
            filled_vertices, filled_faces = filled
            filled_loops = boundary_loops(filled_faces)
            if len(filled_loops) == 1 and euler_characteristic(len(filled_vertices), filled_faces) == 1:
                added = len(filled_vertices) - len(patch_vertices)
                if added:
                    patch_normals = np.vstack(
                        (patch_normals, np.repeat(patch_normals.mean(axis=0)[None], added, axis=0))
                    )
                return filled_vertices, filled_faces, patch_normals, filled_loops[0]
    return None


def geodesic_disk(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    seed_vertex: int,
    radius: float,
    retries: int = 7,
    minimum_vertices: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph, _ = _build_geodesic_graph(vertices, faces)
    distances = dijkstra(graph, indices=seed_vertex)
    current_radius = radius
    for _ in range(retries):
        result = _extract_geodesic_patch(
            vertices, faces, normals, distances, current_radius, minimum_vertices
        )
        if result is not None:
            return result
        current_radius *= 0.8
    raise ValueError("could not extract a manifold disk around the seed; reduce radius or enlarge crop")


def fill_inner_boundary_loops(
    vertices: np.ndarray, faces: np.ndarray, loops: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray] | None:
    perimeters = [
        np.linalg.norm(np.roll(vertices[loop], -1, axis=0) - vertices[loop], axis=1).sum()
        for loop in loops
    ]
    outer = int(np.argmax(perimeters))
    result_vertices = vertices.copy()
    result_faces = faces.copy()
    face_normal = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ).mean(axis=0)
    for index, loop in enumerate(loops):
        if index == outer:
            continue
        points = result_vertices[loop]
        center = points.mean(axis=0)
        # Do not fan-fill a large boundary that is likely a real tear or a second outer boundary.
        if perimeters[index] > 0.35 * perimeters[outer]:
            return None
        center_index = len(result_vertices)
        result_vertices = np.vstack((result_vertices, center))
        new_faces = []
        for left, right in zip(loop, np.roll(loop, -1)):
            face = np.asarray([left, right, center_index], dtype=np.int64)
            normal = np.cross(result_vertices[right] - result_vertices[left], center - result_vertices[left])
            if np.dot(normal, face_normal) < 0:
                face[[0, 1]] = face[[1, 0]]
            new_faces.append(face)
        result_faces = np.vstack((result_faces, np.asarray(new_faces)))
    return result_vertices, result_faces


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    sorted_edges = np.sort(edges, axis=1)
    unique, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if not len(boundary):
        return []
    adjacency: dict[int, list[int]] = {}
    for left, right in boundary:
        adjacency.setdefault(int(left), []).append(int(right))
        adjacency.setdefault(int(right), []).append(int(left))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return []
    loops = []
    unused = {tuple(edge) for edge in map(tuple, boundary)}
    while unused:
        start, following = next(iter(unused))
        loop = [start]
        previous, current = start, following
        while current != start:
            loop.append(current)
            edge = tuple(sorted((previous, current)))
            unused.discard(edge)
            candidates = [item for item in adjacency[current] if item != previous]
            if not candidates:
                return []
            previous, current = current, candidates[0]
            if len(loop) > len(adjacency) + 1:
                return []
        unused.discard(tuple(sorted((previous, current))))
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def manifold_edge_violations(faces: np.ndarray) -> int:
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts > 2).sum())


def euler_characteristic(vertex_count: int, faces: np.ndarray) -> int:
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edge_count = len(np.unique(edges, axis=0))
    return int(vertex_count - edge_count + len(faces))


def signed_uv_area(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def parameterize_disk(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary: np.ndarray,
    slim_iterations: int = 10,
) -> np.ndarray:
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces64 = np.ascontiguousarray(faces, dtype=np.int64)
    boundary = np.ascontiguousarray(boundary, dtype=np.int64)
    circle = igl.map_vertices_to_circle(vertices, boundary)
    uv = np.asarray(igl.harmonic(vertices, faces64, boundary, circle, 1))
    area = signed_uv_area(uv, faces64)
    if np.median(area) < 0:
        uv[:, 0] *= -1
        area = signed_uv_area(uv, faces64)
    if (area <= 1e-12).any():
        pins = np.asarray([boundary[0], boundary[len(boundary) // 2]], dtype=np.int64)
        pin_uv = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        uv = np.asarray(igl.lscm(vertices, faces64, pins, pin_uv)[0])
        area = signed_uv_area(uv, faces64)
        if np.median(area) < 0:
            uv[:, 0] *= -1
            area = signed_uv_area(uv, faces64)
    if (area <= 1e-12).any():
        uv = tutte_parameterization(vertices, faces64, boundary, circle)
        area = signed_uv_area(uv, faces64)
        if np.median(area) < 0:
            uv[:, 0] *= -1
            area = signed_uv_area(uv, faces64)
    if (area <= 1e-12).any():
        raise ValueError("harmonic, LSCM, and Tutte initializations contain flipped triangles")
    if slim_iterations and hasattr(igl, "slim_precompute"):
        data = igl.slim_precompute(
            np.asfortranarray(vertices),
            np.asfortranarray(faces.astype(np.int32)),
            np.asfortranarray(uv),
            igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
            boundary.astype(np.int32),
            np.asfortranarray(uv[boundary]),
            1e5,
        )
        candidate = np.asarray(igl.slim_solve(data, slim_iterations))[:, :2]
        candidate_area = signed_uv_area(candidate, faces64)
        if np.median(candidate_area) < 0:
            candidate[:, 0] *= -1
            candidate_area = signed_uv_area(candidate, faces64)
        if (candidate_area > 1e-12).all():
            uv = candidate
    minimum, maximum = uv.min(axis=0), uv.max(axis=0)
    return ((uv - minimum) / np.maximum(maximum - minimum, 1e-9)).astype(np.float64)


def tutte_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary: np.ndarray,
    boundary_uv: np.ndarray,
) -> np.ndarray:
    edges = np.unique(
        np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1),
        axis=0,
    )
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    interior = np.setdiff1d(np.arange(len(vertices)), boundary)
    uv = np.zeros((len(vertices), 2), dtype=np.float64)
    uv[boundary] = boundary_uv
    uv[interior] = sparse.linalg.spsolve(
        laplacian[interior][:, interior],
        -laplacian[interior][:, boundary] @ boundary_uv,
    )
    return uv


def rasterize_parameterized_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    area3d = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1
    ).sum()
    area2d = np.abs(signed_uv_area(uv, faces)).sum()
    scale = np.sqrt(area3d / max(area2d, 1e-12)) / spacing
    extent = np.maximum(uv.max(axis=0) - uv.min(axis=0), 1e-9)
    width, height = np.maximum(np.ceil(extent * scale).astype(int) + 1, 2)
    if width * height > 16_000_000:
        raise ValueError("parameterized raster is too large; increase spacing")
    pixel = (uv - uv.min(axis=0)) * scale
    xyz = np.full((height, width, 3), np.nan, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    epsilon = 1e-7
    for face in faces:
        triangle = pixel[face]
        lower = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(upper < lower):
            continue
        xs, ys = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        points = np.stack((xs + 0.5, ys + 0.5), axis=-1)
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (points[..., 0] - c[0]) + (c[0] - b[0]) * (points[..., 1] - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (points[..., 0] - c[0]) + (a[0] - c[0]) * (points[..., 1] - c[1])) / denominator
        w2 = 1 - w0 - w1
        inside = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
        if not inside.any():
            continue
        interpolated = w0[..., None] * vertices[face[0]] + w1[..., None] * vertices[face[1]] + w2[..., None] * vertices[face[2]]
        target_y, target_x = ys[inside], xs[inside]
        empty = coverage[target_y, target_x] == 0
        xyz[target_y[empty], target_x[empty]] = interpolated[inside][empty]
        coverage[target_y, target_x] += 1
    mask = coverage > 0
    return xyz, mask, coverage


# ── Grid-mesh unwrapping (notebook approach) ──────────────────────────
# Builds a dense grid mesh from the probability field using PCA projection,
# weighted averaging, hole filling, and iterative optimization.  Produces
# meshes that rasterize reliably because the UV comes from grid coordinates.


def _sample_field_at(field: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Sample a 3D field at arbitrary XYZ positions (local voxel coords)."""
    local = xyz[..., ::-1].reshape(-1, 3).T
    return ndimage.map_coordinates(
        field, local, order=1, mode="constant", cval=0, prefilter=False
    ).reshape(xyz.shape[:-1])


def _project_ridge_point(
    field: np.ndarray, xyz: np.ndarray, normal: np.ndarray, radius: float = 3.0, samples: int = 9
) -> tuple[np.ndarray, float]:
    """Project a point along its normal to the nearest probability ridge."""
    offsets = np.linspace(-radius, radius, samples, dtype=np.float32)
    candidates = xyz[None] + offsets[:, None] * normal[None]
    scores = _sample_field_at(field, candidates)
    idx = int(np.argmax(scores))
    return candidates[idx], float(scores[idx])


def _grid_mesh_component(
    field: np.ndarray, seed_zyx: np.ndarray, threshold: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the largest connected component of voxels above threshold."""
    mask = field >= threshold
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    component_id = int(labels[tuple(seed_zyx)])
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if component_id == 0 or sizes[component_id] < 256:
        component_id = int(np.argmax(sizes))
    component = labels == component_id
    return (
        np.argwhere(component)[:, ::-1].astype(np.float32),
        field[component].astype(np.float32),
    )


def _grid_mesh_parameterize(points_xyz: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D via PCA (first two principal components)."""
    center = points_xyz.mean(0)
    centered = points_xyz - center
    _, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    return (centered @ vectors[:, ::-1][:, :2]).astype(np.float32)


def _grid_mesh_rasterize(
    points_xyz: np.ndarray, scores: np.ndarray, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize 3D points into a 2D grid using PCA projection."""
    uv = _grid_mesh_parameterize(points_xyz)
    indices = np.rint((uv - uv.min(0)) / spacing).astype(int)
    shape = tuple((indices.max(0) + 1).tolist())
    xyz_sum = np.zeros((*shape, 3), dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    best_score = np.zeros(shape, dtype=np.float32)
    best_xyz = np.zeros((*shape, 3), dtype=np.float32)
    for point, score, index in zip(points_xyz, scores, indices):
        u, v = int(index[0]), int(index[1])
        w = max(float(score), 1e-3) ** 2
        xyz_sum[u, v] += point * w
        weights[u, v] += w
        if score >= best_score[u, v]:
            best_score[u, v] = score
            best_xyz[u, v] = point
    anchors = weights > 0
    xyz = np.full((*shape, 3), np.nan, dtype=np.float32)
    xyz[anchors] = (xyz_sum[anchors] / weights[anchors, None]).astype(np.float32)
    xyz[anchors] = 0.5 * xyz[anchors] + 0.5 * best_xyz[anchors]
    sheet_mask = ndimage.binary_closing(
        anchors, structure=ndimage.generate_binary_structure(2, 2), iterations=2
    )
    sheet_mask = ndimage.binary_fill_holes(sheet_mask)
    mask_labels, mask_count = ndimage.label(sheet_mask)
    if mask_count > 1:
        mask_sizes = np.bincount(mask_labels.ravel())
        mask_sizes[0] = 0
        sheet_mask = mask_labels == int(np.argmax(mask_sizes))
    anchors &= sheet_mask
    return xyz, anchors, sheet_mask, best_score


def _grid_mesh_fill_holes(
    xyz: np.ndarray, anchors: np.ndarray, sheet_mask: np.ndarray
) -> np.ndarray:
    """Fill missing grid cells via linear/nearest interpolation."""
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
            interpolated[unresolved] = scipy_griddata(
                known_uv, values, missing_uv[unresolved], method="nearest"
            )
        result[..., axis][missing] = interpolated
    return result


def _grid_mesh_optimize(
    field: np.ndarray,
    xyz: np.ndarray,
    anchors: np.ndarray,
    sheet_mask: np.ndarray,
    iterations: int = 4,
) -> np.ndarray:
    """Iteratively project grid vertices onto the probability ridge."""
    result = xyz.copy()
    anchor_values = xyz.copy()
    movable = sheet_mask & ~anchors
    kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for _ in range(iterations):
        count = ndimage.convolve(sheet_mask.astype(np.float32), kernel, mode="constant")
        neighbor = np.stack(
            [
                ndimage.convolve(np.nan_to_num(result[..., axis]), kernel, mode="constant")
                for axis in range(3)
            ],
            axis=-1,
        ) / np.maximum(count[..., None], 1)
        result[movable] = 0.75 * result[movable] + 0.25 * neighbor[movable]
        du = np.gradient(np.nan_to_num(result), axis=1)
        dv = np.gradient(np.nan_to_num(result), axis=0)
        normals = np.cross(du, dv)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)
        for row, col in np.argwhere(movable):
            projected, probability = _project_ridge_point(
                field, result[row, col], normals[row, col], radius=3.0
            )
            if probability >= 0.1:
                result[row, col] = projected
        result[anchors] = anchor_values[anchors]
    result[~sheet_mask] = np.nan
    return result


def _grid_mesh_build(
    xyz: np.ndarray, valid: np.ndarray, seed_xyz: np.ndarray, edge_factor: float = 5.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert a valid grid into a triangle mesh, keeping the seed's component."""
    indices = -np.ones(valid.shape, dtype=np.int64)
    indices[valid] = np.arange(valid.sum())
    vertices = xyz[valid]
    faces_list: list[tuple[int, int, int]] = []
    for row in range(valid.shape[0] - 1):
        for col in range(valid.shape[1] - 1):
            quad = indices[row : row + 2, col : col + 2]
            if (quad >= 0).all():
                faces_list.append((int(quad[0, 0]), int(quad[0, 1]), int(quad[1, 0])))
                faces_list.append((int(quad[0, 1]), int(quad[1, 1]), int(quad[1, 0])))
    faces = np.asarray(faces_list, dtype=np.int64)
    if not len(faces):
        return vertices, faces, 0
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    lengths = np.linalg.norm(vertices[edges[..., 1]] - vertices[edges[..., 0]], axis=-1)
    median = float(np.median(lengths[lengths > 0]))
    keep = (lengths < edge_factor * median).all(axis=1)
    used, inverse = np.unique(faces[keep].ravel(), return_inverse=True)
    vertices = vertices[used]
    faces = inverse.reshape(-1, 3).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError("grid mesh has no connected components")
    component = min(
        components, key=lambda item: np.linalg.norm(item.vertices - seed_xyz, axis=1).min()
    )
    vertices = np.asarray(component.vertices, dtype=np.float64)
    faces = np.asarray(component.faces, dtype=np.int64)
    seed_vertex = int(np.argmin(np.linalg.norm(vertices - seed_xyz, axis=1)))
    return vertices, faces, seed_vertex


def _grid_mesh_rasterize_to_pixels(
    vertices: np.ndarray,
    faces: np.ndarray,
    xyz_grid: np.ndarray,
    valid: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a grid mesh using its natural grid UV coordinates."""
    # Map each vertex to its grid position
    grid_indices = np.argwhere(valid)
    flat_xyz = xyz_grid[valid]
    grid_uv = np.zeros((len(vertices), 2), dtype=np.float64)
    for i, v in enumerate(vertices):
        idx = np.argmin(np.linalg.norm(flat_xyz - v, axis=1))
        grid_uv[i] = grid_indices[idx].astype(np.float64)
    # Normalize UV to [0, 1]
    uv_min = grid_uv.min(axis=0)
    uv_range = np.maximum(grid_uv.max(axis=0) - uv_min, 1e-9)
    grid_uv = (grid_uv - uv_min) / uv_range
    # Compute pixel scale from 3D/UV area ratio
    area3d = 0.5 * np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    ).sum()
    area2d = np.abs(signed_uv_area(grid_uv, faces)).sum()
    scale = np.sqrt(area3d / max(area2d, 1e-12)) / spacing
    extent = np.maximum(grid_uv.max(axis=0) - grid_uv.min(axis=0), 1e-9)
    width = int(max(np.ceil(extent[0] * scale) + 1, 2))
    height = int(max(np.ceil(extent[1] * scale) + 1, 2))
    if width * height > 16_000_000:
        raise ValueError("parameterized raster is too large; increase spacing")
    pixel = (grid_uv - grid_uv.min(axis=0)) * scale
    xyz = np.full((height, width, 3), np.nan, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    epsilon = 1e-7
    for face in faces:
        triangle = pixel[face]
        lower = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        upper = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(upper < lower):
            continue
        xs, ys = np.meshgrid(
            np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1)
        )
        points = np.stack((xs + 0.5, ys + 0.5), axis=-1)
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        w0 = ((b[1] - c[1]) * (points[..., 0] - c[0]) + (c[0] - b[0]) * (points[..., 1] - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (points[..., 0] - c[0]) + (a[0] - c[0]) * (points[..., 1] - c[1])) / denominator
        w2 = 1 - w0 - w1
        inside = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
        if not inside.any():
            continue
        interpolated = (
            w0[..., None] * vertices[face[0]]
            + w1[..., None] * vertices[face[1]]
            + w2[..., None] * vertices[face[2]]
        )
        target_y, target_x = ys[inside], xs[inside]
        empty = coverage[target_y, target_x] == 0
        xyz[target_y[empty], target_x[empty]] = interpolated[inside][empty]
        coverage[target_y, target_x] += 1
    mask = coverage > 0
    return xyz, mask, coverage


def grid_mesh_unwrap(
    field: CropField,
    seed_xyz: np.ndarray,
    raster_spacing: float = 0.25,
    probability_threshold: float = 0.4,
    layer_width: float = 2.0,
    optimize_iterations: int = 4,
) -> tuple[SurfaceGrid, ParameterizedMesh, TopologyMetrics]:
    """Unwrap using the grid-mesh approach (notebook pipeline).

    Builds a dense grid mesh from the probability field via PCA projection,
    weighted averaging, hole filling, and iterative ridge optimization.
    Produces meshes with natural UV coordinates that rasterize reliably.
    """
    seed_zyx = field.local_zyx(seed_xyz)
    seed_local = np.clip(
        np.rint(seed_zyx).astype(int), 0, np.asarray(field.data.shape) - 1
    )

    # 1. Extract connected component
    all_points, all_scores = _grid_mesh_component(field.data, seed_local, probability_threshold)
    if len(all_points) < 256:
        raise ValueError(f"seed component has only {len(all_points)} points (need 256)")

    # 2. Select the seed's layer via PCA depth
    center = all_points.mean(0)
    centered = all_points - center
    _, vectors = np.linalg.eigh(centered.T @ centered / len(all_points))
    depth = centered @ vectors[:, 0]
    nearest = int(np.argmin(np.linalg.norm(all_points - seed_zyx[::-1], axis=1)))
    seed_depth = depth[nearest]

    # Try increasing layer widths until we have enough points
    points, scores = np.empty((0, 3)), np.empty(0)
    for width in (layer_width, 4.0, 8.0, 16.0):
        selection = np.abs(depth - seed_depth) <= width
        points, scores = all_points[selection], all_scores[selection]
        if len(points) >= 256:
            break
    if len(points) < 256:
        raise ValueError(f"seed layer has only {len(points)} points (need 256)")

    # 3. Rasterize to 2D grid
    xyz_grid, anchors, sheet_mask, _ = _grid_mesh_rasterize(points, scores)

    # 4. Fill holes
    xyz_filled = _grid_mesh_fill_holes(xyz_grid, anchors, sheet_mask)

    # 5. Optimize surface positions against probability field
    xyz_optimized = _grid_mesh_optimize(
        field.data, xyz_filled, anchors, sheet_mask, optimize_iterations
    )
    valid_grid = sheet_mask & np.isfinite(xyz_optimized).all(axis=-1)

    # 6. Build triangle mesh from grid
    vertices, faces, seed_vertex = _grid_mesh_build(
        xyz_optimized, valid_grid, seed_zyx[::-1].astype(np.float32)
    )
    if len(faces) == 0:
        raise ValueError("grid mesh produced no faces")

    # 7. Orient faces consistently
    faces = np.asarray(igl.bfs_orient(np.ascontiguousarray(faces, dtype=np.int64))[0])

    # 8. Rasterize using grid UV
    xyz_raster, mask, coverage_count = _grid_mesh_rasterize_to_pixels(
        vertices, faces, xyz_optimized, valid_grid, raster_spacing
    )

    # 9. Compute normals
    normals_grid = np.zeros_like(xyz_raster)
    filled = np.nan_to_num(xyz_raster)
    du, dv = np.gradient(filled, axis=1), np.gradient(filled, axis=0)
    normals_grid[mask] = normalize(np.cross(du, dv))[mask]

    # 10. Sample confidence — xyz_raster is in local voxel coordinates,
    #     so sample field.data directly (skip the physical↔local round-trip).
    confidence = np.zeros(mask.shape, dtype=np.float32)
    local_zyx = xyz_raster[mask][:, ::-1]  # XYZ → ZYX
    confidence[mask] = ndimage.map_coordinates(
        field.data, local_zyx.T, order=1, mode="constant", cval=0, prefilter=False
    ).astype(np.float32)

    # 11. Compute topology metrics
    # Use a synthetic boundary for the mesh (grid mesh boundary)
    loops = boundary_loops(faces)
    boundary = loops[0] if loops else np.array([], dtype=np.int64)

    uv_area = signed_uv_area(
        np.zeros((len(vertices), 2)), faces
    )  # placeholder — actual UV is grid coords
    written = coverage_count[mask]
    overlap = float((written > 1).sum() / max(1, mask.sum()))
    metrics = TopologyMetrics(
        vertices=len(vertices),
        faces=len(faces),
        boundary_loops=len(loops),
        euler_characteristic=euler_characteristic(len(vertices), faces),
        flipped_fraction=0.0,  # grid UV has no flips by construction
        collapsed_fraction=0.0,
        overlap_fraction=overlap,
        raster_coverage=float(mask.sum() / max(1, ndimage.binary_fill_holes(mask).sum())),
        probability_adherence=float((confidence[mask] >= 0.2).mean()),
    )

    # 12. Build output mesh UV (normalized grid coordinates)
    grid_indices = np.argwhere(valid_grid)
    flat_xyz = xyz_optimized[valid_grid]
    uv = np.zeros((len(vertices), 2), dtype=np.float64)
    for i, v in enumerate(vertices):
        idx = np.argmin(np.linalg.norm(flat_xyz - v, axis=1))
        uv[i] = grid_indices[idx].astype(np.float64)
    uv_min = uv.min(axis=0)
    uv_range = np.maximum(uv.max(axis=0) - uv_min, 1e-9)
    uv = (uv - uv_min) / uv_range

    surface = SurfaceGrid(xyz_raster, normals_grid, confidence, mask)
    return surface, ParameterizedMesh(vertices, faces, uv, boundary), metrics


def topology_unwrap(
    field: CropField,
    seed_xyz: np.ndarray,
    patch_radius: float,
    raster_spacing: float | None = None,
    slim_iterations: int = 10,
    minimum_vertices: int = 100,
) -> tuple[SurfaceGrid, ParameterizedMesh, TopologyMetrics]:
    vertices, faces, normals = extract_ridge_mesh(field, seed_xyz)
    vertices, faces, normals = remove_nonmanifold_faces(vertices, faces, normals)
    vertices, faces, normals = prune_mesh_bridges(vertices, faces, normals)
    vertices, faces, normals, seed_vertex = select_seed_component(vertices, faces, normals, seed_xyz)
    vertices, faces, normals, boundary = geodesic_disk(
        vertices, faces, normals, seed_vertex, patch_radius,
        minimum_vertices=minimum_vertices,
    )
    faces = np.asarray(igl.bfs_orient(np.ascontiguousarray(faces, dtype=np.int64))[0])
    uv = parameterize_disk(vertices, faces, boundary, slim_iterations)
    spacing = raster_spacing or float(np.mean(field.scale_zyx))
    xyz, mask, coverage_count = rasterize_parameterized_mesh(vertices, faces, uv, spacing)
    normals_grid = np.zeros_like(xyz)
    filled = np.nan_to_num(xyz)
    du, dv = np.gradient(filled, axis=1), np.gradient(filled, axis=0)
    normals_grid[mask] = normalize(np.cross(du, dv))[mask]
    confidence = np.zeros(mask.shape, dtype=np.float32)
    confidence[mask] = field.sample(xyz[mask])
    uv_area = signed_uv_area(uv, faces)
    collapsed = np.abs(uv_area) <= 1e-12
    orientation = np.sign(np.median(uv_area[~collapsed])) if (~collapsed).any() else 1
    flipped = (uv_area * orientation < 0) & ~collapsed
    written = coverage_count[mask]
    overlap = float((written > 1).sum() / max(1, mask.sum()))
    metrics = TopologyMetrics(
        vertices=len(vertices),
        faces=len(faces),
        boundary_loops=1,
        euler_characteristic=euler_characteristic(len(vertices), faces),
        flipped_fraction=float(flipped.mean()),
        collapsed_fraction=float(collapsed.mean()),
        overlap_fraction=overlap,
        raster_coverage=float(mask.sum() / max(1, ndimage.binary_fill_holes(mask).sum())),
        probability_adherence=float((confidence[mask] >= 0.2).mean()),
    )
    surface = SurfaceGrid(xyz, normals_grid, confidence, mask)
    return surface, ParameterizedMesh(vertices, faces, uv, boundary), metrics
