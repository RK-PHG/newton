# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-topology differentiable rigid contact based on dense mesh SDFs."""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp

from .sdf_utils import get_distance_to_mesh_normal
from .types import GeoType, Mesh


@wp.struct
class _DenseSDFData:
    values: wp.array3d[float]
    lower: wp.vec3
    upper: wp.vec3
    inv_voxel_size: wp.vec3


@wp.kernel(enable_backward=False)
def _bake_dense_sdf(
    mesh: wp.uint64,
    lower: wp.vec3,
    voxel_size: wp.vec3,
    values: wp.array3d[float],
):
    i, j, k = wp.tid()
    point = lower + wp.cw_mul(wp.vec3(float(i), float(j), float(k)), voxel_size)
    values[i, j, k] = get_distance_to_mesh_normal(mesh, point, 1.0e6)


@wp.func
def _cubic_bspline_weights(t: float) -> tuple[wp.vec4, wp.vec4]:
    one_minus_t = 1.0 - t
    t2 = t * t
    t3 = t2 * t
    weights = wp.vec4(
        one_minus_t * one_minus_t * one_minus_t / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    )
    derivatives = wp.vec4(
        -0.5 * one_minus_t * one_minus_t,
        1.5 * t2 - 2.0 * t,
        -1.5 * t2 + t + 0.5,
        0.5 * t2,
    )
    return weights, derivatives


@wp.func
def _sample_dense_sdf_gradient(sdf: _DenseSDFData, local_pos: wp.vec3) -> tuple[float, wp.vec3]:
    """Evaluate a C2 cubic B-spline field and its analytical gradient."""
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.lower[0], sdf.upper[0]),
        wp.clamp(local_pos[1], sdf.lower[1], sdf.upper[1]),
        wp.clamp(local_pos[2], sdf.lower[2], sdf.upper[2]),
    )
    outside_delta = local_pos - clamped
    outside_distance = wp.length(outside_delta)

    grid = wp.cw_mul(clamped - sdf.lower, sdf.inv_voxel_size)
    nx = sdf.values.shape[0]
    ny = sdf.values.shape[1]
    nz = sdf.values.shape[2]
    cell_x = int(wp.floor(grid[0]))
    cell_y = int(wp.floor(grid[1]))
    cell_z = int(wp.floor(grid[2]))
    tx = grid[0] - float(cell_x)
    ty = grid[1] - float(cell_y)
    tz = grid[2] - float(cell_z)
    weights_x, derivatives_x = _cubic_bspline_weights(tx)
    weights_y, derivatives_y = _cubic_bspline_weights(ty)
    weights_z, derivatives_z = _cubic_bspline_weights(tz)

    distance = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    gz = float(0.0)
    for x_offset in range(4):
        ix = wp.clamp(cell_x + x_offset - 1, 0, nx - 1)
        for y_offset in range(4):
            iy = wp.clamp(cell_y + y_offset - 1, 0, ny - 1)
            for z_offset in range(4):
                iz = wp.clamp(cell_z + z_offset - 1, 0, nz - 1)
                value = sdf.values[ix, iy, iz]
                distance = distance + value * weights_x[x_offset] * weights_y[y_offset] * weights_z[z_offset]
                gx = gx + value * derivatives_x[x_offset] * weights_y[y_offset] * weights_z[z_offset]
                gy = gy + value * weights_x[x_offset] * derivatives_y[y_offset] * weights_z[z_offset]
                gz = gz + value * weights_x[x_offset] * weights_y[y_offset] * derivatives_z[z_offset]

    gradient = wp.cw_mul(wp.vec3(gx, gy, gz), sdf.inv_voxel_size)
    if outside_distance > 0.0:
        distance = distance + outside_distance
        gradient = outside_delta / outside_distance
    return distance, gradient


@wp.func
def _smooth_normal_force(distance: float, stiffness: float, margin: float) -> float:
    """Return a C1 penalty force with compact support at ``margin``."""
    if distance >= margin:
        return 0.0
    if distance > 0.0:
        activation = margin - distance
        return 0.5 * stiffness * activation * activation / margin
    return stiffness * (0.5 * margin - distance)


@wp.func
def _shape_world_transform(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    body: int,
    shape: int,
) -> wp.transform:
    transform = shape_transform[shape]
    if body >= 0:
        transform = wp.transform_multiply(body_q[body], transform)
    return transform


@wp.kernel
def _eval_differentiable_rigid_pair(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_material_ke: wp.array[float],
    shape_a: int,
    shape_b: int,
    samples_a: wp.array[wp.vec3],
    samples_b: wp.array[wp.vec3],
    weights_a: wp.array[float],
    weights_b: wp.array[float],
    sdf_a: _DenseSDFData,
    sdf_b: _DenseSDFData,
    margin: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Integrate two directed surface-to-SDF contact energies for one pair."""
    tid = wp.tid()
    sample_count = samples_a.shape[0]
    direction = tid // sample_count
    sample_index = tid - direction * sample_count

    source_shape = shape_a
    target_shape = shape_b
    sample_local = samples_a[sample_index]
    sample_weight = weights_a[sample_index]
    if direction == 1:
        source_shape = shape_b
        target_shape = shape_a
        sample_local = samples_b[sample_index]
        sample_weight = weights_b[sample_index]

    source_body = shape_body[source_shape]
    target_body = shape_body[target_shape]
    source_xform = _shape_world_transform(body_q, shape_transform, source_body, source_shape)
    target_xform = _shape_world_transform(body_q, shape_transform, target_body, target_shape)
    sample_world = wp.transform_point(source_xform, sample_local)
    sample_target = wp.transform_point(wp.transform_inverse(target_xform), sample_world)

    distance = float(0.0)
    normal_target = wp.vec3(0.0)
    if direction == 0:
        distance, normal_target = _sample_dense_sdf_gradient(sdf_b, sample_target)
    else:
        distance, normal_target = _sample_dense_sdf_gradient(sdf_a, sample_target)
    normal_length = wp.length(normal_target)
    stiffness = 0.5 * (shape_material_ke[shape_a] + shape_material_ke[shape_b])
    force_magnitude = _smooth_normal_force(distance, stiffness, margin)

    if force_magnitude > 0.0 and normal_length > 1.0e-8:
        normal_target = normal_target / normal_length
        normal_world = wp.quat_rotate(wp.transform_get_rotation(target_xform), normal_target)
        # The two directed integrals share the pair energy. Each direction has
        # unit-sum quadrature weights, so material stiffness retains N/m units.
        force = normal_world * force_magnitude * sample_weight * 0.5

        if source_body >= 0:
            source_com = wp.transform_point(body_q[source_body], body_com[source_body])
            source_torque = wp.cross(sample_world - source_com, force)
            wp.atomic_add(body_f, source_body, wp.spatial_vector(force, source_torque))
        if target_body >= 0:
            target_com = wp.transform_point(body_q[target_body], body_com[target_body])
            target_force = -force
            target_torque = wp.cross(sample_world - target_com, target_force)
            wp.atomic_add(body_f, target_body, wp.spatial_vector(target_force, target_torque))


@dataclasses.dataclass
class _PreparedShape:
    samples: wp.array
    weights: wp.array
    sdf: _DenseSDFData


@dataclasses.dataclass
class _PreparedPair:
    shape_a: int
    shape_b: int
    shape_a_data: _PreparedShape
    shape_b_data: _PreparedShape
    margin: float


@dataclasses.dataclass
class DifferentiableRigidContactData:
    """Internal device data attached to a :class:`Contacts` instance."""

    pairs: tuple[_PreparedPair, ...]
    pair_keys: wp.array


def find_differentiable_rigid_pairs(model, candidate_pairs: wp.array | None = None) -> list[tuple[int, int]]:
    """Return fixed mesh-mesh pairs supported by the differentiable path."""
    if candidate_pairs is None:
        candidate_pairs = model.shape_contact_pairs
    if candidate_pairs is None or candidate_pairs.shape[0] == 0:
        return []

    shape_types = model.shape_type.numpy()
    shape_bodies = model.shape_body.numpy()
    supported_types = (int(GeoType.MESH), int(GeoType.CONVEX_MESH))
    pairs = []
    for shape_a_value, shape_b_value in candidate_pairs.numpy().reshape((-1, 2)):
        shape_a = int(shape_a_value)
        shape_b = int(shape_b_value)
        if shape_types[shape_a] not in supported_types or shape_types[shape_b] not in supported_types:
            continue
        if not isinstance(model.shape_source[shape_a], Mesh) or not isinstance(model.shape_source[shape_b], Mesh):
            continue
        if shape_bodies[shape_a] == shape_bodies[shape_b]:
            continue
        pairs.append((min(shape_a, shape_b), max(shape_a, shape_b)))
    return sorted(set(pairs))


def _prepare_shape(model, shape: int, resolution: int, padding: float, sample_count: int) -> _PreparedShape:
    mesh = model.shape_source[shape]
    scale = np.asarray(model.shape_scale.numpy()[shape], dtype=np.float32)
    if np.any(np.abs(scale) < 1.0e-8):
        raise ValueError(f"Differentiable rigid contact shape {shape} has a degenerate scale {scale.tolist()}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale[None, :]
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape((-1, 3)).copy()
    if float(np.prod(scale)) < 0.0:
        indices[:, [1, 2]] = indices[:, [2, 1]]

    triangles = vertices[indices]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    total_area = float(np.sum(triangle_areas))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError(f"Differentiable rigid contact shape {shape} has no non-degenerate triangle area")

    cumulative_area = np.cumsum(triangle_areas, dtype=np.float64)
    area_targets = (np.arange(sample_count, dtype=np.float64) + 0.5) * (total_area / sample_count)
    sampled_triangles = np.searchsorted(cumulative_area, area_targets, side="left")
    # Area stratification chooses triangles; two irrational rotations then
    # distribute repeated selections inside each triangle instead of emitting
    # duplicate centroids on low-face-count meshes such as boxes.
    sample_ids = np.arange(sample_count, dtype=np.float64) + 0.5
    u = np.mod(sample_ids * 0.6180339887498949, 1.0)
    v = np.mod(sample_ids * 0.7548776662466927, 1.0)
    sqrt_u = np.sqrt(u)
    barycentric = np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v))
    samples = np.sum(triangles[sampled_triangles] * barycentric[:, :, None], axis=1).astype(np.float32)
    weights = np.full(sample_count, 1.0 / sample_count, dtype=np.float32)

    lower = vertices.min(axis=0) - padding
    requested_upper = vertices.max(axis=0) + padding
    extent = requested_upper - lower
    voxel_size_scalar = float(np.max(extent)) / float(resolution)
    dimensions = np.ceil(extent / voxel_size_scalar).astype(np.int32) + 1
    voxel_size = np.full(3, voxel_size_scalar, dtype=np.float32)
    upper = lower + (dimensions - 1) * voxel_size

    device = model.device
    values = wp.zeros(
        shape=tuple(int(value) for value in dimensions),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    query_points = wp.array(vertices, dtype=wp.vec3, device=device)
    query_indices = wp.array(indices.reshape(-1), dtype=wp.int32, device=device)
    query_mesh = wp.Mesh(points=query_points, indices=query_indices)
    wp.launch(
        _bake_dense_sdf,
        dim=tuple(int(value) for value in dimensions),
        inputs=[query_mesh.id, wp.vec3(lower), wp.vec3(voxel_size)],
        outputs=[values],
        device=device,
    )
    # The bake owns a temporary query BVH; finish before releasing it.
    wp.synchronize_device(device)

    sdf = _DenseSDFData()
    sdf.values = values
    sdf.lower = wp.vec3(lower)
    sdf.upper = wp.vec3(upper)
    sdf.inv_voxel_size = wp.vec3(1.0 / voxel_size)
    return _PreparedShape(
        samples=wp.array(samples, dtype=wp.vec3, device=device),
        weights=wp.array(weights, dtype=float, device=device),
        sdf=sdf,
    )


def prepare_differentiable_rigid_contacts(
    model,
    pairs: list[tuple[int, int]],
    *,
    sdf_resolution: int,
    surface_sample_count: int,
    contact_margin: float,
) -> DifferentiableRigidContactData:
    """Bake per-shape data and assemble fixed differentiable contact pairs."""
    shape_margins = model.shape_margin.numpy()
    max_shape_margin = max((float(shape_margins[shape]) for pair in pairs for shape in pair), default=0.0)
    shape_scales = model.shape_scale.numpy()
    max_extent = 0.0
    for pair in pairs:
        for shape in pair:
            mesh = model.shape_source[shape]
            vertices = np.asarray(mesh.vertices, dtype=np.float32) * shape_scales[shape][None, :]
            max_extent = max(max_extent, float(np.max(vertices.max(axis=0) - vertices.min(axis=0))))
    padding = contact_margin + max_shape_margin + 2.0 * max_extent / float(sdf_resolution)

    prepared_shapes = {}
    prepared_pairs = []
    for shape_a, shape_b in pairs:
        for shape in (shape_a, shape_b):
            if shape not in prepared_shapes:
                prepared_shapes[shape] = _prepare_shape(
                    model,
                    shape,
                    sdf_resolution,
                    padding,
                    surface_sample_count,
                )
        prepared_pairs.append(
            _PreparedPair(
                shape_a=shape_a,
                shape_b=shape_b,
                shape_a_data=prepared_shapes[shape_a],
                shape_b_data=prepared_shapes[shape_b],
                margin=contact_margin + float(shape_margins[shape_a]) + float(shape_margins[shape_b]),
            )
        )

    pair_keys_np = np.asarray([shape_a * model.shape_count + shape_b for shape_a, shape_b in pairs], dtype=np.int64)
    return DifferentiableRigidContactData(
        pairs=tuple(prepared_pairs),
        pair_keys=wp.array(pair_keys_np, dtype=wp.int64, device=model.device),
    )


def eval_differentiable_rigid_contact_forces(model, state, data, body_f: wp.array) -> None:
    """Evaluate all prepared pairs into the solver's body-force accumulator."""
    if data is None:
        return
    for pair in data.pairs:
        wp.launch(
            _eval_differentiable_rigid_pair,
            dim=2 * pair.shape_a_data.samples.shape[0],
            inputs=[
                state.body_q,
                model.body_com,
                model.shape_transform,
                model.shape_body,
                model.shape_material_ke,
                pair.shape_a,
                pair.shape_b,
                pair.shape_a_data.samples,
                pair.shape_b_data.samples,
                pair.shape_a_data.weights,
                pair.shape_b_data.weights,
                pair.shape_a_data.sdf,
                pair.shape_b_data.sdf,
                pair.margin,
            ],
            outputs=[body_f],
            device=model.device,
        )
