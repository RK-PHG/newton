# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""RS1Dist-style differentiable witness points for rigid mesh contact.

This module adapts the distance-based first-order randomized smoothing method
from "Robust Differentiable Collision Detection for General Objects"
(Chen et al., 2026, arXiv:2511.06267) to Newton and Warp.

The ordinary Newton narrow phase remains the source of truth in the forward
pass. Surface samples only define a surrogate derivative for each discrete
witness point, so enabling this path does not replace collision detection or
alter the forward contact points, normals, or penalty law.

The paper's target-dependent adaptive sampling is not suitable for a generic
simulation pipeline because downstream objectives are unavailable during
collision detection. Newton therefore uses the paper's fixed-radius sampling
variant. Coupled cross-witness Jacobians are evaluated at the discrete
witnesses and solved as the paper's 3-by-3 implicit system.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import warp as wp

from .types import GeoType, Mesh


@wp.func
def _exact_forward_smooth_backward(exact: wp.vec3, smooth: wp.vec3) -> wp.vec3:
    """Return ``exact`` while routing its adjoint through ``smooth``."""
    return exact


@wp.func_grad(_exact_forward_smooth_backward)
def _adj_exact_forward_smooth_backward(exact: wp.vec3, smooth: wp.vec3, adj_ret: wp.vec3):
    wp.adjoint[smooth] += adj_ret


@wp.func
def _exact_forward_smooth_backward_scalar(exact: float, smooth: float) -> float:
    """Return an exact scalar while routing its adjoint through a surrogate."""
    return exact


@wp.func_grad(_exact_forward_smooth_backward_scalar)
def _adj_exact_forward_smooth_backward_scalar(exact: float, smooth: float, adj_ret: float):
    wp.adjoint[smooth] += adj_ret


@wp.func
def _stop_gradient(value: wp.vec3) -> wp.vec3:
    """Return a value without propagating an adjoint to its source."""
    return value


@wp.func_grad(_stop_gradient)
def _adj_stop_gradient(value: wp.vec3, adj_ret: wp.vec3):
    pass


@wp.func
def _implicit_witness_a(
    exact_a: wp.vec3,
    smooth_a: wp.vec3,
    smooth_b: wp.vec3,
    cross_a_b: wp.mat33,
    cross_b_a: wp.mat33,
) -> wp.vec3:
    """Return exact witness A with the coupled RS1Dist adjoint."""
    return exact_a


@wp.func_grad(_implicit_witness_a)
def _adj_implicit_witness_a(
    exact_a: wp.vec3,
    smooth_a: wp.vec3,
    smooth_b: wp.vec3,
    cross_a_b: wp.mat33,
    cross_b_a: wp.mat33,
    adj_ret: wp.vec3,
):
    system = wp.identity(3, float) - cross_a_b * cross_b_a
    adjoint_a = wp.inverse(wp.transpose(system)) * adj_ret
    adjoint_b = wp.transpose(cross_a_b) * adjoint_a
    wp.adjoint[smooth_a] += adjoint_a
    wp.adjoint[smooth_b] += adjoint_b


@wp.func
def _implicit_witness_b(
    exact_b: wp.vec3,
    smooth_a: wp.vec3,
    smooth_b: wp.vec3,
    cross_a_b: wp.mat33,
    cross_b_a: wp.mat33,
) -> wp.vec3:
    """Return exact witness B with the coupled RS1Dist adjoint."""
    return exact_b


@wp.func_grad(_implicit_witness_b)
def _adj_implicit_witness_b(
    exact_b: wp.vec3,
    smooth_a: wp.vec3,
    smooth_b: wp.vec3,
    cross_a_b: wp.mat33,
    cross_b_a: wp.mat33,
    adj_ret: wp.vec3,
):
    system = wp.identity(3, float) - cross_b_a * cross_a_b
    adjoint_b = wp.inverse(wp.transpose(system)) * adj_ret
    adjoint_a = wp.transpose(cross_b_a) * adjoint_b
    wp.adjoint[smooth_a] += adjoint_a
    wp.adjoint[smooth_b] += adjoint_b


@wp.func
def _contains_pair(pair_keys: wp.array[wp.int64], pair_count: int, shape_count: int, shape_a: int, shape_b: int):
    if pair_count == 0 or shape_a < 0 or shape_b < 0:
        return False
    shape_low = wp.min(shape_a, shape_b)
    shape_high = wp.max(shape_a, shape_b)
    key = wp.int64(shape_low) * wp.int64(shape_count) + wp.int64(shape_high)
    lower = int(0)
    upper = pair_count
    while lower < upper:
        middle = lower + (upper - lower) // 2
        candidate = pair_keys[middle]
        if candidate < key:
            lower = middle + 1
        else:
            upper = middle
    return lower < pair_count and pair_keys[lower] == key


@wp.func
def _body_transform(body_q: wp.array[wp.transform], body: int) -> wp.transform:
    if body >= 0:
        return body_q[body]
    return wp.transform_identity()


@wp.func
def _safe_length(value: wp.vec3, epsilon: float) -> float:
    return wp.sqrt(wp.dot(value, value) + epsilon * epsilon)


@wp.func
def _candidate_world(
    slot: int,
    selected_sample_indices: wp.array[int],
    selected_base: int,
    sample_points: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    shape_world: wp.transform,
    exact_world: wp.vec3,
    exact_normal_world: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
    sample = selected_sample_indices[selected_base + slot]
    if sample < 0:
        return exact_world, exact_normal_world
    point = wp.transform_point(shape_world, sample_points[sample])
    normal = wp.transform_vector(shape_world, sample_normals[sample])
    return point, normal


@wp.func
def _distance_softmax_witness(
    shape_world: wp.transform,
    other_witness: wp.vec3,
    exact_world: wp.vec3,
    exact_normal_world: wp.vec3,
    sample_points: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    selected_sample_indices: wp.array[int],
    selected_sample_weights: wp.array[float],
    selected_temperature_coefficients: wp.array[float],
    selected_base: int,
    selected_count: int,
    temperature: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Evaluate RS1Dist with an analytic softmax-weight surrogate."""
    point_sum = wp.vec3(0.0)
    normal_sum = wp.vec3(0.0)
    smooth_temperature = float(0.0)
    for slot in range(selected_count):
        point, normal = _candidate_world(
            slot,
            selected_sample_indices,
            selected_base,
            sample_points,
            sample_normals,
            shape_world,
            exact_world,
            exact_normal_world,
        )
        weight = selected_sample_weights[selected_base + slot]
        point_sum += weight * point
        normal_sum += weight * normal
        distance = _safe_length(point - other_witness, 1.0e-6)
        smooth_temperature += selected_temperature_coefficients[selected_base + slot] * distance

    differentiable_temperature = _exact_forward_smooth_backward_scalar(temperature, smooth_temperature)

    # For softmax weights w_i=softmax(l_i),
    #   ds = sum_i w_i dp_i + sum_i w_i (p_i-s) dl_i.
    # The zero-forward proxies below add exactly the second term without
    # differentiating the exp/reduction implementation itself. This avoids
    # unstable generated adjoints for dynamic reductions in Warp.
    point_correction = wp.vec3(0.0)
    normal_correction = wp.vec3(0.0)
    for slot in range(selected_count):
        point, normal = _candidate_world(
            slot,
            selected_sample_indices,
            selected_base,
            sample_points,
            sample_normals,
            shape_world,
            exact_world,
            exact_normal_world,
        )
        weight = selected_sample_weights[selected_base + slot]
        logit = -_safe_length(point - other_witness, 1.0e-6) / differentiable_temperature
        point_coefficient = _stop_gradient(weight * (point - point_sum))
        normal_coefficient = _stop_gradient(weight * (normal - normal_sum))
        point_correction += _exact_forward_smooth_backward(wp.vec3(0.0), point_coefficient * logit)
        normal_correction += _exact_forward_smooth_backward(wp.vec3(0.0), normal_coefficient * logit)

    return point_sum + point_correction, normal_sum + normal_correction


@wp.kernel(enable_backward=False)
def _select_local_samples(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_count: int,
    pair_keys: wp.array[wp.int64],
    pair_count: int,
    sample_points: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    shape_sample_start: wp.array[int],
    shape_sample_count: wp.array[int],
    shape_sample_radius: wp.array[float],
    normal_cosine_threshold: float,
    local_sample_count: int,
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    temperature_epsilon: float,
    selected_sample_indices: wp.array[int],
    selected_sample_counts: wp.array[int],
    selected_sample_temperatures: wp.array[float],
    selected_sample_weights: wp.array[float],
    selected_temperature_coefficients: wp.array[float],
    selected_cross_jacobians: wp.array[wp.mat33],
):
    directed_contact = wp.tid()
    contact = directed_contact // 2
    direction = directed_contact - contact * 2
    if contact >= contact_count[0]:
        return

    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    if not _contains_pair(pair_keys, pair_count, shape_count, shape_a, shape_b):
        return

    shape = shape_a
    point_body = contact_point0[contact]
    normal_world = contact_normal[contact]
    if direction == 1:
        shape = shape_b
        point_body = contact_point1[contact]
        normal_world = -normal_world

    body = shape_body[shape]
    body_xform = _body_transform(body_q, body)
    shape_world = wp.transform_multiply(body_xform, shape_transform[shape])
    witness_world = wp.transform_point(body_xform, point_body)
    inverse_shape_world = wp.transform_inverse(shape_world)
    witness_local = wp.transform_point(inverse_shape_world, witness_world)
    normal_local = wp.transform_vector(inverse_shape_world, normal_world)

    output_base = directed_contact * local_sample_count
    selected_sample_indices[output_base] = -1
    selected_count = int(1)
    sample_start = shape_sample_start[shape]
    sample_count = shape_sample_count[shape]
    radius = shape_sample_radius[shape]

    # The global sequence is deterministically interleaved over vertices and
    # faces, so taking the first valid samples avoids frame-to-frame RNG noise.
    for local_index in range(sample_count):
        if selected_count < local_sample_count:
            sample = sample_start + local_index
            delta = sample_points[sample] - witness_local
            normal_alignment = wp.dot(sample_normals[sample], normal_local)
            if wp.length(delta) <= radius and normal_alignment >= normal_cosine_threshold:
                selected_sample_indices[output_base + selected_count] = sample
                selected_count += 1

    # If the filters return too few candidates, expand to unfiltered samples,
    # matching the official implementation's threshold-relaxation fallback.
    fallback_index = int(0)
    while selected_count < local_sample_count and fallback_index < sample_count:
        selected_sample_indices[output_base + selected_count] = sample_start + fallback_index
        selected_count += 1
        fallback_index += 1
    selected_sample_counts[directed_contact] = selected_count

    other_shape = shape_b
    other_point_body = contact_point1[contact]
    if direction == 1:
        other_shape = shape_a
        other_point_body = contact_point0[contact]
    other_body_xform = _body_transform(body_q, shape_body[other_shape])
    other_witness_world = wp.transform_point(other_body_xform, other_point_body)

    distance_mean = float(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        distance_mean += _safe_length(point - other_witness_world, temperature_epsilon)
    distance_mean /= float(selected_count)

    distance_variance = float(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        distance = _safe_length(point - other_witness_world, temperature_epsilon)
        centered = distance - distance_mean
        distance_variance += centered * centered
    distance_std = wp.sqrt(distance_variance / float(selected_count - 1) + temperature_epsilon * temperature_epsilon)
    temperature = wp.sqrt(distance_std + temperature_epsilon)
    selected_sample_temperatures[directed_contact] = temperature

    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        distance = _safe_length(point - other_witness_world, temperature_epsilon)
        selected_temperature_coefficients[output_base + slot] = (distance - distance_mean) / (
            2.0 * float(selected_count - 1) * distance_std * temperature
        )

    max_logit = float(-1.0e30)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        max_logit = wp.max(
            max_logit,
            -_safe_length(point - other_witness_world, temperature_epsilon) / temperature,
        )

    weight_sum = float(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        logit = -_safe_length(point - other_witness_world, temperature_epsilon) / temperature
        weight = wp.exp(logit - max_logit)
        selected_sample_weights[output_base + slot] = weight
        weight_sum += weight
    for slot in range(selected_count):
        selected_sample_weights[output_base + slot] /= weight_sum

    point_sum = wp.vec3(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        point_sum += selected_sample_weights[output_base + slot] * point

    temperature_gradient = wp.vec3(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        delta = point - other_witness_world
        temperature_gradient -= (
            selected_temperature_coefficients[output_base + slot] * delta / _safe_length(delta, temperature_epsilon)
        )

    cross_jacobian = wp.mat33(0.0)
    for slot in range(selected_count):
        point, _normal = _candidate_world(
            slot,
            selected_sample_indices,
            output_base,
            sample_points,
            sample_normals,
            shape_world,
            witness_world,
            normal_world,
        )
        delta = point - other_witness_world
        distance = _safe_length(delta, temperature_epsilon)
        logit_gradient = delta / (temperature * distance) + distance * temperature_gradient / (
            temperature * temperature
        )
        coefficient = selected_sample_weights[output_base + slot] * (point - point_sum)
        cross_jacobian += wp.outer(coefficient, logit_gradient)
    selected_cross_jacobians[directed_contact] = cross_jacobian


@wp.kernel
def _eval_rs1dist_contacts(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_kf: wp.array[float],
    shape_material_ka: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_count: int,
    pair_keys: wp.array[wp.int64],
    pair_count: int,
    sample_points: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    selected_sample_indices: wp.array[int],
    selected_sample_counts: wp.array[int],
    selected_sample_temperatures: wp.array[float],
    selected_sample_weights: wp.array[float],
    selected_temperature_coefficients: wp.array[float],
    selected_cross_jacobians: wp.array[wp.mat33],
    local_sample_count: int,
    temperature_epsilon: float,
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    rigid_contact_stiffness: wp.array[float],
    rigid_contact_damping: wp.array[float],
    rigid_contact_friction_scale: wp.array[float],
    friction_smoothing: float,
    body_f: wp.array[wp.spatial_vector],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    if not _contains_pair(pair_keys, pair_count, shape_count, shape_a, shape_b):
        return

    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    body_xform_a = _body_transform(body_q, body_a)
    body_xform_b = _body_transform(body_q, body_b)
    shape_world_a = wp.transform_multiply(body_xform_a, shape_transform[shape_a])
    shape_world_b = wp.transform_multiply(body_xform_b, shape_transform[shape_b])

    exact_normal_ab = contact_normal[contact]
    exact_point_a = wp.transform_point(body_xform_a, contact_point0[contact])
    exact_point_b = wp.transform_point(body_xform_b, contact_point1[contact])

    ke = 0.5 * (shape_material_ke[shape_a] + shape_material_ke[shape_b])
    kd = 0.5 * (shape_material_kd[shape_a] + shape_material_kd[shape_b])
    kf = 0.5 * (shape_material_kf[shape_a] + shape_material_kf[shape_b])
    ka = 0.5 * (shape_material_ka[shape_a] + shape_material_ka[shape_b])
    mu = 0.5 * (shape_material_mu[shape_a] + shape_material_mu[shape_b])
    if rigid_contact_stiffness:
        contact_ke = rigid_contact_stiffness[contact]
        contact_kd = rigid_contact_damping[contact]
        contact_mu = rigid_contact_friction_scale[contact]
        if contact_ke > 0.0:
            ke = contact_ke
        if contact_kd > 0.0:
            kd = contact_kd
        if contact_mu > 0.0:
            mu *= contact_mu

    # Speculative contacts that are outside the force law must not enter the
    # surrogate. Their zero output adjoint can otherwise still encounter
    # singular intermediate derivatives during Warp's generated backward.
    exact_force_normal = -exact_normal_ab
    exact_force_point_a = exact_point_a - contact_margin0[contact] * exact_force_normal
    exact_force_point_b = exact_point_b + contact_margin1[contact] * exact_force_normal
    if wp.dot(exact_force_normal, exact_force_point_a - exact_force_point_b) >= ka:
        return

    base_a = (contact * 2) * local_sample_count
    base_b = (contact * 2 + 1) * local_sample_count
    count_a = selected_sample_counts[contact * 2]
    count_b = selected_sample_counts[contact * 2 + 1]

    # These terms contain the direct/self Jacobians. The custom witness proxies
    # below solve the paper's coupled 3-by-3 cross-witness system in reverse.
    smooth_point_a, smooth_normal_a = _distance_softmax_witness(
        shape_world_a,
        _stop_gradient(exact_point_b),
        exact_point_a,
        exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_a,
        count_a,
        selected_sample_temperatures[contact * 2],
    )
    smooth_point_b, smooth_normal_b = _distance_softmax_witness(
        shape_world_b,
        _stop_gradient(exact_point_a),
        exact_point_b,
        -exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_b,
        count_b,
        selected_sample_temperatures[contact * 2 + 1],
    )
    cross_a_b = selected_cross_jacobians[contact * 2]
    cross_b_a = selected_cross_jacobians[contact * 2 + 1]
    point_a = _implicit_witness_a(exact_point_a, smooth_point_a, smooth_point_b, cross_a_b, cross_b_a)
    point_b = _implicit_witness_b(exact_point_b, smooth_point_a, smooth_point_b, cross_a_b, cross_b_a)

    # Re-evaluate the normal fields at the exact forward witnesses. Their
    # cross-witness adjoints flow through the same implicit point proxies.
    _unused_point_a, smooth_normal_a = _distance_softmax_witness(
        shape_world_a,
        point_b,
        exact_point_a,
        exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_a,
        count_a,
        selected_sample_temperatures[contact * 2],
    )
    _unused_point_b, smooth_normal_b = _distance_softmax_witness(
        shape_world_b,
        point_a,
        exact_point_b,
        -exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_b,
        count_b,
        selected_sample_temperatures[contact * 2 + 1],
    )
    smooth_normal_delta = smooth_normal_a - smooth_normal_b
    smooth_normal_ab = smooth_normal_delta / _safe_length(smooth_normal_delta, temperature_epsilon)
    if wp.dot(smooth_normal_ab, exact_normal_ab) < 0.0:
        smooth_normal_ab = -smooth_normal_ab
    normal_ab = _exact_forward_smooth_backward(exact_normal_ab, smooth_normal_ab)

    # Match eval_body_contact's forward penalty law; only its witness-point
    # derivative has been replaced.
    normal = -normal_ab
    point_a -= contact_margin0[contact] * normal
    point_b += contact_margin1[contact] * normal

    distance = wp.dot(normal, point_a - point_b)

    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    point_velocity_a = wp.vec3(0.0)
    point_velocity_b = wp.vec3(0.0)
    if body_a >= 0:
        r_a = point_a - wp.transform_point(body_xform_a, body_com[body_a])
        velocity_a = body_qd[body_a]
        point_velocity_a = wp.spatial_top(velocity_a) + wp.cross(wp.spatial_bottom(velocity_a), r_a)
    if body_b >= 0:
        r_b = point_b - wp.transform_point(body_xform_b, body_com[body_b])
        velocity_b = body_qd[body_b]
        point_velocity_b = wp.spatial_top(velocity_b) + wp.cross(wp.spatial_bottom(velocity_b), r_b)
    relative_velocity = point_velocity_a - point_velocity_b
    normal_velocity = wp.dot(normal, relative_velocity)
    tangent_velocity = relative_velocity - normal * normal_velocity

    elastic_force = distance * ke
    damping_force = wp.min(normal_velocity, 0.0) * kd * wp.step(distance)
    friction_force = wp.vec3(0.0)
    if distance < 0.0:
        tangent_speed = wp.norm_huber(tangent_velocity, delta=friction_smoothing)
        if tangent_speed > 0.0:
            friction_direction = tangent_velocity / tangent_speed
            friction_force = friction_direction * wp.min(kf * tangent_speed, -mu * (elastic_force + damping_force))

    total_force = normal * (elastic_force + damping_force) + friction_force
    if body_a >= 0:
        wp.atomic_sub(body_f, body_a, wp.spatial_vector(total_force, wp.cross(r_a, total_force)))
    if body_b >= 0:
        wp.atomic_add(body_f, body_b, wp.spatial_vector(total_force, wp.cross(r_b, total_force)))


@wp.kernel
def _eval_rs1dist_kinematics(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_count: int,
    pair_keys: wp.array[wp.int64],
    pair_count: int,
    sample_points: wp.array[wp.vec3],
    sample_normals: wp.array[wp.vec3],
    selected_sample_indices: wp.array[int],
    selected_sample_counts: wp.array[int],
    selected_sample_temperatures: wp.array[float],
    selected_sample_weights: wp.array[float],
    selected_temperature_coefficients: wp.array[float],
    selected_cross_jacobians: wp.array[wp.mat33],
    local_sample_count: int,
    temperature_epsilon: float,
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    out_distance: wp.array[float],
    out_point0_world: wp.array[wp.vec3],
    out_point1_world: wp.array[wp.vec3],
):
    """Write exact contact kinematics with RS1Dist witness adjoints."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    shape_a = contact_shape0[contact]
    shape_b = contact_shape1[contact]
    body_a = int(-1)
    body_b = int(-1)
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]
    body_xform_a = _body_transform(body_q, body_a)
    body_xform_b = _body_transform(body_q, body_b)
    exact_normal_ab = contact_normal[contact]
    exact_point_a = wp.transform_point(body_xform_a, contact_point0[contact])
    exact_point_b = wp.transform_point(body_xform_b, contact_point1[contact])

    if not _contains_pair(pair_keys, pair_count, shape_count, shape_a, shape_b):
        if out_distance.shape[0] > 0:
            thickness = contact_margin0[contact] + contact_margin1[contact]
            out_distance[contact] = wp.dot(exact_normal_ab, exact_point_b - exact_point_a) - thickness
        if out_point0_world.shape[0] > 0:
            out_point0_world[contact] = exact_point_a
        if out_point1_world.shape[0] > 0:
            out_point1_world[contact] = exact_point_b
        return

    shape_world_a = wp.transform_multiply(body_xform_a, shape_transform[shape_a])
    shape_world_b = wp.transform_multiply(body_xform_b, shape_transform[shape_b])

    base_a = (contact * 2) * local_sample_count
    base_b = (contact * 2 + 1) * local_sample_count
    smooth_point_a, _smooth_normal_a = _distance_softmax_witness(
        shape_world_a,
        _stop_gradient(exact_point_b),
        exact_point_a,
        exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_a,
        selected_sample_counts[contact * 2],
        selected_sample_temperatures[contact * 2],
    )
    smooth_point_b, _smooth_normal_b = _distance_softmax_witness(
        shape_world_b,
        _stop_gradient(exact_point_a),
        exact_point_b,
        -exact_normal_ab,
        sample_points,
        sample_normals,
        selected_sample_indices,
        selected_sample_weights,
        selected_temperature_coefficients,
        base_b,
        selected_sample_counts[contact * 2 + 1],
        selected_sample_temperatures[contact * 2 + 1],
    )
    cross_a_b = selected_cross_jacobians[contact * 2]
    cross_b_a = selected_cross_jacobians[contact * 2 + 1]
    point_a = _implicit_witness_a(exact_point_a, smooth_point_a, smooth_point_b, cross_a_b, cross_b_a)
    point_b = _implicit_witness_b(exact_point_b, smooth_point_a, smooth_point_b, cross_a_b, cross_b_a)

    if out_distance.shape[0] > 0:
        thickness = contact_margin0[contact] + contact_margin1[contact]
        out_distance[contact] = wp.dot(exact_normal_ab, point_b - point_a) - thickness
    if out_point0_world.shape[0] > 0:
        out_point0_world[contact] = point_a
    if out_point1_world.shape[0] > 0:
        out_point1_world[contact] = point_b


@dataclasses.dataclass
class DifferentiableRigidContactData:
    """Prepared RS1Dist samples and per-contact work buffers."""

    pair_keys: wp.array
    sample_points: wp.array
    sample_normals: wp.array
    shape_sample_start: wp.array
    shape_sample_count: wp.array
    shape_sample_radius: wp.array
    selected_sample_indices: wp.array | None
    selected_sample_counts: wp.array | None
    selected_sample_temperatures: wp.array | None
    selected_sample_weights: wp.array | None
    selected_temperature_coefficients: wp.array | None
    selected_cross_jacobians: wp.array | None
    local_sample_count: int
    normal_cosine_threshold: float
    temperature_epsilon: float

    def with_workspace(self, rigid_contact_max: int) -> DifferentiableRigidContactData:
        """Return a view sharing immutable samples with private contact work buffers."""
        device = self.sample_points.device
        return dataclasses.replace(
            self,
            selected_sample_indices=wp.full(
                rigid_contact_max * 2 * self.local_sample_count,
                value=-1,
                dtype=int,
                device=device,
            ),
            selected_sample_counts=wp.zeros(rigid_contact_max * 2, dtype=int, device=device),
            selected_sample_temperatures=wp.zeros(rigid_contact_max * 2, dtype=float, device=device),
            selected_sample_weights=wp.zeros(
                rigid_contact_max * 2 * self.local_sample_count,
                dtype=float,
                device=device,
            ),
            selected_temperature_coefficients=wp.zeros(
                rigid_contact_max * 2 * self.local_sample_count,
                dtype=float,
                device=device,
            ),
            selected_cross_jacobians=wp.zeros(rigid_contact_max * 2, dtype=wp.mat33, device=device),
        )


def find_differentiable_rigid_pairs(model, candidate_pairs: wp.array | None = None) -> list[tuple[int, int]]:
    """Return collidable mesh pairs supported by the RS1Dist path."""
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


def _sample_surface(mesh: Mesh, scale: np.ndarray, sample_count: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a deterministic vertex/surface candidate set like DiffCollision."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale[None, :]
    indices = np.asarray(mesh.indices, dtype=np.int32).reshape((-1, 3)).copy()
    if float(np.prod(scale)) < 0.0:
        indices[:, [1, 2]] = indices[:, [2, 1]]
    triangles = vertices[indices]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(cross, axis=1)
    surface_area = 0.5 * float(np.sum(double_areas))
    if not np.isfinite(surface_area) or surface_area <= 0.0:
        raise ValueError("Differentiable rigid contact mesh has no non-degenerate triangle area")
    face_normals = cross / np.maximum(double_areas[:, None], 1.0e-12)

    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, indices[:, corner], cross)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1.0e-12)

    vertex_count = min((sample_count + 1) // 2, len(vertices))
    surface_count = sample_count - vertex_count
    if vertex_count > 0:
        vertex_ids = np.floor((np.arange(vertex_count) + 0.5) * len(vertices) / vertex_count).astype(np.int32)
        vertex_points = vertices[vertex_ids]
        vertex_sample_normals = vertex_normals[vertex_ids]
    else:
        vertex_points = np.empty((0, 3), dtype=np.float32)
        vertex_sample_normals = np.empty((0, 3), dtype=np.float32)

    if surface_count > 0:
        cumulative_area = np.cumsum(double_areas, dtype=np.float64)
        targets = (np.arange(surface_count, dtype=np.float64) + 0.5) * (2.0 * surface_area / surface_count)
        triangle_ids = np.searchsorted(cumulative_area, targets, side="left")
        sample_ids = np.arange(surface_count, dtype=np.float64) + 0.5
        u = np.mod(sample_ids * 0.6180339887498949, 1.0)
        v = np.mod(sample_ids * 0.7548776662466927, 1.0)
        sqrt_u = np.sqrt(u)
        barycentric = np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v))
        surface_points = np.sum(triangles[triangle_ids] * barycentric[:, :, None], axis=1).astype(np.float32)
        surface_normals = face_normals[triangle_ids].astype(np.float32)
    else:
        surface_points = np.empty((0, 3), dtype=np.float32)
        surface_normals = np.empty((0, 3), dtype=np.float32)

    points = []
    normals = []
    for sample in range(max(vertex_count, surface_count)):
        if sample < vertex_count:
            points.append(vertex_points[sample])
            normals.append(vertex_sample_normals[sample])
        if sample < surface_count:
            points.append(surface_points[sample])
            normals.append(surface_normals[sample])
    return np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32), surface_area


def prepare_differentiable_rigid_contacts(
    model,
    pairs: list[tuple[int, int]],
    *,
    global_sample_count: int,
    local_sample_count: int,
    sample_radius_scale: float,
    min_sample_radius: float,
    normal_angle: float,
) -> DifferentiableRigidContactData:
    """Pre-sample mesh surfaces and allocate fixed-size RS1Dist work buffers."""
    shape_scales = model.shape_scale.numpy()
    participating_shapes = sorted({shape for pair in pairs for shape in pair})
    shape_sample_start = np.zeros(model.shape_count, dtype=np.int32)
    shape_sample_count = np.zeros(model.shape_count, dtype=np.int32)
    shape_sample_radius = np.zeros(model.shape_count, dtype=np.float32)
    packed_points = []
    packed_normals = []
    packed_count = 0

    for shape in participating_shapes:
        mesh = model.shape_source[shape]
        scale = np.asarray(shape_scales[shape], dtype=np.float32)
        if np.any(np.abs(scale) < 1.0e-8):
            raise ValueError(f"Differentiable rigid contact shape {shape} has a degenerate scale {scale.tolist()}")
        # The official implementation reserves the final global-sample slot
        # for the exact discrete witness before local sampling.
        points, normals, surface_area = _sample_surface(mesh, scale, global_sample_count - 1)
        shape_sample_start[shape] = packed_count
        shape_sample_count[shape] = len(points)
        object_scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        safe_radius = 2.0 * math.sqrt(local_sample_count * surface_area / (math.pi * global_sample_count))
        shape_sample_radius[shape] = max(sample_radius_scale * object_scale, min_sample_radius, safe_radius)
        packed_points.append(points)
        packed_normals.append(normals)
        packed_count += len(points)

    points_np = np.concatenate(packed_points, axis=0)
    normals_np = np.concatenate(packed_normals, axis=0)
    pair_keys_np = np.asarray([shape_a * model.shape_count + shape_b for shape_a, shape_b in pairs], dtype=np.int64)
    device = model.device
    return DifferentiableRigidContactData(
        pair_keys=wp.array(pair_keys_np, dtype=wp.int64, device=device),
        sample_points=wp.array(points_np, dtype=wp.vec3, device=device),
        sample_normals=wp.array(normals_np, dtype=wp.vec3, device=device),
        shape_sample_start=wp.array(shape_sample_start, dtype=int, device=device),
        shape_sample_count=wp.array(shape_sample_count, dtype=int, device=device),
        shape_sample_radius=wp.array(shape_sample_radius, dtype=float, device=device),
        selected_sample_indices=None,
        selected_sample_counts=None,
        selected_sample_temperatures=None,
        selected_sample_weights=None,
        selected_temperature_coefficients=None,
        selected_cross_jacobians=None,
        local_sample_count=local_sample_count,
        normal_cosine_threshold=math.cos(normal_angle),
        temperature_epsilon=1.0e-4,
    )


def _prepare_contact_witnesses(model, state, contacts, data: DifferentiableRigidContactData) -> None:
    """Select local samples and precompute their RS1Dist coefficients."""
    if (
        data.selected_sample_indices is None
        or data.selected_sample_counts is None
        or data.selected_sample_temperatures is None
        or data.selected_sample_weights is None
        or data.selected_temperature_coefficients is None
        or data.selected_cross_jacobians is None
    ):
        raise RuntimeError("Differentiable rigid contact workspace was not initialized for this Contacts object")
    wp.launch(
        _select_local_samples,
        dim=contacts.rigid_contact_max * 2,
        inputs=[
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_count,
            data.pair_keys,
            data.pair_keys.shape[0],
            data.sample_points,
            data.sample_normals,
            data.shape_sample_start,
            data.shape_sample_count,
            data.shape_sample_radius,
            data.normal_cosine_threshold,
            data.local_sample_count,
            contacts.rigid_contact_count,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            data.temperature_epsilon,
        ],
        outputs=[
            data.selected_sample_indices,
            data.selected_sample_counts,
            data.selected_sample_temperatures,
            data.selected_sample_weights,
            data.selected_temperature_coefficients,
            data.selected_cross_jacobians,
        ],
        device=model.device,
        record_tape=False,
    )


def eval_differentiable_rigid_contact_kinematics(
    model,
    state,
    contacts,
    data: DifferentiableRigidContactData,
    *,
    out_distance: wp.array | None,
    out_point0_world: wp.array | None,
    out_point1_world: wp.array | None,
) -> None:
    """Write exact rigid-contact kinematics with RS1Dist witness gradients."""
    _prepare_contact_witnesses(model, state, contacts, data)
    wp.launch(
        _eval_rs1dist_kinematics,
        dim=contacts.rigid_contact_max,
        inputs=[
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_count,
            data.pair_keys,
            data.pair_keys.shape[0],
            data.sample_points,
            data.sample_normals,
            data.selected_sample_indices,
            data.selected_sample_counts,
            data.selected_sample_temperatures,
            data.selected_sample_weights,
            data.selected_temperature_coefficients,
            data.selected_cross_jacobians,
            data.local_sample_count,
            data.temperature_epsilon,
            contacts.rigid_contact_count,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
        ],
        outputs=[out_distance, out_point0_world, out_point1_world],
        device=model.device,
    )


def eval_differentiable_rigid_contact_forces(
    model,
    state,
    contacts,
    data: DifferentiableRigidContactData | None,
    body_f: wp.array,
    friction_smoothing: float,
) -> None:
    """Apply ordinary contact forces with RS1Dist surrogate witness gradients."""
    if data is None or contacts is None:
        return
    _prepare_contact_witnesses(model, state, contacts, data)
    wp.launch(
        _eval_rs1dist_contacts,
        dim=contacts.rigid_contact_max,
        inputs=[
            state.body_q,
            state.body_qd,
            model.body_com,
            model.shape_transform,
            model.shape_body,
            model.shape_material_ke,
            model.shape_material_kd,
            model.shape_material_kf,
            model.shape_material_ka,
            model.shape_material_mu,
            model.shape_count,
            data.pair_keys,
            data.pair_keys.shape[0],
            data.sample_points,
            data.sample_normals,
            data.selected_sample_indices,
            data.selected_sample_counts,
            data.selected_sample_temperatures,
            data.selected_sample_weights,
            data.selected_temperature_coefficients,
            data.selected_cross_jacobians,
            data.local_sample_count,
            data.temperature_epsilon,
            contacts.rigid_contact_count,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            contacts.rigid_contact_stiffness,
            contacts.rigid_contact_damping,
            contacts.rigid_contact_friction,
            friction_smoothing,
        ],
        outputs=[body_f],
        device=model.device,
    )
