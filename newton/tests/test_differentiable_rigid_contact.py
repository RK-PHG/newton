# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


@wp.kernel
def _target_linear_velocity_loss(body_qd: wp.array[wp.spatial_vector], loss: wp.array[float]):
    velocity = wp.spatial_top(body_qd[1])
    loss[0] = wp.dot(velocity, velocity)


@wp.kernel
def _contact_point_loss(point0_world: wp.array[wp.vec3], loss: wp.array[float]):
    point = point0_world[0]
    loss[0] = wp.dot(point, point)


def _make_overlapping_mesh_boxes(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=100.0, ke=2.0e3, kd=10.0, kf=0.0, mu=0.0)
    mesh = newton.Mesh.create_box(
        0.15,
        0.10,
        0.10,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
    )
    mesh.build_sdf(device=device, max_resolution=32)
    body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.08, -0.01, 0.0)))
    body_b = builder.add_body(
        xform=wp.transform(
            wp.vec3(0.08, 0.01, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.15),
        )
    )
    builder.add_shape_mesh(body_a, mesh=mesh, cfg=shape_cfg)
    builder.add_shape_mesh(body_b, mesh=mesh, cfg=shape_cfg)
    return builder.finalize(device=device, requires_grad=True)


def test_config_validation(test, device):
    config_type = newton.CollisionPipeline.DifferentiableRigidContactConfig
    with test.assertRaisesRegex(ValueError, "global_sample_count"):
        config_type(global_sample_count=1)
    with test.assertRaisesRegex(ValueError, "local_sample_count"):
        config_type(global_sample_count=8, local_sample_count=9)
    with test.assertRaisesRegex(ValueError, "normal_angle"):
        config_type(normal_angle=0.0)


def test_contact_workspaces_are_private(test, device):
    model = _make_overlapping_mesh_boxes(device)
    pipeline = newton.CollisionPipeline(
        model,
        differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
            global_sample_count=64,
            local_sample_count=8,
        ),
    )
    contacts_a = pipeline.contacts()
    contacts_b = pipeline.contacts()
    data_a = contacts_a._differentiable_rigid_contact_data
    data_b = contacts_b._differentiable_rigid_contact_data

    test.assertIs(data_a.sample_points, data_b.sample_points)
    test.assertIs(data_a.pair_keys, data_b.pair_keys)
    test.assertIsNot(data_a.selected_sample_indices, data_b.selected_sample_indices)
    test.assertIsNot(data_a.selected_sample_weights, data_b.selected_sample_weights)


def _step_once(model, pipeline, *, record_tape):
    contacts = pipeline.contacts()
    state_0 = model.state(requires_grad=record_tape)
    state_1 = model.state(requires_grad=record_tape)
    solver = newton.solvers.SolverSemiImplicit(model, enable_tri_contact=False)
    loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=record_tape)

    if record_tape:
        tape = wp.Tape()
        with tape:
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 600.0)
            wp.launch(
                _target_linear_velocity_loss, dim=1, inputs=[state_1.body_qd], outputs=[loss], device=model.device
            )
        tape.backward(loss)
        return state_1.body_q.numpy(), state_1.body_qd.numpy(), state_0.body_q.grad.numpy(), contacts

    state_0.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 600.0)
    return state_1.body_q.numpy(), state_1.body_qd.numpy(), None, contacts


def test_exact_forward_and_finite_surrogate_gradient(test, device):
    model = _make_overlapping_mesh_boxes(device)
    config = newton.CollisionPipeline.DifferentiableRigidContactConfig(
        global_sample_count=64,
        local_sample_count=8,
    )
    baseline_pipeline = newton.CollisionPipeline(model)
    differentiable_pipeline = newton.CollisionPipeline(model, differentiable_rigid_contact_config=config)

    baseline_q, baseline_qd, _, baseline_contacts = _step_once(model, baseline_pipeline, record_tape=False)
    smooth_q, smooth_qd, pose_gradient, smooth_contacts = _step_once(
        model,
        differentiable_pipeline,
        record_tape=True,
    )

    test.assertGreater(int(baseline_contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertEqual(
        int(baseline_contacts.rigid_contact_count.numpy()[0]),
        int(smooth_contacts.rigid_contact_count.numpy()[0]),
    )
    np.testing.assert_allclose(smooth_q, baseline_q, atol=1.0e-6)
    np.testing.assert_allclose(smooth_qd, baseline_qd, atol=2.0e-5)
    test.assertTrue(np.all(np.isfinite(pose_gradient)))
    test.assertGreater(float(np.linalg.norm(pose_gradient)), 1.0e-8)


def test_contact_kinematics_preserves_forward_and_smooths_gradient(test, device):
    """Preserve exact contact points while exposing RS1Dist pose gradients."""
    model = _make_overlapping_mesh_boxes(device)
    pipeline = newton.CollisionPipeline(
        model,
        differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
            global_sample_count=64,
            local_sample_count=8,
        ),
    )
    contacts = pipeline.contacts()
    state = model.state(requires_grad=True)
    pipeline.collide(state, contacts)
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(contact_count, 0)

    exact_points = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device)
    smooth_points = wp.empty(
        contacts.rigid_contact_max,
        dtype=wp.vec3,
        device=device,
        requires_grad=True,
    )
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_point0_world=exact_points)

    tape = wp.Tape()
    with tape:
        newton.eval_rigid_contact_kinematics(
            model,
            state,
            contacts,
            out_point0_world=smooth_points,
            use_smooth_witness_gradients=True,
        )
        wp.launch(_contact_point_loss, dim=1, inputs=[smooth_points], outputs=[loss], device=device)
    tape.backward(loss)

    np.testing.assert_allclose(
        smooth_points.numpy()[:contact_count],
        exact_points.numpy()[:contact_count],
        atol=1.0e-6,
    )
    pose_gradient = state.body_q.grad.numpy()
    test.assertTrue(np.all(np.isfinite(pose_gradient)))
    test.assertGreater(float(np.linalg.norm(pose_gradient)), 1.0e-8)


class TestDifferentiableRigidContact(unittest.TestCase):
    pass


devices = get_cuda_test_devices(mode="basic")
add_function_test(TestDifferentiableRigidContact, "test_config_validation", test_config_validation, devices=devices)
add_function_test(
    TestDifferentiableRigidContact,
    "test_contact_workspaces_are_private",
    test_contact_workspaces_are_private,
    devices=devices,
)
add_function_test(
    TestDifferentiableRigidContact,
    "test_contact_kinematics_preserves_forward_and_smooths_gradient",
    test_contact_kinematics_preserves_forward_and_smooths_gradient,
    devices=devices,
)
add_function_test(
    TestDifferentiableRigidContact,
    "test_exact_forward_and_finite_surrogate_gradient",
    test_exact_forward_and_finite_surrogate_gradient,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
