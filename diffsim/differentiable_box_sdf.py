# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optimize a box impact through Newton's differentiable rigid-contact path.

Two rectangular mesh boxes collide obliquely in free space. Gradient descent
adjusts the striker's initial X/Y velocity and angular velocity so that the
target box leaves the collision with a desired linear and angular velocity.

Run from the repository root::

    uv run python diffsim/differentiable_box_sdf.py --viewer gl
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

newton.use_coord_layout_targets = True

STRIKER_BODY = 0
TARGET_BODY = 1

BOX_HALF_EXTENTS = wp.vec3(0.14, 0.08, 0.08)
STRIKER_START = wp.vec3(-1.0, -0.10, 0.35)
TARGET_START = wp.vec3(0.0, 0.0, 0.35)
TARGET_INITIAL_ANGLE = math.radians(15.0)
TARGET_LINEAR_VELOCITY = wp.vec3(1.4, 0.3, 0.0)
TARGET_ANGULAR_VELOCITY = wp.vec3(0.0, 0.0, -1.5)
ANGULAR_VELOCITY_LOSS_WEIGHT = 0.25


@wp.kernel
def terminal_velocity_loss(
    body_qd: wp.array[wp.spatial_vector],
    target_linear_velocity: wp.vec3,
    target_angular_velocity: wp.vec3,
    angular_weight: float,
    loss: wp.array[float],
):
    """Compute target-box post-impact linear and angular velocity error."""
    target_velocity = body_qd[TARGET_BODY]
    linear_delta = wp.spatial_top(target_velocity) - target_linear_velocity
    angular_delta = wp.spatial_bottom(target_velocity) - target_angular_velocity
    loss[0] = wp.dot(linear_delta, linear_delta) + angular_weight * wp.dot(angular_delta, angular_delta)


@wp.kernel
def update_striker_velocity(
    body_qd: wp.array[wp.spatial_vector],
    gradient: wp.array[wp.spatial_vector],
    linear_rate: float,
    angular_rate: float,
):
    """Update only the three action coordinates used by this scene."""
    velocity = body_qd[STRIKER_BODY]
    velocity_gradient = gradient[STRIKER_BODY]
    body_qd[STRIKER_BODY] = wp.spatial_vector(
        velocity[0] - linear_rate * velocity_gradient[0],
        velocity[1] - linear_rate * velocity_gradient[1],
        0.0,
        0.0,
        0.0,
        velocity[5] - angular_rate * velocity_gradient[5],
    )


class Example:
    """Train and visualize one differentiable box-box collision."""

    def __init__(self, viewer, args):
        self.viewer = viewer

        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.train_iter = 0
        self.linear_train_rate = 6.0e-4
        self.angular_train_rate = 5.0e-4
        self.loss = wp.zeros(1, dtype=float, device=args.device, requires_grad=True)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        volume = 8.0 * BOX_HALF_EXTENTS[0] * BOX_HALF_EXTENTS[1] * BOX_HALF_EXTENTS[2]
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=1.0 / volume,
            has_shape_collision=True,
            ke=4.8e4,
            kd=0.0,
            kf=0.0,
            mu=0.0,
        )
        box_mesh = newton.Mesh.create_box(
            BOX_HALF_EXTENTS[0],
            BOX_HALF_EXTENTS[1],
            BOX_HALF_EXTENTS[2],
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
        )

        striker = builder.add_body(
            xform=wp.transform(STRIKER_START, wp.quat_identity()),
            label="striker",
        )
        target = builder.add_body(
            xform=wp.transform(
                TARGET_START,
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), TARGET_INITIAL_ANGLE),
            ),
            label="target",
        )
        builder.add_shape_mesh(striker, mesh=box_mesh, cfg=shape_cfg, color=wp.vec3(0.1, 0.4, 0.94))
        builder.add_shape_mesh(target, mesh=box_mesh, cfg=shape_cfg, color=wp.vec3(0.94, 0.3, 0.1))

        self.model = builder.finalize(device=args.device, requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
                sdf_resolution=64,
                surface_sample_count=192,
                contact_margin=0.02,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()
        self.states = [self.model.state(requires_grad=True) for _ in range(self.sim_steps * self.sim_substeps + 1)]
        initial_velocity = np.zeros((self.model.body_count, 6), dtype=np.float32)
        initial_velocity[STRIKER_BODY, [0, 1, 5]] = (2.0, 0.12, 0.0)
        self.states[0].body_qd.assign(initial_velocity)

        self.linear_velocity_error = 0.0
        self.angular_velocity_error = 0.0

        self.viewer.set_model(self.model)
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(1.6, -2.4, 1.8), pitch=-28.0, yaw=125.0)
        print(
            "desired target velocity: "
            f"linear=({TARGET_LINEAR_VELOCITY[0]:.2f}, {TARGET_LINEAR_VELOCITY[1]:.2f}, "
            f"{TARGET_LINEAR_VELOCITY[2]:.2f}), "
            f"angular=({TARGET_ANGULAR_VELOCITY[0]:.2f}, {TARGET_ANGULAR_VELOCITY[1]:.2f}, "
            f"{TARGET_ANGULAR_VELOCITY[2]:.2f})"
        )
        self.capture()

    def capture(self):
        """Capture the fixed forward/backward rollout in one CUDA graph."""
        if not self.model.device.is_cuda:
            self.graph = None
            return
        with wp.ScopedCapture() as capture:
            self.forward_backward()
        self.graph = capture.graph

    def forward_backward(self):
        """Record the rollout on a Tape and propagate the terminal loss."""
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def forward(self):
        """Roll out one collision and compute post-impact velocity loss."""
        for step in range(self.sim_steps * self.sim_substeps):
            self.states[step].clear_forces()
            self.collision_pipeline.collide(self.states[step], self.contacts)
            self.solver.step(
                self.states[step],
                self.states[step + 1],
                self.control,
                self.contacts,
                self.sim_dt,
            )

        wp.launch(
            terminal_velocity_loss,
            dim=1,
            inputs=[
                self.states[-1].body_qd,
                TARGET_LINEAR_VELOCITY,
                TARGET_ANGULAR_VELOCITY,
                ANGULAR_VELOCITY_LOSS_WEIGHT,
            ],
            outputs=[self.loss],
            device=self.model.device,
        )

    def step(self):
        """Run one differentiable rollout and one optimizer update."""
        if self.graph is None:
            self.forward_backward()
        else:
            wp.capture_launch(self.graph)

        loss = float(self.loss.numpy()[0])
        target_velocity = self.states[-1].body_qd.numpy()[TARGET_BODY]
        linear_velocity = target_velocity[:3]
        angular_velocity = target_velocity[3:]
        linear_delta = linear_velocity - np.asarray(TARGET_LINEAR_VELOCITY)
        angular_delta = angular_velocity - np.asarray(TARGET_ANGULAR_VELOCITY)
        self.linear_velocity_error = float(np.linalg.norm(linear_delta))
        self.angular_velocity_error = float(np.linalg.norm(angular_delta))
        action = self.states[0].body_qd.numpy()[STRIKER_BODY]
        print(
            f"iter={self.train_iter:03d} loss={loss:.6f} "
            f"linear=({linear_velocity[0]:.3f}, {linear_velocity[1]:.3f}, {linear_velocity[2]:.3f}) "
            f"angular=({angular_velocity[0]:.3f}, {angular_velocity[1]:.3f}, {angular_velocity[2]:.3f}) "
            f"action=({action[0]:.3f}, {action[1]:.3f}, {action[5]:.3f})"
        )

        wp.launch(
            update_striker_velocity,
            dim=1,
            inputs=[
                self.states[0].body_qd,
                self.states[0].body_qd.grad,
                self.linear_train_rate,
                self.angular_train_rate,
            ],
            device=self.model.device,
        )
        self.tape.zero()
        self.train_iter += 1

    def render(self):
        """Replay the current collision rollout."""
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        for sim_step in range(self.sim_steps + 1):
            state = self.states[sim_step * self.sim_substeps]
            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
            self.viewer.log_scalar("/loss", float(self.loss.numpy()[0]))
            self.viewer.log_scalar("/linear_velocity_error", self.linear_velocity_error)
            self.viewer.log_scalar("/angular_velocity_error", self.angular_velocity_error)
            self.viewer.end_frame()
            self.frame += 1


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
