# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optimize a polyhedron's initial velocity for a target contact point.

A convex icosahedron falls into a fixed concave trough. The objective is the
impact witness point on the trough, rather than a body center or a point rigidly
attached to the moving body. Exact contact points are used in the forward pass
and RS1Dist witness derivatives are used in the backward pass.

Run from the repository root::

    uv run python diffsim/differentiable_contact_point_target.py --viewer gl
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

newton.use_coord_layout_targets = True

PROJECTILE_BODY = 0
TARGET_CONTACT_POINT = wp.vec3(0.18, 0.10, 0.05)
INITIAL_POSITION = wp.vec3(-0.35, -0.10, 1.00)
INITIAL_VELOCITY = wp.vec3(0.75, 0.16, -0.10)


def _cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    epsilon = 1.0e-10
    return (
        _cross_2d(a, b, point) >= -epsilon and _cross_2d(b, c, point) >= -epsilon and _cross_2d(c, a, point) >= -epsilon
    )


def _triangulate_polygon(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a small counter-clockwise concave polygon by ear clipping."""
    remaining = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear_found = False
        for slot, current in enumerate(remaining):
            previous = remaining[slot - 1]
            following = remaining[(slot + 1) % len(remaining)]
            if _cross_2d(polygon[previous], polygon[current], polygon[following]) <= 0.0:
                continue
            if any(
                candidate not in (previous, current, following)
                and _point_in_triangle(
                    polygon[candidate],
                    polygon[previous],
                    polygon[current],
                    polygon[following],
                )
                for candidate in remaining
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[slot]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Failed to triangulate the trough cross-section")
    triangles.append(tuple(remaining))
    return triangles


def _create_trough_mesh() -> newton.Mesh:
    """Create one closed concave mesh whose top surface forms a trough."""
    # Counter-clockwise solid cross-section in the X/Z plane. The indentation
    # between x=-0.55 and x=0.55 is the open trough.
    cross_section = np.asarray(
        [
            (-1.00, -0.25),
            (1.00, -0.25),
            (1.00, 0.45),
            (0.55, 0.45),
            (0.38, 0.05),
            (-0.38, 0.05),
            (-0.55, 0.45),
            (-1.00, 0.45),
        ],
        dtype=np.float32,
    )
    half_width = 0.45
    vertices = np.asarray(
        [(x, -half_width, z) for x, z in cross_section] + [(x, half_width, z) for x, z in cross_section],
        dtype=np.float32,
    )
    cap_triangles = _triangulate_polygon(cross_section)
    count = len(cross_section)
    triangles: list[tuple[int, int, int]] = []
    triangles.extend(cap_triangles)  # front cap, outward normal -Y
    triangles.extend((a + count, c + count, b + count) for a, b, c in cap_triangles)
    for index in range(count):
        following = (index + 1) % count
        triangles.append((index, index + count, following))
        triangles.append((following, index + count, following + count))
    return newton.Mesh(vertices, np.asarray(triangles, dtype=np.int32), compute_inertia=False)


def _create_icosahedron_mesh(radius: float) -> newton.Mesh:
    """Create a low-resolution convex polyhedron with twenty triangular faces."""
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float32,
    )
    vertices *= radius / np.linalg.norm(vertices[0])
    triangles = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    return newton.Mesh(vertices, triangles)


@wp.kernel
def average_contact_point(
    contact_count: wp.array[int],
    point0_world: wp.array[wp.vec3],
    measured_point: wp.array[wp.vec3],
):
    """Average all active trough witnesses without depending on contact order."""
    contact = wp.tid()
    count = contact_count[0]
    if contact < count:
        wp.atomic_add(measured_point, 0, point0_world[contact] / float(count))


@wp.kernel
def contact_point_loss(
    contact_count: wp.array[int],
    measured_point: wp.array[wp.vec3],
    target: wp.vec3,
    loss: wp.array[float],
):
    """Minimize the mean contact witness on the static trough."""
    if contact_count[0] > 0:
        point = measured_point[0]
        delta = point - target
        loss[0] = wp.dot(delta, delta)
    else:
        loss[0] = 4.0


@wp.kernel(enable_backward=False)
def update_initial_velocity(
    body_qd: wp.array[wp.spatial_vector],
    gradient: wp.array[wp.spatial_vector],
    learning_rate: float,
):
    """Update horizontal launch velocity while keeping vertical motion fixed."""
    velocity = body_qd[PROJECTILE_BODY]
    velocity_gradient = gradient[PROJECTILE_BODY]
    body_qd[PROJECTILE_BODY] = wp.spatial_vector(
        wp.clamp(velocity[0] - learning_rate * velocity_gradient[0], 0.2, 1.8),
        wp.clamp(velocity[1] - learning_rate * velocity_gradient[1], -0.6, 0.8),
        INITIAL_VELOCITY[2],
        0.0,
        0.0,
        0.0,
    )


class Example:
    """Train and visualize contact-point targeting in a fixed trough."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 32
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.objective_step = 200
        self.frame = 0
        self.train_iter = 0
        self.learning_rate = 1.0e-1
        self.loss_history: list[float] = []
        self.current_loss = 0.0
        self.current_contact_count = 0
        self.current_velocity = np.asarray(INITIAL_VELOCITY, dtype=np.float32)
        self.current_contact_point = np.zeros(3, dtype=np.float32)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.002
        trough_mesh = _create_trough_mesh()
        projectile_mesh = _create_icosahedron_mesh(radius=0.11)
        trough_mesh.build_sdf(device=args.device, max_resolution=96)
        projectile_mesh.build_sdf(device=args.device, max_resolution=48)

        material = newton.ModelBuilder.ShapeConfig(ke=1.2e4, kd=180.0, kf=20.0, mu=0.15)
        builder.add_shape_mesh(
            body=-1,
            mesh=trough_mesh,
            cfg=material,
            color=wp.vec3(0.32, 0.37, 0.46),
            label="fixed_trough_sdf",
        )
        projectile = builder.add_body(
            xform=wp.transform(INITIAL_POSITION, wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 0.5)), 0.35)),
            label="falling_icosahedron",
        )
        builder.add_shape_mesh(
            body=projectile,
            mesh=projectile_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=900.0,
                ke=material.ke,
                kd=material.kd,
                kf=material.kf,
                mu=material.mu,
            ),
            color=wp.vec3(0.12, 0.48, 0.95),
            label="falling_icosahedron_sdf",
        )

        self.model = builder.finalize(device=args.device, requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
                global_sample_count=512,
                local_sample_count=16,
            ),
        )
        rollout_steps = self.sim_steps * self.sim_substeps
        self.contacts = [self.collision_pipeline.contacts() for _ in range(rollout_steps)]
        self.objective_contacts = self.contacts[self.objective_step]
        self.states = [self.model.state(requires_grad=True) for _ in range(rollout_steps + 1)]
        self.control = self.model.control()

        initial_qd = np.zeros((self.model.body_count, 6), dtype=np.float32)
        initial_qd[PROJECTILE_BODY, :3] = np.asarray(INITIAL_VELOCITY)
        self.states[0].body_qd.assign(initial_qd)

        contact_max = self.objective_contacts.rigid_contact_max
        self.contact_point0_world = wp.empty(contact_max, dtype=wp.vec3, device=args.device, requires_grad=True)
        self.contact_point1_world = wp.empty(contact_max, dtype=wp.vec3, device=args.device, requires_grad=True)
        self.measured_contact_point = wp.zeros(1, dtype=wp.vec3, device=args.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=args.device, requires_grad=True)
        self.target_marker = wp.array([TARGET_CONTACT_POINT], dtype=wp.vec3, device=args.device)
        self.actual_marker = wp.zeros(1, dtype=wp.vec3, device=args.device)
        self.marker_radii = wp.array([0.035], dtype=float, device=args.device)
        self.target_color = wp.array([wp.vec3(0.10, 0.95, 0.20)], dtype=wp.vec3, device=args.device)
        self.actual_color = wp.array([wp.vec3(1.00, 0.25, 0.05)], dtype=wp.vec3, device=args.device)

        self.viewer.set_model(self.model)
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(1.75, -2.35, 1.55), pitch=-24.0, yaw=126.0)

    def forward(self):
        """Simulate, then evaluate the final contact witness directly."""
        for step, contacts in enumerate(self.contacts):
            self.states[step].clear_forces()
            self.collision_pipeline.collide(self.states[step], contacts)
            self.solver.step(
                self.states[step],
                self.states[step + 1],
                self.control,
                contacts,
                self.sim_dt,
            )

        objective_state = self.states[self.objective_step]
        newton.eval_rigid_contact_kinematics(
            self.model,
            objective_state,
            self.objective_contacts,
            out_point0_world=self.contact_point0_world,
            out_point1_world=self.contact_point1_world,
            use_smooth_witness_gradients=True,
        )
        wp.launch(
            average_contact_point,
            dim=self.objective_contacts.rigid_contact_max,
            inputs=[
                self.objective_contacts.rigid_contact_count,
                self.contact_point0_world,
            ],
            outputs=[self.measured_contact_point],
            device=self.model.device,
        )
        wp.launch(
            contact_point_loss,
            dim=1,
            inputs=[
                self.objective_contacts.rigid_contact_count,
                self.measured_contact_point,
                TARGET_CONTACT_POINT,
            ],
            outputs=[self.loss],
            device=self.model.device,
        )

    def forward_backward(self):
        """Record the rollout and propagate contact-point loss."""
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def step(self):
        """Execute one rollout and one velocity update."""
        self.forward_backward()

        loss = float(self.loss.numpy()[0])
        contact_count = int(self.objective_contacts.rigid_contact_count.numpy()[0])
        contact_point = self.measured_contact_point.numpy()[0]
        action = self.states[0].body_qd.numpy()[PROJECTILE_BODY]
        gradient = self.states[0].body_qd.grad.numpy()[PROJECTILE_BODY]
        self.actual_marker.assign(contact_point.reshape(1, 3))
        self.current_loss = loss
        self.current_contact_count = contact_count
        self.current_velocity = action[:3].copy()
        self.current_contact_point = contact_point.copy()
        self.loss_history.append(loss)
        print(
            f"iter={self.train_iter:03d} loss={loss:.6f} contacts={contact_count} "
            f"contact=({contact_point[0]:.3f}, {contact_point[1]:.3f}, {contact_point[2]:.3f}) "
            f"velocity=({action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}) "
            f"grad=({gradient[0]:.6g}, {gradient[1]:.6g})"
        )

        wp.launch(
            update_initial_velocity,
            dim=1,
            inputs=[self.states[0].body_qd, self.states[0].body_qd.grad, self.learning_rate],
            device=self.model.device,
        )
        self.tape.zero()
        self.loss.zero_()
        self.measured_contact_point.zero_()
        self.train_iter += 1

    def render(self):
        """Replay the rollout with target and measured contact markers."""
        if self.viewer.is_paused():
            self.viewer.begin_frame(self.viewer.time)
            self.viewer.end_frame()
            return

        for sim_step in range(self.sim_steps + 1):
            state = self.states[sim_step * self.sim_substeps]
            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
            if sim_step < self.sim_steps:
                contacts = self.contacts[sim_step * self.sim_substeps]
                self.viewer.log_contacts(contacts, state)
            self.viewer.log_points("/target_contact", self.target_marker, self.marker_radii, self.target_color)
            if self.current_contact_count > 0:
                self.viewer.log_points("/actual_contact", self.actual_marker, self.marker_radii, self.actual_color)
            self.viewer.log_scalar("/contact_point_loss", self.current_loss)
            self.viewer.end_frame()
            self.frame += 1

    def gui(self, ui):
        """Show the optimization variable and contact-point objective."""
        target = np.asarray(TARGET_CONTACT_POINT, dtype=np.float32)
        delta = self.current_contact_point - target
        distance = float(np.linalg.norm(delta))
        ui.text(f"Training iteration: {self.train_iter}")
        ui.text(f"Contact-point loss: {self.current_loss:.6f}")
        ui.text(f"Initial velocity XY: ({self.current_velocity[0]:.4f}, {self.current_velocity[1]:.4f}) m/s")
        ui.text("GREEN target contact (fixed)")
        ui.text(f"  X={target[0]:.4f}  Y={target[1]:.4f}  Z={target[2]:.4f} m")
        ui.text("RED impact contact (optimized)")
        ui.text(
            f"  X={self.current_contact_point[0]:.4f}  "
            f"Y={self.current_contact_point[1]:.4f}  Z={self.current_contact_point[2]:.4f} m"
        )
        ui.text(f"RED - GREEN: dX={delta[0]:+.4f}  dY={delta[1]:+.4f}  dZ={delta[2]:+.4f} m")
        ui.text(f"Contact-point distance: {distance:.4f} m")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
