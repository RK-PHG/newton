# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualize a fixed two-stone curling collision on a low-friction lane.

The blue stone is thrown obliquely into the stationary red stone.  Its score
increases as its planar distance to the center of the house decreases.  This
file intentionally contains only the forward scene; it is the baseline for a
later differentiable-simulation experiment.

Run from the repository root::

    uv run python diffsim/curling_sdf_scene.py --viewer gl
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

newton.use_coord_layout_targets = True

OPPONENT_BODY = 0
OUR_BODY = 1

FPS = 60
SIM_SUBSTEPS = 12
CYCLE_SECONDS = 3.5

LANE_CENTER_X = 0.25
LANE_HALF_LENGTH = 3.25
LANE_HALF_WIDTH = 1.05
HOUSE_CENTER = np.array([2.35, 0.0], dtype=np.float32)
HOUSE_RADIUS = 0.58

STONE_RADIUS = 0.18
STONE_HALF_HEIGHT = 0.055
STONE_CENTER_HEIGHT = STONE_HALF_HEIGHT + 0.003
OPPONENT_START = wp.vec3(0.65, 0.52, STONE_CENTER_HEIGHT)
OUR_START = wp.vec3(-2.35, 0.20, STONE_CENTER_HEIGHT)
OUR_INITIAL_VELOCITY = wp.vec3(2.00, 0.0, 0.0)


def _transform(position: tuple[float, float, float] | wp.vec3) -> wp.transform:
    return wp.transform(position, wp.quat_identity())


def _add_lane_visuals(builder: newton.ModelBuilder) -> None:
    """Add non-colliding lane, boundary lines, and house rings."""
    builder.add_shape_box(
        -1,
        xform=_transform((LANE_CENTER_X, 0.0, -0.012)),
        hx=LANE_HALF_LENGTH,
        hy=LANE_HALF_WIDTH,
        hz=0.012,
        as_site=True,
        color=wp.vec3(0.62, 0.76, 0.82),
        label="ice_lane",
    )

    for y in (-LANE_HALF_WIDTH, LANE_HALF_WIDTH):
        builder.add_shape_box(
            -1,
            xform=_transform((LANE_CENTER_X, y, 0.002)),
            hx=LANE_HALF_LENGTH,
            hy=0.018,
            hz=0.004,
            as_site=True,
            color=wp.vec3(0.92, 0.96, 0.98),
            label=f"side_line_{y:+.0f}",
        )

    # Nested non-colliding disks form a clearly visible curling house.
    rings = (
        (HOUSE_RADIUS, 0.0020, wp.vec3(0.08, 0.30, 0.78), "house_blue"),
        (0.42, 0.0026, wp.vec3(0.94, 0.96, 0.96), "house_white"),
        (0.25, 0.0032, wp.vec3(0.84, 0.12, 0.12), "house_red"),
        (0.075, 0.0038, wp.vec3(0.96, 0.96, 0.96), "house_button"),
    )
    for radius, z, color, label in rings:
        builder.add_shape_cylinder(
            -1,
            xform=_transform((float(HOUSE_CENTER[0]), float(HOUSE_CENTER[1]), z)),
            radius=radius,
            half_height=0.001,
            as_site=True,
            color=color,
            label=label,
        )

    # Hog and tee lines make direction and distance easy to judge in the viewer.
    for x, color, label in (
        (-0.85, wp.vec3(0.84, 0.12, 0.12), "hog_line"),
        (float(HOUSE_CENTER[0]), wp.vec3(0.08, 0.30, 0.78), "tee_line"),
    ):
        builder.add_shape_box(
            -1,
            xform=_transform((x, 0.0, 0.0045)),
            hx=0.012,
            hy=LANE_HALF_WIDTH,
            hz=0.001,
            as_site=True,
            color=color,
            label=label,
        )


def _add_stone(
    builder: newton.ModelBuilder,
    mesh: newton.Mesh,
    position: wp.vec3,
    body_color: wp.vec3,
    handle_color: wp.vec3,
    label: str,
) -> int:
    """Add one mesh-backed cylinder and a non-colliding orientation handle."""
    body = builder.add_body(xform=_transform(position), label=label)
    stone_cfg = newton.ModelBuilder.ShapeConfig(
        density=1800.0,
        ke=2.0e5,
        kd=2.0e3,
        kf=2.0e3,
        mu=0.025,
        mu_torsional=0.0005,
        mu_rolling=0.0001,
        restitution=0.08,
        gap=0.002,
    )
    builder.add_shape_mesh(body, mesh=mesh, cfg=stone_cfg, color=body_color, label=f"{label}_sdf")
    builder.add_shape_box(
        body,
        xform=_transform((0.0, 0.0, STONE_HALF_HEIGHT + 0.022)),
        hx=0.070,
        hy=0.022,
        hz=0.016,
        as_site=True,
        color=handle_color,
        label=f"{label}_handle",
    )
    return body


class Example:
    """Run and visualize the fixed curling collision baseline."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.cycle_frame = 0
        self.cycle_frames = int(CYCLE_SECONDS * FPS)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.002
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e5,
            kd=2.0e3,
            kf=2.0e3,
            mu=0.025,
            mu_torsional=0.0005,
            mu_rolling=0.0001,
            restitution=0.0,
            gap=0.002,
        )
        builder.add_ground_plane(cfg=ground_cfg)
        _add_lane_visuals(builder)

        stone_mesh = newton.Mesh.create_cylinder(
            STONE_RADIUS,
            STONE_HALF_HEIGHT,
            up_axis=newton.Axis.Z,
            segments=48,
            compute_normals=True,
            compute_uvs=False,
        )
        # Mesh-backed collision shapes own their SDF; ShapeConfig.sdf_* is
        # reserved for primitive SDF generation.
        stone_mesh.build_sdf(max_resolution=64, device=args.device)
        opponent = _add_stone(
            builder,
            stone_mesh,
            OPPONENT_START,
            wp.vec3(0.78, 0.10, 0.08),
            wp.vec3(0.98, 0.82, 0.18),
            "opponent_stone",
        )
        ours = _add_stone(
            builder,
            stone_mesh,
            OUR_START,
            wp.vec3(0.06, 0.30, 0.82),
            wp.vec3(0.92, 0.95, 0.98),
            "our_stone",
        )
        if opponent != OPPONENT_BODY or ours != OUR_BODY:
            raise RuntimeError("Curling body order no longer matches score constants")
        builder.body_qd[OUR_BODY] = wp.spatial_vector(
            OUR_INITIAL_VELOCITY[0],
            OUR_INITIAL_VELOCITY[1],
            OUR_INITIAL_VELOCITY[2],
            0.0,
            0.0,
            1.25,
        )

        self.model = builder.finalize(device=args.device)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=10,
            enable_restitution=True,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_body_q = wp.clone(self.state_0.body_q)
        self.initial_body_qd = wp.clone(self.state_0.body_qd)

        self.distance_to_house = 0.0
        self.score = 0.0
        self.viewer.set_model(self.model)
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(0.0, -4.6, 4.4), pitch=-43.0, yaw=90.0)

        self.capture()
        self.reset_cycle()

    def capture(self) -> None:
        """Capture one display frame of simulation when running on CUDA."""
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        """Advance one display frame with sufficiently small contact steps."""
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset_cycle(self) -> None:
        """Restore the same throw so the collision repeats for inspection."""
        wp.copy(self.state_0.body_q, self.initial_body_q)
        wp.copy(self.state_0.body_qd, self.initial_body_qd)
        wp.copy(self.state_1.body_q, self.initial_body_q)
        wp.copy(self.state_1.body_qd, self.initial_body_qd)
        self.state_0.clear_forces()
        self.state_1.clear_forces()
        self.cycle_frame = 0

    def step(self) -> None:
        """Advance or restart the fixed throw."""
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.cycle_frame += 1
        if self.cycle_frame >= self.cycle_frames:
            self.reset_cycle()

    def _update_score(self) -> None:
        pose = self.state_0.body_q.numpy()[OUR_BODY]
        delta = pose[:2] - HOUSE_CENTER
        self.distance_to_house = float(np.linalg.norm(delta))
        self.score = math.exp(-((self.distance_to_house / HOUSE_RADIUS) ** 2))

    def gui(self, ui) -> None:
        """Expose the fixed throw and its eventual optimization objective."""
        ui.text("Curling SDF scene")
        ui.separator()
        ui.text(
            "Our initial velocity [m/s]: "
            f"({OUR_INITIAL_VELOCITY[0]:.2f}, {OUR_INITIAL_VELOCITY[1]:.2f}, {OUR_INITIAL_VELOCITY[2]:.2f})"
        )
        ui.text(f"Distance to house center: {self.distance_to_house:.3f} m")
        ui.text(f"Normalized score: {self.score:.4f}")
        ui.text("Score = exp(-(distance / house radius)^2)")
        ui.separator()
        ui.text("Blue: our thrown stone")
        ui.text("Red: stationary opponent stone")

    def render(self) -> None:
        self._update_score()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if not math.isfinite(self.distance_to_house) or not math.isfinite(self.score):
            raise AssertionError("Curling score became non-finite")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
