# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Large, fixed particle-to-analytic-primitive soft-contact scene.

A 4096-particle curtain moves along +X and strikes a 4-by-8 wall of static
analytic primitives. Particle-box soft contact is the dominant workload;
capsules, cylinders, and spheres provide a smaller curved-surface sample.

Run from the repository root::

    uv run python diffsim/particle_primitive_wall.py
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

PARTICLE_COUNT = 4096
PARTICLE_RADIUS = 0.012
PARTICLE_MASS = 0.05
PARTICLE_SPEED = 1.5
PARTICLE_START_X = -0.9

SHAPE_COLUMNS = 8
SHAPE_ROWS = 4
SHAPE_COUNT = SHAPE_COLUMNS * SHAPE_ROWS
SHAPE_SPACING = 0.9
WALL_CENTER_Z = 1.7
SHAPE_RADIUS = 0.3
CAPSULE_HALF_HEIGHT = 0.08
CYLINDER_HALF_HEIGHT = SHAPE_RADIUS

PATCH_COLUMNS = 16
PATCH_ROWS = 8
PARTICLES_PER_SHAPE = PATCH_COLUMNS * PATCH_ROWS
PATCH_HALF_EXTENT = 0.18

FPS = 60
SIM_SUBSTEPS = 8
FRAME_DT = 1.0 / FPS
SIM_DT = FRAME_DT / SIM_SUBSTEPS
CYCLE_FRAMES = 75

CONTACT_KE = 2.5e3
CONTACT_KD = 5.0
SOFT_CONTACT_MARGIN = 0.005
CONTACT_SNAPSHOT_PENETRATION = 0.002

# Curved primitives are deliberately a minority so particle-box contact remains dominant.
CAPSULE_INDICES = frozenset((5, 10, 21, 26))
CYLINDER_INDICES = frozenset((12, 19))
SPHERE_INDICES = frozenset((14, 17))

BOX_COLOR = (0.18, 0.42, 0.82)
CAPSULE_COLOR = (0.95, 0.55, 0.16)
CYLINDER_COLOR = (0.22, 0.72, 0.42)
SPHERE_COLOR = (0.88, 0.22, 0.18)


def _shape_kind(shape_index: int) -> str:
    """Return the fixed analytic primitive type for one wall cell."""
    if shape_index in CAPSULE_INDICES:
        return "capsule"
    if shape_index in CYLINDER_INDICES:
        return "cylinder"
    if shape_index in SPHERE_INDICES:
        return "sphere"
    return "box"


def _wall_center(shape_index: int) -> tuple[float, float]:
    """Return the Y/Z center of one cell in the fixed obstacle wall."""
    column = shape_index % SHAPE_COLUMNS
    row = shape_index // SHAPE_COLUMNS
    y = (column - 0.5 * (SHAPE_COLUMNS - 1)) * SHAPE_SPACING
    z = WALL_CENTER_Z + (row - 0.5 * (SHAPE_ROWS - 1)) * SHAPE_SPACING
    return y, z


def _build_particle_curtain() -> tuple[np.ndarray, np.ndarray]:
    """Build 32 separated particle patches, one aimed at each wall primitive."""
    local_y = np.linspace(-PATCH_HALF_EXTENT, PATCH_HALF_EXTENT, PATCH_COLUMNS, dtype=np.float32)
    local_z = np.linspace(-PATCH_HALF_EXTENT, PATCH_HALF_EXTENT, PATCH_ROWS, dtype=np.float32)

    positions = np.empty((PARTICLE_COUNT, 3), dtype=np.float32)
    velocities = np.empty((PARTICLE_COUNT, 3), dtype=np.float32)
    particle_index = 0
    for shape_index in range(SHAPE_COUNT):
        center_y, center_z = _wall_center(shape_index)
        for z_offset in local_z:
            for y_offset in local_y:
                positions[particle_index] = (PARTICLE_START_X, center_y + y_offset, center_z + z_offset)
                velocities[particle_index] = (PARTICLE_SPEED, 0.0, 0.0)
                particle_index += 1

    if particle_index != PARTICLE_COUNT:
        raise RuntimeError(f"Particle curtain construction produced {particle_index} particles")
    return positions, velocities


def build_contact_snapshot() -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed state where every particle touches its assigned primitive."""
    positions, velocities = _build_particle_curtain()
    particle_index = 0
    for shape_index in range(SHAPE_COUNT):
        center_y, center_z = _wall_center(shape_index)
        kind = _shape_kind(shape_index)
        for _ in range(PARTICLES_PER_SHAPE):
            local_y = float(positions[particle_index, 1] - center_y)
            local_z = float(positions[particle_index, 2] - center_z)

            if kind == "box":
                surface_x = -SHAPE_RADIUS
            elif kind == "cylinder":
                surface_x = -np.sqrt(max(SHAPE_RADIUS**2 - local_y**2, 0.0))
            else:
                cap_z = max(abs(local_z) - CAPSULE_HALF_HEIGHT, 0.0) if kind == "capsule" else local_z
                surface_x = -np.sqrt(max(SHAPE_RADIUS**2 - local_y**2 - cap_z**2, 0.0))

            positions[particle_index, 0] = surface_x - PARTICLE_RADIUS + CONTACT_SNAPSHOT_PENETRATION
            particle_index += 1

    return positions, velocities


def build_model(device=None, requires_grad: bool = False) -> newton.Model:
    """Build the fixed model using only particle-static-shape soft contact."""
    particle_q, particle_qd = _build_particle_curtain()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.add_particles(
        pos=particle_q.tolist(),
        vel=particle_qd.tolist(),
        mass=[PARTICLE_MASS] * PARTICLE_COUNT,
        radius=[PARTICLE_RADIUS] * PARTICLE_COUNT,
    )

    shape_cfg = newton.ModelBuilder.ShapeConfig(
        ke=CONTACT_KE,
        kd=CONTACT_KD,
        kf=0.0,
        mu=0.0,
        has_shape_collision=False,
        has_particle_collision=True,
    )

    for shape_index in range(SHAPE_COUNT):
        center_y, center_z = _wall_center(shape_index)
        xform = wp.transform(wp.vec3(0.0, center_y, center_z), wp.quat_identity())
        kind = _shape_kind(shape_index)
        common = {
            "body": -1,
            "xform": xform,
            "cfg": shape_cfg,
            "label": f"{kind}_{shape_index}",
        }
        if kind == "capsule":
            builder.add_shape_capsule(
                **common,
                radius=SHAPE_RADIUS,
                half_height=CAPSULE_HALF_HEIGHT,
                color=CAPSULE_COLOR,
            )
        elif kind == "cylinder":
            builder.add_shape_cylinder(
                **common,
                radius=SHAPE_RADIUS,
                half_height=CYLINDER_HALF_HEIGHT,
                color=CYLINDER_COLOR,
            )
        elif kind == "sphere":
            builder.add_shape_sphere(**common, radius=SHAPE_RADIUS, color=SPHERE_COLOR)
        else:
            builder.add_shape_box(
                **common,
                hx=SHAPE_RADIUS,
                hy=SHAPE_RADIUS,
                hz=SHAPE_RADIUS,
                color=BOX_COLOR,
            )

    model = builder.finalize(device=device, requires_grad=requires_grad)
    model.particle_grid = None
    model.soft_contact_ke = CONTACT_KE
    model.soft_contact_kd = CONTACT_KD
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    model.soft_contact_restitution = 0.0
    return model


class Example:
    """Run and render the fixed particle-primitive wall scene."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.cycle_frame = 0
        self.test_mode = args.test
        self.max_soft_contact_count = 0

        self.model = build_model(device=args.device)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            enable_soft_contact_candidates=getattr(args, "soft_contact_candidates", False),
            enable_fused_soft_contact=getattr(args, "fused_soft_contact", False),
            requires_grad=False,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.control = self.model.control(requires_grad=False)
        self.state_0 = self.model.state(requires_grad=False)
        self.state_1 = self.model.state(requires_grad=False)

        self.candidate_count = self.collision_pipeline.soft_rigid_contact_pair_count
        self.expected_candidate_count = PARTICLE_COUNT * SHAPE_COUNT

        self.viewer.show_particles = True
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-6.0, -7.0, 4.2), pitch=-18.0, yaw=49.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 70.0

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
            self.reset()

        print("ParticlePrimitiveWall")
        print(f"  particles                 : {self.model.particle_count}")
        print(f"  analytic primitive shapes : {self.model.shape_count}")
        print("  shape mix                 : 24 box, 4 capsule, 2 cylinder, 2 sphere")
        print("  enabled contact type       : soft particle-static-shape")
        print(f"  candidate pairs per step   : {self.candidate_count}")
        print(
            "  candidate compaction       : "
            f"{'enabled' if self.collision_pipeline.enable_soft_contact_candidates else 'disabled'}"
        )
        print(
            "  fused soft contact         : "
            f"{'enabled' if self.collision_pipeline.enable_fused_soft_contact else 'disabled'}"
        )
        print(f"  soft contact capacity      : {self.contacts.soft_contact_max}")

    def reset(self):
        """Restore the fixed initial particle curtain without rebuilding the model."""
        wp.copy(self.state_0.particle_q, self.model.particle_q)
        wp.copy(self.state_0.particle_qd, self.model.particle_qd)
        self.state_0.clear_forces()
        self.cycle_frame = 0

    def simulate(self):
        """Advance one rendered frame using the standard semi-implicit loop."""
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Run one frame and periodically restart the approach-impact-bounce cycle."""
        if self.cycle_frame >= CYCLE_FRAMES:
            self.reset()

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        if self.test_mode:
            count = min(int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_max)
            self.max_soft_contact_count = max(self.max_soft_contact_count, count)

        self.cycle_frame += 1
        self.sim_time += FRAME_DT

    def render(self):
        """Render through the repository's standard viewer state path."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that only the intended large soft-contact workload is present."""
        assert self.model.particle_count == PARTICLE_COUNT
        assert self.model.shape_count == SHAPE_COUNT
        assert self.model.body_count == 0
        assert self.model.tri_count == 0
        assert self.model.edge_count == 0
        assert self.model.tet_count == 0
        assert self.model.spring_count == 0
        assert self.model.particle_grid is None
        assert self.model.shape_contact_pair_count == 0
        assert self.candidate_count == self.expected_candidate_count
        assert self.contacts.soft_contact_max == self.expected_candidate_count
        assert self.max_soft_contact_count > 0


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--soft-contact-candidates",
        action="store_true",
        help="Enable BVH candidate compaction before soft-contact generation.",
    )
    parser.add_argument(
        "--fused-soft-contact",
        action="store_true",
        help="Fuse deferred soft-contact detection and SemiImplicit response.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
