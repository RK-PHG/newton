# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Refine an Allegro grasp through differentiable rigid contact.

This example adapts the multi-witness grasp-refinement task from
"Robust Differentiable Collision Detection for General Objects" to Newton's
in-pipeline RS1Dist implementation. Four fingertip joint chains are optimized
so their object-side witness points reach four reference contacts on a
non-convex star-prism mesh.

Run from the repository root::

    uv run python diffsim/differentiable_contact_point_target.py --viewer gl
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

newton.use_coord_layout_targets = True

LOG_DIR = Path(__file__).resolve().parent / "log"
FINGER_NAMES = ("index", "middle", "ring", "thumb")
FINGER_BODY_SUFFIXES = (
    "/index_link_3",
    "/middle_link_3",
    "/ring_link_3",
    "/thumb_link_3",
)
GEAR_CENTER = np.asarray((0.000, -0.170, 0.580), dtype=np.float32)
REFERENCE_FINGERS = np.asarray(
    (
        (-0.198, 0.331, 0.283, 1.231),
        (0.160, 0.146, 0.321, 1.226),
        (0.114, 0.131, 0.658, 1.231),
        (1.275, 0.700, 1.100, 0.900),
    ),
    dtype=np.float32,
)
INITIAL_FINGERS = np.asarray(
    (
        (-0.198, 0.331, 0.246, 1.231),
        (0.160, 0.146, 0.271, 1.226),
        (0.052, 0.131, 0.658, 1.231),
        (1.325, 0.700, 1.100, 0.900),
    ),
    dtype=np.float32,
)


def _format_values(values: np.ndarray | list[float]) -> str:
    """Format every component without suppressing small gradients."""
    return "[" + ", ".join(f"{float(value):+.9e}" for value in np.asarray(values).flat) + "]"


def _create_gear_mesh(tooth_count: int = 14) -> newton.Mesh:
    """Create a closed non-convex star prism with unambiguous SDF winding."""
    root_radius = 0.032
    outer_radius = 0.040
    half_thickness = 0.035
    segments = tooth_count * 4
    angles = np.arange(segments, dtype=np.float32) * (2.0 * np.pi / segments)
    phase = np.arange(segments) % 4
    radii = np.where((phase == 1) | (phase == 2), outer_radius, root_radius)
    outer_xy = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).astype(np.float32)

    vertices = [(float(x), float(y), -half_thickness) for x, y in outer_xy]
    vertices.extend((float(x), float(y), half_thickness) for x, y in outer_xy)
    vertices.extend(((0.0, 0.0, -half_thickness), (0.0, 0.0, half_thickness)))

    outer_bottom = 0
    outer_top = segments
    bottom_center = 2 * segments
    top_center = bottom_center + 1

    triangles: list[tuple[int, int, int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        ob0, ob1 = outer_bottom + index, outer_bottom + following
        ot0, ot1 = outer_top + index, outer_top + following
        triangles.extend(((bottom_center, ob1, ob0), (top_center, ot0, ot1)))
        triangles.extend(((ob0, ob1, ot0), (ob1, ot1, ot0)))

    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
        compute_inertia=False,
    )


def _joint_coordinate_indices(builder: newton.ModelBuilder) -> list[int]:
    """Return the Allegro revolute coordinates in finger-chain order."""
    coordinates = []
    for joint, joint_type in enumerate(builder.joint_type):
        if joint_type == newton.JointType.REVOLUTE:
            coordinates.append(builder.joint_q_start[joint])
    return coordinates


def _grasp_joint_q(
    builder: newton.ModelBuilder,
    finger_values: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Insert four finger configurations into the articulation coordinates."""
    coordinate_indices = _joint_coordinate_indices(builder)
    joint_q = np.asarray(builder.joint_q, dtype=np.float32)
    if len(coordinate_indices) != finger_values.size:
        raise RuntimeError(
            f"Expected {finger_values.size} Allegro revolute coordinates, found {len(coordinate_indices)}"
        )
    joint_q[coordinate_indices] = finger_values.reshape(-1)
    return joint_q, coordinate_indices


def _find_fingertip_collision_shapes(builder: newton.ModelBuilder) -> list[int]:
    """Find the four collision proxies attached to distal Allegro links."""
    body_by_suffix = {}
    for body, label in enumerate(builder.body_label):
        for suffix in FINGER_BODY_SUFFIXES:
            if label.endswith(suffix):
                body_by_suffix[suffix] = body

    missing = [suffix for suffix in FINGER_BODY_SUFFIXES if suffix not in body_by_suffix]
    if missing:
        raise RuntimeError(f"Allegro asset is missing fingertip bodies: {missing}")

    fingertip_shapes = []
    for suffix in FINGER_BODY_SUFFIXES:
        body = body_by_suffix[suffix]
        matches = [
            shape
            for shape, shape_body in enumerate(builder.shape_body)
            if shape_body == body and "/collisions" in builder.shape_label[shape]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one collision shape on {suffix}, found {len(matches)}")
        fingertip_shapes.append(matches[0])
    return fingertip_shapes


def _find_hand_collision_shapes(builder: newton.ModelBuilder) -> list[int]:
    """Return every physical collision proxy attached to the Allegro hand."""
    return [
        shape
        for shape, body in enumerate(builder.shape_body)
        if body >= 0 and "/collisions" in builder.shape_label[shape]
    ]


@wp.kernel(enable_backward=False)
def find_minimum_fingertip_distances(
    contact_count: wp.array[int],
    distance: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
    finger_slot_by_shape: wp.array[int],
    minimum_distances: wp.array[float],
):
    """Find the closest contact candidate for each fingertip."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    slot = finger_slot_by_shape[shape0[contact]]
    if slot < 0:
        slot = finger_slot_by_shape[shape1[contact]]
    if slot >= 0:
        wp.atomic_min(minimum_distances, slot, distance[contact])


@wp.kernel(enable_backward=False)
def find_closest_fingertip_contacts(
    contact_count: wp.array[int],
    distance: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
    finger_slot_by_shape: wp.array[int],
    minimum_distances: wp.array[float],
    contact_indices: wp.array[int],
):
    """Resolve each fingertip's closest distance to one contact index."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    slot = finger_slot_by_shape[shape0[contact]]
    if slot < 0:
        slot = finger_slot_by_shape[shape1[contact]]
    if slot >= 0 and distance[contact] <= minimum_distances[slot] + 1.0e-7:
        wp.atomic_min(contact_indices, slot, contact)


@wp.kernel
def gather_closest_fingertip_witnesses(
    contact_count: wp.array[int],
    contact_indices: wp.array[int],
    distance: wp.array[float],
    point0_world: wp.array[wp.vec3],
    point1_world: wp.array[wp.vec3],
    shape0: wp.array[int],
    finger_slot_by_shape: wp.array[int],
    object_witnesses: wp.array[wp.vec3],
    fingertip_witnesses: wp.array[wp.vec3],
    selected_distances: wp.array[float],
    witness_counts: wp.array[int],
):
    """Gather one exact object/fingertip witness pair per finger."""
    finger = wp.tid()
    contact = contact_indices[finger]
    if contact >= contact_count[0]:
        return

    if finger_slot_by_shape[shape0[contact]] == finger:
        fingertip_witnesses[finger] = point0_world[contact]
        object_witnesses[finger] = point1_world[contact]
    else:
        object_witnesses[finger] = point0_world[contact]
        fingertip_witnesses[finger] = point1_world[contact]
    selected_distances[finger] = distance[contact]
    witness_counts[finger] = 1


@wp.kernel
def grasp_witness_loss(
    witnesses: wp.array[wp.vec3],
    witness_counts: wp.array[int],
    targets: wp.array[wp.vec3],
    missing_contact_penalty: float,
    target_loss: wp.array[float],
):
    """Match each selected fingertip witness to its target on the gear surface."""
    finger = wp.tid()
    if witness_counts[finger] > 0:
        delta = witnesses[finger] - targets[finger]
        wp.atomic_add(target_loss, 0, wp.dot(delta, delta) / 4.0)
    else:
        wp.atomic_add(target_loss, 0, missing_contact_penalty / 4.0)


@wp.kernel
def grasp_contact_constraints(
    contact_count: wp.array[int],
    distance: wp.array[float],
    shape0: wp.array[int],
    shape1: wp.array[int],
    finger_slot_by_shape: wp.array[int],
    fingertip_distance_sums: wp.array[float],
    nonfinger_penetration_loss: wp.array[float],
):
    """Close fingertip gaps and penalize the supplied safety-contact set."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    slot = finger_slot_by_shape[shape0[contact]]
    if slot < 0:
        slot = finger_slot_by_shape[shape1[contact]]
    signed_distance = distance[contact]
    if slot >= 0:
        wp.atomic_add(fingertip_distance_sums, slot, signed_distance)
    elif signed_distance < 0.0:
        wp.atomic_add(nonfinger_penetration_loss, 0, signed_distance * signed_distance)


@wp.kernel
def closest_fingertip_gap_loss(
    selected_distances: wp.array[float],
    witness_counts: wp.array[int],
    fingertip_gap_loss: wp.array[float],
):
    """Drive each selected fingertip witness pair to zero separation."""
    finger = wp.tid()
    if witness_counts[finger] > 0:
        distance = selected_distances[finger]
        wp.atomic_add(fingertip_gap_loss, 0, distance * distance / 4.0)


@wp.kernel
def combine_grasp_loss(
    target_loss: wp.array[float],
    fingertip_gap_loss: wp.array[float],
    nonfinger_penetration_loss: wp.array[float],
    target_weight: float,
    fingertip_gap_weight: float,
    nonfinger_penetration_weight: float,
    loss: wp.array[float],
):
    """Combine independently observable grasp-refinement constraints."""
    loss[0] = (
        target_weight * target_loss[0]
        + fingertip_gap_weight * fingertip_gap_loss[0]
        + nonfinger_penetration_weight * nonfinger_penetration_loss[0]
    )


class Example:
    """Optimize an articulated four-finger grasp using contact witnesses."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.display_frame = 0
        self.train_iter = 0
        self.frames_per_iteration = 3 if args.viewer == "gl" and not args.headless else 1
        self.max_iterations = max(1, (args.num_frames + self.frames_per_iteration - 1) // self.frames_per_iteration)
        self.learning_rate = 0.5
        self.target_weight = 1.0
        self.fingertip_gap_weight = 10.0
        self.nonfinger_penetration_weight = 25.0
        self.current_loss = 0.0
        self.current_target_loss = 0.0
        self.current_fingertip_gap_loss = 0.0
        self.current_nonfinger_penetration_loss = 0.0
        self.current_max_penetration = 0.0
        self.current_gradient_norm = 0.0
        self.current_joint_q = np.zeros(16, dtype=np.float32)
        self.current_joint_gradient = np.zeros(16, dtype=np.float32)
        self.current_witnesses = np.zeros((4, 3), dtype=np.float32)
        self.current_fingertip_witnesses = np.zeros((4, 3), dtype=np.float32)
        self.current_fingertip_distances = np.zeros(4, dtype=np.float32)
        self.current_counts = np.zeros(4, dtype=np.int32)
        self.initial_loss = np.inf
        self.initial_max_fingertip_gap = np.inf
        self.best_loss = np.inf
        self.best_joint_q = None
        self.best_target_loss = np.inf
        self.best_fingertip_gap_loss = np.inf
        self.best_nonfinger_penetration_loss = np.inf
        self.best_max_penetration = np.inf
        self.best_witnesses = np.zeros((4, 3), dtype=np.float32)
        self.best_fingertip_witnesses = np.zeros((4, 3), dtype=np.float32)
        self.best_fingertip_distances = np.zeros(4, dtype=np.float32)
        self.best_counts = np.zeros(4, dtype=np.int32)
        self.iterations_without_improvement = 0
        self.early_stop_patience = 30
        self.penetration_tolerance = 1.0e-6
        self.current_max_joint_delta_degrees = 0.0
        self.converged = False

        asset_path = newton.utils.download_asset("wonik_allegro")
        asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.015
        builder.add_usd(
            asset_file,
            enable_self_collisions=False,
            ignore_paths=[
                ".*Dummy",
                ".*CollisionPlane",
                ".*DexCube.*",
                ".*goal_marker.*",
                ".*/ground/.*",
            ],
            hide_collision_shapes=True,
        )

        reference_joint_q, self.optimized_coordinates = _grasp_joint_q(builder, REFERENCE_FINGERS)
        initial_joint_q, _ = _grasp_joint_q(builder, INITIAL_FINGERS)
        builder.joint_q = reference_joint_q.tolist()
        fingertip_shapes = _find_fingertip_collision_shapes(builder)
        hand_collision_shapes = _find_hand_collision_shapes(builder)
        if not hand_collision_shapes:
            raise RuntimeError("Allegro asset has no hand collision shapes")
        for shape in hand_collision_shapes:
            builder.shape_gap[shape] = 0.012
        for shape in fingertip_shapes:
            builder.shape_gap[shape] = 0.15

        gear_mesh = _create_gear_mesh()
        gear_mesh.build_sdf(device=args.device, max_resolution=72)
        gear_shape = builder.add_shape_mesh(
            body=-1,
            mesh=gear_mesh,
            xform=wp.transform(wp.vec3(*GEAR_CENTER), wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(gap=0.012, margin=0.001),
            color=wp.vec3(0.12, 0.43, 0.88),
            label="fixed_nonconvex_gear_sdf",
        )

        self.model = builder.finalize(device=args.device, requires_grad=True)
        fingertip_pair_values = np.asarray([(shape, gear_shape) for shape in fingertip_shapes], dtype=np.int32)
        fingertip_pairs = wp.array(fingertip_pair_values, dtype=wp.vec2i, device=self.model.device)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=fingertip_pairs,
            reduce_contacts=True,
            differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
                global_sample_count=768,
                local_sample_count=24,
            ),
        )
        self.contacts = self.collision_pipeline.contacts()
        nonfinger_shapes = [shape for shape in hand_collision_shapes if shape not in fingertip_shapes]
        nonfinger_pair_values = np.asarray([(shape, gear_shape) for shape in nonfinger_shapes], dtype=np.int32)
        nonfinger_pairs = wp.array(nonfinger_pair_values, dtype=wp.vec2i, device=self.model.device)
        self.safety_collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=nonfinger_pairs,
            reduce_contacts=True,
            differentiable_rigid_contact_config=newton.CollisionPipeline.DifferentiableRigidContactConfig(
                global_sample_count=512,
                local_sample_count=16,
            ),
        )
        self.safety_contacts = self.safety_collision_pipeline.contacts()
        self.state = self.model.state(requires_grad=True)

        finger_slot_by_shape = np.full(self.model.shape_count, -1, dtype=np.int32)
        for slot, shape in enumerate(fingertip_shapes):
            finger_slot_by_shape[shape] = slot
        self.finger_slot_by_shape = wp.array(finger_slot_by_shape, dtype=int, device=self.model.device)
        self.target_radii = wp.full(4, 0.0025, dtype=float, device=self.model.device)
        self.actual_radii = wp.full(4, 0.0020, dtype=float, device=self.model.device)
        self.fingertip_radii = wp.full(4, 0.0018, dtype=float, device=self.model.device)
        self.target_colors = wp.full(4, wp.vec3(1.00, 0.85, 0.05), dtype=wp.vec3, device=self.model.device)
        self.actual_colors = wp.full(4, wp.vec3(1.00, 0.20, 0.06), dtype=wp.vec3, device=self.model.device)
        self.fingertip_colors = wp.full(4, wp.vec3(0.05, 0.95, 1.00), dtype=wp.vec3, device=self.model.device)
        self.witness_line_colors = wp.full(4, wp.vec3(1.00, 1.00, 1.00), dtype=wp.vec3, device=self.model.device)

        contact_max = self.contacts.rigid_contact_max
        safety_contact_max = self.safety_contacts.rigid_contact_max
        self.distance = wp.empty(contact_max, dtype=float, device=self.model.device, requires_grad=True)
        self.safety_distance = wp.empty(
            safety_contact_max,
            dtype=float,
            device=self.model.device,
            requires_grad=True,
        )
        self.point0_world = wp.empty(contact_max, dtype=wp.vec3, device=self.model.device, requires_grad=True)
        self.point1_world = wp.empty(contact_max, dtype=wp.vec3, device=self.model.device, requires_grad=True)
        self.minimum_fingertip_distances = wp.full(4, 1.0e6, dtype=float, device=self.model.device)
        self.closest_contact_indices = wp.full(4, contact_max, dtype=int, device=self.model.device)
        self.witness_counts = wp.zeros(4, dtype=int, device=self.model.device)
        self.witnesses = wp.zeros(4, dtype=wp.vec3, device=self.model.device, requires_grad=True)
        self.fingertip_witnesses = wp.zeros(4, dtype=wp.vec3, device=self.model.device, requires_grad=True)
        self.selected_fingertip_distances = wp.zeros(4, dtype=float, device=self.model.device, requires_grad=True)
        self.fingertip_distance_sums = wp.zeros(4, dtype=float, device=self.model.device, requires_grad=True)
        self.target_loss = wp.zeros(1, dtype=float, device=self.model.device, requires_grad=True)
        self.fingertip_gap_loss = wp.zeros(1, dtype=float, device=self.model.device, requires_grad=True)
        self.nonfinger_penetration_loss = wp.zeros(1, dtype=float, device=self.model.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=self.model.device, requires_grad=True)
        self.loss_gradient_trace = wp.zeros(1, dtype=float, device=self.model.device)
        self.witness_gradient_trace = wp.zeros(4, dtype=wp.vec3, device=self.model.device)
        self.fingertip_witness_gradient_trace = wp.zeros(4, dtype=wp.vec3, device=self.model.device)
        self.point0_gradient_trace = wp.zeros(contact_max, dtype=wp.vec3, device=self.model.device)
        self.point1_gradient_trace = wp.zeros(contact_max, dtype=wp.vec3, device=self.model.device)
        self.body_q_gradient_trace = wp.zeros(
            self.model.body_count,
            dtype=wp.transform,
            device=self.model.device,
        )

        self.target_contact_points = self._capture_reference_targets()
        self.targets = wp.array(self.target_contact_points, dtype=wp.vec3, device=self.model.device)
        self.model.joint_q.assign(initial_joint_q)
        self.initial_optimized_joint_q = initial_joint_q[self.optimized_coordinates].copy()
        self._clear_objective_buffers()
        self._create_gradient_log()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.set_model(self.model)
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(0.27, -0.48, 0.72), pitch=-23.0, yaw=129.0)

    def _clear_objective_buffers(self):
        """Reset all arrays written by one objective evaluation."""
        self.minimum_fingertip_distances.fill_(1.0e6)
        self.closest_contact_indices.fill_(self.contacts.rigid_contact_max)
        self.witness_counts.zero_()
        self.witnesses.zero_()
        self.fingertip_witnesses.zero_()
        self.selected_fingertip_distances.zero_()
        self.fingertip_distance_sums.zero_()
        self.target_loss.zero_()
        self.fingertip_gap_loss.zero_()
        self.nonfinger_penetration_loss.zero_()
        self.loss.zero_()

    def _select_closest_witnesses(self):
        """Select and gather one exact witness pair for each fingertip."""
        wp.launch(
            find_minimum_fingertip_distances,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.distance,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.finger_slot_by_shape,
            ],
            outputs=[self.minimum_fingertip_distances],
            device=self.model.device,
            record_tape=False,
        )
        wp.launch(
            find_closest_fingertip_contacts,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.distance,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.finger_slot_by_shape,
                self.minimum_fingertip_distances,
            ],
            outputs=[self.closest_contact_indices],
            device=self.model.device,
            record_tape=False,
        )
        wp.launch(
            gather_closest_fingertip_witnesses,
            dim=4,
            inputs=[
                self.contacts.rigid_contact_count,
                self.closest_contact_indices,
                self.distance,
                self.point0_world,
                self.point1_world,
                self.contacts.rigid_contact_shape0,
                self.finger_slot_by_shape,
            ],
            outputs=[
                self.witnesses,
                self.fingertip_witnesses,
                self.selected_fingertip_distances,
                self.witness_counts,
            ],
            device=self.model.device,
        )

    def _capture_reference_targets(self) -> np.ndarray:
        """Sample reachable targets from the reference pose and current mesh."""
        self._clear_objective_buffers()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.collision_pipeline.collide(self.state, self.contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.contacts,
            out_distance=self.distance,
            out_point0_world=self.point0_world,
            out_point1_world=self.point1_world,
        )
        self._select_closest_witnesses()
        counts = self.witness_counts.numpy()
        if np.any(counts == 0):
            raise RuntimeError(
                f"Reference grasp does not contact every fingertip on the current object: counts={counts.tolist()}"
            )
        self.reference_fingertip_distances = self.selected_fingertip_distances.numpy().copy()
        self.safety_collision_pipeline.collide(self.state, self.safety_contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.safety_contacts,
            out_distance=self.safety_distance,
        )
        safety_count = int(self.safety_contacts.rigid_contact_count.numpy()[0])
        safety_distances = self.safety_distance.numpy()[:safety_count]
        fingertip_count = int(self.contacts.rigid_contact_count.numpy()[0])
        fingertip_distances = self.distance.numpy()[:fingertip_count]
        active_distances = np.concatenate((fingertip_distances, safety_distances))
        self.reference_max_penetration = float(max(0.0, -float(np.min(active_distances))))
        self.reference_deepest_shape = "none"
        if self.reference_max_penetration > 0.0:
            deepest = int(np.argmin(active_distances))
            source_contacts = self.contacts
            source_index = deepest
            if deepest >= fingertip_count:
                source_contacts = self.safety_contacts
                source_index = deepest - fingertip_count
            shape0 = int(source_contacts.rigid_contact_shape0.numpy()[source_index])
            shape1 = int(source_contacts.rigid_contact_shape1.numpy()[source_index])
            hand_shape = shape0 if self.model.shape_body.numpy()[shape0] >= 0 else shape1
            self.reference_deepest_shape = self.model.shape_label[hand_shape]
        if self.reference_max_penetration > 1.0e-6:
            raise RuntimeError(
                "Reference grasp penetrates the gear: "
                f"depth={self.reference_max_penetration:.6e} m, shape={self.reference_deepest_shape}"
            )
        return self.witnesses.numpy().copy()

    def _create_gradient_log(self):
        """Create one gradient log for this process invocation."""
        started_at = datetime.now()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        self.gradient_log_path = LOG_DIR / f"differentiable_allegro_grasp_{timestamp}.log"
        lines = [
            "Differentiable Allegro grasp-refinement gradient log",
            f"started_at={started_at.isoformat(timespec='microseconds')}",
            f"device={self.model.device}",
            "objective=closest surface-witness alignment + fingertip closure + full-hand nonpenetration",
            "gradient_chain=loss -> selected exact witnesses -> RS1Dist contact witnesses -> body_q -> FK -> joint_q",
            f"target_contact_xyz={_format_values(self.target_contact_points)}",
            f"reference_fingertip_signed_distance={_format_values(self.reference_fingertip_distances)}",
            f"reference_max_hand_penetration={self.reference_max_penetration:.12e}",
            f"reference_deepest_hand_shape={self.reference_deepest_shape}",
            f"learning_rate={self.learning_rate:.12e}",
            f"target_weight={self.target_weight:.12e}",
            f"fingertip_gap_weight={self.fingertip_gap_weight:.12e}",
            f"nonfinger_penetration_weight={self.nonfinger_penetration_weight:.12e}",
            "",
        ]
        with self.gradient_log_path.open("x", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines))
        print(f"Gradient log: {self.gradient_log_path}")

    def _write_iteration_log(self):
        body_q = self.state.body_q.numpy()
        body_q_gradient = self.body_q_gradient_trace.numpy()
        point0 = self.point0_world.numpy()
        point1 = self.point1_world.numpy()
        point0_gradient = self.point0_gradient_trace.numpy()
        point1_gradient = self.point1_gradient_trace.numpy()
        distances = self.distance.numpy()
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        safety_contact_count = int(self.safety_contacts.rigid_contact_count.numpy()[0])
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contact_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contact_count]

        lines = [
            "=" * 100,
            f"OPTIMIZATION_ITERATION {self.train_iter:06d} BEGIN",
            "[objective]",
            f"loss={self.current_loss:.12e}",
            f"target_loss={self.current_target_loss:.12e}",
            f"fingertip_gap_loss={self.current_fingertip_gap_loss:.12e}",
            f"nonfingertip_penetration_loss={self.current_nonfinger_penetration_loss:.12e}",
            f"max_penetration={self.current_max_penetration:.12e}",
            f"dL_d_loss={float(self.loss_gradient_trace.numpy()[0]):+.12e}",
            f"target_contact_xyz={_format_values(self.target_contact_points)}",
            f"gear_witness_xyz={_format_values(self.current_witnesses)}",
            f"fingertip_witness_xyz={_format_values(self.current_fingertip_witnesses)}",
            f"selected_signed_distance={_format_values(self.current_fingertip_distances)}",
            f"dL_d_gear_witness={_format_values(self.witness_gradient_trace.numpy())}",
            f"dL_d_fingertip_witness={_format_values(self.fingertip_witness_gradient_trace.numpy())}",
            f"contact_count_by_finger={_format_values(self.current_counts)}",
            f"nonfingertip_contact_count={safety_contact_count}",
            "[optimizer]",
            f"joint_q={_format_values(self.current_joint_q)}",
            f"dL_d_joint_q={_format_values(self.current_joint_gradient)}",
            f"gradient_norm={self.current_gradient_norm:.12e}",
            f"max_joint_delta_degrees={self.current_max_joint_delta_degrees:.12e}",
            f"learning_rate={self.learning_rate:.12e}",
            "[contact_witnesses]",
        ]
        for contact in range(contact_count):
            lines.extend(
                (
                    f"contact={contact} shape0={int(shape0[contact])} shape1={int(shape1[contact])}",
                    f"  signed_distance={float(distances[contact]):+.12e}",
                    f"  point0_world={_format_values(point0[contact])}",
                    f"  dL_d_point0_world={_format_values(point0_gradient[contact])}",
                    f"  point1_world={_format_values(point1[contact])}",
                    f"  dL_d_point1_world={_format_values(point1_gradient[contact])}",
                )
            )
        lines.append("[articulated_gradient_chain]")
        for body in range(self.model.body_count):
            lines.extend(
                (
                    f"body={body} label={self.model.body_label[body]}",
                    f"  body_q={_format_values(body_q[body])}",
                    f"  dL_d_body_q={_format_values(body_q_gradient[body])}",
                )
            )
        lines.extend((f"OPTIMIZATION_ITERATION {self.train_iter:06d} END", ""))
        with self.gradient_log_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(lines))

    def _record_gradient_snapshot(self, sources: list[wp.array], destinations: list[wp.array]):
        """Preserve an intermediate adjoint before its producer consumes it."""

        def snapshot():
            for source, destination in zip(sources, destinations, strict=True):
                destination.assign(source.grad)

        self.tape.record_func(snapshot, sources)

    def _restore_best_state(self):
        """Restore the best accepted articulation and its exact contacts."""
        if self.best_joint_q is None:
            return
        self.model.joint_q.assign(self.best_joint_q)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.collision_pipeline.collide(self.state, self.contacts)
        self.safety_collision_pipeline.collide(self.state, self.safety_contacts)
        self.current_loss = self.best_loss
        self.current_target_loss = self.best_target_loss
        self.current_fingertip_gap_loss = self.best_fingertip_gap_loss
        self.current_nonfinger_penetration_loss = self.best_nonfinger_penetration_loss
        self.current_max_penetration = self.best_max_penetration
        self.current_witnesses = self.best_witnesses.copy()
        self.current_fingertip_witnesses = self.best_fingertip_witnesses.copy()
        self.current_fingertip_distances = self.best_fingertip_distances.copy()
        self.current_counts = self.best_counts.copy()
        self.current_joint_q = self.best_joint_q[self.optimized_coordinates].copy()
        self.current_max_joint_delta_degrees = float(
            np.degrees(np.max(np.abs(self.current_joint_q - self.initial_optimized_joint_q)))
        )
        self.witnesses.assign(self.best_witnesses)
        self.fingertip_witnesses.assign(self.best_fingertip_witnesses)

    def forward(self):
        """Evaluate FK, collision, smooth witnesses, and the grasp loss."""
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self._record_gradient_snapshot([self.state.body_q], [self.body_q_gradient_trace])
        self.collision_pipeline.collide(self.state, self.contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.contacts,
            out_point0_world=self.point0_world,
            out_point1_world=self.point1_world,
            use_smooth_witness_gradients=True,
        )
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.contacts,
            out_distance=self.distance,
        )
        self._record_gradient_snapshot(
            [self.point0_world, self.point1_world],
            [self.point0_gradient_trace, self.point1_gradient_trace],
        )
        self._select_closest_witnesses()
        self._record_gradient_snapshot(
            [self.witnesses, self.fingertip_witnesses],
            [self.witness_gradient_trace, self.fingertip_witness_gradient_trace],
        )
        wp.launch(
            grasp_witness_loss,
            dim=4,
            inputs=[self.witnesses, self.witness_counts, self.targets, 0.01],
            outputs=[self.target_loss],
            device=self.model.device,
        )
        wp.launch(
            closest_fingertip_gap_loss,
            dim=4,
            inputs=[self.selected_fingertip_distances, self.witness_counts],
            outputs=[self.fingertip_gap_loss],
            device=self.model.device,
        )
        self.safety_collision_pipeline.collide(self.state, self.safety_contacts)
        newton.eval_rigid_contact_kinematics(
            self.model,
            self.state,
            self.safety_contacts,
            out_distance=self.safety_distance,
            use_smooth_witness_gradients=True,
        )
        wp.launch(
            grasp_contact_constraints,
            dim=self.safety_contacts.rigid_contact_max,
            inputs=[
                self.safety_contacts.rigid_contact_count,
                self.safety_distance,
                self.safety_contacts.rigid_contact_shape0,
                self.safety_contacts.rigid_contact_shape1,
                self.finger_slot_by_shape,
            ],
            outputs=[self.fingertip_distance_sums, self.nonfinger_penetration_loss],
            device=self.model.device,
        )
        wp.launch(
            combine_grasp_loss,
            dim=1,
            inputs=[
                self.target_loss,
                self.fingertip_gap_loss,
                self.nonfinger_penetration_loss,
                self.target_weight,
                self.fingertip_gap_weight,
                self.nonfinger_penetration_weight,
            ],
            outputs=[self.loss],
            device=self.model.device,
        )
        self._record_gradient_snapshot([self.loss], [self.loss_gradient_trace])

    def step(self):
        """Run one differentiable grasp refinement iteration."""
        if self.converged or self.display_frame % self.frames_per_iteration != 0:
            self.display_frame += 1
            self.sim_time += self.frame_dt
            return

        self._clear_objective_buffers()
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

        self.current_loss = float(self.loss.numpy()[0])
        self.current_target_loss = float(self.target_loss.numpy()[0])
        self.current_fingertip_gap_loss = float(self.fingertip_gap_loss.numpy()[0])
        self.current_nonfinger_penetration_loss = float(self.nonfinger_penetration_loss.numpy()[0])
        if self.train_iter == 0:
            self.initial_loss = self.current_loss
        self.current_counts = self.witness_counts.numpy().copy()
        self.current_witnesses = self.witnesses.numpy().copy()
        self.current_fingertip_witnesses = self.fingertip_witnesses.numpy().copy()
        self.current_fingertip_distances = self.selected_fingertip_distances.numpy().copy()
        if self.train_iter == 0:
            self.initial_max_fingertip_gap = float(np.max(np.abs(self.current_fingertip_distances)))
        safety_contact_count = int(self.safety_contacts.rigid_contact_count.numpy()[0])
        safety_distances = self.safety_distance.numpy()[:safety_contact_count]
        fingertip_contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        fingertip_distances = self.distance.numpy()[:fingertip_contact_count]
        active_distances = np.concatenate((fingertip_distances, safety_distances))
        self.current_max_penetration = float(max(0.0, -float(np.min(active_distances))))
        joint_q = self.model.joint_q.numpy()
        gradient = self.model.joint_q.grad.numpy()
        optimized_gradient = gradient[self.optimized_coordinates]
        self.current_joint_q = joint_q[self.optimized_coordinates].copy()
        self.current_joint_gradient = optimized_gradient.copy()
        self.current_gradient_norm = float(np.linalg.norm(optimized_gradient))
        self.current_max_joint_delta_degrees = float(
            np.degrees(np.max(np.abs(self.current_joint_q - self.initial_optimized_joint_q)))
        )
        feasible = self.current_max_penetration <= self.penetration_tolerance
        improved = feasible and self.current_loss < self.best_loss
        worsened = self.best_joint_q is not None and self.current_loss > self.best_loss + 1.0e-10
        rejected = self.best_joint_q is not None and (not feasible or worsened)
        if improved:
            self.best_loss = self.current_loss
            self.best_joint_q = joint_q.copy()
            self.best_target_loss = self.current_target_loss
            self.best_fingertip_gap_loss = self.current_fingertip_gap_loss
            self.best_nonfinger_penetration_loss = self.current_nonfinger_penetration_loss
            self.best_max_penetration = self.current_max_penetration
            self.best_witnesses = self.current_witnesses.copy()
            self.best_fingertip_witnesses = self.current_fingertip_witnesses.copy()
            self.best_fingertip_distances = self.current_fingertip_distances.copy()
            self.best_counts = self.current_counts.copy()
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

        print(
            f"iter={self.train_iter:03d} total={self.current_loss:.7f} target={self.current_target_loss:.7f} "
            f"tip_gap={self.current_fingertip_gap_loss:.7f} penetration={self.current_max_penetration:.6f}m "
            f"motion={self.current_max_joint_delta_degrees:.2f}deg contacts={self.current_counts.tolist()} "
            f"gaps={np.round(self.current_fingertip_distances * 1000.0, 2).tolist()}mm "
            f"|dL/dq|={self.current_gradient_norm:.6e}{' REJECT' if rejected else ''}"
        )
        self._write_iteration_log()

        if rejected:
            self.learning_rate *= 0.5
            self._restore_best_state()
        elif (
            np.all(self.current_counts > 0)
            and self.current_target_loss < 2.5e-6
            and float(np.max(np.abs(self.current_fingertip_distances))) < 2.5e-4
            and self.current_max_penetration <= self.penetration_tolerance
        ):
            self.converged = True
        elif self.train_iter + 1 >= self.max_iterations:
            self.converged = True
        elif self.iterations_without_improvement >= self.early_stop_patience:
            self.converged = True
        elif self.current_gradient_norm > 1.0e-10:
            step = self.learning_rate * optimized_gradient
            joint_q[self.optimized_coordinates] -= step
            lower = self.model.joint_limit_lower.numpy()[self.optimized_coordinates]
            upper = self.model.joint_limit_upper.numpy()[self.optimized_coordinates]
            joint_q[self.optimized_coordinates] = np.clip(joint_q[self.optimized_coordinates], lower, upper)
            self.model.joint_q.assign(joint_q)

        if self.converged:
            self._restore_best_state()

        self.tape.zero()
        self.train_iter += 1
        self.display_frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        """Render only the latest optimization state; rendering is outside the tape."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.log_contacts(self.safety_contacts, self.state)
        self.viewer.log_points("/target_contacts", self.targets, self.target_radii, self.target_colors)
        if np.any(self.current_counts > 0):
            self.viewer.log_points("/gear_witnesses", self.witnesses, self.actual_radii, self.actual_colors)
            self.viewer.log_points(
                "/fingertip_witnesses",
                self.fingertip_witnesses,
                self.fingertip_radii,
                self.fingertip_colors,
            )
            self.viewer.log_lines(
                "/witness_separation",
                self.witnesses,
                self.fingertip_witnesses,
                self.witness_line_colors,
            )
        self.viewer.log_scalar("/grasp_witness_loss", self.current_loss)
        self.viewer.log_scalar("/target_loss", self.current_target_loss)
        self.viewer.log_scalar("/nonfingertip_penetration_loss", self.current_nonfinger_penetration_loss)
        self.viewer.end_frame()

    def gui(self, ui):
        """Expose the multi-contact objective without affecting optimization."""
        ui.text(f"Grasp refinement iteration: {self.train_iter}/{self.max_iterations}")
        ui.text(f"Total constrained loss: {self.current_loss:.7f}")
        ui.text(f"Target witness loss: {self.current_target_loss:.7f}")
        ui.text(f"Fingertip gap loss: {self.current_fingertip_gap_loss:.7f}")
        ui.text(f"Non-tip penetration loss: {self.current_nonfinger_penetration_loss:.7f}")
        ui.text(f"Max hand penetration: {self.current_max_penetration * 1000.0:.3f} mm")
        ui.text(f"Best total loss: {self.best_loss:.7f}")
        ui.text(f"|dL/d joint_q|: {self.current_gradient_norm:.6e}")
        ui.text(f"Max joint motion: {self.current_max_joint_delta_degrees:.2f} deg")
        ui.text("YELLOW: fixed targets on the gear surface")
        ui.text("RED: exact gear-side witnesses")
        ui.text("CYAN: exact fingertip-side witnesses")
        ui.text("WHITE: signed separation for each selected pair")
        for finger, name in enumerate(FINGER_NAMES):
            target = self.target_contact_points[finger]
            witness = self.current_witnesses[finger]
            error = float(np.linalg.norm(witness - target))
            ui.text(
                f"{name:>6}: gap={self.current_fingertip_distances[finger] * 1000.0:+.2f} mm "
                f"target error={error * 1000.0:.2f} mm"
            )
            ui.text(
                f"        target=({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}) "
                f"red=({witness[0]:+.3f}, {witness[1]:+.3f}, {witness[2]:+.3f})"
            )
        if self.converged:
            ui.text("Optimization stopped (converged or iteration limit reached)")

    def test_final(self):
        """Verify that the contact topology and articulated gradient remain useful."""
        assert np.all(self.current_counts > 0), f"Missing fingertip contacts: {self.current_counts.tolist()}"
        assert np.isfinite(self.current_loss), f"Non-finite witness loss: {self.current_loss}"
        assert np.isfinite(self.current_gradient_norm), f"Non-finite joint gradient: {self.current_gradient_norm}"
        assert self.current_gradient_norm > 0.0, "Expected a non-zero contact-to-joint gradient"
        assert self.current_max_penetration <= self.penetration_tolerance, (
            f"Hand-object penetration exceeds tolerance: {self.current_max_penetration}"
        )
        assert self.current_max_joint_delta_degrees >= 0.75, (
            f"Finger motion is not visually meaningful: {self.current_max_joint_delta_degrees} degrees"
        )
        assert float(np.max(np.abs(self.current_fingertip_distances))) < 0.2 * self.initial_max_fingertip_gap, (
            "Fingertips did not close around the gear: "
            f"initial={self.initial_max_fingertip_gap}, final={self.current_fingertip_distances.tolist()}"
        )
        assert self.best_loss < 0.9 * self.initial_loss, (
            f"Grasp refinement did not reduce loss enough: initial={self.initial_loss}, best={self.best_loss}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=390)
    viewer, args = newton.examples.init(parser)
    if args.viewer == "gl" and args.render_fps is None:
        args.render_fps = 30.0
    newton.examples.run(Example(viewer, args), args)
