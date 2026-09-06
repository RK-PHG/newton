# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualize four independently scheduled collision partitions.

Four dense particle curtains move along +X and strike four spatially separated
rows of mixed analytic primitives. Each colored row owns an independent Newton
model, contact buffer, state, CUDA Graph, and CUDA stream. The scene loops after
the impact so that the partitioning and collision response remain visible.

Run from the repository root::

    uv run python diffsim/partitioned_contact_overlap.py --viewer gl
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp
from particle_primitive_wall import CYCLE_FRAMES, FRAME_DT, SIM_DT, SIM_SUBSTEPS, SOFT_CONTACT_MARGIN
from profile_partitioned_contact_overlap import (
    PARTICLES_PER_PARTITION,
    PARTITION_COUNT,
    SHAPE_COUNT,
    SHAPES_PER_PARTITION,
    build_partition_model,
)

import newton
import newton.examples

PARTITION_COLORS = (
    (0.12, 0.42, 0.95),
    (0.95, 0.42, 0.10),
    (0.16, 0.78, 0.35),
    (0.68, 0.28, 0.92),
)


class VisualContactPartition:
    """One independently simulated and rendered contact-island row."""

    def __init__(
        self,
        shape_indices: Sequence[int],
        color: tuple[float, float, float],
        device: wp.Device,
    ):
        self.shape_indices = tuple(shape_indices)
        self.device = device
        self.model = build_partition_model(
            self.shape_indices,
            device,
            contact_snapshot=False,
            partition_color=color,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            enable_soft_contact_candidates=True,
            enable_fused_soft_contact=True,
            requires_grad=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.control = self.model.control(requires_grad=False)
        self.state_0 = self.model.state(requires_grad=False)
        self.state_1 = self.model.state(requires_grad=False)
        self.graph: wp.Graph | None = None

        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                self._simulate_frame()
            self.graph = capture.graph
            self.reset()

    def _simulate_frame(self) -> None:
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> None:
        wp.copy(self.state_0.particle_q, self.model.particle_q)
        wp.copy(self.state_0.particle_qd, self.model.particle_qd)
        self.state_0.clear_forces()

    def launch(self, stream: wp.Stream | None = None) -> None:
        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph, stream=stream)


class Example:
    """Run and render the four collision partitions on independent streams."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device(args.device)
        self.sim_time = 0.0
        self.cycle_frame = 0
        self.main_stream = self.device.stream
        self.partitions = []

        for partition_index, color in enumerate(PARTITION_COLORS):
            begin = partition_index * SHAPES_PER_PARTITION
            end = begin + SHAPES_PER_PARTITION
            partition = VisualContactPartition(range(begin, end), color, self.device)
            self.partitions.append(partition)

            self.viewer.activate(f"Partition {partition_index}")
            self.viewer.set_model(partition.model)
            self.viewer.show_particles = True
            self.viewer.show_ground = False

        self.worker_streams = tuple(wp.Stream(self.device) for _ in self.partitions) if self.device.is_cuda else ()
        self.fork_events = tuple(wp.Event(self.device) for _ in self.partitions) if self.device.is_cuda else ()
        self.join_events = tuple(wp.Event(self.device) for _ in self.partitions) if self.device.is_cuda else ()

        self.viewer.set_camera(pos=wp.vec3(-5.5, -7.8, 4.3), pitch=-16.0, yaw=52.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 68.0

        print("PartitionedContactOverlap")
        print(f"  partitions / CUDA streams    : {PARTITION_COUNT} / {len(self.worker_streams)}")
        print(f"  total particles / shapes     : {PARTITION_COUNT * PARTICLES_PER_PARTITION} / {SHAPE_COUNT}")
        print(f"  particles / shapes per block : {PARTICLES_PER_PARTITION} / {SHAPES_PER_PARTITION}")
        print("  each colored row is one independent collision partition")

    def gui(self, ui):
        ui.text("Partitioned collision scene")
        ui.text(f"Partitions: {PARTITION_COUNT}")
        ui.text(f"Particles: {PARTITION_COUNT * PARTICLES_PER_PARTITION}")
        ui.text(f"Shapes: {SHAPE_COUNT}")
        ui.separator()
        ui.text("Each colored wall row is one partition.")
        ui.text("Every partition owns:")
        ui.text("  - state and force buffers")
        ui.text("  - contact buffers")
        ui.text("  - CUDA Graph and CUDA stream")
        ui.separator()
        ui.text("All curtains move toward +X.")
        ui.text("The scene restarts after impact.")

    def _launch_overlap(self) -> None:
        if not self.device.is_cuda:
            for partition in self.partitions:
                partition.launch()
            return

        for partition, stream, event in zip(
            self.partitions,
            self.worker_streams,
            self.fork_events,
            strict=True,
        ):
            stream.wait_stream(self.main_stream, event=event)
            partition.launch(stream)
        for stream, event in zip(self.worker_streams, self.join_events, strict=True):
            self.main_stream.wait_stream(stream, event=event)

    def reset(self) -> None:
        for partition in self.partitions:
            partition.reset()
        self.cycle_frame = 0

    def step(self) -> None:
        if self.cycle_frame >= CYCLE_FRAMES:
            self.reset()
        self._launch_overlap()
        self.cycle_frame += 1
        self.sim_time += FRAME_DT

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        for partition_index, partition in enumerate(self.partitions):
            self.viewer.activate(f"Partition {partition_index}")
            self.viewer.log_state(partition.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        for partition in self.partitions:
            assert partition.model.particle_count == PARTICLES_PER_PARTITION
            assert partition.model.shape_count == SHAPES_PER_PARTITION


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(device="cuda:0", viewer="gl")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
