# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare serial and multi-stream execution of independent contact islands.

The fixed workload contains 32 particle-to-static-primitive contact regions.
The unoptimized baseline uses one model and the original two-operator soft-
contact path. Intermediate measurements enable fusion, then split the scene
into four independent models. The final path launches the four captured CUDA
graphs on four CUDA streams.

Run from the repository root::

    uv run python diffsim/profile_partitioned_contact_overlap.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp
from particle_primitive_wall import (
    BOX_COLOR,
    CAPSULE_COLOR,
    CAPSULE_HALF_HEIGHT,
    CAPSULE_INDICES,
    CONTACT_KD,
    CONTACT_KE,
    CONTACT_SNAPSHOT_PENETRATION,
    CYLINDER_COLOR,
    CYLINDER_HALF_HEIGHT,
    CYLINDER_INDICES,
    PARTICLE_MASS,
    PARTICLE_RADIUS,
    PARTICLE_SPEED,
    PARTICLE_START_X,
    SHAPE_COLUMNS,
    SHAPE_COUNT,
    SHAPE_RADIUS,
    SHAPE_SPACING,
    SIM_DT,
    SOFT_CONTACT_MARGIN,
    SPHERE_COLOR,
    SPHERE_INDICES,
    WALL_CENTER_Z,
)

import newton

try:
    import nvtx as _nvtx  # noqa: F401

    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

PARTITION_COUNT = 4
SHAPES_PER_PARTITION = SHAPE_COUNT // PARTITION_COUNT
PATCH_COLUMNS = 64
PATCH_ROWS = 16
PARTICLES_PER_SHAPE = PATCH_COLUMNS * PATCH_ROWS
PARTICLES_PER_PARTITION = SHAPES_PER_PARTITION * PARTICLES_PER_SHAPE
PATCH_HALF_EXTENT = 0.18

WARMUP_ITERATIONS = 10
TIMING_ITERATIONS = 30
LOG_DIR = Path(__file__).resolve().parent / "log"


def _shape_kind(shape_index: int) -> str:
    if shape_index in CAPSULE_INDICES:
        return "capsule"
    if shape_index in CYLINDER_INDICES:
        return "cylinder"
    if shape_index in SPHERE_INDICES:
        return "sphere"
    return "box"


def _wall_center(shape_index: int) -> tuple[float, float]:
    column = shape_index % SHAPE_COLUMNS
    row = shape_index // SHAPE_COLUMNS
    y = (column - 0.5 * (SHAPE_COLUMNS - 1)) * SHAPE_SPACING
    z = WALL_CENTER_Z + (row - 1.5) * SHAPE_SPACING
    return y, z


def _surface_x(kind: str, local_y: float, local_z: float) -> float:
    if kind == "box":
        return -SHAPE_RADIUS
    if kind == "cylinder":
        return -np.sqrt(max(SHAPE_RADIUS**2 - local_y**2, 0.0))
    cap_z = max(abs(local_z) - CAPSULE_HALF_HEIGHT, 0.0) if kind == "capsule" else local_z
    return -np.sqrt(max(SHAPE_RADIUS**2 - local_y**2 - cap_z**2, 0.0))


def build_partition_model(
    shape_indices: Sequence[int],
    device: wp.Device,
    *,
    contact_snapshot: bool = True,
    partition_color: tuple[float, float, float] | None = None,
) -> newton.Model:
    """Build one spatially independent row of the fixed wall scene."""
    local_y = np.linspace(-PATCH_HALF_EXTENT, PATCH_HALF_EXTENT, PATCH_COLUMNS, dtype=np.float32)
    local_z = np.linspace(-PATCH_HALF_EXTENT, PATCH_HALF_EXTENT, PATCH_ROWS, dtype=np.float32)
    particle_count = len(shape_indices) * PARTICLES_PER_SHAPE
    positions = np.empty((particle_count, 3), dtype=np.float32)
    velocities = np.zeros((particle_count, 3), dtype=np.float32)
    velocities[:, 0] = PARTICLE_SPEED

    particle_index = 0
    for shape_index in shape_indices:
        center_y, center_z = _wall_center(shape_index)
        kind = _shape_kind(shape_index)
        for z_offset in local_z:
            for y_offset in local_y:
                x = PARTICLE_START_X
                if contact_snapshot:
                    x = (
                        _surface_x(kind, float(y_offset), float(z_offset))
                        - PARTICLE_RADIUS
                        + CONTACT_SNAPSHOT_PENETRATION
                    )
                positions[particle_index] = (
                    x,
                    center_y + y_offset,
                    center_z + z_offset,
                )
                particle_index += 1

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.add_particles(
        pos=positions.tolist(),
        vel=velocities.tolist(),
        mass=[PARTICLE_MASS] * particle_count,
        radius=[PARTICLE_RADIUS] * particle_count,
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        ke=CONTACT_KE,
        kd=CONTACT_KD,
        kf=0.0,
        mu=0.0,
        has_shape_collision=False,
        has_particle_collision=True,
    )

    for shape_index in shape_indices:
        center_y, center_z = _wall_center(shape_index)
        common = {
            "body": -1,
            "xform": wp.transform(wp.vec3(0.0, center_y, center_z), wp.quat_identity()),
            "cfg": shape_cfg,
            "label": f"{_shape_kind(shape_index)}_{shape_index}",
        }
        kind = _shape_kind(shape_index)
        color = partition_color
        if kind == "capsule":
            builder.add_shape_capsule(
                **common,
                radius=SHAPE_RADIUS,
                half_height=CAPSULE_HALF_HEIGHT,
                color=color or CAPSULE_COLOR,
            )
        elif kind == "cylinder":
            builder.add_shape_cylinder(
                **common,
                radius=SHAPE_RADIUS,
                half_height=CYLINDER_HALF_HEIGHT,
                color=color or CYLINDER_COLOR,
            )
        elif kind == "sphere":
            builder.add_shape_sphere(**common, radius=SHAPE_RADIUS, color=color or SPHERE_COLOR)
        else:
            builder.add_shape_box(
                **common,
                hx=SHAPE_RADIUS,
                hy=SHAPE_RADIUS,
                hz=SHAPE_RADIUS,
                color=color or BOX_COLOR,
            )

    model = builder.finalize(device=device, requires_grad=False)
    model.particle_grid = None
    model.soft_contact_ke = CONTACT_KE
    model.soft_contact_kd = CONTACT_KD
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    model.soft_contact_restitution = 0.0
    return model


class ContactPartition:
    """One independently writable collision-and-solve graph."""

    def __init__(
        self,
        shape_indices: Sequence[int],
        device: wp.Device,
        *,
        enable_fused_soft_contact: bool = True,
    ):
        self.shape_indices = tuple(shape_indices)
        self.device = device
        self.enable_fused_soft_contact = enable_fused_soft_contact
        self.model = build_partition_model(self.shape_indices, device)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            enable_soft_contact_candidates=True,
            enable_fused_soft_contact=enable_fused_soft_contact,
            requires_grad=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.control = self.model.control(requires_grad=False)
        self.state_in = self.model.state(requires_grad=False)
        self.state_out = self.model.state(requires_grad=False)
        self.graph: wp.Graph | None = None

    def step(self) -> None:
        self.clear_forces()
        self.collide()
        self.solve()

    def clear_forces(self) -> None:
        self.state_in.clear_forces()

    def collide(self) -> None:
        self.pipeline.collide(self.state_in, self.contacts)

    def solve(self) -> None:
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, SIM_DT)

    def prepare_solve(self) -> None:
        """Prepare valid contacts and clean force storage outside solver timing."""
        self.clear_forces()
        self.collide()

    def capture(self) -> None:
        with wp.ScopedCapture(device=self.device) as capture:
            self.step()
        self.graph = capture.graph

    def launch(self, stream: wp.Stream | None = None) -> None:
        if self.graph is None:
            raise RuntimeError("Partition graph has not been captured")
        wp.capture_launch(self.graph, stream=stream)


@dataclass
class TimingStats:
    samples_ms: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples_ms))

    @property
    def minimum(self) -> float:
        return float(np.min(self.samples_ms))

    @property
    def median(self) -> float:
        return float(np.median(self.samples_ms))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples_ms, 95))


class PartitionedScheduler:
    """Launch the same captured partitions serially or on independent streams."""

    def __init__(self, partitions: Sequence[ContactPartition], device: wp.Device):
        self.partitions = tuple(partitions)
        self.device = device
        self.main_stream = device.stream
        self.worker_streams = tuple(wp.Stream(device) for _ in self.partitions)
        self.fork_events = tuple(wp.Event(device) for _ in self.partitions)
        self.join_events = tuple(wp.Event(device) for _ in self.partitions)
        self.graph_fork_events = tuple(wp.Event(device) for _ in self.partitions)
        self.graph_join_events = tuple(wp.Event(device) for _ in self.partitions)
        self.overlap_graph = self._capture_overlap_graph()

    def _capture_overlap_graph(self) -> wp.Graph:
        """Capture all stream branches and their dependencies as one graph."""
        with wp.ScopedCapture(device=self.device, stream=self.main_stream) as capture:
            for partition, stream, event in zip(
                self.partitions,
                self.worker_streams,
                self.graph_fork_events,
                strict=True,
            ):
                stream.wait_stream(self.main_stream, event=event)
                with wp.ScopedStream(stream, sync_enter=False):
                    partition.step()
            for stream, event in zip(self.worker_streams, self.graph_join_events, strict=True):
                self.main_stream.wait_stream(stream, event=event)
        return capture.graph

    def launch_serial(self) -> None:
        for partition in self.partitions:
            partition.launch()

    def launch_overlap(self) -> None:
        """Launch four child graphs and coordinate them from the host."""
        for partition, stream, event in zip(
            self.partitions,
            self.worker_streams,
            self.fork_events,
            strict=True,
        ):
            stream.wait_stream(self.main_stream, event=event)
            partition.launch(stream=stream)
        for stream, event in zip(self.worker_streams, self.join_events, strict=True):
            self.main_stream.wait_stream(stream, event=event)

    def launch_overlap_graph(self) -> None:
        """Launch the complete multi-stream schedule as one parent graph."""
        wp.capture_launch(self.overlap_graph)

    def _measure_once(self, operation: Callable[[], None]) -> float:
        start_event = wp.Event(self.device, enable_timing=True)
        end_event = wp.Event(self.device, enable_timing=True)
        self.main_stream.record_event(start_event)
        operation()
        self.main_stream.record_event(end_event)
        return float(wp.get_event_elapsed_time(start_event, end_event))

    def measure_schedules(
        self,
        baseline: ContactPartition,
        fused_unpartitioned: ContactPartition,
    ) -> tuple[TimingStats, TimingStats, TimingStats, TimingStats, TimingStats]:
        """Interleave all schedules to reduce clock and thermal bias."""
        for _ in range(WARMUP_ITERATIONS):
            baseline.launch()
            fused_unpartitioned.launch()
            self.launch_serial()
            self.launch_overlap()
            self.launch_overlap_graph()
        wp.synchronize_device(self.device)

        baseline_samples_ms = []
        fused_samples_ms = []
        serial_samples_ms = []
        overlap_samples_ms = []
        overlap_graph_samples_ms = []
        for sample_index in range(TIMING_ITERATIONS):
            schedules = (
                (baseline_samples_ms, baseline.launch),
                (fused_samples_ms, fused_unpartitioned.launch),
                (serial_samples_ms, self.launch_serial),
                (overlap_samples_ms, self.launch_overlap),
                (overlap_graph_samples_ms, self.launch_overlap_graph),
            )
            rotation = sample_index % len(schedules)
            for samples, operation in schedules[rotation:] + schedules[:rotation]:
                samples.append(self._measure_once(operation))
        return (
            TimingStats(baseline_samples_ms),
            TimingStats(fused_samples_ms),
            TimingStats(serial_samples_ms),
            TimingStats(overlap_samples_ms),
            TimingStats(overlap_graph_samples_ms),
        )

    def measure_stage(
        self,
        operation: Callable[[], None],
        *,
        prepare: Callable[[], None] | None = None,
    ) -> TimingStats:
        for _ in range(WARMUP_ITERATIONS):
            if prepare is not None:
                prepare()
            operation()
        wp.synchronize_device(self.device)

        samples_ms = []
        for _ in range(TIMING_ITERATIONS):
            if prepare is not None:
                prepare()
            samples_ms.append(self._measure_once(operation))
        return TimingStats(samples_ms)


@contextmanager
def _profile_range(device: wp.Device):
    if hasattr(wp, "ScopedCudaProfiler"):
        with wp.ScopedCudaProfiler(device):
            yield
    else:
        yield


@contextmanager
def _nvtx_range(name: str):
    if NVTX_AVAILABLE:
        with wp.ScopedTimer(name, print=False, use_nvtx=True, skip_tape=True):
            yield
    else:
        yield


def _outputs(partitions: Sequence[ContactPartition]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (partition.state_out.particle_q.numpy().copy(), partition.state_out.particle_qd.numpy().copy())
        for partition in partitions
    ]


def _validate_results(
    partitions: Sequence[ContactPartition],
    serial_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
    overlap_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    max_position_error = 0.0
    max_velocity_error = 0.0
    for partition, serial, overlap in zip(partitions, serial_outputs, overlap_outputs, strict=True):
        contact_count = min(int(partition.contacts.soft_contact_count.numpy()[0]), partition.contacts.soft_contact_max)
        if contact_count != PARTICLES_PER_PARTITION:
            raise RuntimeError(
                f"Partition {partition.shape_indices} produced {contact_count} contacts; "
                f"expected {PARTICLES_PER_PARTITION}"
            )
        max_position_error = max(max_position_error, float(np.max(np.abs(serial[0] - overlap[0]))))
        max_velocity_error = max(max_velocity_error, float(np.max(np.abs(serial[1] - overlap[1]))))

    if max_position_error > 1.0e-6 or max_velocity_error > 1.0e-6:
        raise RuntimeError("Multi-stream scheduling changed the integrated particle state")
    return max_position_error, max_velocity_error


def _validate_unpartitioned_baseline(
    baseline: ContactPartition,
    partitioned_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    baseline_contact_count = min(
        int(baseline.contacts.soft_contact_count.numpy()[0]), baseline.contacts.soft_contact_max
    )
    expected_contact_count = PARTITION_COUNT * PARTICLES_PER_PARTITION
    if baseline_contact_count != expected_contact_count:
        raise RuntimeError(f"Baseline produced {baseline_contact_count} contacts; expected {expected_contact_count}")

    baseline_q = baseline.state_out.particle_q.numpy()
    baseline_qd = baseline.state_out.particle_qd.numpy()
    partitioned_q = np.concatenate([output[0] for output in partitioned_outputs])
    partitioned_qd = np.concatenate([output[1] for output in partitioned_outputs])
    max_position_error = float(np.max(np.abs(baseline_q - partitioned_q)))
    max_velocity_error = float(np.max(np.abs(baseline_qd - partitioned_qd)))
    if max_position_error > 1.0e-6 or max_velocity_error > 1.0e-6:
        raise RuntimeError("Scene partitioning changed the integrated particle state")
    return max_position_error, max_velocity_error


def _format_samples(samples_ms: Sequence[float]) -> str:
    return ", ".join(f"{sample:.6f}" for sample in samples_ms)


def _write_report(report: str, run_started_at: datetime) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S_%f")
    path = LOG_DIR / f"partitioned_contact_overlap_{timestamp}.log"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
        output.write("\n")
    return path


def main() -> None:
    run_started_at = datetime.now().astimezone()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("The partitioned overlap profile requires a CUDA device")

    baseline = ContactPartition(range(SHAPE_COUNT), device, enable_fused_soft_contact=False)
    fused_unpartitioned = ContactPartition(range(SHAPE_COUNT), device, enable_fused_soft_contact=True)
    partitions = []
    for partition_index in range(PARTITION_COUNT):
        begin = partition_index * SHAPES_PER_PARTITION
        end = begin + SHAPES_PER_PARTITION
        partitions.append(ContactPartition(range(begin, end), device))

    # Compile and allocate outside capture, then capture one fixed graph per workload.
    workloads = [baseline, fused_unpartitioned, *partitions]
    for workload in workloads:
        workload.step()
    wp.synchronize_device(device)
    for workload in workloads:
        workload.capture()
    wp.synchronize_device(device)

    scheduler = PartitionedScheduler(partitions, device)
    baseline_stats, fused_stats, serial_stats, overlap_stats, overlap_graph_stats = scheduler.measure_schedules(
        baseline, fused_unpartitioned
    )

    def partitioned_collide_serial() -> None:
        for partition in partitions:
            partition.collide()

    def prepare_partitioned_solve() -> None:
        for partition in partitions:
            partition.prepare_solve()

    def partitioned_solve_serial() -> None:
        for partition in partitions:
            partition.solve()

    baseline_collision_stats = scheduler.measure_stage(baseline.collide)
    baseline_solver_stats = scheduler.measure_stage(baseline.solve, prepare=baseline.prepare_solve)
    fused_collision_stats = scheduler.measure_stage(fused_unpartitioned.collide)
    fused_solver_stats = scheduler.measure_stage(
        fused_unpartitioned.solve,
        prepare=fused_unpartitioned.prepare_solve,
    )
    partitioned_collision_stats = scheduler.measure_stage(partitioned_collide_serial)
    partitioned_solver_stats = scheduler.measure_stage(
        partitioned_solve_serial,
        prepare=prepare_partitioned_solve,
    )

    baseline.launch()
    wp.synchronize_device(device)
    scheduler.launch_serial()
    wp.synchronize_device(device)
    serial_outputs = _outputs(partitions)
    scheduler.launch_overlap()
    wp.synchronize_device(device)
    overlap_outputs = _outputs(partitions)
    max_position_error, max_velocity_error = _validate_results(partitions, serial_outputs, overlap_outputs)
    scheduler.launch_overlap_graph()
    wp.synchronize_device(device)
    overlap_graph_outputs = _outputs(partitions)
    graph_position_error, graph_velocity_error = _validate_results(
        partitions,
        serial_outputs,
        overlap_graph_outputs,
    )
    baseline_position_error, baseline_velocity_error = _validate_unpartitioned_baseline(
        baseline,
        overlap_graph_outputs,
    )
    fused_position_error, fused_velocity_error = _validate_unpartitioned_baseline(
        fused_unpartitioned,
        overlap_graph_outputs,
    )

    # Emit synchronized ranges so NSYS shows the complete GPU duration under each marker.
    with _profile_range(device):
        with _nvtx_range("baseline/unpartitioned_two_operator/full_pipeline"):
            baseline.launch()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/fused_unpartitioned/full_pipeline"):
            fused_unpartitioned.launch()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/partitioned_serial/full_pipeline"):
            scheduler.launch_serial()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/partitioned_overlap/full_pipeline"):
            scheduler.launch_overlap()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/partitioned_overlap_parent_graph/full_pipeline"):
            scheduler.launch_overlap_graph()
            wp.synchronize_device(device)
        baseline.clear_forces()
        wp.synchronize_device(device)
        with _nvtx_range("baseline/unpartitioned_two_operator/collide"):
            baseline.collide()
            wp.synchronize_device(device)
        with _nvtx_range("baseline/unpartitioned_two_operator/solve"):
            baseline.solve()
            wp.synchronize_device(device)
        fused_unpartitioned.clear_forces()
        wp.synchronize_device(device)
        with _nvtx_range("optimized/fused_unpartitioned/collide"):
            fused_unpartitioned.collide()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/fused_unpartitioned/solve"):
            fused_unpartitioned.solve()
            wp.synchronize_device(device)
        for partition in partitions:
            partition.clear_forces()
        wp.synchronize_device(device)
        with _nvtx_range("optimized/partitioned_serial/collide"):
            partitioned_collide_serial()
            wp.synchronize_device(device)
        with _nvtx_range("optimized/partitioned_serial/solve"):
            partitioned_solve_serial()
            wp.synchronize_device(device)

    fusion_speedup = baseline_stats.mean / fused_stats.mean
    partition_speedup = fused_stats.mean / serial_stats.mean
    overlap_speedup = serial_stats.mean / overlap_stats.mean
    graph_launch_speedup = overlap_stats.mean / overlap_graph_stats.mean
    total_speedup = baseline_stats.mean / overlap_graph_stats.mean
    baseline_candidate_pairs = baseline.pipeline.soft_rigid_contact_pair_count
    partitioned_candidate_pairs = sum(partition.pipeline.soft_rigid_contact_pair_count for partition in partitions)
    baseline_compact_candidates = int(baseline.pipeline.soft_contact_candidate_count.numpy()[0])
    partitioned_compact_candidates = sum(
        int(partition.pipeline.soft_contact_candidate_count.numpy()[0]) for partition in partitions
    )
    report = "\n".join(
        [
            "=" * 88,
            "Partitioned contact-island overlap profile",
            f"timestamp                    : {run_started_at.isoformat(timespec='seconds')}",
            f"device                       : {device} / {device.name}",
            f"warp                         : {wp.__version__}",
            f"newton                       : {newton.__version__}",
            f"partitions / CUDA streams    : {PARTITION_COUNT} / {PARTITION_COUNT}",
            f"total particles / shapes     : {PARTITION_COUNT * PARTICLES_PER_PARTITION} / {SHAPE_COUNT}",
            f"particles / shapes per block : {PARTICLES_PER_PARTITION} / {SHAPES_PER_PARTITION}",
            f"baseline pair launch bound   : {baseline_candidate_pairs}",
            f"partitioned pair launch bound: {partitioned_candidate_pairs}",
            f"baseline compact candidates  : {baseline_compact_candidates}",
            f"partition compact candidates : {partitioned_compact_candidates}",
            "soft-contact candidates      : enabled",
            "baseline fused soft contact  : disabled",
            "optimized fused soft contact : enabled",
            f"NVTX markers available       : {NVTX_AVAILABLE}",
            f"warmup / timing samples      : {WARMUP_ITERATIONS} / {TIMING_ITERATIONS}",
            "",
            "End-to-end CUDA Graph timings",
            "  schedule                  mean ms     min ms     p50 ms     p95 ms",
            f"  baseline two-operator     {baseline_stats.mean:9.4f}  {baseline_stats.minimum:9.4f}  "
            f"{baseline_stats.median:9.4f}  {baseline_stats.p95:9.4f}",
            f"  fused unpartitioned      {fused_stats.mean:9.4f}  {fused_stats.minimum:9.4f}  "
            f"{fused_stats.median:9.4f}  {fused_stats.p95:9.4f}",
            f"  partitioned serial       {serial_stats.mean:9.4f}  {serial_stats.minimum:9.4f}  "
            f"{serial_stats.median:9.4f}  {serial_stats.p95:9.4f}",
            f"  partitioned overlap      {overlap_stats.mean:9.4f}  {overlap_stats.minimum:9.4f}  "
            f"{overlap_stats.median:9.4f}  {overlap_stats.p95:9.4f}",
            f"  overlap parent graph     {overlap_graph_stats.mean:9.4f}  {overlap_graph_stats.minimum:9.4f}  "
            f"{overlap_graph_stats.median:9.4f}  {overlap_graph_stats.p95:9.4f}",
            "",
            "Pipeline speedups",
            f"  operator fusion only       : {fusion_speedup:.3f}x",
            f"  partitioning after fusion  : {partition_speedup:.3f}x",
            f"  overlap after partitioning : {overlap_speedup:.3f}x",
            f"  parent Graph launch        : {graph_launch_speedup:.3f}x",
            f"  baseline / final optimized : {total_speedup:.3f}x",
            "",
            "Synchronized stage timings",
            "  stage                              mean ms     min ms     p50 ms     p95 ms",
            f"  baseline collide                 {baseline_collision_stats.mean:9.4f}  "
            f"{baseline_collision_stats.minimum:9.4f}  {baseline_collision_stats.median:9.4f}  "
            f"{baseline_collision_stats.p95:9.4f}",
            f"  baseline solve                   {baseline_solver_stats.mean:9.4f}  "
            f"{baseline_solver_stats.minimum:9.4f}  {baseline_solver_stats.median:9.4f}  "
            f"{baseline_solver_stats.p95:9.4f}",
            f"  fused unpartitioned collide      {fused_collision_stats.mean:9.4f}  "
            f"{fused_collision_stats.minimum:9.4f}  {fused_collision_stats.median:9.4f}  "
            f"{fused_collision_stats.p95:9.4f}",
            f"  fused unpartitioned solve        {fused_solver_stats.mean:9.4f}  "
            f"{fused_solver_stats.minimum:9.4f}  {fused_solver_stats.median:9.4f}  "
            f"{fused_solver_stats.p95:9.4f}",
            f"  partitioned serial collide       {partitioned_collision_stats.mean:9.4f}  "
            f"{partitioned_collision_stats.minimum:9.4f}  {partitioned_collision_stats.median:9.4f}  "
            f"{partitioned_collision_stats.p95:9.4f}",
            f"  partitioned serial solve         {partitioned_solver_stats.mean:9.4f}  "
            f"{partitioned_solver_stats.minimum:9.4f}  {partitioned_solver_stats.median:9.4f}  "
            f"{partitioned_solver_stats.p95:9.4f}",
            "",
            "Raw baseline-two-operator samples [ms]",
            f"  {_format_samples(baseline_stats.samples_ms)}",
            "Raw fused-unpartitioned samples [ms]",
            f"  {_format_samples(fused_stats.samples_ms)}",
            "Raw partitioned-serial samples [ms]",
            f"  {_format_samples(serial_stats.samples_ms)}",
            "Raw partitioned-overlap samples [ms]",
            f"  {_format_samples(overlap_stats.samples_ms)}",
            "Raw overlap-parent-graph samples [ms]",
            f"  {_format_samples(overlap_graph_stats.samples_ms)}",
            "",
            "Correctness",
            f"  active contacts per block : {PARTICLES_PER_PARTITION}",
            f"  max position difference   : {max_position_error:.8e}",
            f"  max velocity difference   : {max_velocity_error:.8e}",
            "  serial / overlap state    : PASS",
            f"  baseline position diff    : {baseline_position_error:.8e}",
            f"  baseline velocity diff    : {baseline_velocity_error:.8e}",
            "  baseline / partition state: PASS",
            f"  fused position diff       : {fused_position_error:.8e}",
            f"  fused velocity diff       : {fused_velocity_error:.8e}",
            "  fused / partition state   : PASS",
            f"  parent Graph position diff: {graph_position_error:.8e}",
            f"  parent Graph velocity diff: {graph_velocity_error:.8e}",
            "  parent Graph state        : PASS",
            "",
            "Measurement scope",
            "  - The baseline uses the original two-operator soft-contact path in one model.",
            "  - Fused unpartitioned isolates operator fusion from spatial decomposition.",
            "  - Partitioned serial isolates decomposition from stream overlap.",
            "  - Partitioned overlap launches four child graphs from the host.",
            "  - The parent Graph captures the complete multi-stream fork/join schedule.",
            "  - Each graph preserves collide() followed by solver.step() within its partition.",
            "  - Independent partitions own separate state, contact, and force buffers.",
        ]
    )
    print(report)
    log_path = _write_report(report, run_started_at)
    print(f"\nWrote report to: {log_path}")


if __name__ == "__main__":
    main()
