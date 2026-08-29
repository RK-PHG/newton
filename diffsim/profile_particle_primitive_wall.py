# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile the fixed particle-to-primitive soft-contact workload.

The script keeps the scene parameters fixed and writes one timestamped report
per run under ``diffsim/log``. The standard Newton viewer is enabled by default;
pass ``--viewer null`` for uncontaminated performance measurements.

Run from the repository root::

    uv run python diffsim/profile_particle_primitive_wall.py
"""

from __future__ import annotations

import platform
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp
from particle_primitive_wall import (
    PARTICLE_COUNT,
    SHAPE_COUNT,
    SIM_DT,
    SOFT_CONTACT_MARGIN,
    build_contact_snapshot,
    build_model,
)
from particle_primitive_wall import (
    Example as SceneExample,
)

import newton
import newton.examples

WARMUP_ITERATIONS = 5
TIMING_ITERATIONS = 20
LOG_DIR = Path(__file__).resolve().parent / "log"

try:
    import nvtx as _nvtx  # noqa: F401

    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False


@wp.kernel
def particle_velocity_loss(particle_qd: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    """Make the one-step soft-contact response observable by the reverse pass."""
    particle_index = wp.tid()
    velocity = particle_qd[particle_index]
    wp.atomic_add(loss, 0, wp.dot(velocity, velocity))


@dataclass
class StageStats:
    """Synchronized elapsed-time samples for one profiling stage."""

    name: str
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


class Workload:
    """Fixed, all-particles-in-contact profiling state."""

    def __init__(
        self,
        device: wp.Device,
        enable_soft_contact_candidates: bool,
        enable_fused_soft_contact: bool,
    ):
        self.device = device
        self.enable_soft_contact_candidates = enable_soft_contact_candidates
        self.enable_fused_soft_contact = enable_fused_soft_contact
        self.model = build_model(device=device, requires_grad=True)
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            enable_soft_contact_candidates=enable_soft_contact_candidates,
            enable_fused_soft_contact=enable_fused_soft_contact,
            requires_grad=True,
        )
        self.contacts = self.pipeline.contacts()

        self.fusion_baseline_pipeline = None
        self.fusion_baseline_contacts = None
        if enable_fused_soft_contact:
            self.fusion_baseline_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                soft_contact_margin=SOFT_CONTACT_MARGIN,
                enable_soft_contact_candidates=enable_soft_contact_candidates,
                enable_fused_soft_contact=False,
                requires_grad=True,
            )
            self.fusion_baseline_contacts = self.fusion_baseline_pipeline.contacts()

        self.collision_only_pipeline = self.fusion_baseline_pipeline or self.pipeline
        self.collision_only_contacts = self.collision_only_pipeline.contacts()
        self.baseline_pipeline = None
        self.baseline_contacts = None
        if enable_soft_contact_candidates:
            self.baseline_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                soft_contact_margin=SOFT_CONTACT_MARGIN,
                enable_soft_contact_candidates=False,
                requires_grad=True,
            )
            self.baseline_contacts = self.baseline_pipeline.contacts()
        self.control = self.model.control(requires_grad=True)
        self.state_in = self.model.state(requires_grad=True)
        self.state_out = self.model.state(requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        self.contact_body_pos_seed = wp.full(
            self.collision_only_contacts.soft_contact_max,
            value=wp.vec3(1.0, 1.0, 1.0),
            dtype=wp.vec3,
            device=device,
        )

        particle_q, particle_qd = build_contact_snapshot()
        self.state_in.particle_q.assign(particle_q)
        self.state_in.particle_qd.assign(particle_qd)

    def clear_forces(self) -> None:
        self.state_in.clear_forces()

    def collide(self) -> None:
        self.pipeline.collide(self.state_in, self.contacts)

    def collide_baseline(self) -> None:
        if self.baseline_pipeline is None or self.baseline_contacts is None:
            raise RuntimeError("Baseline pipeline is only available when candidates are enabled")
        self.baseline_pipeline.collide(self.state_in, self.baseline_contacts)

    def solve(self) -> None:
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, SIM_DT)

    def prepare_solver_timing(self) -> None:
        """Reset force outputs and the deferred counter before isolated solver timing."""
        self.clear_forces()
        if self.enable_fused_soft_contact:
            self.contacts.soft_contact_count.zero_()

    def simulation_forward(self) -> None:
        self.clear_forces()
        self.collide()
        self.solve()

    def fusion_baseline_forward(self) -> None:
        """Run the original two-operator path paired with the fused pipeline."""
        if self.fusion_baseline_pipeline is None or self.fusion_baseline_contacts is None:
            raise RuntimeError("Fusion baseline is only available when fused soft contact is enabled")
        self.clear_forces()
        self.fusion_baseline_pipeline.collide(self.state_in, self.fusion_baseline_contacts)
        self.solver.step(self.state_in, self.state_out, self.control, self.fusion_baseline_contacts, SIM_DT)

    def record_collision_forward(self) -> tuple[wp.Tape, list[wp.TimingResult]]:
        """Record only differentiable soft-contact generation."""
        tape = wp.Tape()
        activity_timer = _activity_timer("collision_forward")
        with _nvtx_range("diffsim/collision_forward"), activity_timer, tape:
            self.collision_only_pipeline.collide(self.state_in, self.collision_only_contacts)
        wp.synchronize_device(self.device)
        return tape, activity_timer.timing_results

    def collision_backward(self, tape: wp.Tape) -> None:
        """Backpropagate through collision generation without solver kernels."""
        tape.backward(
            grads={
                self.collision_only_contacts.soft_contact_body_pos: self.contact_body_pos_seed,
            }
        )

    def record_differentiable_forward(
        self,
        operation: Callable[[], None] | None = None,
        name: str = "differentiable_forward",
    ) -> tuple[wp.Tape, list[wp.TimingResult]]:
        """Record one forward pass and collect its individual CUDA activities."""
        if operation is None:
            operation = self.simulation_forward
        self.loss.zero_()
        tape = wp.Tape()
        activity_timer = _activity_timer(name)
        with _nvtx_range(f"diffsim/{name}"), activity_timer, tape:
            operation()
            wp.launch(
                particle_velocity_loss,
                dim=PARTICLE_COUNT,
                inputs=[self.state_out.particle_qd],
                outputs=[self.loss],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        return tape, activity_timer.timing_results


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when Warp's optional NVTX dependency is present."""
    if NVTX_AVAILABLE:
        with wp.ScopedTimer(name, print=False, use_nvtx=True, skip_tape=True):
            yield
    else:
        yield


def _activity_timer(name: str) -> wp.ScopedTimer:
    """Create a silent Warp CUDA-activity collector."""
    return wp.ScopedTimer(
        name,
        print=False,
        synchronize=True,
        cuda_filter=wp.TIMING_ALL,
        skip_tape=True,
    )


def _measure_stage(
    name: str,
    operation: Callable[[], None],
    device: wp.Device,
    *,
    prepare: Callable[[], None] | None = None,
) -> StageStats:
    samples_ms: list[float] = []
    for _ in range(TIMING_ITERATIONS):
        if prepare is not None:
            prepare()
        wp.synchronize_device(device)
        start = time.perf_counter()
        with _nvtx_range(f"diffsim/measure/{name}"):
            operation()
        wp.synchronize_device(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return StageStats(name=name, samples_ms=samples_ms)


def _capture_activity(name: str, operation: Callable[[], None], device: wp.Device) -> list[wp.TimingResult]:
    timer = _activity_timer(name)
    with _nvtx_range(f"diffsim/activity/{name}"), timer:
        operation()
    wp.synchronize_device(device)
    return timer.timing_results


def _activity_kind(activity_filter: int) -> str:
    kinds = (
        (wp.TIMING_KERNEL, "kernel"),
        (wp.TIMING_KERNEL_BUILTIN, "builtin"),
        (wp.TIMING_MEMCPY, "memcpy"),
        (wp.TIMING_MEMSET, "memset"),
        (wp.TIMING_GRAPH, "graph"),
    )
    for flag, name in kinds:
        if activity_filter == flag:
            return name
    return f"type-{activity_filter}"


def _aggregate_activities(results: list[wp.TimingResult]) -> list[tuple[str, str, int, float, float]]:
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result in results:
        aggregate[(_activity_kind(result.filter), result.name)].append(float(result.elapsed))

    rows = []
    for (kind, name), samples in aggregate.items():
        rows.append((kind, name, len(samples), sum(samples), max(samples)))
    rows.sort(key=lambda row: row[3], reverse=True)
    return rows


def _activity_elapsed(results: list[wp.TimingResult], name_fragment: str) -> float:
    """Sum elapsed CUDA activity time for names containing ``name_fragment``."""
    return sum(float(result.elapsed) for result in results if name_fragment in result.name)


def _active_soft_contact_pairs(contacts: newton.Contacts) -> np.ndarray:
    """Return active particle-shape pairs in a stable order for correctness checks."""
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    particles = contacts.soft_contact_particle.numpy()[:count]
    shapes = contacts.soft_contact_shape.numpy()[:count]
    pairs = np.column_stack((particles, shapes))
    if count == 0:
        return pairs
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _format_activity_table(title: str, results: list[wp.TimingResult]) -> list[str]:
    rows = _aggregate_activities(results)
    total_ms = sum(row[3] for row in rows)
    lines = [title, "  share   total ms  calls    avg ms    max ms  type      activity"]
    if not rows:
        lines.append("  no CUDA activities captured")
        return lines

    for kind, name, count, elapsed, maximum in rows:
        share = 100.0 * elapsed / total_ms if total_ms else 0.0
        lines.append(
            f"  {share:5.1f}%  {elapsed:9.4f}  {count:5d}  {elapsed / count:8.4f}  {maximum:8.4f}  {kind:<8}  {name}"
        )
    lines.append(f"  activity total: {total_ms:.4f} ms (CUDA-event instrumentation enabled)")
    return lines


def _format_cuda_version(version: int | tuple[int, ...] | None) -> str:
    if version is None:
        return "unknown"
    if isinstance(version, tuple):
        return ".".join(str(component) for component in version)
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor} ({version})"


def _device_description(device: wp.Device) -> str:
    name = getattr(device, "name", None)
    arch = getattr(device, "arch", None)
    details = [str(device)]
    if name:
        details.append(str(name))
    if arch:
        details.append(f"sm_{arch}")
    return " / ".join(details)


def _write_report(report: str, run_started_at: datetime) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S_%f")
    log_path = LOG_DIR / f"particle_primitive_wall_profile_{timestamp}.log"
    with log_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(report)
        output.write("\n")
    return log_path


def run_profile(
    device_name: str,
    viewer_name: str,
    run_started_at: datetime,
    enable_soft_contact_candidates: bool,
    enable_fused_soft_contact: bool,
) -> str:
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise RuntimeError("This performance profile requires a CUDA device")

    workload = Workload(device, enable_soft_contact_candidates, enable_fused_soft_contact)
    for _ in range(WARMUP_ITERATIONS):
        workload.simulation_forward()
        if enable_fused_soft_contact:
            workload.fusion_baseline_forward()
    wp.synchronize_device(device)

    workload.simulation_forward()
    wp.synchronize_device(device)
    active_contacts = min(int(workload.contacts.soft_contact_count.numpy()[0]), workload.contacts.soft_contact_max)
    if active_contacts != PARTICLE_COUNT:
        raise RuntimeError(f"Expected {PARTICLE_COUNT} active soft contacts, got {active_contacts}")

    workload.collision_only_pipeline.collide(workload.state_in, workload.collision_only_contacts)
    wp.synchronize_device(device)
    optimized_contact_pairs = _active_soft_contact_pairs(workload.collision_only_contacts)

    compact_candidate_count = candidate_count = workload.pipeline.soft_rigid_contact_pair_count
    baseline_activities: list[wp.TimingResult] = []
    contact_pairs_match = True
    if enable_soft_contact_candidates:
        compact_candidate_count = int(workload.pipeline.soft_contact_candidate_count.numpy()[0])
        for _ in range(WARMUP_ITERATIONS):
            workload.collide_baseline()
        wp.synchronize_device(device)
        baseline_contact_pairs = _active_soft_contact_pairs(workload.baseline_contacts)
        contact_pairs_match = np.array_equal(optimized_contact_pairs, baseline_contact_pairs)
        if not contact_pairs_match:
            raise RuntimeError("Candidate compaction changed the active particle-shape contact pairs")
        baseline_activities = _capture_activity("baseline_collision_forward", workload.collide_baseline, device)

    fusion_baseline_force = None
    fused_force = None
    fusion_force_match = True
    fusion_state_match = True
    fusion_count_match = True
    if enable_fused_soft_contact:
        workload.fusion_baseline_forward()
        wp.synchronize_device(device)
        fusion_baseline_force = workload.state_in.particle_f.numpy().copy()
        fusion_baseline_q = workload.state_out.particle_q.numpy().copy()
        fusion_baseline_qd = workload.state_out.particle_qd.numpy().copy()
        fusion_baseline_count = min(
            int(workload.fusion_baseline_contacts.soft_contact_count.numpy()[0]),
            workload.fusion_baseline_contacts.soft_contact_max,
        )

        workload.simulation_forward()
        wp.synchronize_device(device)
        fused_force = workload.state_in.particle_f.numpy().copy()
        fused_q = workload.state_out.particle_q.numpy().copy()
        fused_qd = workload.state_out.particle_qd.numpy().copy()
        fused_count = min(int(workload.contacts.soft_contact_count.numpy()[0]), workload.contacts.soft_contact_max)

        fusion_force_match = np.allclose(fused_force, fusion_baseline_force, rtol=1.0e-5, atol=1.0e-6)
        fusion_state_match = np.allclose(fused_q, fusion_baseline_q, rtol=1.0e-5, atol=1.0e-6) and np.allclose(
            fused_qd, fusion_baseline_qd, rtol=1.0e-5, atol=1.0e-6
        )
        fusion_count_match = fused_count == fusion_baseline_count
        if not (fusion_force_match and fusion_state_match and fusion_count_match):
            raise RuntimeError("Fused soft-contact forward results do not match the original two-operator path")

    collision_stats = _measure_stage("collision_forward", workload.collide, device)
    solver_stats = _measure_stage(
        "solver_forward",
        workload.solve,
        device,
        prepare=workload.prepare_solver_timing,
    )
    forward_stats = _measure_stage("simulation_forward", workload.simulation_forward, device)

    collision_tape, collision_forward_activities = workload.record_collision_forward()
    collision_tape.backward(
        grads={
            workload.collision_only_contacts.soft_contact_body_pos: workload.contact_body_pos_seed,
        }
    )
    wp.synchronize_device(device)
    collision_tape.zero()
    wp.synchronize_device(device)

    collision_backward_stats = _measure_stage(
        "collision_backward",
        lambda: workload.collision_backward(collision_tape),
        device,
        prepare=collision_tape.zero,
    )
    collision_tape.zero()
    wp.synchronize_device(device)
    collision_backward_activities = _capture_activity(
        "collision_backward",
        lambda: workload.collision_backward(collision_tape),
        device,
    )
    collision_tape.zero()
    wp.synchronize_device(device)

    fusion_baseline_forward_activities: list[wp.TimingResult] = []
    fusion_baseline_backward_activities: list[wp.TimingResult] = []
    fusion_loss_match = True
    fusion_gradient_match = True
    fusion_baseline_loss = 0.0
    fused_loss = 0.0
    if enable_fused_soft_contact:
        fusion_baseline_tape, fusion_baseline_forward_activities = workload.record_differentiable_forward(
            workload.fusion_baseline_forward,
            "fusion_baseline_forward",
        )
        fusion_baseline_tape.backward(workload.loss)
        wp.synchronize_device(device)
        fusion_baseline_loss = float(workload.loss.numpy()[0])
        fusion_baseline_gradient = workload.state_in.particle_q.grad.numpy().copy()
        fusion_baseline_tape.zero()
        wp.synchronize_device(device)
        fusion_baseline_backward_activities = _capture_activity(
            "fusion_baseline_backward",
            lambda: fusion_baseline_tape.backward(workload.loss),
            device,
        )
        fusion_baseline_tape.zero()
        wp.synchronize_device(device)

    simulation_tape, forward_activities = workload.record_differentiable_forward()
    simulation_tape.backward(workload.loss)
    wp.synchronize_device(device)
    if enable_fused_soft_contact:
        fused_loss = float(workload.loss.numpy()[0])
        fused_gradient = workload.state_in.particle_q.grad.numpy().copy()
        fusion_loss_match = bool(np.isclose(fused_loss, fusion_baseline_loss, rtol=1.0e-5, atol=1.0e-6))
        fusion_gradient_match = np.allclose(
            fused_gradient,
            fusion_baseline_gradient,
            rtol=1.0e-4,
            atol=1.0e-5,
        )
        if not (fusion_loss_match and fusion_gradient_match):
            raise RuntimeError("Fused soft-contact loss or particle-position gradient does not match baseline")
    simulation_tape.zero()
    wp.synchronize_device(device)

    backward_stats = _measure_stage(
        "simulation_backward",
        lambda: simulation_tape.backward(workload.loss),
        device,
        prepare=simulation_tape.zero,
    )
    backward_activities = _capture_activity(
        "simulation_backward",
        lambda: simulation_tape.backward(workload.loss),
        device,
    )
    simulation_tape.zero()
    wp.synchronize_device(device)

    profiler_api_available = hasattr(wp, "ScopedCudaProfiler")
    if profiler_api_available:
        with wp.ScopedCudaProfiler(device), _nvtx_range("diffsim/profile_capture"):
            with _nvtx_range("diffsim/profile_capture/collision_forward"):
                workload.clear_forces()
                workload.collide()
            with _nvtx_range("diffsim/profile_capture/solver_forward"):
                workload.solve()
            with _nvtx_range("diffsim/profile_capture/collision_backward"):
                collision_tape.zero()
                workload.collision_backward(collision_tape)
            with _nvtx_range("diffsim/profile_capture/simulation_backward"):
                simulation_tape.zero()
                simulation_tape.backward(workload.loss)
            wp.synchronize_device(device)

    stage_stats = (
        collision_stats,
        solver_stats,
        forward_stats,
        collision_backward_stats,
        backward_stats,
    )
    collision_forward_rows = _aggregate_activities(collision_forward_activities)
    collision_backward_rows = _aggregate_activities(collision_backward_activities)
    forward_rows = _aggregate_activities(forward_activities)
    backward_rows = _aggregate_activities(backward_activities)
    top_collision_forward = collision_forward_rows[0][1] if collision_forward_rows else "not captured"
    top_collision_backward = collision_backward_rows[0][1] if collision_backward_rows else "not captured"
    top_forward = forward_rows[0][1] if forward_rows else "not captured"
    top_backward = backward_rows[0][1] if backward_rows else "not captured"
    collision_share = 100.0 * collision_stats.mean / forward_stats.mean
    backward_ratio = backward_stats.mean / forward_stats.mean
    baseline_soft_ms = _activity_elapsed(baseline_activities, "create_soft_contacts")
    candidate_kernel_ms = _activity_elapsed(collision_forward_activities, "build_soft_contact_candidates")
    candidate_soft_ms = _activity_elapsed(collision_forward_activities, "create_soft_contacts")
    candidate_operator_sum_ms = candidate_kernel_ms + candidate_soft_ms
    comparison_available = baseline_soft_ms > 0.0 and candidate_kernel_ms > 0.0 and candidate_soft_ms > 0.0
    candidate_comparison_passed = comparison_available and candidate_operator_sum_ms < baseline_soft_ms
    fusion_baseline_create_ms = _activity_elapsed(fusion_baseline_forward_activities, "create_soft_contacts")
    fusion_baseline_response_ms = _activity_elapsed(
        fusion_baseline_forward_activities, "eval_particle_body_contact_"
    )
    fusion_baseline_operator_sum_ms = fusion_baseline_create_ms + fusion_baseline_response_ms
    fused_operator_ms = _activity_elapsed(forward_activities, "eval_particle_body_contact_fused")
    fusion_comparison_available = (
        fusion_baseline_create_ms > 0.0 and fusion_baseline_response_ms > 0.0 and fused_operator_ms > 0.0
    )
    fusion_comparison_passed = fusion_comparison_available and fused_operator_ms < fusion_baseline_operator_sum_ms
    fusion_baseline_create_backward_ms = _activity_elapsed(
        fusion_baseline_backward_activities, "create_soft_contacts"
    )
    fusion_baseline_response_backward_ms = _activity_elapsed(
        fusion_baseline_backward_activities, "eval_particle_body_contact_"
    )
    fusion_baseline_backward_sum_ms = fusion_baseline_create_backward_ms + fusion_baseline_response_backward_ms
    fused_backward_ms = _activity_elapsed(backward_activities, "eval_particle_body_contact_fused")

    lines = [
        "=" * 96,
        "ParticlePrimitiveWall collision profile",
        f"timestamp              : {run_started_at.isoformat(timespec='seconds')}",
        f"platform               : {platform.platform()}",
        f"python                 : {sys.version.split()[0]}",
        f"newton                 : {newton.__version__}",
        f"warp                   : {wp.__version__}",
        f"device                 : {_device_description(device)}",
        f"CUDA driver            : {_format_cuda_version(wp.get_cuda_driver_version())}",
        f"CUDA toolkit (Warp)    : {_format_cuda_version(wp.get_cuda_toolkit_version())}",
        f"NVTX Python available  : {NVTX_AVAILABLE}",
        f"nsys executable found  : {shutil.which('nsys') or 'no'}",
        f"viewer                 : {viewer_name} ({'enabled' if viewer_name != 'null' else 'disabled'})",
        f"soft candidates        : {'enabled' if enable_soft_contact_candidates else 'disabled'}",
        f"fused soft contact     : {'enabled' if enable_fused_soft_contact else 'disabled'}",
        f"warmup / samples       : {WARMUP_ITERATIONS} / {TIMING_ITERATIONS}",
        "",
        "Fixed workload",
        f"  particles             : {PARTICLE_COUNT}",
        f"  static primitives     : {SHAPE_COUNT} (24 box, 4 capsule, 2 cylinder, 2 sphere)",
        "  enabled contact type  : soft particle-static-shape only",
        f"  candidate pairs       : {candidate_count}",
        f"  compact candidates    : {compact_candidate_count}",
        f"  soft-contact capacity : {workload.contacts.soft_contact_max}",
        f"  active contacts       : {active_contacts}",
        f"  candidates / active   : {candidate_count / active_contacts:.1f}x",
        "",
        "Synchronized stage timings",
        "  stage                    mean ms     min ms     p50 ms     p95 ms   samples",
    ]
    for stats in stage_stats:
        lines.append(
            f"  {stats.name:<24}  {stats.mean:9.4f}  {stats.minimum:9.4f}  "
            f"{stats.median:9.4f}  {stats.p95:9.4f}  {len(stats.samples_ms):8d}"
        )

    if enable_soft_contact_candidates:
        comparison_status = "PASS" if candidate_comparison_passed else "FAIL"
        if not comparison_available:
            comparison_status = "N/A"
        speedup = baseline_soft_ms / candidate_operator_sum_ms if candidate_operator_sum_ms > 0.0 else 0.0
        lines.extend(
            [
                "",
                "Candidate optimization comparison",
                f"  baseline create_soft_contacts       : {baseline_soft_ms:.4f} ms",
                f"  candidate operator                  : {candidate_kernel_ms:.4f} ms",
                f"  candidate create_soft_contacts      : {candidate_soft_ms:.4f} ms",
                f"  candidate + soft-contact            : {candidate_operator_sum_ms:.4f} ms",
                f"  operator speedup                    : {speedup:.2f}x",
                f"  active contact pairs match baseline : {'PASS' if contact_pairs_match else 'FAIL'}",
                f"  candidate + soft < baseline soft    : {comparison_status}",
            ]
        )

    if enable_fused_soft_contact:
        fusion_status = "PASS" if fusion_comparison_passed else "FAIL"
        if not fusion_comparison_available:
            fusion_status = "N/A"
        fusion_speedup = (
            fusion_baseline_operator_sum_ms / fused_operator_ms if fused_operator_ms > 0.0 else 0.0
        )
        fusion_backward_speedup = (
            fusion_baseline_backward_sum_ms / fused_backward_ms if fused_backward_ms > 0.0 else 0.0
        )
        max_force_error = float(np.max(np.abs(fused_force - fusion_baseline_force)))
        lines.extend(
            [
                "",
                "Soft-contact fusion comparison",
                f"  baseline create_soft_contacts       : {fusion_baseline_create_ms:.4f} ms",
                f"  baseline eval_particle_body_contact : {fusion_baseline_response_ms:.4f} ms",
                f"  baseline two-operator sum           : {fusion_baseline_operator_sum_ms:.4f} ms",
                f"  fused detection + response          : {fused_operator_ms:.4f} ms",
                f"  forward operator speedup            : {fusion_speedup:.2f}x",
                f"  fused < baseline two-operator sum   : {fusion_status}",
                f"  baseline backward operator sum      : {fusion_baseline_backward_sum_ms:.4f} ms",
                f"  fused backward operator             : {fused_backward_ms:.4f} ms",
                f"  backward operator speedup           : {fusion_backward_speedup:.2f}x",
                f"  active contact count match          : {'PASS' if fusion_count_match else 'FAIL'}",
                f"  particle force match                : {'PASS' if fusion_force_match else 'FAIL'}",
                f"  integrated state match              : {'PASS' if fusion_state_match else 'FAIL'}",
                f"  one-step loss match                 : {'PASS' if fusion_loss_match else 'FAIL'}",
                f"  particle_q gradient match           : {'PASS' if fusion_gradient_match else 'FAIL'}",
                f"  max abs particle-force error        : {max_force_error:.8e}",
                f"  baseline / fused loss               : {fusion_baseline_loss:.8f} / {fused_loss:.8f}",
            ]
        )

    lines.extend(
        [
            "",
            "Derived analysis",
            f"  collision / simulation forward : {collision_share:.1f}%",
            f"  backward / simulation forward  : {backward_ratio:.2f}x",
            f"  top collision forward activity : {top_collision_forward}",
            f"  top collision backward activity: {top_collision_backward}",
            f"  top forward CUDA activity       : {top_forward}",
            f"  top backward CUDA activity      : {top_backward}",
            "  interpretation                  : baseline soft-contact evaluates particle x shape",
            "                                    even though one contact per particle is active here",
            "",
        ]
    )
    if baseline_activities:
        lines.extend(_format_activity_table("Baseline collision CUDA activities", baseline_activities))
        lines.append("")
    if fusion_baseline_forward_activities:
        lines.extend(_format_activity_table("Fusion baseline forward CUDA activities", fusion_baseline_forward_activities))
        lines.append("")
    if fusion_baseline_backward_activities:
        lines.extend(
            _format_activity_table("Fusion baseline backward CUDA activities", fusion_baseline_backward_activities)
        )
        lines.append("")
    lines.extend(_format_activity_table("Collision-only forward CUDA activities", collision_forward_activities))
    lines.append("")
    lines.extend(_format_activity_table("Collision-only backward CUDA activities", collision_backward_activities))
    lines.append("")
    lines.extend(_format_activity_table("Differentiable forward CUDA activities", forward_activities))
    lines.append("")
    lines.extend(_format_activity_table("Tape backward CUDA activities", backward_activities))
    lines.extend(
        [
            "",
            "Measurement notes",
            "  - Stage timings synchronize the CUDA device and exclude first-use compilation via warm-up.",
            "  - CUDA activity timings inject events; use them to rank kernels, not as stage totals.",
            "  - simulation_forward includes force clear, collision, contact solve, and integration.",
            "  - collision_backward seeds soft_contact_body_pos and excludes solver adjoints.",
            "  - simulation_backward includes collision, contact-force, integration, and loss adjoints.",
            "  - fused mode preserves collide(); solver.step() while deferring soft-contact geometry to the solver.",
            "  - fused mode does not materialize soft-contact geometry arrays.",
            "  - Gradient zeroing runs before each backward sample and is not included in its timing.",
            f"  - External profiler capture markers available: {profiler_api_available}.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = newton.examples.create_parser()
    parser.description = __doc__
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
    parser.set_defaults(device="cuda:0", viewer="gl")
    viewer, args = newton.examples.init(parser)

    run_started_at = datetime.now().astimezone()
    report = run_profile(
        args.device,
        args.viewer,
        run_started_at,
        args.soft_contact_candidates,
        args.fused_soft_contact,
    )
    print(report)
    log_path = _write_report(report, run_started_at)
    print(f"\nWrote report to: {log_path}")

    if args.viewer == "null":
        viewer.close()
    else:
        newton.examples.run(SceneExample(viewer, args), args)


if __name__ == "__main__":
    main()
