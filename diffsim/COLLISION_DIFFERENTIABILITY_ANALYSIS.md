# Newton 碰撞可微分链路源码分析

> 分析基线：仓库提交 `71452a4e`，2026-08-25。本文关注 Newton 原生碰撞管线、接触响应和 Warp 反向传播；不把“代码能在 `wp.Tape` 中运行”与“整个碰撞过程数学上处处可微”混为一谈。

## 1. 结论先行

Newton 当前没有对刚体碰撞做 broad phase → narrow phase → contact set 的端到端求导。真实实现分成两条链路：

1. **刚体—刚体接触：冻结检测结果，求接触响应的路径导数。**
   broad phase、narrow phase、接触排序/匹配等离散步骤不进入 Tape。它们输出形体索引、body-local 接触点、world-space 法向和 margin。后续接触几何辅助函数或求解器再用可微的 `state.body_q/body_qd` 重建世界坐标、间距和力。因此梯度能回到位姿、速度、质量/材料等连续量，但不会穿过接触集合、法向选择、特征选择和 GJK/MPR/BVH 的离散决策。

2. **软体粒子—刚体形体接触：检测 kernel 可以进入 Tape。**
   `create_soft_contacts()` 读取 `particle_q`、`body_q` 并写入可带梯度的 soft-contact 数组。原子计数分配通过 `@wp.func_replay` 在反向时复用前向线程对应的接触槽位。即使把接触几何冻结，求解器仍会用“当前粒子位置 - 冻结表面点”重新计算 gap，所以存在一条稳定、直接的梯度路径。

3. **仿真前向/反向的官方主路径是 `SolverSemiImplicit`，`SolverFeatherstone` 也标为 basic differentiable。**
   `SolverXPBD` 虽有一些为 `requires_grad` 准备的非原地缓冲逻辑，但当前官方能力矩阵仍标为不可微；`SolverVBD`、`SolverKamino` 等关键模块显式关闭了 backward，不能据局部数学函数可微就推断整个 solver 可微。

4. **`eval_rigid_contact_kinematics()` 只是接触几何的可微读取接口，不是 solver 必经节点。**
   `SolverSemiImplicit` 会直接在 `eval_body_contact()` 中用相同的 body-local 点和实时 `body_q` 重建接触点并算力。辅助接口适合自定义 loss（距离、世界接触点），其输出不需要再喂给内置 solver。

整体上，Newton 给出的是固定活动集下的分段/路径导数，而不是跨接触生成、消失和特征切换的全局光滑导数。

## 2. 总体数据流

```text
ModelBuilder.finalize(requires_grad=True)
  ├─ Model 中连续参数数组可带梯度
  └─ model.state(requires_grad=True)
       ├─ body_q, body_qd, body_f
       └─ particle_q, particle_qd, particle_f

每个仿真子步（在 wp.Tape 内）
  state[t].clear_forces()
  CollisionPipeline.collide(state[t], contacts[t])
    ├─ 刚体检测：AABB → broad phase → narrow phase
    │    ├─ record_tape=False / enable_backward=False
    │    └─ frozen Contacts:
    │         shape0/1, point0/1(local), normal(world), margin0/1
    ├─ 可选刚体增强：local point + live body_q → distance/world points
    │    └─ 进入 Tape，但不改变内置 solver 的输入
    └─ 软接触检测：create_soft_contacts / edge-face SDF passes
         ├─ 进入 Tape
         ├─ counter_increment_replay 固定反向槽位映射
         └─ soft contact body_pos/body_vel/normal/barycentric

  SolverSemiImplicit.step(state[t], state[t+1], control, contacts[t], dt)
    ├─ eval_body_contact              # 刚体接触力
    ├─ eval_particle_body_contact     # 粒子—形体接触力
    ├─ integrate_particles
    └─ integrate_bodies

  loss_kernel(state[T] → scalar loss)

tape.backward(loss)
  loss
   ← integration adjoints
   ← contact-force adjoints
   ← live state / model / control arrays
   ←（仅软接触及显式刚体几何辅助）collision-derived continuous outputs
  不经过刚体 broad phase / narrow phase / contact-set decisions
```

## 3. 梯度开关与内存对象

### 3.1 Model 和 State

入口是 [`ModelBuilder.finalize()`](newton/_src/sim/builder.py#L11243)。`requires_grad=True` 被保存为 `model.requires_grad`，并用于创建粒子、刚体、形体材料、质量、惯量、关节等连续数组，见 [`builder.py:11330`](newton/_src/sim/builder.py#L11330) 及随后各数组的构造。

[`Model.state()`](newton/_src/sim/model.py#L1779) 默认继承 `model.requires_grad`，分别 clone/allocate：

- `particle_q / particle_qd / particle_f`
- `body_q / body_qd / body_f`
- `joint_q / joint_qd`

多步反传时，每个时间点都应有独立的 `State` 数组，形成 `state[0] → state[1] → ... → state[T]` 的计算图。

### 3.2 CollisionPipeline.requires_grad 的准确含义

[`CollisionPipeline.__init__()`](newton/_src/sim/collide.py#L1111) 中 `requires_grad=None` 会继承 `model.requires_grad`。该标志主要控制：

- soft-contact 输出数组是否 `requires_grad=True`；
- 已弃用的 `contacts.rigid_contact_diff_*` 兼容数组是否分配并自动填充。

它**不控制**以下能力：

- 显式调用 `eval_rigid_contact_kinematics()`；该接口只要求输入/输出数组按需带梯度；
- solver 通过实时 `state.body_q/body_qd` 对冻结刚体接触几何求导。

因此 `CollisionPipeline(..., requires_grad=False)` 仍可用于“刚体检测 + 显式刚体接触距离 loss”的场景；但想让 soft-contact 检测输出参与反向，就需要 `requires_grad=True`。

### 3.3 Contacts 的梯度布局

[`Contacts.__init__()`](newton/_src/sim/contacts.py#L164) 明确区分两类数组：

- 刚体原始接触数组从不要求梯度，因为 narrow phase 不反传：`rigid_contact_point0/1`、`normal`、`shape0/1`、`margin0/1` 等是冻结数据，见 [`contacts.py:244`](newton/_src/sim/contacts.py#L244)。
- soft-contact 的 `barycentric`、`body_pos`、`body_vel`、`normal` 会继承 pipeline 的 `requires_grad`，见 [`contacts.py:349`](newton/_src/sim/contacts.py#L349)。

这不是说刚体接触对状态没有梯度；它表示梯度从 solver 对 `body_q/body_qd` 的重新计算产生，而不是从 narrow-phase 输出数组产生。

## 4. 前向链路：刚体碰撞

### 4.1 检测阶段被刻意移出 Tape

公共入口是 [`CollisionPipeline.collide()`](newton/_src/sim/collide.py#L1765)。刚体部分依次执行：

1. `compute_shape_aabbs`：由 `state.body_q` 和 shape transform 计算 AABB，同时清 contact counter；调用显式设置 `record_tape=False`。
2. `BroadPhaseAllPairs` / `BroadPhaseSAP` / `BroadPhaseExplicit`：生成候选 shape pairs。相关 launch 均设置 `record_tape=False`，例如 [`broad_phase_nxn.py:426`](newton/_src/geometry/broad_phase_nxn.py#L426) 和 [`broad_phase_sap.py:765`](newton/_src/geometry/broad_phase_sap.py#L765)。
3. `NarrowPhase.launch_custom_write()`：按形体类型进入 primitive、GJK、MPR、mesh/heightfield、SDF/hydroelastic 等路径，写入 `Contacts`。该类在 [`narrow_phase.py:2640`](newton/_src/geometry/narrow_phase.py#L2640) 说明所有内部 launch 都使用 `record_tape=False`；许多 kernel 还显式 `enable_backward=False`。
4. 可选的排序、匹配、sticky replay、contact report 仍属于离散接触管理，不构成状态梯度路径。

检测输出的核心语义是：

- `rigid_contact_shape0/1`：接触的形体索引；
- `rigid_contact_point0/1`：body-local 支撑点；
- `rigid_contact_normal`：从 shape 0 指向 shape 1 的 world-space 单位法向；
- `rigid_contact_margin0/1`：有效表面厚度。

### 4.2 两种可微消费方式

#### A. 自定义接触几何 loss

[`eval_rigid_contact_kinematics()`](newton/_src/sim/contact_kinematics.py#L35) 调用 [`rigid_contact_kinematics_kernel()`](newton/_src/geometry/differentiable_contacts.py#L30)：

```text
p_a = transform_point(body_q[body_a], point0_local)
p_b = transform_point(body_q[body_b], point1_local)
d   = dot(n_frozen, p_b - p_a) - margin0 - margin1
```

可选输出为 `distance`、`point0_world`、`point1_world`。kernel 正常进入 Tape，因此 loss 对 `state.body_q` 有梯度；法向只是复制/读取的冻结常量，梯度不经过法向方向。

#### B. 内置接触响应

`SolverSemiImplicit.step()` 在 [`solver_semi_implicit.py:124`](newton/_src/solvers/semi_implicit/solver_semi_implicit.py#L124) 调用 [`eval_body_contact_forces()`](newton/_src/solvers/semi_implicit/kernels_contact.py#L601)，最终执行 [`eval_body_contact()`](newton/_src/solvers/semi_implicit/kernels_contact.py#L382)。该 kernel：

1. 从 shape index 找到两个 body 和材料参数；
2. 用实时 `body_q` 把冻结的 local point 转回 world point；
3. 加入 margin 得到有效表面位置并计算 signed separation `d`；
4. 用实时 `body_qd` 算接触点相对速度、法向速度和切向速度；
5. 计算弹性、阻尼和经 Huber norm 平滑的摩擦力；
6. 通过 `wp.atomic_add/sub` 累加到 `state.body_f`。

所以刚体动力学的连续梯度链是：

```text
body_q, body_qd, material parameters
  → world contact points / relative velocity / penetration
  → normal+damping+friction force
  → body_f
  → integrate_bodies
  → next body_q, body_qd
```

注意：这条链完全不读取 `rigid_contact_diff_distance`；兼容增强数组和 solver 接触力是并行消费者。

## 5. 前向链路：软体粒子—刚体碰撞

### 5.1 检测 kernel 进入 Tape

刚体检测结束后，`CollisionPipeline.collide()` 在 [`collide.py:2152`](newton/_src/sim/collide.py#L2152) 启动 [`create_soft_contacts()`](newton/_src/geometry/kernels.py#L1103)。每个预计算的 particle-shape pair：

1. 从 `particle_q` 取得粒子位置，从 `body_q * shape_transform` 得到 shape frame；
2. 把粒子转到 shape-local 坐标；
3. 对 sphere/box/capsule/cylinder/cone/ellipsoid/plane/heightfield 或 mesh query 计算 SDF、最近点和法向；
4. 若 `d < soft_contact_margin + shape_margin + particle_radius`，分配一个 contact slot；
5. 写入 shape index、body-local surface point/velocity、world normal，并用 `(p,-1,-1)` 与 `(1,0,0)` 表示粒子特征。

edge/face 全表面接触由 [`soft_contacts_sdf.py`](newton/_src/geometry/soft_contacts_sdf.py) 的 SDF 优化路径追加，记录 corner indices 与 barycentric weights。当前只有 `SolverVBD` 消费完整 edge/face contact，但 VBD solver 本身不支持 backward；这些数组仍可供自定义可微目标使用。

### 5.2 原子槽位分配为什么需要 replay

接触数和输出下标由 [`counter_increment()`](newton/_src/geometry/kernels.py#L1068) 的 `wp.atomic_add` 产生。反向 kernel 不能重新做一次原子分配，否则线程到 contact slot 的映射可能变化。

[`counter_increment_replay()`](newton/_src/geometry/kernels.py#L1094) 用 `@wp.func_replay` 返回前向保存在 `soft_contact_tids[tid]` 中的槽位。`Contacts` 为它分配的长度覆盖 particle + edge + face 的所有候选线程，而不是只按最终 contact capacity 分配。

这个机制保证的是“反向使用与前向相同的写入索引”，并不让接触数量、阈值比较或候选拓扑变成光滑函数。

### 5.3 接触响应阶段

`SolverSemiImplicit` 通过 [`eval_particle_body_contact_forces()`](newton/_src/solvers/semi_implicit/kernels_contact.py#L650) 调用 [`eval_particle_body_contact()`](newton/_src/solvers/semi_implicit/kernels_contact.py#L144)。关键计算为：

```text
px = live particle_q[particle]
bx = transform_point(live body_q[body], contact_body_pos)
c  = dot(contact_normal, px - bx) - particle_radius
```

随后由 `c`、相对速度和材料参数得到 normal/damping/friction force，分别原子累加到 `particle_f` 和 `body_f`。

这里存在两层梯度：

- 若 soft-contact 检测在 Tape 内且输出数组带梯度，可经 `body_pos/body_vel/normal` 回到检测输入；
- 即使把接触几何作为常量，`px` 和 `body_q/body_qd` 仍是实时状态，gap/force 对它们可微。这也是固定接触点模型最重要、通常更稳的梯度路径。

对 edge/face contact，SDF argmin 或离散特征可能让“接触点位置”路径为零，但 solver-relevant gap
`dot(n, Σ bary_i * particle_q[i] - bx)` 对参与顶点仍有 `bary_i * n` 的梯度。

## 6. 仿真反向链路

以 SemiImplicit 多步 rollout 为例，前向第 `t` 步是：

```text
x_t
  ├─ collide(x_t) → frozen rigid contacts + differentiable soft contacts
  ├─ contact kernels(x_t, contacts_t) → f_t
  └─ symplectic Euler(x_t, f_t, dt) → x_{t+1}
```

积分代码在 [`solver.py`](newton/_src/solvers/solver.py)：

- [`integrate_particles()`](newton/_src/solvers/solver.py#L24)：`v1 = v0 + a*dt`，`x1 = x0 + v1*dt`；
- [`integrate_rigid_body()`](newton/_src/solvers/solver.py#L64)：更新 COM 线速度/位置、body-frame 角速度、四元数并归一化；
- [`integrate_bodies()`](newton/_src/solvers/solver.py#L112)：逐 body 写下一状态。

当末端 loss 调用 `tape.backward(loss)` 时，Warp 按 launch 的逆序运行自动生成的 adjoint：

```text
∂L/∂x_T
  → integration backward
  → ∂L/∂f_t 和 ∂L/∂x_t（直接动力学项）
  → contact-force backward
  → ∂L/∂body_q_t, body_qd_t, particle_q_t, particle_qd_t
  → soft-contact detection backward（若该 launch 被记录）
  → 继续传播到 x_{t-1}
```

刚体检测 launch 不在 Tape 中，因此反向到接触力 kernel 后，会把 `point0/1`、`normal`、shape index 当常量，不再进入 broad/narrow phase。软接触反向则会使用 replay 的接触槽位映射。

[`test_multistep_gradient_flow()`](newton/tests/test_differentiable_contacts.py#L516) 实际覆盖了四个子步的 `collide → SolverSemiImplicit.step → final loss → tape.backward`，并把初始 `body_q.z` 的解析梯度与中心有限差分比较。

## 7. Solver 支持边界

以 [`docs/solvers/index.rst`](docs/solvers/index.rst#L60) 的官方 feature matrix 为准：

| Solver | 当前官方可微状态 | 碰撞相关说明 |
|---|---:|---|
| `SolverSemiImplicit` | basic | 本文详述的主链路；刚体与 particle-shape penalty contact kernel 可进入 Tape。粒子—粒子 hash-grid contact 的梯度有已知警告，DiffSim 示例会关闭 `model.particle_grid`。 |
| `SolverFeatherstone` | basic | 复用 SemiImplicit 的 `kernels_contact`；刚体接触力先在 maximal body 空间计算，再经 Featherstone/RNEA/ABA 链路进入 generalized state。需逐案例做有限差分。 |
| `SolverXPBD` | no | 源码含 `requires_grad` 下避免原地覆盖的缓冲策略，但官方矩阵未承诺可微，不能视为受支持的碰撞反传后端。 |
| `SolverVBD` | no | `particle_vbd_kernels.py` 和 `rigid_vbd_kernels.py` 模块级 `enable_backward=False`。检测或自定义 gap 的局部梯度不等于 VBD step 可反传。 |
| `SolverKamino` | no | 大量模块级 `enable_backward=False`，包括 core、geometry、integrator 和 solver。 |
| `SolverMuJoCo` | no | MuJoCo/native contact 路径不在 Newton 的受支持 Tape 链中。即使用 Newton contacts，也不能据此推导整个 MuJoCo step 可微。 |
| `SolverStyle3D` / `SolverImplicitMPM` | no | 当前 feature matrix 未提供可微仿真承诺。 |

源码中“某个 kernel 没有写 `enable_backward=False`”只代表 Warp 可以尝试生成 adjoint，不等于 solver 在数组生命周期、迭代更新、外部后端和数值分支上建立了完整的可微支持契约。

## 8. 正确使用模板

### 8.1 动态接触、多步反传

```python
model = builder.finalize(requires_grad=True)
solver = newton.solvers.SolverSemiImplicit(model)
pipeline = newton.CollisionPipeline(model, requires_grad=True)

states = [model.state(requires_grad=True) for _ in range(num_steps + 1)]
control = model.control(requires_grad=True)
contacts_per_step = [pipeline.contacts() for _ in range(num_steps)]
loss = wp.zeros(1, dtype=float, requires_grad=True)

with wp.Tape() as tape:
    for t in range(num_steps):
        states[t].clear_forces()
        pipeline.collide(states[t], contacts_per_step[t])
        solver.step(states[t], states[t + 1], control, contacts_per_step[t], dt)

    wp.launch(loss_kernel, dim=1, inputs=[states[-1].body_q], outputs=[loss])

tape.backward(loss)
grad_initial_pose = states[0].body_q.grad
```

每个子步使用独立 `Contacts`，可以避免同一组 contact 输出数组在一个 Tape 中被后续碰撞 pass 覆盖。仓库的 [`example_diffsim_soft_body.py:267`](newton/examples/diffsim/example_diffsim_soft_body.py#L267) 和多步回归测试都采用此方式。做 CUDA graph capture 时应在 capture 前预分配这些 buffer。

### 8.2 只对刚体接触距离构造 loss

```python
model = builder.finalize(requires_grad=True)
state = model.state(requires_grad=True)

# False 仅关闭 soft-contact grad 数组和旧兼容输出；显式 helper 仍可微。
pipeline = newton.CollisionPipeline(model, requires_grad=False)
contacts = pipeline.contacts()
distance = wp.empty(
    contacts.rigid_contact_max,
    dtype=float,
    requires_grad=True,
)

with wp.Tape() as tape:
    pipeline.collide(state, contacts)
    newton.eval_rigid_contact_kinematics(
        model,
        state,
        contacts,
        out_distance=distance,
    )

# 实际使用时只应 seed active contact rows；其余 capacity 行不是有效接触。
seed = wp.zeros_like(distance)
# 用 kernel 根据 rigid_contact_count 填充 seed[0:count] = 1
tape.backward(grads={distance: seed})
```

不要继续依赖已弃用的 `Contacts.rigid_contact_diff_*`；新代码应显式调用 `eval_rigid_contact_kinematics()` 并只分配需要的输出。

## 9. 不可微点与数值风险

### 9.1 接触拓扑不在梯度里

以下变化不会被刚体解析梯度捕获：

- 接触从无到有或从有到无；
- broad-phase pair 出现/消失；
- GJK/MPR 支撑特征、mesh triangle、edge/face 发生切换；
- manifold contact 数变化；
- contact reduction、sort、match 的离散选择变化。

有限差分扰动若跨过这些边界，会与 Tape 梯度明显不一致。这不是 adjoint 实现错误，而是固定活动集近似的边界。

### 9.2 法向和局部点冻结

刚体接触法向、body-local point、margin 都来自不反传的 narrow phase。位姿发生微小变化时，Tape 假设这些量不随状态变化，只让 local point 随 body transform 运动。因此结果是切平面近似。

### 9.3 分支、clamp 和摩擦

接触激活条件、`min/max/step/clamp`、Coulomb 上限、零切向速度附近的归一化都会产生分段梯度。SemiImplicit 使用 Huber norm 改善摩擦在零速附近的稳定性，但不能消除 stick/slip 和接触激活的非光滑性。

### 9.4 原子累加和确定性

接触力通过 atomic 累加。浮点加法次序会影响末位，进而影响长 rollout 的状态和梯度。需要可复现实验时，应使用 solver/pipeline 提供的 deterministic 选项，并仍以有限差分检查目标量。

### 9.5 contact buffer 的生命周期

- 一般动态接触：每个子步重新 `collide()`，并为一个 Tape 内的各时间步保留独立 `Contacts`。
- 简单静态平面且 margin 足够大：可以像 `example_diffsim_ball.py` 一样在 Tape 外检测一次，再把接触记录当固定几何复用；此时没有检测梯度，只有 live particle/body state 进入 force kernel 的梯度。
- contact capacity 是上限，loss/seed 只能使用 `contact_count` 以内的有效行。

## 10. 测试证据与建议验证方式

核心回归集中在 [`newton/tests/test_differentiable_contacts.py`](newton/tests/test_differentiable_contacts.py)：

- `test_eval_rigid_contact_kinematics`：显式刚体 kinematics 输出和 `body_q` 梯度；
- `test_finite_difference_distance_gradient` / two-body finite difference：刚体距离解析梯度对中心差分；
- `test_multistep_gradient_flow`：碰撞响应跨多个仿真子步回到初始位姿；
- `test_particle_soft_contact_gap_differentiable`：粒子 gap 对位置的梯度；
- `test_full_surface_gap_differentiable`：edge/face frozen barycentric gap 梯度；
- [`test_soft_contact_detection_differentiable_through_collide()`](newton/tests/test_differentiable_contacts.py#L851)：验证 `create_soft_contacts → counter_increment replay → contact output` 的 Tape 链路。

建议本地验证：

```powershell
uv run --extra dev -m newton.tests -k test_differentiable_contacts
```

对新的优化问题，还应对少量参数做中心有限差分：

```text
g_fd[i] = (L(x + eps*e_i) - L(x - eps*e_i)) / (2*eps)
```

比较时必须确认正负扰动下接触数、shape pair、特征和 solver 分支保持一致；若活动集已变化，固定接触集的 Tape 梯度与跨拓扑有限差分没有可比性。

## 11. 源码导航

| 层次 | 文件 / 符号 |
|---|---|
| 梯度模型与状态 | [`builder.py::ModelBuilder.finalize`](newton/_src/sim/builder.py#L11243), [`model.py::Model.state`](newton/_src/sim/model.py#L1779) |
| 碰撞总入口 | [`collide.py::CollisionPipeline`](newton/_src/sim/collide.py#L1061), [`CollisionPipeline.collide`](newton/_src/sim/collide.py#L1765) |
| Contact buffer | [`contacts.py::Contacts`](newton/_src/sim/contacts.py#L117) |
| Broad phase | [`broad_phase_nxn.py`](newton/_src/geometry/broad_phase_nxn.py), [`broad_phase_sap.py`](newton/_src/geometry/broad_phase_sap.py) |
| Narrow phase | [`narrow_phase.py::NarrowPhase.launch_custom_write`](newton/_src/geometry/narrow_phase.py#L2610) |
| 刚体几何辅助 | [`contact_kinematics.py`](newton/_src/sim/contact_kinematics.py), [`differentiable_contacts.py`](newton/_src/geometry/differentiable_contacts.py) |
| Soft contact + replay | [`kernels.py::counter_increment`](newton/_src/geometry/kernels.py#L1068), [`kernels.py::create_soft_contacts`](newton/_src/geometry/kernels.py#L1103), [`soft_contacts_sdf.py`](newton/_src/geometry/soft_contacts_sdf.py) |
| SemiImplicit 接触力 | [`kernels_contact.py`](newton/_src/solvers/semi_implicit/kernels_contact.py) |
| SemiImplicit 调度 | [`solver_semi_implicit.py::SolverSemiImplicit.step`](newton/_src/solvers/semi_implicit/solver_semi_implicit.py#L124) |
| 积分 | [`solver.py`](newton/_src/solvers/solver.py) |
| 官方 solver 能力 | [`docs/solvers/index.rst`](docs/solvers/index.rst#L60) |
| 可微接触测试 | [`test_differentiable_contacts.py`](newton/tests/test_differentiable_contacts.py) |

## 12. 现有可微链路的性能问题

本节是源码静态分析，不包含 Nsight/实际 GPU 计时结论。优先级依据 launch 规模、数组容量和 rollout 长度的渐近关系判断；合入优化前仍应按第 15 节补齐 benchmark。

### 12.1 P0：soft contact 的候选对和默认容量是 particle × shape

[`_build_soft_particle_rigid_contact_pairs()`](newton/_src/sim/collide.py#L882) 在 pipeline 构造时生成所有 world-compatible particle-shape pair。单 world 且没有 global/local 特殊情况时：

```text
N_candidate = N_particle × N_shape
```

随后：

- `create_soft_contacts` 的 launch dim 是 `soft_rigid_contact_pair_count`；
- 未显式设置 `soft_contact_max` 时，默认 capacity 也是这个候选数；
- `soft_contact_tids` 为 backward replay 按候选线程数分配，而不是按实际接触数分配；
- 每个 rollout 时间步若使用独立 `Contacts`，上述内存再乘以 `T`。

实际接触通常只占候选对很小一部分。假设 100k 个 soft vertices、10 个刚体 shape，即使每个时刻只有几百个顶点靠近表面，当前路径仍会启动 1M 个逻辑线程，并默认预留 1M 个 contact rows。

全表面 contact 还会追加：

```text
N_edge_candidate ≈ N_edge × N_capable_shape
N_face_candidate ≈ N_face × N_capable_shape
```

edge/face kernel 内部又包含 SDF 优化迭代；如果被 Tape 记录，反向也要重放对应的连续计算。

### 12.2 P0：长 rollout 的状态、接触和 Tape 内存线性增长

正确反传要求保留每个时间点的 state；动态 contact 通常还需要每步独立 `Contacts`，防止后续 `collide()` 覆盖 Tape 反向所需的前向输出。因此峰值显存近似为：

```text
M_peak ≈ (T + 1) × M_state
       + T × M_contacts
       + M_tape_saved_values
       + M_model_and_grad
```

按当前 `Contacts` 字段粗略估算，不含 allocator 对齐、自定义属性和 solver scratch：

- rigid 基础接触数据约 `96 × rigid_contact_max` bytes/时间步；
- deprecated rigid differentiable outputs 的值与 adjoint 合计再约 `56 × rigid_contact_max` bytes/时间步；
- soft contact 的值与四个可微字段的 adjoint 合计约 `116 × soft_contact_max` bytes/时间步；
- replay 映射另需约 `4 × soft_candidate_thread_count` bytes/时间步。

例如 `soft_contact_max=1,000,000` 时，仅 soft contact 值/梯度和 replay 就约 120 MB/时间步；几十到上百步时会迅速成为主导。这个估算还没有算粒子状态、FEM 中间量和 Tape 保存值。

[`example_diffsim_soft_body.py`](newton/examples/diffsim/example_diffsim_soft_body.py#L70) 使用 60×16=960 个子步，并在 `simulate()` 中为每个子步创建新 `Contacts`。它在 CUDA graph capture 时只执行一次 Python 分配，但这些 buffer 会成为图的长期存储；不使用 graph、反复调用 forward 时则还会产生 allocator churn。

### 12.3 P0：`finalize(requires_grad=True)` 是整模型级开关

[`ModelBuilder.finalize()`](newton/_src/sim/builder.py#L11243) 的单个 bool 会让大量连续模型数组一起 require grad：

- 粒子质量/半径、shape transform/scale/margin/gap；
- 全套 shape material；
- body mass/inertia/COM；
- spring/triangle/tet/muscle 参数；
- joint frame、axis、状态、target、limit 等。

如果优化目标只有初始速度或 control，整模型 adjoint 大多没有用途，却仍会增加梯度数组和反向数据流。当前 API 把“state 需要反传”“soft contacts 需要记录”“某几个 model 参数需要梯度”三个需求合并成了一个常用入口。

已有 API 实际允许更细一点的应用侧配置：

```python
# 模型参数作为常量；只对 rollout state/control 和 soft contacts 求导。
model = builder.finalize(requires_grad=False)
states = [model.state(requires_grad=True) for _ in range(num_steps + 1)]
control = model.control(requires_grad=True)
pipeline = newton.CollisionPipeline(model, requires_grad=True)
```

如果 loss 不依赖 control，也应把 control 设为 `requires_grad=False`。需要优化个别 model 参数时，目前通常只能单独构造/替换相应 Warp array；缺少正式的 attribute-level gradient policy。

### 12.4 P1：soft-contact 梯度会连带开启已弃用刚体输出

`CollisionPipeline(requires_grad=True)` 对 soft detection 是必要开关，但 [`Contacts.__init__()`](newton/_src/sim/contacts.py#L280) 同时分配：

- `_rigid_contact_diff_distance`
- `_rigid_contact_diff_point0_world`
- `_rigid_contact_diff_point1_world`

而 [`CollisionPipeline.collide()`](newton/_src/sim/collide.py#L2140) 每步都会在 `rigid_contact_max` 上运行 `launch_differentiable_contact_augment()`。内置 solver 不消费这些数组；它们只是 Newton 1.6 deprecation window 的兼容输出。

于是纯 particle-shape DiffSim 也会承担：

- 三个 rigid-capacity 数组及其 adjoint；
- 每步一个 rigid kinematics forward launch；
- Tape 中一条通常不连接最终 loss 的记录。

这是最明确、风险最低的代码级优化点之一：让 soft autodiff 与 deprecated rigid outputs 解耦。

### 12.5 P1：接触响应按 capacity 而不是 active count 启动

SemiImplicit 的两个包装函数分别使用：

- `eval_body_contact`: `dim=contacts.rigid_contact_max`，见 [`kernels_contact.py:620`](newton/_src/solvers/semi_implicit/kernels_contact.py#L620)；
- `eval_particle_body_contact`: `dim=contacts.soft_contact_max`，见 [`kernels_contact.py:668`](newton/_src/solvers/semi_implicit/kernels_contact.py#L668)。

kernel 内再读取 device-side count 并 early return。固定 capacity 有 CUDA graph 友好、无需 host sync 的优点，但当 `active_count << capacity` 时，forward 和 backward 都要调度大量空逻辑线程。P×S 默认 capacity 会放大这个问题。

不能简单把 `contact_count.numpy()` 读回 host 再作为 dim：这会强制同步，破坏异步执行和 graph capture。需要 device-side work scheduling 或更紧的 capacity。

### 12.6 P1：soft autodiff 粒度过粗

当前 `requires_grad=True` 会让以下 soft 字段全部带梯度：

- `soft_contact_barycentric`
- `soft_contact_body_pos`
- `soft_contact_body_vel`
- `soft_contact_normal`

但常见的 SemiImplicit particle-only 路径不读取 `barycentric`；很多优化只需要 live gap 对 `particle_q/body_q` 的梯度，并不需要对 SDF 最近点和法向反传。全量模式会记录更多检测计算，也可能引入法向/argmin 的高噪声梯度。

当前没有 `full / frozen_geometry / off` 这样的 soft-contact gradient policy，性能和梯度语义只能一起开关。

### 12.7 P2：每子步重复碰撞检测

rigid broad/narrow phase 不进 Tape，所以没有 backward 成本，但仍有每步 forward 成本。soft detection 还会增加 forward、Tape 和 backward 成本。

对静态平面、接触特征稳定、margin 足够大的任务，可以像 [`example_diffsim_ball.py`](newton/examples/diffsim/example_diffsim_ball.py#L100) 一样在 Tape 外检测一次并复用冻结 contact；solver 仍从 live state 计算 gap/force。更一般的折中是每 K 个子步刷新一次。

代价是可能漏碰撞或使用过时的 local point/normal，且梯度变成显式的 frozen-contact 近似。这个优化必须由任务的最大位移、曲率和 margin 决定，不能作为无条件默认值。

### 12.8 P2：可关闭但容易被忽略的前向功能

以下选项主要影响 forward，但长 rollout 中会累积：

- `verify_buffers=True`：每次 narrow phase 末尾多一个诊断 kernel；容量稳定后可关闭。
- `deterministic=True`：增加 sort/gather；只有需要 bitwise/reproducible ordering 时才开启。
- `contact_matching/contact_report/sticky`：增加匹配、历史 buffer 和多个 launch；纯 DiffSim 若不消费历史应关闭。
- `enable_rigid_soft_full_surface_contact=True`：追加 edge/face pair 和迭代 SDF；只有 coarse mesh 确实存在漏碰撞时才开启。
- 过大的 `soft_contact_margin`：不会改变 pair launch 数，但会增加 active contact 数、接触力工作和 Tape 有效路径；one-shot contact 技巧常靠大 margin 换检测频率，需衡量两者。

## 13. 建议修改点与优先级

这里必须区分两个概念：

- **性能影响优先级**：优先解决哪一项最可能改变复杂度和总体耗时；
- **落地顺序/quick win**：哪些改动小、兼容风险低，可以先提交。

如果目标是优化 soft-contact 可微仿真，而不是刚体—刚体碰撞，那么
`particle × shape` 全候选枚举是首要性能问题。下面的 legacy rigid
augmentation 只是容易先落地的附带优化，不能取代 soft broad phase。

### 13.1 Quick win：解耦 soft autodiff 与 legacy rigid augmentation

短期兼容方案可给 `CollisionPipeline` 增加显式 opt-out，例如：

```python
CollisionPipeline(
    model,
    requires_grad=True,
    populate_deprecated_rigid_contact_outputs=False,
)
```

实现上把该 flag 传给 `Contacts`：不分配三个 `_rigid_contact_diff_*` 数组，并跳过 `collide.py:2140` 的 augmentation。默认值在 deprecation window 内保持 `True` 以避免破坏旧代码；兼容周期结束后删除旧字段和这个临时 flag。

预期收益：对每个时间步省约 `56 × rigid_contact_max` bytes 的值/adjoint，并去掉一个 capacity-sized launch。不会改变 solver 的刚体接触梯度，因为 solver 从未读取这些输出。

需要新增测试：

- opt-out 时三个私有兼容数组为 `None`，soft arrays 仍 require grad；
- `test_soft_contact_detection_differentiable_through_collide` 在 opt-out 下仍通过；
- SemiImplicit rigid multistep gradient 不变。

### 13.2 第一优先级：给 soft contact 加空间 broad phase

目标是把：

```text
O(N_particle × N_shape)
```

改成与空间邻近候选数相关。可复用 model 已构建的 particle BVH，或按 shape AABB 查询粒子 BVH；也可以为粒子建 spatial hash，再让每个 shape 查询扩张 AABB。

建议的数据流：

```text
particle positions + expanded shape AABBs
  → non-differentiable soft broad phase (record_tape=False)
  → compact candidate_pair buffer + candidate_count
  → differentiable create_soft_contacts over candidate capacity
  → replay tids indexed by compact candidate slot
```

关键约束：

- 候选生成是离散操作，可以像 rigid broad phase 一样不进 Tape；
- backward 前不能覆盖 `candidate_pair`，所以多步 rollout 仍需每步独立候选 buffer，或 Tape 支持保存/重算；
- capacity 必须有 overflow 检查，并保持 CUDA graph 可捕获；
- AABB 要按 `soft_contact_margin + shape_margin + particle_radius` 保守扩张，不能漏掉接触；
- ACTIVE/COLLIDE_PARTICLES/world filtering 的运行时语义必须保留。

这是 **soft-contact 性能影响优先级最高** 的仓库级修改，尤其适合大量
soft vertices、少量局部接触的场景。前一节的 legacy rigid opt-out 更容易
实现，但只消除附带开销，不会改变 `O(N_particle × N_shape)` 的主复杂度。

### 13.3 第一优先级：引入 selective gradient policy

建议不要继续扩展一个全局 bool，而是增加明确的梯度选择：

```python
model = builder.finalize(requires_grad=False)
model.enable_grad("shape_material_ke", "shape_material_kd")

state = model.state(requires_grad={"particle_q", "particle_qd"})
control = model.control(requires_grad={"tet_activations"})
```

若暂时不做新 API，至少应修改 DiffSim 文档/示例：只有优化 model 参数时才 `finalize(requires_grad=True)`；只优化初始 state/control 时使用上一节的 selective pattern。

实现难点是中间量仍必须 require grad，例如 `body_f/particle_f` 和下一状态；因此 state policy 不能简单关闭所有“用户不优化”的字段，而应区分 leaf gradient 与 graph intermediate。一个更稳妥的第一版是只对 `Model` 和 `Control` 做 attribute-level policy，保持 `State(requires_grad=True)` 的内部完整性。

### 13.4 第二优先级：增加 frozen soft-geometry 模式

建议把 soft contact 梯度语义显式化：

```python
soft_contact_grad_mode = "full"             # 当前行为
soft_contact_grad_mode = "frozen_geometry"  # detection 不进 Tape，solver 的 live gap 仍可微
soft_contact_grad_mode = "none"             # 非 DiffSim
```

`frozen_geometry` 模式可以：

- soft detection launch 使用 `record_tape=False`；
- soft contact geometry 数组不 require grad；
- 不分配 `soft_contact_tids` replay buffer；
- solver contact kernel 仍记录，对 live `particle_q/body_q/body_qd` 反传。

这通常比 full SDF/normal differentiation 更省显存、更稳定。它会改变梯度定义，必须是显式 opt-in，并配有限差分测试；不能悄悄替换当前 `requires_grad=True` 行为。

### 13.5 第二优先级：优化 capacity-sized consumer kernel

可评估两种方案：

1. **应用/API 侧紧容量。** 先收集实际峰值，再用 `soft_contact_max` / `rigid_contact_max` 设置峰值加安全余量。这最简单，但 soft replay 数组仍按候选 pair 数分配。
2. **device-side grid-stride work queue。** 只启动固定数量 worker，每个 worker 循环处理 `i < contact_count` 的 active rows。这样无需 host sync，仍可 capture；需要验证 Warp adjoint 对循环、atomic 和 replay 的生成效率，并在 active count 很大时避免降低占用率。

无论哪种方案，都要以 `active_count/capacity` 比值驱动决策，不能只比较 kernel launch 次数。

### 13.6 第二优先级：checkpoint/recompute 长 rollout

仅靠 contact 优化无法消除 `O(T)` 状态和 Tape 存储。可增加分段 rollout 工具：每 K 步保存 checkpoint，backward 时重算段内 forward，用额外计算换显存。

碰撞场景的重算必须保证：

- deterministic contact ordering，或保存足够的 frozen contact identity；
- 随机控制/扰动可复现；
- 重算时接触活动集与原 forward 一致。

这是中长期能力，优先级低于 soft broad phase，但对数百/数千步优化是最终必须解决的问题。

### 13.7 低风险示例修改：预分配 per-step Contacts

`example_diffsim_soft_body.py` 当前在 `simulate()` 中调用 `contacts()`。可以在 `__init__`、capture 之前一次性建立：

```python
self.contacts = [
    self.collision_pipeline.contacts()
    for _ in range(self.sim_steps * self.sim_substeps)
]
```

然后：

```python
contacts = self.contacts[t]
self.collision_pipeline.collide(self.states[t], contacts)
```

这不会减少 rollout 必需的 contact 总显存，但会：

- 把分配生命周期写清楚；
- 避免非 graph 模式下每轮 Python/allocator churn；
- 确保 capture 内没有隐式资源创建；
- 便于在启动前打印和审计总容量。

这是适合先落地的示例级改动，不涉及数值语义。

## 14. 推荐优先级

### 14.1 按性能影响排序

| 性能优先级 | 修改 | 主要解决的问题 | 收益预期 | 风险/实现量 |
|---:|---|---|---|---|
| 1 | **soft particle-shape spatial broad phase** | `particle × shape` 候选、默认 capacity、replay 和检测 forward/backward | 很高 | 中等语义风险、实现量大 |
| 2 | frozen soft-geometry 显式模式 | SDF 最近点/法向检测进入 Tape 的计算和保存 | 高 | 改变梯度定义、实现量中 |
| 3 | rollout checkpoint/recompute | 数百到数千步 State/Contacts/Tape 的线性显存 | 很高，主要省显存 | 重算一致性风险高、实现量大 |
| 4 | selective model/control gradient policy | 整模型无关 adjoint | 中到高 | API 设计、实现量中到大 |
| 5 | active-count device work queue | contact consumer 按 capacity 启动 | 场景相关 | 需 benchmark adjoint、实现量中 |
| 6 | soft autodiff opt-out legacy rigid outputs | soft DiffSim 连带产生的 deprecated rigid 开销 | 中；rigid capacity 小时较低 | 低风险、实现量小 |
| 7 | 预分配 per-step Contacts、收紧并打印容量 | allocator churn、隐式资源创建和过量 capacity | 中 | 低风险、实现量小 |

对大量 soft vertices 且局部接触稀疏的场景，第一项应直接作为主线工作，
而不是等 quick win 全部完成后再开始。

### 14.2 按低风险落地顺序

如果希望先用小 PR 建立测量基线，可以依次做：

1. 示例预分配 Contacts，报告 candidate/capacity/active 数量；
2. 推广 selective model/state/control 配置；
3. opt-out deprecated rigid outputs；
4. 建立第 15 节 benchmark；
5. 在 benchmark 保护下实现 soft spatial broad phase。

这个顺序只表示工程落地容易程度，不表示第 1–3 项比 soft broad phase 的
性能影响更大。若 profiling 显示 `soft_candidate/active_contact` 比值很高，
应并行或直接推进 soft broad phase；若主要是长 Tape OOM，则同时推进
frozen geometry 和 checkpointing。

## 15. 性能 benchmark 设计

不能先改实现再随便找场景。第一步应构造一个满足以下条件的受控场景：

- 只包含 soft particle-shape contact，不混入 rigid-rigid、粒子自碰撞、FEM 或关节开销；
- 增加 shape 数量时，实际接触数、接触力、loss 和梯度保持不变；
- 当前实现的候选数严格按 `particle × shape` 增长；
- 空间 broad phase 能安全剔除新增 shape；
- 接触曲面具有变化法向，能够真正覆盖 soft detection 的连续反向路径。

### 15.1 主场景：粒子球壳接触一个有效球体，附加远处干扰 Shape

场景命名建议：`SoftContactDistractorShell`。

#### 几何

在原点放置一个静态有效球体：

```text
active sphere radius R = 1.0 m
body = -1
has_shape_collision = False
has_particle_collision = True
```

用确定性的 Fibonacci sphere 生成 `P` 个互不连接的粒子，均匀分布在球壳：

```text
particle radius r_p = 0.01 m
penetration delta = 0.002 m
particle distance to origin = R + r_p - delta
initial velocity = -0.05 * radial_normal m/s
gravity = 0
```

所有粒子都与中心球形成一个稳定 soft contact，因此预期：

```text
active_contact_count = P
```

再加入 `D` 个静态干扰 sphere 或 box，放在距离粒子球壳足够远的位置，例如：

```text
(10 + 3*i, 0, 0), i = 0 ... D-1
```

干扰 Shape 与有效球处于同一 world，并保持
`has_particle_collision=True`，确保当前
[`_build_soft_particle_rigid_contact_pairs()`](newton/_src/sim/collide.py#L882)
确实为它们生成候选对；同时关闭 shape-shape collision，避免引入刚体 broad phase 工作。

于是当前实现应满足：

```text
soft_candidate_pair_count = P × (D + 1)
soft_active_contact_count  = P
candidate / active ratio  = D + 1
```

而空间 soft broad phase 的理想结果是只保留中心球附近的 `P` 个候选；增加
`D` 不应明显改变 contact detection 和 backward 时间。

#### 为什么不用平面作为主场景

无限平面必然与所有粒子的空间查询范围相关，且法向恒定。它适合测 fixed-gap
contact，但不能充分覆盖 `body_pos/normal` 对粒子位置的检测梯度。球面法向
`n = normalize(x)` 会随粒子位置变化，更适合验证修改没有意外截断 full soft
detection 的反向路径。

#### 排除其他物理开销

```python
model.particle_grid = None
solver = newton.solvers.SolverSemiImplicit(
    model,
    enable_tri_contact=False,
)
```

- 不创建 edge/triangle/tet/spring；
- 摩擦设为零，只保留 normal stiffness/damping；
- 所有 shape 静态，场景不存在刚体状态和刚体—刚体接触；
- `model = builder.finalize(requires_grad=False)`，只让 state、contacts 和必要的 loss 输出带梯度。

这样观察到的随 `D` 增长的开销，可以归因于 soft candidate pipeline，而不是其他 solver 子系统。

### 15.2 前向、反向和 loss

主性能测试先使用一个仿真子步，避免 rollout 内存掩盖候选对开销：

```text
state_0
  → collide(state_0, contacts)
  → SolverSemiImplicit.step(..., dt=1/240)
  → radial loss(state_1)
  → tape.backward(loss)
```

loss 定义为所有粒子到目标半径的均方误差：

```text
L = mean_i (length(q_i(T)) - R_target)^2
```

`R_target` 取当前球壳外侧，例如 `R + r_p + 0.02`。检查初始
`particle_qd` 的梯度；有限差分只扰动一个粒子的 radial velocity，避免
对全部 `3P` 个自由度做昂贵数值差分。

球面 `body_pos` 和 normal 都依赖粒子位置，接触响应又影响 `state_1`，因此该
loss 会覆盖：

```text
create_soft_contacts
  → replay slot mapping
  → soft contact geometry
  → eval_particle_body_contact
  → particle integration
  → loss
```

### 15.3 参数扫描

先用小规模 correctness 组：

```text
P = 1,024
D = 0, 8, 32
T = 1
```

再用性能组：

```text
P = 4,096 / 16,384
D = 0 / 8 / 32 / 64
T = 1
```

最后单独做 horizon/显存组，不与 shape scaling 混在一起：

```text
P = 4,096
D = 32
T = 1 / 8 / 32
```

一次只改变一个主要尺度，才能判断优化解决的是 candidate scaling 还是 Tape
horizon scaling。不要一开始上真实布料、几百个 shape 和 1000 步 rollout；那样即使变快，也无法确认是哪条链路贡献。

### 15.4 必须同时保留的三组基线

1. **Current/default capacity**

   ```text
   soft_contact_max = P × (D + 1)
   replay tids      = P × (D + 1)
   detection dim    = P × (D + 1)
   ```

   反映当前默认用户体验，但较大组合可能先 OOM。

2. **Current/tight contact capacity**

   ```text
   soft_contact_max ≈ 1.1 × P
   replay tids      = P × (D + 1)
   detection dim    = P × (D + 1)
   ```

   它只压缩 contact rows 和 solver consumer launch，不改变候选检测及 replay
   数量，用于分离“默认 capacity 过大”和“候选算法复杂度过大”。

3. **Proposed/spatial soft broad phase**

   ```text
   candidate capacity ≈ 1.1 × P
   soft_contact_max   ≈ 1.1 × P
   expected candidates = P
   ```

   用于证明性能不再随远处干扰 Shape 数 `D` 线性增长。

这三组必须使用完全相同的模型初值、材料、dt、loss 和 gradient seed。

### 15.5 验收不变量

对每个 `D` 和每种实现，必须验证：

```text
active_contact_count == P
no candidate/contact buffer overflow
loss 与 D=0 基线一致
final particle_q/qd 与 D=0 基线一致
gradient norm/checksum 与 D=0 基线一致
单粒子 radial velocity 梯度通过中心有限差分
```

空间 broad phase 允许 candidate ordering 不同，因此不要要求 contact row
逐行相等；应比较物理结果、loss 和梯度。若需要逐 contact 调试，可在小规模组
开启 deterministic sorting。

### 15.6 性能指标

每组记录：

```text
soft_candidate_pair_count
soft_contact_max
observed active contact count
candidate / active ratio
collide forward time
solver forward time
tape.backward time
peak allocated/reserved device memory
analytic-vs-finite-difference error
```

主比较只改变候选生成和 capacity，不同时混入 frozen geometry、legacy-output
opt-out 或 checkpointing。那些优化应在主场景通过后作为独立实验逐项叠加。

benchmark 可放在现有 ASV 目录，例如
`asv/benchmarks/simulation/bench_differentiable_contacts.py`，沿用现有 benchmark
的 CUDA skip、warm-up 和 graph capture 习惯。计时前要完成 kernel compilation
warm-up，计时区间显式同步；显存同时报告 active allocation 与 memory-pool
reserved，避免把缓存池误判为泄漏。

### 15.7 第二阶段真实场景

受控球壳场景证明复杂度和梯度正确后，再增加一个真实验证场景，例如“一块
中等分辨率 cloth 落到由多个刚体组成的障碍物上”。真实场景只用于确认收益能
迁移，不用于定位 P×S 根因。若受控场景没有先建立，不应直接根据真实场景总耗时
决定 soft broad phase 是否有效。
