# Newton 碰撞与可微仿真工作总结

> 快照日期：2026-10-28  
> 当前分支：`diffsim_Rigid_body_collision`  
> 当前提交：`c4d72fa6`（`diffsim in newton`）  
> 对比基线：`origin/main` / `71452a4e`

## 1. 总体结论

当前工作可以分为两条主线：

1. **粒子—刚体 soft contact 性能优化**
   - 已实现基于 shape BVH/AABB 的候选对压缩，固定测试场景中候选算子与 soft-contact 算子之和相对原始 soft-contact 算子约有 `2.09x–2.26x` 加速，接触结果一致。
   - 已实现 soft-contact 检测与响应融合算子，并保留 `collide(); solver.step()` 两阶段调用语义。融合结果数值一致，但性能收益不稳定：一组运行前向 `1.25x`、反向 `1.68x`，另一组运行前向仅 `0.75x`、反向约 `1.00x`。因此它仍是实验性优化，不能宣称稳定达到 `1.5x`。
   - 已建立固定规模场景、逐次独立日志、Warp CUDA activity、NVTX、Nsight Systems/Nsight Compute 分析流程。

2. **刚体 mesh—mesh 可微碰撞管线**
   - 已将 RS1Dist 风格的平滑 witness 导数接入 Newton 的 `CollisionPipeline`、接触运动学和 Semi-Implicit Solver。
   - 前向仍使用 Newton 原窄相位给出的离散接触点、法向和接触力；反向把 witness point 的导数替换为基于表面采样、距离 softmax 和耦合隐式系统的平滑代理导数。
   - 当前支持 `MESH` / `CONVEX_MESH` 对；候选对、接触数量、接触拓扑、局部样本选择仍是离散决策，不可对其求导。因此当前实现是局部/分段的 surrogate gradient，而不是全局连续的碰撞检测。
   - 已有两个 box mesh 的单元测试，以及 Allegro 四指手抓取非凸齿轮的可视化验证场景。

## 2. 提交演进

| 提交 | 主要内容 |
|---|---|
| `5ba34b53` `fused contact and solve` | 固定粒子矩阵性能场景、候选对压缩、soft-contact 融合算子、性能日志、NSYS/NCU 资料。 |
| `838ea918` `newton rigid diffsim` | 第一版刚体可微碰撞旁路及其与碰撞、求解代码的初步连接。 |
| `d79518bf` `平滑witness的可微管线` | 将平滑 witness 正式接入 `CollisionPipeline`、接触运动学和 solver；加入测试和早期场景。 |
| `c4d72fa6` `diffsim in newton` | 清理 curling、partition overlap 等探索场景；把当前验证场景扩展为 Allegro 手—非凸齿轮优化。 |

当前分支相对 `origin/main` 主要增加或修改了 38 个文件，核心源码集中在：

- `newton/_src/geometry/differentiable_rigid_contact.py`
- `newton/_src/sim/collide.py`
- `newton/_src/sim/contact_kinematics.py`
- `newton/_src/sim/contacts.py`
- `newton/_src/solvers/semi_implicit/kernels_contact.py`
- `newton/_src/solvers/semi_implicit/solver_semi_implicit.py`
- `newton/tests/test_differentiable_rigid_contact.py`

## 3. 性能优化工作

### 3.1 固定性能场景

保留的性能场景为 `diffsim/particle_primitive_wall.py`：

- `4096` 个粒子；
- `32` 个静态解析基本体（8 列 × 4 行，混合 sphere、box、capsule、cylinder、cone）；
- 原始容量为 `4096 × 32 = 131072` 个 particle-shape 对；
- 接触快照中实际有效接触约为 `4096` 个；
- 固定粒子数、形状数、初态、时间步和接触参数，避免用调参替代优化；
- viewer 使用 Newton examples 的标准逻辑，性能测试时使用 `--viewer null`。

该场景用于验证以下指标：

- 候选压缩前后的有效接触对是否一致；
- 融合前后的接触数量、粒子力、积分状态、一步 loss 和 `particle_q` 梯度是否一致；
- 单个 CUDA kernel 和完整前向/反向阶段耗时。

### 3.2 候选对压缩

问题来源：原始 soft contact 默认按 `particle_count × particle-colliding shape_count` 分配并遍历潜在对。在粒子只靠近少量 shape 的场景中，大部分工作都在做必然失败的 SDF/AABB 查询。

实现方式：

- `CollisionPipeline(enable_soft_contact_candidates=True)` 打开优化；
- `build_soft_contact_candidates` 使用 model shape BVH，对每个粒子的 AABB 查询可能相交的 shape；
- 将通过 AABB 的 particle-shape pair 压缩到紧凑数组；
- 原有 `create_soft_contacts` 只在紧凑候选上执行；
- 候选集合的生成是离散过程，不对候选集合本身求导；一旦 pair 被接受，原 soft-contact 几何和反向链路保持不变。

代表性原始结果：

```text
# particle_primitive_wall_profile_20260825_221558_604173.log
baseline create_soft_contacts       : 0.0440 ms
candidate operator                  : 0.0082 ms
candidate create_soft_contacts      : 0.0113 ms
candidate + soft-contact            : 0.0195 ms
operator speedup                    : 2.26x
active contact pairs match baseline : PASS
candidate + soft < baseline soft    : PASS
```

另一组结果：

```text
# particle_primitive_wall_profile_20260825_225820_725130.log
baseline create_soft_contacts       : 0.0440 ms
candidate + soft-contact            : 0.0211 ms
operator speedup                    : 2.09x
active contact pairs match baseline : PASS
candidate + soft < baseline soft    : PASS
```

结论：候选对优化是目前最明确、最稳定的性能成果。它主要适用于“大量粒子 × 多个刚体 shape，但空间上局部接触”的 soft-contact 场景。

### 3.3 soft-contact 检测/响应融合

原始流程：

```text
collide(): create_soft_contacts
    -> 写出 contact point / velocity / normal / body index 等中间数组
solver.step(): eval_particle_body_contact
    -> 再从 Global Memory 读取中间数组并计算接触力
```

融合流程：

```text
collide(): 只登记/defer particle-shape pair
solver.step(): eval_particle_body_contact_fused
    -> 在一个 kernel 中重算接触几何并直接产生接触响应
```

这样仍保留用户侧 `collide(); solver.step()` 的语义边界，但内部不再物化完整 soft-contact 几何数组。对应开关为：

```python
newton.CollisionPipeline(model, enable_fused_soft_contact=True)
```

正确性验证包含：

- active contact count；
- particle force；
- integrated state；
- one-step loss；
- `particle_q` gradient。

较好的一次原始结果：

```text
# particle_primitive_wall_profile_20260825_225820_725130.log
baseline create_soft_contacts       : 0.0123 ms
baseline contact response           : 0.0082 ms
fused detection + response          : 0.0164 ms
forward operator speedup            : 1.25x
fused < baseline two-operator sum   : PASS
fused backward operator             : 0.0358 ms
backward operator speedup           : 1.68x
active contact count match          : PASS
particle force match                : PASS
integrated state match              : PASS
one-step loss match                 : PASS
particle_q gradient match           : PASS
```

但另一组原始结果为：

```text
# particle_primitive_wall_profile_20260826_095350_007282.log
baseline create_soft_contacts       : 0.0532 ms
baseline contact response           : 0.0102 ms
fused detection + response          : 0.0850 ms
forward operator speedup            : 0.75x
fused < baseline two-operator sum   : FAIL
fused backward operator             : 0.3612 ms
backward operator speedup           : 1.00x
```

结论：融合设计和数值链路已打通，但当前 kernel 的寄存器压力、重复几何计算、场景规模和 profiling 波动可能抵消 Global Memory 收益。它不能作为已经稳定完成的性能结论，后续必须结合 NCU 的 occupancy、register、memory throughput 和 instruction 指标继续拆解。

### 3.4 分块、stream overlap 和 CUDA Graph

曾经制作过 `partitioned_contact_overlap.py` 和对应 profiler，用于探索场景分块、不同 CUDA stream 上的碰撞/求解 overlap，以及 CUDA Graph 降低 launch 开销。

当前结论：

- Newton/Warp 已有 CUDA Graph 能力，因此“仅接入 CUDA Graph”不是独立的核心创新点；
- 分块 overlap 要求不同分区之间不存在接触和写冲突，否则必须引入 halo、同步或冲突归并；
- 早期总体结果没有形成稳定正收益，因此相关场景和脚本已从当前版本删除；
- 这个方向仍可继续，但应先用明确的空间独立分区和可复现 benchmark 重新实现，不能把早期探索写成已完成的 `1.5x` 结果。

### 3.5 性能测量设施

`diffsim/profile_particle_primitive_wall.py` 提供：

- 每次运行生成一个带微秒时间戳的独立日志；
- 5 次 warmup、20 次同步采样；
- collision forward、solver forward、simulation forward、collision backward、simulation backward 分段计时；
- Warp CUDA activity 表，保留 kernel 原始名字、调用次数和耗时；
- 候选模式和融合模式各自与对应 baseline 比较；
- viewer 开关，测试时关闭 viewer 以避免渲染污染测量；
- NVTX / CUDA profiler capture，可导入 NSYS；
- 已保存 baseline/fused 的 `.nsys-rep` 和 `.sqlite`。

运行方式：

```powershell
# 原始管线
uv run python diffsim/profile_particle_primitive_wall.py --viewer null

# 只测候选对优化
uv run python diffsim/profile_particle_primitive_wall.py --viewer null --soft-contact-candidates

# 只测融合算子；候选对保持关闭
uv run python diffsim/profile_particle_primitive_wall.py --viewer null --fused-soft-contact

# 仅查看固定场景
uv run python diffsim/particle_primitive_wall.py --viewer gl
```

详细 NSYS/NCU 命令见 `diffsim/docs/NSYS_NCU_FUSED_SOFT_CONTACT_PROFILING.md`。

## 4. 刚体可微碰撞管线

### 4.1 要解决的问题

普通刚体碰撞窄相位通常包含：候选三角形选择、最近特征切换、接触增删、接触裁剪/规约等离散操作。即使某个固定接触特征内部可微，接触特征切换处仍会产生跳变，Warp 自动微分无法直接给出稳定的整体梯度。

当前方案的目标不是改变前向物理结果，而是：

- 前向继续使用 Newton 原始窄相位；
- 为已生成的 mesh contact witness 构造一个随位姿平滑变化的局部代理；
- 反向时把 loss 对精确 witness 的伴随量路由到平滑代理上；
- 由平滑 witness 把梯度传到刚体位姿，再通过积分或 FK 传到初态、速度或关节变量。

### 4.2 前向链路

```text
body_q
  -> Newton broad phase / narrow phase
  -> discrete contact topology
  -> exact point0 / point1 / normal / distance
  -> 原接触力模型
  -> solver / state update / loss
```

开启可微配置后，前向的接触点、法向和 penalty law 仍与普通管线一致。`_exact_forward_smooth_backward()` 和 `_implicit_witness_a/b()` 在前向直接返回 exact witness，因此不会为了平滑梯度而改变接触几何。

### 4.3 反向链路

每个 mesh shape 在初始化阶段进行确定性的表面预采样。对窄相位给出的每个双向 witness：

1. 将精确 witness 和法向转换到 shape local frame；
2. 在固定半径内筛选位置接近、法向夹角满足阈值的表面样本；
3. 样本不足时退化为无过滤样本，保证局部集合有足够元素；
4. 按样本到另一侧 witness 的距离计算 softmax 权重；
5. 得到平滑位置与平滑法向，并解析计算 softmax 权重对距离的导数；
6. 对 A、B 两侧 witness 的交叉依赖构造 `3×3` 耦合系统；
7. 自定义 Warp adjoint 中求解转置隐式系统，将 `dL/d witness` 传到两侧平滑 witness；
8. Warp 再把梯度传播到 shape/body transform，以及后续的初态、速度或关节坐标。

核心反向关系在代码中表现为：

```text
system_A = I - J_AB J_BA
adj_A    = inverse(transpose(system_A)) * upstream_A
adj_B    = transpose(J_AB) * adj_A
```

另一侧使用对称形式。实现位于 `newton/_src/geometry/differentiable_rigid_contact.py`。

### 4.4 管线接入位置

对外配置：

```python
config = newton.CollisionPipeline.DifferentiableRigidContactConfig(
    global_sample_count=1024,
    local_sample_count=16,
    sample_radius_scale=1.0,
    min_sample_radius=0.01,
    normal_angle=2.0 * math.pi / 3.0,
)

pipeline = newton.CollisionPipeline(
    model,
    differentiable_rigid_contact_config=config,
)
```

接入点：

- `CollisionPipeline.__init__()` 查找可微 mesh pair、预采样表面并创建只读数据；
- `contacts()` 为每个 contacts 实例创建独立的可写 workspace，避免多 contacts 之间覆盖；
- `collide()` 仍执行原碰撞流程，同时为当前 exact contacts 选择局部样本并准备平滑 witness 数据；
- `eval_rigid_contact_kinematics(..., use_smooth_witness_gradients=True)` 前向输出 exact point/distance，反向使用平滑 witness；
- `SolverSemiImplicit.step()` 对配置过的 mesh pair 调用可微接触力路径；
- 未配置的 pair 继续使用原有接触 kernel；
- Featherstone 路径只做了 pair-key/兼容性传递，完整的 RS1Dist 接触力消费目前以 Semi-Implicit Solver 为主。

### 4.5 支持范围和边界

当前支持：

- `MESH` 与 `CONVEX_MESH` 的组合；
- mesh 需要已有几何和表面数据，场景中通常同时构建 SDF 供 Newton 窄相位使用；
- 静态—动态、动态—动态 mesh pair；
- contact kinematics objective；
- Semi-Implicit Solver 中的刚体接触力反传；
- 通过 FK 向 articulated joint coordinate 反传。

当前不支持或未解决：

- broad phase pair 的出现/消失不可微；
- contact topology、contact count、triangle/feature 选择不可微；
- 局部样本集合的筛选和 fallback 不可微；
- primitive—primitive、primitive—mesh 没有走该 RS1Dist 路径；
- 接触切换附近只能得到代理梯度，不能保证与 exact finite difference 一致；
- 当前配置是实验 API，尚未完成公共 API、文档、changelog 和 upstream PR 的完整整理。

## 5. 自动测试

`newton/tests/test_differentiable_rigid_contact.py` 当前包含四类 CUDA 测试：

1. 配置参数校验；
2. 多个 `contacts()` 共享只读表面样本、但拥有独立可写 workspace；
3. contact kinematics 的平滑路径保持 exact forward，同时产生有限且非零的 pose gradient；
4. 两个重叠 box mesh 做一步 Semi-Implicit 求解：可微管线与 baseline 的 forward `q/qd` 一致，同时初始 pose gradient 有限且非零。

这些测试证明：

- 配置和 workspace 生命周期基本正确；
- exact-forward / smooth-backward 机制生效；
- solver 中的反向链路能够走通。

它们尚未证明：

- surrogate gradient 与 exact finite difference 在任意场景中一致；
- 接触拓扑变化时梯度连续；
- 多步高速碰撞和复杂动态接触的优化一定收敛。

定向测试命令：

```powershell
uv run --extra dev -m newton.tests -k test_differentiable_rigid_contact
```

## 6. 当前保留的可微验证场景

当前场景为 `diffsim/differentiable_contact_point_target.py`，虽然文件名仍沿用早期 contact-point-target 名称，实际内容已经是：

- Wonik Allegro 左手，16 个可优化 revolute joint coordinate；
- 一个固定的非凸 14 齿 star-prism mesh，构建 dense SDF；
- 四个 fingertip 与齿轮之间使用显式 shape pair 和可微 collision pipeline；
- 其余手部 collision shapes 与齿轮使用单独 safety pipeline 检测穿透；
- 参考手势产生四个齿轮表面目标 witness；
- 从略微张开的初始手势开始，用梯度下降优化四指关节角。

目标函数为：

```text
L = 1.0 * L_target_witness
  + 10.0 * L_fingertip_gap
  + 25.0 * L_nonfinger_penetration
```

其中：

- `L_target_witness`：当前齿轮侧 witness 到固定目标 witness 的平方距离；
- `L_fingertip_gap`：四个 fingertip 选中接触对的 signed distance 平方；
- `L_nonfinger_penetration`：非指尖手部 shape 对齿轮的穿透惩罚。

可视化含义：

- 黄色小点：参考手势产生的固定齿轮表面目标；
- 红色小点：当前齿轮侧 exact witness；
- 青色小点：当前 fingertip 侧 exact witness；
- 白线：当前所选接触对的 signed separation；
- UI 显示每根手指的 gap、target error、目标/当前坐标、总 loss、各子 loss、梯度范数和最大关节运动。

viewer 渲染完全放在 Tape 和优化计算之外；GL 模式每个优化状态保持 3 帧，只降低展示频率，不改变优化计算本身。

运行方式：

```powershell
uv run python diffsim/differentiable_contact_point_target.py --viewer gl
```

### 6.1 当前实际结果

取日志 `differentiable_allegro_grasp_20260916_104355_393375.log`：

```text
iteration 0:
  total loss              = 6.034666512278e-05
  target loss             = 7.045144201356e-08
  fingertip gap loss      = 6.027621111571e-06
  gaps                    = [2.574, 2.676, 2.438, 2.093] mm
  |dL/d joint_q|          = 2.555062063038e-03

iteration 125:
  total loss              = 6.057074415367e-07
  target loss             = 3.863538609039e-07
  fingertip gap loss      = 2.193535664219e-08
  gaps                    = [0.041, 0.250, 0.111, 0.106] mm
  max penetration         = 0.0
  |dL/d joint_q|          = 1.243847364094e-04
  max joint delta         = 1.0266 deg
```

可以确认的结论：

- 四个 fingertip 都有接触候选；
- 接触相关梯度成功传到 16 个 joint coordinates；
- 四指间隙明显减小；
- 总 loss 下降约两个数量级；
- 当前记录中未发生手—齿轮穿透；
- 手指运动在 viewer 中可见。

必须保留的限制：

- 下降主要由 `L_fingertip_gap` 驱动；
- `target loss` 从 `7.05e-8` 增至 `3.86e-7`，并没有证明目标 witness 对齐得到改善；
- 该场景是静态 FK + contact witness 的 grasp refinement，不是带时间积分的多步动态 rollout；
- 因而当前场景能验证“接触 witness → body pose → FK → joint_q”的反向链路，但不能单独证明“跨多个仿真时间步的碰撞动力学梯度”已经充分验证。

## 7. 可微场景日志结构

每次运行生成一个独立文件：

```text
diffsim/log/differentiable_allegro_grasp_YYYYMMDD_HHMMSS_microseconds.log
```

文件头记录：

- device；
- objective 和完整 gradient chain 描述；
- 固定 target contact 坐标；
- 参考手势的 gap/penetration；
- learning rate 和三个 loss 权重。

每个 `OPTIMIZATION_ITERATION` 分块记录：

1. `[objective]`
   - total/sub-loss；
   - target、gear witness、fingertip witness；
   - selected signed distance；
   - `dL/d gear_witness`、`dL/d fingertip_witness`；
   - 每指 contact count。
2. `[optimizer]`
   - 当前 `joint_q`；
   - `dL/d joint_q`；
   - gradient norm、学习率、累计最大关节位移。
3. `[contact_witnesses]`
   - 每个 contact 的 shape pair、distance、point0/point1；
   - `dL/d point0_world` 和 `dL/d point1_world`。
4. `[articulated_gradient_chain]`
   - 每个刚体的 `body_q`；
   - 对应 `dL/d body_q`；
   - 可直接观察梯度从指尖刚体沿 FK 回传到关节变量。

注意：完整接触数组会让单个日志达到数 MB，历史目录中还有两个早期日志达到约 23 MB 和 96 MB。后续如果进入正式 PR，建议默认只输出 active/selected contacts，并把全量 trace 改为显式 debug 开关。

## 8. 早期探索与清理结果

已经做过但当前不再保留为正式验证场景的内容：

- 手写 box SDF 旁路：用于快速确认可微碰撞思路，但没有走正式 `CollisionPipeline`；
- bunny SDF 策略场景：暴露过 CUDA illegal memory access，且一次方向梯度检查相对误差约 `0.691`，不能作为正确性证据；
- box 落点/速度策略场景：viewer、物理接触和训练补丁过多，已删除；
- curling SDF 场景：mesh/SDF 构建和地面动力学不稳定，已删除；
- partitioned contact overlap 场景：未形成稳定总体加速，已删除；
- 槽型体 contact-point rollout：曾用于记录跨步梯度，后来被当前 Allegro 场景替换，源码不再保留，但旧日志仍在 `diffsim/log`。

当前 `diffsim` 目录保留的可执行核心场景只有：

- `particle_primitive_wall.py`：性能固定场景；
- `profile_particle_primitive_wall.py`：性能测量；
- `differentiable_contact_point_target.py`：当前 Allegro 可微接触验证。

## 9. 现有文档和数据

| 路径 | 内容 |
|---|---|
| `diffsim/COLLISION_DIFFERENTIABILITY_ANALYSIS.md` | 最初的 Newton 碰撞前向/反向链路和性能问题分析。 |
| `diffsim/PERFORMANCE_PROFILING.md` | 固定性能场景与日志字段说明。 |
| `diffsim/docs/NSYS_NCU_FUSED_SOFT_CONTACT_PROFILING.md` | 精简后的 NSYS/NCU baseline 与 fused 对比操作。 |
| `diffsim/log/particle_primitive_wall_profile_*.log` | 候选和融合算子的原始性能日志。 |
| `diffsim/log/differentiable_allegro_grasp_*.log` | Allegro 每次优化的完整梯度链日志。 |
| `diffsim/profile/nsys/` | baseline/fused 的 NSYS report 与 SQLite 导出。 |

## 10. 当前代码审计中发现的问题

### 10.1 与物理项目无关的 `speculative.py`

`newton/_src/geometry/speculative.py` 当前是一个基于 GPT-2/DistilGPT-2、PyTorch 和 Transformers 的 speculative decoding 脚本，并在模块末尾直接执行模型推理。它：

- 与 Newton 的 speculative contact 无关；
- 没有被碰撞管线导入；
- 引入了 Newton geometry 不应依赖的 `torch` / `transformers`；
- 如果被误导入会下载/加载语言模型并执行推理。

该文件应视为误提交内容，在正式整理分支时删除。当前总结任务未修改源码，因此只在此标记。

### 10.2 提交前仍需处理

- 删除误提交的 `newton/_src/geometry/speculative.py`；
- 清理大量历史日志、NSYS 二进制和过大的梯度 trace，决定哪些应进入版本控制；
- 将当前场景改成符合 Newton examples 规范的注册方式，补 README 命令和 320×320 截图；
- 为用户可见变更补 Towncrier fragment；
- 补充 primitive/mesh 组合或明确限制为 mesh/mesh；
- 增加接触拓扑稳定区间内的方向梯度检查，以及拓扑切换附近的行为测试；
- 增加真正的多步动态 rollout，验证 loss 经 solver 和多帧 state 传回初始 pose/velocity；
- 对融合 kernel 用同一设备、同一时钟状态和多轮统计重新测量，不能选取单次最好结果；
- 运行完整格式化、lint 和回归测试。

建议的提交前命令：

```powershell
uvx pre-commit run -a
uv run --extra dev -m newton.tests -k test_differentiable_rigid_contact
uv run --extra dev -m newton.tests
```

## 11. 下一步优先级建议

1. **P0：建立多步动态梯度基准。** 使用一个固定 mesh 撞击另一个 mesh，loss 直接定义在碰撞后的速度、位姿或接触 witness 上，同时做 Tape directional derivative 与有限差分对比。
2. **P0：修正 Allegro 验证目标。** 提高初始 target error、重新设置 loss 权重，使 target witness loss 本身显著下降，而不是只证明 gap closure。
3. **P0：清理分支。** 删除误提交和大体积实验产物，保证 PR 只包含必要源码、测试、一个性能场景和一个可微场景。
4. **P1：完善梯度边界说明。** 对 contact creation/removal、feature switching、sample-set switching 建立专门测试，明确平滑区间与失败条件。
5. **P1：继续 NCU 分析融合 kernel。** 重点检查 register pressure、occupancy、重复 SDF 查询、branch divergence、global load/store 和 backward kernel 生成质量。
6. **P2：重新设计空间分块 overlap。** 只在 contact graph 可分、写集合独立的分区上并发，显式处理 halo 与同步，然后和单 stream + CUDA Graph baseline 比较。

## 12. 一句话交接

当前最可靠的性能成果是 **soft-contact BVH/AABB 候选压缩约 2x 加速**；最核心的功能成果是 **Newton mesh—mesh 接触保持 exact forward、使用 RS1Dist 风格 smooth witness backward 的实验管线**。融合算子和 Allegro 目标点优化都已打通基本链路，但前者性能不稳定，后者目前主要验证了四指闭合而非目标 witness 对齐，二者仍需要进一步验证后才能形成强结论。
