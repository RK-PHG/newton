# 固定 soft-contact 场景的性能测量

## 默认测量

默认打开标准 OpenGL Viewer，方便确认场景：

```powershell
uv run python diffsim/profile_particle_primitive_wall.py
```

正式做性能对比时关闭 Viewer：

```powershell
uv run python diffsim/profile_particle_primitive_wall.py --viewer null
```

开启 soft-contact 候选压紧优化：

```powershell
uv run python diffsim/profile_particle_primitive_wall.py --viewer null --soft-contact-candidates
```

开启候选时，同一次运行会额外采集一次未优化的原始路径。日志中的
`Candidate optimization comparison` 会直接给出：

```text
baseline create_soft_contacts
candidate operator
candidate create_soft_contacts
candidate + soft-contact
operator speedup
active contact pairs match baseline: PASS/FAIL
candidate + soft < baseline soft: PASS/FAIL
```

当前优化只减少计算量，不缩减 `soft_contact_max` 和接触缓冲区容量。候选集合是离散的，
不对候选选择本身求导；候选确定后仍由原来的可微 `create_soft_contacts` 完成前向和反向。

脚本不启动 Viewer，也不接受场景规模、形状比例或接触参数。它始终测量
`particle_primitive_wall.py` 对应的固定场景：4096 个粒子、32 个静态解析基本体、
131072 个 `particle × shape` 候选对，以及 4096 个实际 soft contact。

每次成功运行会生成一个独立日志，不会和其他运行混在一起：

```text
diffsim/log/particle_primitive_wall_profile_YYYYMMDD_HHMMSS_ffffff.log
```

报告包含：

- `collision_forward`：完整 `CollisionPipeline.collide()`；
- `solver_forward`：soft-contact 力计算和粒子积分；
- `simulation_forward`：清力、碰撞和求解的完整一步；
- `collision_backward`：只对 soft-contact 几何输出反传，不含求解器；
- `simulation_backward`：从速度 loss 经过积分、接触力和接触生成的完整反传；
- Warp 通过 CUDA event 收集的逐 Kernel、内建 Kernel、memcpy 和 memset 明细。

阶段表使用设备同步后的墙钟时间，适合优化前后比较。逐 CUDA activity 的收集会插入
CUDA event，适合判断 Kernel 排名，不应拿它的总和替代阶段表。

## Nsight Systems timeline

脚本通过 Warp 的 `ScopedTimer(use_nvtx=True)` 标记阶段，并用
`ScopedCudaProfiler` 把外部 profiler 的采集范围限制到预热和默认统计之后的一次紧凑
前向/反向迭代。安装 `nsys` 后运行：

```powershell
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=diffsim/log/particle_primitive_wall_timeline uv run python diffsim/profile_particle_primitive_wall.py --viewer null
```

结果为 `diffsim/log/particle_primitive_wall_timeline.nsys-rep`。timeline 中的主要 NVTX
区间是：

```text
diffsim/profile_capture
├── collision_forward
├── solver_forward
├── collision_backward
└── simulation_backward
```

`nsys` 可以实现真正的 CUDA timeline。当前脚本已经接入两部分：

- `ScopedCudaProfiler` 提供采集开始/停止边界，排除编译、预热和 20 次统计采样；
- `ScopedTimer(use_nvtx=True)` 给前向、求解和两种反向添加可读区间。

如果文本日志中的 `NVTX Python available` 为 `False`，`nsys` 仍能采集 CUDA Kernel
timeline，但不会显示上述命名阶段。安装当前 Warp 的 `nvtx` optional extra 后即可显示。

当前开发环境只检测到 Nsight Compute 的 `ncu`，没有检测到 `nsys`，所以这里没有生成
伪造或跨机器的 `.nsys-rep` 基线。
