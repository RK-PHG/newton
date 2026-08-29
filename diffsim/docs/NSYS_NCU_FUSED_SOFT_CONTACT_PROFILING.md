# Fused soft-contact 的 NSYS / NCU 分析

所有命令都在 Newton 仓库根目录执行。对比融合算子时保持候选点对关闭：

- baseline：不传 `--fused-soft-contact`
- fused：传入 `--fused-soft-contact`
- 两边都不要传 `--soft-contact-candidates`
- 两边都使用 `--viewer null`

先建立输出目录：

```powershell
New-Item -ItemType Directory -Force diffsim\profile\nsys,diffsim\profile\ncu | Out-Null
```

## 1. NSYS：比较真实运行时间

### Baseline

```powershell
nsys profile -t cuda -s none -c cudaProfilerApi --capture-range-end stop-shutdown -f true -o diffsim\profile\nsys\baseline uv run python diffsim\profile_particle_primitive_wall.py --viewer null
```

### Fused

```powershell
nsys profile -t cuda -s none -c cudaProfilerApi --capture-range-end stop-shutdown -f true -o diffsim\profile\nsys\fused uv run python diffsim\profile_particle_primitive_wall.py --viewer null --fused-soft-contact
```

脚本中的 `wp.ScopedCudaProfiler` 负责启动和停止采集，因此这里使用
`-c cudaProfilerApi`。Nsight Systems 2026.4.1 不支持
`--profile-from-start=false`，不要再传该参数。

生成文件：

```text
diffsim/profile/nsys/baseline.nsys-rep
diffsim/profile/nsys/fused.nsys-rep
```

查看 Kernel 汇总：

```powershell
nsys stats --report cuda_gpu_kern_sum diffsim\profile\nsys\baseline.nsys-rep
nsys stats --report cuda_gpu_kern_sum diffsim\profile\nsys\fused.nsys-rep
```

在 NSYS GUI 中比较名称以 `_cuda_kernel_forward` 结尾的条目：

```text
baseline: create_soft_contacts + eval_particle_body_contact
fused:    eval_particle_body_contact_fused
```

报告还包含 backward。fused 报告中的 collision-only backward 检查仍可能出现
`create_soft_contacts ... _cuda_kernel_backward`；它不属于 fused 前向，不要把它
计入下面的前向时间。

前向性能比较：

```text
baseline = create_soft_contacts duration
         + eval_particle_body_contact duration

fused    = eval_particle_body_contact_fused duration
```

同时记录两次 baseline Kernel 之间的 launch gap。NSYS 的 duration 用于判断融合
是否真正变快。

## 2. NCU：分析变快或变慢的原因

使用 `--set detailed` 收集寄存器、occupancy、计算和内存指标。NCU 会 replay
Kernel，因此不要用 NCU 中的时长替代 NSYS 时长。

如果出现 `ERR_NVGPUCTRPERM`，需要使用管理员 PowerShell，或在 NVIDIA Control
Panel 中允许访问 GPU Performance Counters；这属于计数器权限问题，不是命令错误。

### Baseline：接触生成

```powershell
ncu --profile-from-start off --set detailed -k regex:create_soft_contacts -c 1 -f -o diffsim\profile\ncu\baseline_create uv run python diffsim\profile_particle_primitive_wall.py --viewer null
```

### Baseline：接触响应

```powershell
ncu --profile-from-start off --set detailed -k regex:eval_particle_body_contact_ -c 1 -f -o diffsim\profile\ncu\baseline_response uv run python diffsim\profile_particle_primitive_wall.py --viewer null
```

### Fused

```powershell
ncu --profile-from-start off --set detailed -k regex:eval_particle_body_contact_fused -c 1 -f -o diffsim\profile\ncu\fused uv run python diffsim\profile_particle_primitive_wall.py --viewer null --fused-soft-contact
```

生成文件：

```text
diffsim/profile/ncu/baseline_create.ncu-rep
diffsim/profile/ncu/baseline_response.ncu-rep
diffsim/profile/ncu/fused.ncu-rep
```

重点比较：

- Registers Per Thread
- Local Memory Per Thread
- Theoretical / Achieved Occupancy
- DRAM Read/Write Bytes 和 DRAM Throughput
- L1/TEX、L2 命中率
- SM Throughput 和执行指令数

如果需要完整的 warp stall 分析，把对应命令中的 `--set detailed` 改成
`--set full`；完整采集会明显更慢。

## 3. 结果判断

- fused duration 小于 baseline 两个 Kernel duration 之和：融合达到性能目标。
- DRAM 流量下降，但 fused 仍更慢：重点检查寄存器、occupancy 和指令数。
- Local Memory 或 spill 上升：寄存器压力抵消了中间数组读写的节省。
- Registers Per Thread 上升且 occupancy 下降：融合 Kernel 过大或变量 live range
  过长。

分析完成后保留两个 `.nsys-rep`、三个 `.ncu-rep`，以及 baseline/fused 各自生成
的 `diffsim/log` 时间戳日志。
