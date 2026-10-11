# EdgeAI-Robotic-Arm-Control

**面向工业场景的端侧 AI 系统与模型推理加速研究**  
*Industry-Oriented Edge-AI System and On-Device LLM Inference Acceleration on RK3588 (6 TOPS NPU)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%7C%20Ubuntu%2022.04%20aarch64-orange.svg)](https://github.com/kimkingyu/EdgeAI-Robotic-Arm-Control)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5-purple.svg)](https://github.com/QwenLM/Qwen2.5)
[![NPU Accelerator](https://img.shields.io/badge/NPU%20Engine-RKLLM%20%26%20RKNPU2-red.svg)](https://github.com/airockchip/rknpu2)
[![Author](https://img.shields.io/badge/Author-kimkingyu-lightgrey.svg)](https://github.com/kimkingyu)

---

## 📌 项目背景与研究定位

本项目为个人独立自主研发的工业边缘端具身智能与视觉伺服控制系统，作为硕士毕业课题的前置核心工程预研。

针对现代工业自动化与柔性智造对**毫秒级硬实时、产线核心工艺数据隐私安全、弱网/无网环境下离线自主决策**的迫切需求，开展轻量化大语言模型（Qwen2.5）与工业视觉系统在边缘嵌入式算力（RK3588 / 6 TOPS NPU）上的极限轻量化量化与硬件级推理加速研究。

```text
[工业操作员 自然语言/工况指令]
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   RK3588 边缘端侧智能具身大脑 (Edge Brain)             │
│                                                                        │
│   [Qwen2.5 工业大模型] ──► [端侧任务规划器 TaskPlanner]                │
│            ▲                                                           │
│            │ (RKLLM 端侧推理，量化部署受阻中，见部署约束调研)          │
│            ▼                                                           │
│   [结构化工业动作序列 JSON] (pick / place / inspect / move_safe)        │
└────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   多模态感知与多轴运动控制闭环 (Control Pipeline)       │
│                                                                        │
│   [工业视觉检测 (YOLOv8 INT8)] ──► [手眼标定 Eye-to-Hand 齐次矩阵]     │
│            ▲                                     │                     │
│            │ (3核 NPU 并发实测 104.5 FPS)        ▼                     │
│            │                       [工件空间 3D 物理坐标 (X, Y, Z)]    │
│            │                                     │                     │
│            │                                     ▼                     │
│   [执行反馈] ◄── [I2C / PCA9685 驱动] ◄── [运动学逆解求解器 (IK)]      │
└────────────────────────────────────────────────────────────────────────┘
               │
               ▼
[工业工件自主抓取 / 缺陷分拣 / 自动化装配物理闭环]
```

> **当前进度说明**：上图中 VLM 决策、YOLO 检测、逆运动学与 PCA9685 驱动
> 已在板端实现并各自验证；虚线之后的**物理闭环尚未打通** —— 舵机板未接线、
> 相机未到货，末端执行环节目前只在 Mock 模式下运行。各模块的验证方式与
> 边界见下方各节标注。

---

## 🚀 核心关键技术与攻坚成果

### 1. 工业大模型指令微调与端侧任务规划 (Task Planning)
* 针对工业阀芯、法兰、轴承等典型工件规格与流水线装配逻辑，构建工业领域指令微调数据集；
* 基于 Qwen2.5 设计专用工业 Prompt 格式与 Function Calling 解析器，将非结构化自然语言指令秒级转化为严格可执行的底层动作状态机。

### 2. INT4/INT8 极限压缩与非对称量化校准 (PTQ & Calibration)
* **大模型量化（规划中，尚未实测）**：目标采用 `rkllm-toolkit` 对 Qwen2.5 实施 W4A16 混合量化。当前受阻于该工具链仅提供 `linux_x86_64` wheel，无 aarch64 版本，详见 [部署约束调研](docs/QWEN_DEPLOYMENT_BLOCKER.md)；
* **视觉骨干量化**：构建 100 张真实场景校准集。原始 YOLOv8n 的共享输出 INT8 量化虽把体积压至 **4.73MB（缩减 61.2%）**，但分类分支恒为 0，已停用；现改为**拆分输出头 INT8**（坐标 `[1,4,8400]`、分类 `[1,80,8400]` 分开量化），模型仍约 4.7MB，真实输出恢复，单实例 P50 **42.4ms / 23.6 FPS**。

### 3. RK3588 3核 NPU 异构并发调度与系统级调优
* **证伪常见误解**：实测发现单实例即便绑定 `NPU_CORE_0_1_2` 也只有 1.03x 加速 —— 因 `RKNNLite.inference()` 是同步阻塞调用，`core_mask` 仅声明"允许用哪些核"，并不会把单个任务拆分并行；
* **正确解法**：为每个物理核心创建独立 RKNNLite 实例，配合任务队列多线程并发喂帧，实测达成 **2.71x 近线性加速（38.5 → 104.5 FPS）**，封装为 `RKNNParallelVisionPool` 后实跑 **112.99 FPS**；
* **瓶颈定位**：NPU 全程满频 1GHz、温度仅 46℃，真正卡点在 CPU 侧的输入拷贝与输出反量化被 `ondemand` 压在 1.2GHz。交付 `scripts/set_performance.sh` 全线锁频后，单帧时延降 **21.5%**，P99 抖动降 **89%**。

### 4. 空间手眼标定与物理控制闭环（算法就绪，**实机未联调**）

> ⚠️ 本节为算法与驱动层成果，**全部基于合成数据与假总线验证**。
> USB 相机尚未到货，PCA9685 舵机板已到货但尚未接线（`i2c-7` 与 `0x40~0x47`
> 探测无响应）。舵机从未通电，因此**没有任何实物抓取或定位精度数据**。

* **手眼标定**：实现 Eye-to-Hand 平面单应标定（`cv2.findHomography`，>4 点用
  RANSAC 抑制示教误差）。合成真值反解验证：含示教噪声（像素 ±1.5px、
  机械臂 ±0.8mm）下独立测试点平均定位误差 **0.64mm**、最大 0.90mm，
  正反变换回环 0.0000px。该数值来自合成数据，**不代表实机定位精度**；
  退化输入（共线点、NaN、损坏标定文件）均已加防护；
* **舵机驱动**：基于 Linux I2C 直接操作 PCA9685 寄存器，PWM 周期由写入的
  prescale 与振荡频率推算而非硬编码 20ms。C++/Python 双实现在 721 个角度上
  换算完全一致，0/90/180° 脉宽偏差 0.043°/0.130°/0.216°。以上为公式复算与
  假总线捕获，**未经示波器校准**；
* **运动学**：3-DOF 几何逆解，含工作空间与关节行程双重边界防护。C++ 与
  Python 双实现在 1018 个目标点上可达判定与关节角零分歧，FK/IK 回环误差
  1.8e-13mm（纯数学）；
* **C++ 主控链路**：`HandEyeTransformer` 已实现平面单应变换与 Python 标定
  文件载入，`src/main.cpp` 控制线程已接入手眼变换、逆解与 PCA9685 驱动。
  与 Python 侧跨语言比对：378 个像素点坐标解算最大偏差 0.00e+00mm、
  正反变换回环 5.76e-14px、退化标定文件双方均拒绝载入。
  **安全默认**：未载入标定则拒绝解算而非退回近似映射；驱动舵机需
  `--i2c` 与 `--drive` 同时显式指定。板端实测 3 秒解算 172 次、下发 0 次；
  舵机板未接线时降级为仅解算。**仍无实机抓取与定位精度数据**。

### 5. MLIR 视觉预处理编译器（**实验模块，未取得加速**）

> ⚠️ 结论先行：自建的 MLIR 编译器能正确生成并跑通板端算子，但**在两个生产
> profile 上都慢于 OpenCV**（满频条件下 4.38 倍）。计划设定的"至少改善 10%"
> 门槛未达成，因此 **OpenCV 保持默认后端**，MLIR 仅作显式可选实验后端。
> 本节记录的是一次结论为负的完整优化尝试，不是已生效的性能优化。

* **编译器本体**：固定 `llvmorg-20.1.8`（锁 SHA256），用 MLIR Builder 生成 IR 而非文本拼接；自写三个 Pass —— `edge-fuse-resize-color` 消除 resize 与换色间的整图中间张量、`edge-prepare-preprocess-destination` 做 destination-passing、`edge-tile/vectorize-preprocess` 做受限分块与向量化。AOT 产出不链接 libMLIR 的独立 `.so`，运行期不带编译器；
* **正确性**：52 个内核全部通过逐字节比对（对独立 Q11 定点参考）、链接期 `malloc` 包装的分配审计与保护页检查；8 个因输出尺寸不大于分块尺寸标为不适用，不计入通过。融合＋目标复用版本实测**零分配**；
* **发现向量化的语义缺陷**：上游对 `tensor.extract` 生成的 `vector.gather` 按紧凑布局算偏移、无视行 stride，带 5 字节行填充的输入最大误差达 249。没有改用无填充输入掩盖，而是让 Pass 默认拒绝该结果，须显式开关才生成；
* **纠正自己的测量错误**：边界快照显示各阶段恒定 2.4GHz，但补上 50ms 连续采样后发现混合负载下 **69.8% 的采样低于满频** —— NPU 推理占 38ms、预处理仅 9.5ms，CPU 等待时被 `ondemand` 降频。OpenCV 与 MLIR 受影响程度不同（2.20x vs 1.52x），把比值从 4.38 **压缩**成 3.02。原先报的 3.0 倍是污染结果，已订正；
* **±1 灰度级不等于等价**：MLIR 与 OpenCV 的预处理输出最大仅差 1 个灰度级，但真实 RKNN 原始输出并不相同（YOLOv8n 最大绝对差 180.2）。经对照实验确认 NPU 本身确定、且 MLIR 与参考逐字节一致（偏离参考的是 OpenCV 舍入），故判定不可做无缝默认替换；
* **可复现**：全新目录重建，312 个 IR/汇编产物 SHA256 与原产物逐字节一致。

详见 [编译与实测记录](docs/MLIR_PREPROCESS_BUILD.md)、[M4 消融汇总](docs/benchmarks/mlir_m4_summary.json)、[M5/M6 汇总](docs/benchmarks/mlir_m56_summary.json)。

---

## 📊 性能评测基准 (Benchmark)

*测试硬件：香橙派 OrangePi 5 Pro (RK3588S, 16GB LPDDR5) / Ubuntu 22.04 LTS (Kernel 5.10.160)*

### 1. 端侧视觉多模态大模型 (Qwen3-VL-4B, w8a8)

> 板端真实推理实测。完整数据见 **[Qwen3-VL 实测报告](docs/benchmarks/QWEN_VL_BENCHMARK_REPORT.md)**，  
> 部署约束与证伪过程见 **[部署约束调研](docs/QWEN_DEPLOYMENT_BLOCKER.md)**。

| 指标项 | 实测值 |
| :--- | :--- |
| 模型组成 | 语言 `4.51 GB` + 视觉编码器 `0.81 GB` = **5.32 GB** |
| 模型加载耗时 | **5.0 ~ 6.4 s** |
| 视觉编码单帧 (448×448) | 2071 ms → 196 image tokens |
| **首字时延 TTFT** | **253 ms**（纯文本最优）/ 2766 ms（含图像 prefill） |
| **解码吞吐** | **6.32 tokens/s** |
| **JSON 格式遵循率** | **100%** (3/3 工业指令用例) |
| 推理峰值内存 | ~6.5 GB（板端 15.2GB 可用，余量充足） |

**为什么用 VLM**：相机画面可直接进模型，省去「YOLO 检测 → 坐标转译 → LLM 规划」的两段式流程。实测模型能给出**基于物理属性的抓取推理**：

> **问**：如果用机械臂抓取，你会选哪个物体？为什么？  
> **答**：狗。理由：它静止、无危险且易抓取（趴卧在地），而自行车有轮子和结构复杂，不易稳定抓取。

**关键工程点**：① `rkllm-toolkit` 无 aarch64 版本，板端无法自行量化，故采用社区现成 w8a8 模型；② 运行时由 v1.0.1 升级至 v1.2.3；③ 用 ctypes 绑定替代官方 C++ 方案；④ 视觉编码器输出 4 个 deepstack 特征层，**必须拼接**为 `(196, 10240)`，只取首层会导致生成立即中断。

### 2. 视觉感知引擎板端实测评测

> 以下数据全部由香橙派板端真实跑出（rknn-toolkit2 2.3.2 / librknnrt 2.3.2，300 轮正式测试 + 20 轮预热）。  
> 📊 完整数据与分析：**[NPU 推理加速实测报告](docs/benchmarks/NPU_BENCHMARK_REPORT.md)**  
> 🔧 想自己跑一遍：**[NPU 量化与加速复现指南](docs/NPU_REPRODUCTION_GUIDE.md)**（从裸板到 112 FPS，全程板端，无需 x86 PC）

> ⚠️ **重要更正（2026-09-17 发现，2026-09-20 修复）**：原始单输出 INT8 模型的
> 分类分支在量化后恒为 0，检测实际不可用 —— 坐标算得出来，却永远不知道框里是什么。
> 根因是坐标（0~640）与分数（0~1）拼在同一张量，共享 scale≈2.51，分数必然归零；
> `normal`/`mmse`/`kl_divergence` 三种算法全部如此。后处理实现后才发现这个问题，
> 详见 [根因分析](docs/benchmarks/yolo_int8_classification_failure.json)。
>
> **当前默认已改为拆分输出头 INT8**：坐标 `[1,4,8400]` 与分类 `[1,80,8400]` 分开量化，
> 真实输出恢复（score max 0.744），6 张真实图片检出 12 个目标，单实例 P50 42.4ms / 23.6 FPS。
> FP16 仍保留作对照（P50 99.1ms / 10.1 FPS）。

**① 模型量化压缩效果**

| 模型 | ONNX 原始 | INT8 RKNN | 压缩比 | 算子总数 | NPU 直通率 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **YOLOv8n（选定主干）** | 12.2 MB | **4.73 MB** | **61.2%↓** | 118 | **97.46%** (115/118) |
| YOLOv11n（对照组） | 10.4 MB | 4.54 MB | 56.3%↓ | 160 | 98.75% (158/160) |

两者纯计算算子均 **100% NPU 执行、零真实回退**（回退项仅为输入输出桩算子）。但如上所述，
**这两个 INT8 模型的分类输出均已被量化破坏**（YOLOv11n 结构相同，预计一致，尚未实测）；
算子直通率高不等于模型可用。YOLOv11 的 C3k2+PSA 注意力已被 RKNPU 2.3.2 完整支持，直通率反优于 v8；但其算子数多 **35.6%**，端到端耗时高 **17.2%**，故实时伺服主干仍选定 YOLOv8n。

**② 单路推理时延（YOLOv8n INT8）**

| 部署与调度方案 | 单帧耗时 | P99 | 抖动 (σ) | 吞吐 |
| :--- | ---: | ---: | ---: | ---: |
| 默认 `ondemand` 调频（CPU 1.2GHz） | 29.60 ms | 44.21 ms | ±5.94 ms | 33.79 FPS |
| 单核 `NPU_CORE_0` + 锁频 | 23.23 ms | 27.45 ms | ±1.49 ms | 43.04 FPS |
| 三核 `NPU_CORE_0_1_2` + 锁频 | 23.04 ms | 25.73 ms | ±1.80 ms | 43.40 FPS |
| **`NPU_CORE_AUTO` + 全线锁频 + 绑大核** | **22.61 ms** | **24.05 ms** | **±0.65 ms** | **44.24 FPS** |

**③ 三核并发吞吐（核心突破）**

单实例即便声明 `NPU_CORE_0_1_2` 也只有 **1.03x** 加速 —— 因 `inference()` 同步阻塞，同一时刻仅一路任务在跑。改用**每核独立实例 + 多线程并发**后：

| 并发方案 | 系统吞吐 | 单路时延 | 加速比 |
| :--- | ---: | ---: | ---: |
| 1 路（单核独占） | 38.50 FPS | 23.31 ms | 1.00x |
| 2 路（双核并行） | 75.60 FPS | 23.51 ms | 1.96x |
| **3 路（三核全开）** | **104.50 FPS** | **24.95 ms** | **2.71x** |
| **生产封装 `RKNNParallelVisionPool`** | **112.99 FPS** | 26.26 ms | **2.93x** |

> 达成近线性加速的同时单路时延仅增 7%，可同时支撑 3 路工业相机；P99 抖动 ±0.65ms 满足伺服控制回路的确定性要求。

**当前可用视觉模型的实测**（单实例，`taskset -c 4-7`，100 轮）：

| 模型 | 输出 | P50 | P99 | FPS | 分类输出 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 原始 INT8（已停用） | 1 个 `[1,84,8400]` | 43.69 ms | 51.18 ms | 22.9 | 全零 |
| **拆分头 INT8（默认）** | 2 个 `[1,4,8400]` + `[1,80,8400]` | **42.40 ms** | **49.50 ms** | **23.6** | **可用** |
| FP16（对照） | 1 个 `[1,84,8400]` | 99.14 ms | 104.96 ms | 10.1 | 可用 |

拆分头 INT8 用同一套后处理兼容双输出，三核并发池已验证可加载并返回有效分数；完整证据见 [拆分头报告](docs/benchmarks/yolo_split_int8_report.json)。

### 3. MLIR 自建预处理算子 vs OpenCV（**结论为负**）

> 单核（`taskset -c 4`）单线程，480×640 输入，50ms 连续采样确认全程满频 2.4GHz。
> 每次计时调用都校验完整输出，空转或算错的内核无法显得更快。

| 调度配置 | P50 | 相对 OpenCV |
| :--- | ---: | ---: |
| **OpenCV `INTER_LINEAR`（默认，最快）** | **4359.2 us** | **1.00x** |
| MLIR 融合＋目标复用（最优 MLIR 配置） | 19103.1 us | 4.38x 慢 |
| MLIR 分块 4×32＋向量化 | 19012.1 us | 4.36x 慢 |
| MLIR 仅目标复用 | 21005.7 us | 4.82x 慢 |
| MLIR 分块 4×32（无向量化） | 28379.7 us | 6.51x 慢 |

固定尺寸分块反而更慢；向量化确实生成了真实 SIMD（211 个 q 寄存器引用、240 条 `ld1`/`st1`，对比融合版 6 个），但指令数从 164 涨到 4391，最终只追平融合版。**IR 层面变换成功不等于目标机器上有收益** —— 这是本模块最主要的结论。

**PMU 归因（结论与预设假设相反）**：板端没装 perf 且 `apt` 需交互式 sudo，改用 `perf_event_open` 直连 ARMv8 PMU 自测。原以为是内存带宽受限，实测是**指令数**：

| 指标 | OpenCV | 最优 MLIR | 说明 |
| :--- | ---: | ---: | :--- |
| 每输出字节指令数 | **11.59** | **82.38** | **7.1 倍，这就是主因** |
| IPC | 1.79 | **2.33** | MLIR 反而更高 |
| 后端停顿占比 | 0.41 | **0.20** | MLIR 反而更低 |
| L1D 读缺失率 | 0.0052 | **0.0042** | MLIR 反而更低 |

生成代码的**执行效率并不差，是要干的活多了 7 倍** —— 差距在代码生成质量，而非调度策略。

**再往下拆，那 7 倍指令有两个独立来源**：

1. **坐标按字节重算**：融合后的 `linalg.generic` 是三维全 parallel，源坐标与权重计算留在最内层，对每个 (行,列,通道) 都算一遍 —— x/y 方向各冗余 **1920 倍**。最内层循环 74 条指令只产出 1 个字节，其中 41.9% 是坐标与 clamp。但即便全部提出循环外，估算仍慢 **4.13 倍**，只解释了一部分；
2. **向量加载退化**：向量化版本 **1536 条 `ld1` 全是单字节插入通道**（`ld1 {v0.b}[i]`），无一条连续加载 —— `vector.gather` 让凑满一个 128 位向量需要 16 条 `ld1`，而 OpenCV 一条 `ld1 {v0.16b}` 搞定。SIMD 只用在乘加上，**加载侧零收益**。这也解释了分块＋向量化的指令数（83.4）为何反而略高于未分块（82.4）。

参照系也得说清：板端 OpenCV 以 **NEON 为 baseline** 编译，比的是长期手工优化的向量化实现，不是朴素标量代码。要追上需要重新设计算子表达方式（让地址计算 affine 可分析、让向量化产出连续 `ld1`），不是参数调优能解决的。

真实 RKNN 端到端 A/B（`librknnrt 2.3.2` + 真实权重，Mock 推理被显式拒绝）：

| 模型 | OpenCV 预处理 | MLIR 预处理 | 模型原始输出 |
| :--- | ---: | ---: | :--- |
| YOLOv8n INT8 @ 640² | 9544.7 us | 29050.9 us | 8 张量全不一致，最大差 180.2 |
| Qwen3-VL 视觉塔 @ 448² | 2096.5 us | 9708.9 us | 32 张量全不一致，最大差 5.07 |

> 这两行的绝对值受降频影响（见上文第 5 节），**不可用作内核性能比较**，仅说明混合负载下的实际表现。内核相对性能以上表满频数据为准。

---

## 📂 项目模块结构

```text
EdgeAI-Robotic-Arm-Control/
├── compiler/                     # MLIR 预处理编译器（独立 CMake，不入主程序构建）
│   ├── include/edgeai/           # Pass 声明与 pipeline 注册
│   ├── lib/Transforms/           # 自写 Pass：融合 / destination-passing / 分块向量化
│   ├── runtime/                  # 不依赖 LLVM 的稳定 C ABI 运行库
│   ├── profiles/                 # yolo640 / qwen448 / 小尺寸测试规格
│   ├── tests/                    # 正确性、分配审计、消融与频率监控
│   └── toolchain.lock.json       # 固定 llvmorg-20.1.8 与 SHA256
├── configs/
│   └── config.yaml               # 工业工况、大模型与机械臂统一配置文件
├── data/
│   └── calibration/              # 量化校准集样本与指令集
├── docs/
│   ├── PROJECT_ROADMAP.md        # 研发全景蓝图与阶段推进路线图 (Master Roadmap)
│   ├── MLIR_PREPROCESS_BUILD.md  # MLIR 编译器构建与实测记录（含未达成加速的结论）
│   ├── BOARD_SETUP_WALKTHROUGH.md# 香橙派板端环境配置与实战通关指南
│   ├── PCA9685_WIRING_GUIDE.md   # PCA9685 40-Pin 极简硬件接线与引脚定义
│   ├── SINGLE_SERVO_TEST_GUIDE.md# 单舵机免外接电源安全轻测指南
│   ├── NPU_REPRODUCTION_GUIDE.md # NPU 量化与推理加速全流程复现指南
│   ├── QWEN_DEPLOYMENT_BLOCKER.md# Qwen 端侧部署约束调研与路线决策
│   ├── RESUME_GUIDE.md           # 简历项目经历与技术问答参考手册
│   ├── benchmarks/               # 板端实测性能报告与原始 JSON 数据
│   └── dev_logs/                 # 自动化研发技术台账与步骤历史索引
├── models/
│   └── weights/                  # .rkllm (大模型) 与 .rknn (视觉) 量化模型
├── scripts/
│   ├── mlir/                     # LLVM 固定版本拉取、构建与算子产物生成
│   ├── check_board_env.sh        # RK3588 软硬件体检脚本
│   ├── remote_deploy.py          # PC 端一键免密远程部署工具
│   ├── set_performance.sh        # CPU/GPU/NPU/DDR 全线锁频性能模式脚本
│   └── setup_rk3588_base.sh      # 板端初始化加速脚本
├── src/
│   ├── controller/               # 机械臂底层通信驱动 (I2C / PCA9685 / 串口)
│   ├── inference/                # RKLLM 大模型与 RKNN 3核视觉推理引擎
│   ├── kinematics/               # 正逆运动学几何解算器 (IK)
│   ├── llm/                      # Qwen 工业指令微调模板与任务规划器 (Planner)
│   ├── pipeline/                 # 大模型-视觉-控制全闭环任务调度主干
│   └── vision/                   # V4L2 摄像头抓取、检测与统一预处理后端
├── tools/
│   ├── export_rkllm.py           # Qwen 大模型 W4A16 极限压缩转换脚本
│   ├── export_rknn.py            # 视觉模型 INT8 KL 散度量化转换脚本
│   ├── benchmark_quant.py        # 板端端到端性能 Benchmark 评测工具
│   ├── benchmark_npu.py          # NPU 单路多核调度模式时延基准测试
│   ├── benchmark_npu_parallel.py # NPU 三核真并发吞吐上限测试
│   ├── benchmark_preprocess.py   # OpenCV 预处理分段基线（resize/换色/包装分开计时）
│   ├── benchmark_preprocess_rknn.py # 预处理后端的真实 RKNN A/B（拒绝 Mock 推理）
│   └── test_mlir_preprocess.py   # 预处理契约、内存与后端选择测试
├── main.py                       # 工业控制台入口主程序
└── requirements.txt
```

---

## ⚡ 快速上手

### ResNet18 离线分类 demo（CPU，不控制机械臂）

输入一张真实图片，输出 ImageNet Top-5、原始模型与 Conv-BN 融合对照、离线 HTML 和 JSON。这里是**真实 CPU 推理**，不连接摄像头、舵机或 NPU，也不使用 RKNN 的 Mock 延时。分类不提供目标框或抓取坐标。

以下为 Windows / Git Bash 命令，使用项目独立环境（首次创建；已有 `.venv` 时先检查，不要覆盖）：

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-resnet.txt

# 换成你自己的图片；只有显式允许才会下载缺失的官方权重
.venv/Scripts/python.exe tools/run_resnet_demo.py \
    --image path/to/image.jpg --download-weights --compare --benchmark

# 导出静态 FP32 ONNX，并用同一张真实图片做 ONNX Runtime CPU 数值对照
.venv/Scripts/python.exe tools/export_resnet_onnx.py --image path/to/image.jpg

# 离线单元测试（不下载权重，不碰硬件）
.venv/Scripts/python.exe tools/test_resnet_demo.py
```

- 直接打开 `build/resnet-demo/demo.html` 查看结果，`results.json` 保存逐轮计时、输入/权重哈希和环境信息。已有报告不会默认覆盖；重跑时换 `--output-dir`，或显式加 `--overwrite`。
- 模型使用 `ResNet18_Weights.IMAGENET1K_V1` 的 RGB、短边 resize 256、中心裁剪 224、ImageNet mean/std；**不能套用 YOLO 的直接 resize 和归一化**。
- 默认基准为每版本预热 10 次、每轮 50 次、3 轮交错测量，线程数为 1；可用 `--warmup`、`--iterations`、`--rounds`、`--threads` 调整。分类流水线从已解码 RGB 图像开始，不含下载、磁盘 I/O、模型加载或报告写入。单图重复基准不代表摄像头吞吐或数据集准确率。
- Conv-BN 融合是现成优化，可能更快也可能更慢；报告保留实际结果和数值误差。少量图片的 Top-1 一致不等于 ImageNet 精度验证，CPU 数据也不代表 RK3588 NPU 性能。
- ONNX 默认保存到 `models/weights/resnet18-fp32.onnx`，附带同名 JSON 元数据。输入已经是归一化 NCHW FP32，图中不重复归一化；尚未转换或验证 RKNN/INT8。

### 多小模型优化对照（阶段 A，CPU-only）

这里比较**同一个优化对不同模型的作用**，不是只比较模型大小。固定 `ResNet18`、`MobileNetV3-Small`、`ShuffleNetV2 ×1.0` 的官方 ImageNet-1K V1 权重，batch=1，不训练、不访问硬件。沿用上面的项目独立环境，不要安装进板端 RKNN 环境。

| 组别 | 方案 | 对照对象 |
|---|---|---|
| B0 | PyTorch FP32 原始版 | 每模型自己的基线 |
| F1 | Conv-BN 融合 | B0 |
| L1 | channels_last 内存格式 | B0；逻辑形状仍为 NCHW |
| FL1 | 融合 + channels_last | B0、F1、L1 |
| O0 | 同一 ONNX，ORT 图优化关闭 | O1、B0 |
| O1 | 同一 ONNX，ORT 图优化开启 | O0；不能把后端差异归因于文件格式 |

三个模型共 18 个主配置，再给 B0/O1 增加 2/4 线程，共 **30 个唯一配置**。配置见 [configs/classifier_benchmark.json](configs/classifier_benchmark.json)。每个配置和重复块用独立进程；Windows 直接启动真实解释器，同时保留项目虚拟环境，校验实际 worker PID＋操作系统进程创建时间、代码/数据/权重哈希；退出后的 PID 可以被回收，不把裸 PID 绝对唯一当作独立性证据。

```bash
# 只读预检：不执行模型、不下载权重、不写报告
.venv/Scripts/python.exe tools/run_classifier_benchmark.py \
    --mode preflight --image data/resnet-demo/dog.jpg

# 探索性真实运行：仅显式开关允许下载缺失的官方权重
# 独立的流水线/RSS观测和profile不混入推理计时
.venv/Scripts/python.exe tools/run_classifier_benchmark.py \
    --mode smoke --image data/resnet-demo/dog.jpg --download-weights \
    --warmup 5 --iterations 30 --repeats 2 --profile \
    --output-dir build/classifier-benchmark/my-run

# 只重新汇总已有结果，不重新推理；覆盖报告须显式允许
.venv/Scripts/python.exe tools/summarize_classifier_benchmark.py \
    --run-dir build/classifier-benchmark/my-run --overwrite

# 新的离线测试与原有回归：不下载、不碰硬件
.venv/Scripts/python.exe -m unittest \
    tools.test_classification tools.test_classifier_benchmark \
    tools.test_classifier_runner tools.test_resnet_demo tools.test_yolo_postprocess
```

- 结果目录包含 `run.json`、逐任务结果、带 SHA256 的原始 `.samples.jsonl`、JSON/CSV 汇总、自包含 `report.html` 与独立诊断记录。新运行使用新目录，不覆盖旧 ResNet 演示；中断恢复用同参数加 `--resume`，代码、依赖、CPU、清单和任务选择变化会拒绝混用。
- `--models`、`--variants` 可筛选实验；`--no-thread-sweep`、`--no-observations` 可缩小 smoke 运行。完整配置白名单不随筛选改变。
- 推理微基准从已准备的输入开始；流水线另测预处理、输入适配、推理、后处理及总墙钟时间。布局转换和 NumPy 桥接不能藏在计时外。RSS 用独立进程和原生系统接口采样，不是模型独占内存，可能漏掉短暂峰值。
- **单图 smoke 只用于功能和探索对照。** 少量重复块不给正式采用建议；softmax 分数不是正确率；CPU 数据不代表 RK3588 NPU 性能。保留变慢、输出误差、异常及不支持项，不把未运行填成 0。
- 正式模式需要独立 Development 集（128 张，其中固定 64 张测性能）。量化门槛另需 256 张校准图、128 张带标签开发图和至少 1000 张带标签 Holdout；各 split 不能泄漏。阶段 A 的 FP32/单图结果不代替阶段 B 的真实 INT8 与带标签评估。
- 数据通过 `--manifest` 提供 JSON 清单：`schema_version=1`、`root`（相对清单目录）及 `images`。每项含 `id`、相对 `path`、真实文件 `sha256`、`split`、可选 `source_id`/来源/许可和 `label`。标签必须为固定 ImageNet-1K 映射的 0..999，无真值用 `null`，不能把模型预测当真值；清单会核对哈希、路径边界和跨 split 重叠。

### 多图与 INT8 对照（阶段 B，独立数据与冻结评估）

数据选用 Oxford-IIIT Pet 的 **20 个人工冻结映射犬种**，不是完整 ImageNet 验证集。三个模型仍对全部 1000 类输出和排名，不能把候选类别缩成20类提升分数。图片和许可来源、别名、镜像固定版本及内容哈希见 [configs/classifier_dataset.json](configs/classifier_dataset.json) 与运行产生的 `source.json`；独立真值来自公开数据标注，不来自模型预测。

原站归档下载速度不足且不支持 Range，实际准备改用相同 CC BY-SA 4.0 许可的固定公开镜像，核对两个 Parquet 的 LFS SHA256。**不把镜像校验冒充牛津原始 tar 的 MD5 验证**。图片只用于本地研究，不公开再分发；原图所有者版权仍保留。

```bash
# 数据读取依赖只装在项目.venv，不升级现有CPU推理依赖
.venv/Scripts/python.exe -m pip install -r requirements-classifier-data.txt

# 显式允许下载固定revision的许可镜像，不执行远端dataset代码
.venv/Scripts/python.exe tools/download_classifier_mirror.py \
    --download --output-dir data/classifier-mirror-cache

# 校准256、开发128、最终持出1000；同seed/源产生同清单，默认拒绝覆盖
.venv/Scripts/python.exe tools/prepare_classifier_dataset.py \
    --mirror-dir data/classifier-mirror-cache --output-dir data/classifier-pets

# M4：固定开发集64张测性能，独立5块，每块预热20/测量200
# 前后基线哨兵超过10%漂移：至多复测该组一次，旧样本归档不删除
.venv/Scripts/python.exe tools/run_classifier_benchmark.py \
    --mode formal --manifest data/classifier-pets/manifest.json \
    --no-observations --profile --output-dir build/classifier-fp32-formal-v2

# M5/M6：仅接受经过完整性与漂移控制审计的正式FP32输入
.venv/Scripts/python.exe tools/run_classifier_stage_b.py \
    --manifest data/classifier-pets/manifest.json \
    --fp32-run build/classifier-fp32-formal-v2 --output-dir build/classifier-stage-b-v2
```

- Calibration 与 Development 从来源训练部分分层取得，Holdout 只从来源测试部分取得；各 split 做源身份、JPEG字节及 EXIF规范RGB像素哈希去重。不能保证排除所有近重复，不能证明预训练模型从未见过这些图片。
- 量化家族单独保存 **QB/Q1/Q2/Q3**：QB是量化预处理后的FP32图；Q1=MinMax/per-tensor，Q2只改权重per-channel，Q3只改为99.99 Percentile。均为QDQ S8S8、线程1与受控CPU会话，未覆盖算子保留浮点；实际整数核须另看runtime profile，QDQ节点本身不证明INT8执行。
- 校准逐张收集范围/直方图并清空featuremap。Q1/Q2共用同一MinMax范围缓存；缓存与父QB、清单、校准图片列表及参数哈希绑定，图中不重复归一化。
- 方案只在开发集选择。三模型候选、产物和线程全部冻结后才允许最终Holdout评估；冻结SHA、数据SHA或产物/线程不匹配会拒绝执行，最终失败不改选候选。候选采用预算与数据规模只是本项目研究约定，不是普遍性能或精度保证。
- 所有实验不训练、不连接板卡。保留 unsupported、精度超预算、不确定与负加速，既不拼接旧单图数据，也不把小子集成绩称为完整ImageNet准确率。
- Windows 中文工程路径下，本机 ONNX 原生形状推断会缺少 `QB-inferred.onnx`。工具将原图哈希校验副本和计算文件放在系统 ASCII 临时目录，再校验复制到输出盘内并原子发布；不更改图或校准参数、不设置全局环境。若系统 `TEMP/TMP` 也含非 ASCII 字符，会明确拒绝，需先指定合法 ASCII 临时目录。
- `completed_with_failures`、`int8_evidence_status=failed_runtime/unsupported/partial_failure/not_evaluated` 表示未取得完整 INT8 证据，不是“量化没有收益”；只有真实完成开发比较后，才允许给出已测拒绝或采用判定。CLI 对不完整轮返回非零退出码。
- [首轮阶段 B 记录](build/classifier-stage-b/summary.json) 保留作工程诊断：九个 INT8 方案因中文路径失败，旧版“无 INT8 收益”判定无效。该轮已冻结 FP32 候选并访问过 Holdout；路径修复后的重跑保持数据、量化参数、门槛与仅开发集选择规则不变，**不依据旧 Holdout 成绩调参，也不把本数据集说成从未访问过的新持出集**。

#### 本轮正式结果与采用边界

先看 [决策速览 HTML](build/classifier-stage-b-v2/decision-summary.html) 与 [决策 JSON](build/classifier-stage-b-v2/decision-summary.json)；完整数据见 [阶段 B 原始报告](build/classifier-stage-b-v2/report.html)、[完整 JSON](build/classifier-stage-b-v2/summary.json)、[交付审计](build/classifier-stage-b-v2/delivery-audit.json) 和 [M4 正式 FP32 消融](build/classifier-fp32-formal-v2/report.html)。

冻结后独立确认使用 **5 块 × 预热 20 × 测量 500**；加速比是配对块估计，直接对照均为**同一轮、同一模型的 O1.t1**，不是跨轮借用 PyTorch 基线。以下为推理微基准，不含预处理/解码：

| 模型 | 最终冻结方案 | P50 / P95 (ms) | 配对加速比 [95% CI] | Holdout Top-1 / Top-5 |
|---|---|---:|---:|---:|
| ResNet18 | O1 FP32，4 线程 | 6.656 / 9.272 | 4.95× [4.92, 4.98] | 83.4% / 97.9% |
| MobileNetV3-Small | O1 FP32，2 线程 | 1.013 / 1.266 | 1.25× [1.17, 1.33] | 81.4% / 96.3% |
| ShuffleNetV2 ×1.0 | O1 FP32，4 线程 | 1.794 / 2.137 | 1.52× [1.46, 1.56] | 82.9% / 96.9% |

三组持出评估均为 1000 张、完整 1000 类排名，Top-1/Top-5 相对自身 FP32 基线下降均为 0pp。仅覆盖上述 20 个犬种，不能当完整 ImageNet 准确率；旧工程失败轮的 Holdout 访问已披露。五块 CI 仍是探索性不确定性，不表示普遍最优。

- **FP32 因素取舍**：ResNet18 单独融合只有约 1% 变化，布局+融合约 1.27×；ShuffleNetV2 的单独布局约 2.04×、布局+融合约 2.36×（均相对本轮 M4 B0.t1）。MobileNetV3 的 O1/O0 图优化增量约 2.69×。这些比值不能相乘或拼到 M6 确认比值上。
- **线程不是越多越好**：MobileNetV3 的 4 线程在开发筛选中 P95 比 O1.t1 高约 12.9%，未过预设尾延迟门槛，最终选 2 线程。描述性 Pareto 表可能仍出现其更低的名义 P50，不代表可以采用。
- **ResNet18 INT8**：三种方案相对 QB.t1 约 1.17×提速、文件约缩到四分之一；但开发精度下降 CI 跨越 1pp 预算，判为证据不足，未采用。不是“INT8 一点也不快”。
- **MobileNetV3 INT8**：三种固定 S8S8 配方约 0.64×（反而变慢），开发 Top-1/Top-5 均退化到 0，明确拒绝。仅四张开发图的 [数值诊断](build/classifier-stage-b-v2/mobilenet-int8-diagnostic.json) 在关闭/开启 ORT 图优化时也呈同样退化、输出有限；具体逐层根因未定位，不把现象冒充模型结构的普遍因果结论。
- **ShuffleNetV2 INT8**：速度比约 0.99×，未达到 5% 采用收益；Q1 开发 Top-1 下降约 5.47pp，per-channel 的 Q2/Q3 点估计改善但 CI 仍未过预算。文件变小不等于执行更快。
- **代价分开计**：三模型候选的独立分类流水线总墙钟 P50 分别为 10.316 / 2.261 / 2.943ms（已解码 RGB 起点，不含磁盘解码）；RSS 峰值约 154.71 / 115.34 / 119.57MiB，相对各自单线程观测增加约 2MiB，含运行库、64 图输入与分配器。INT8 没有全矩阵 RSS 观测，不把文件压缩等同 RAM 下降。M4 `load_ms` 与 `first_inference_ms` 保留在原始 worker 记录，不能并入稳定推理 P50。

交付复验：285 个独立 OS 进程实例（261 个不同整数 PID，复用由创建时间区分）、36,000 个正式推理样本；30 组漂移均接受，最大因子 1.02424。完整分类/旧入口离线回归 **300 项通过**，日志见 [build/classifier-final-regression.log](build/classifier-final-regression.log)。不训练、不连接硬件、不提交或推送 Git。

### 1. 模拟运行 (脱机测试全流程)
无需外接机械臂硬件，在 PC 或板端直接验证“自然语言指令 ➔ Qwen 解析 ➔ 逆解 ➔ 虚拟执行”：
```bash
python3 main.py --mock
```

### 2. 命令行执行指定工业作业
```bash
python3 main.py --cmd "将3号工位上的待检阀芯抓取并移动至缺陷品收集箱"
```

### 3. 实机 NPU 性能 Benchmark 压测
在香橙派终端运行，自动输出多核调度时延与并发吞吐基准：
```bash
# 先锁定高性能调频（每次重启后需重新执行）
sudo bash scripts/set_performance.sh

# 单路多核调度模式时延对比
taskset -c 4-7 .venv-rknn/bin/python tools/benchmark_npu.py --rounds 300

# 三核真并发吞吐上限测试
taskset -c 4-7 .venv-rknn/bin/python tools/benchmark_npu_parallel.py --rounds 150
```

### 4. 视觉模型导出

**当前默认是拆分头 INT8**：先拆分坐标与分类输出，再分别量化；原始单输出 INT8 不可用。
```bash
.venv-rknn/bin/python tools/split_yolo_output.py \
    --input models/weights/yolov8n_op19.onnx \
    --output /tmp/yolov8n_split.onnx
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx /tmp/yolov8n_split.onnx \
    --output models/weights/yolov8n_split_int8.rknn --algorithm normal
```

FP16 保留作可靠对照：
```bash
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx models/weights/yolov8n_op19.onnx \
    --output models/weights/yolov8n_fp16.rknn --fp16
```

原始单输出 INT8 命令保留备查，但 `normal`/`mmse`/`kl_divergence` 三种算法的分类分支均会归零，
不要把它设为默认：
```bash
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx models/weights/yolov8n_op19.onnx \
    --output models/weights/yolov8n_int8.rknn --algorithm normal
```

验证模型是否可用（不要只看能否加载）：
```bash
# 解码边界测试
.venv-rknn/bin/python tools/test_yolo_postprocess.py
# 真实模型与真实图片验证：应检出非零目标
.venv-rknn/bin/python verify_yolo_decode.py
```

### 5. 启用 MLIR 实验预处理后端（可选，实测更慢）

默认走 OpenCV，**不需要任何 MLIR 依赖**。要用实验后端得显式指定库路径；AOT 内核的输出尺寸是编译期固定的，一个 `.so` 只服务一个 profile，所以 640 和 448 各给一个：

```bash
python3 tools/run_vlm_grasp.py --cmd "..." --mock \
  --preprocess-backend mlir \
  --preprocess-library 640x640=<产物目录>/640x640/fused_destination/libedgeai_preprocess.so \
  --preprocess-library 448x448=<产物目录>/448x448/fused_destination/libedgeai_preprocess.so \
  --preprocess-manifest <产物目录>/manifest.json
```

缺库、架构不符、输入不受支持都会**直接报错**，不会静默退回 OpenCV —— 否则就会把 OpenCV 的耗时记在 MLIR 名下。编译器本身的构建步骤见 [MLIR_PREPROCESS_BUILD.md](docs/MLIR_PREPROCESS_BUILD.md)。

---

## 📄 项目声明与开源协议

* **开发者**: [kimkingyu](https://github.com/kimkingyu)
* **研究方向**: 边缘计算架构、端侧 AI 模型量化加速与机器人运动控制
* **许可证**: 本项目采用 [MIT License](LICENSE) 协议开源。
