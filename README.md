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
* **视觉骨干 INT8 量化**：构建 100 张真实场景校准集，采用逐通道非对称 INT8 训练后量化（PTQ），YOLOv8n 体积由 12.2MB 压至 **4.73MB（缩减 61.2%）**，权重内存仅 3.6MB；攻克 `opset 超限`、`校准集路径二次拼接`、`动态形状无法量化` 三处工程硬伤。

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

**① 模型量化压缩效果**

| 模型 | ONNX 原始 | INT8 RKNN | 压缩比 | 算子总数 | NPU 直通率 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **YOLOv8n（选定主干）** | 12.2 MB | **4.73 MB** | **61.2%↓** | 118 | **97.46%** (115/118) |
| YOLOv11n（对照组） | 10.4 MB | 4.54 MB | 56.3%↓ | 160 | 98.75% (158/160) |

两者纯计算算子均 **100% NPU 执行、零真实回退**（回退项仅为输入输出桩算子）。YOLOv11 的 C3k2+PSA 注意力已被 RKNPU 2.3.2 完整支持，直通率反优于 v8；但其算子数多 **35.6%**，端到端耗时高 **17.2%**，故实时伺服主干仍选定 YOLOv8n。

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

---

## 📂 项目模块结构

```text
EdgeAI-Robotic-Arm-Control/
├── configs/
│   └── config.yaml               # 工业工况、大模型与机械臂统一配置文件
├── data/
│   └── calibration/              # 量化校准集样本与指令集
├── docs/
│   ├── PROJECT_ROADMAP.md        # 研发全景蓝图与阶段推进路线图 (Master Roadmap)
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
│   └── vision/                   # V4L2 摄像头抓取与检测
├── tools/
│   ├── export_rkllm.py           # Qwen 大模型 W4A16 极限压缩转换脚本
│   ├── export_rknn.py            # 视觉模型 INT8 KL 散度量化转换脚本
│   ├── benchmark_quant.py        # 板端端到端性能 Benchmark 评测工具
│   ├── benchmark_npu.py          # NPU 单路多核调度模式时延基准测试
│   └── benchmark_npu_parallel.py # NPU 三核真并发吞吐上限测试
├── main.py                       # 工业控制台入口主程序
└── requirements.txt
```

---

## ⚡ 快速上手

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

### 4. 视觉模型 INT8 量化导出
```bash
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx models/weights/yolov8n_op19.onnx \
    --output models/weights/yolov8n_int8.rknn --algorithm normal
```

---

## 📄 项目声明与开源协议

* **开发者**: [kimkingyu](https://github.com/kimkingyu)
* **研究方向**: 边缘计算架构、端侧 AI 模型量化加速与机器人运动控制
* **许可证**: 本项目采用 [MIT License](LICENSE) 协议开源。
