# EdgeAI-Robotic-Arm-Control

**面向工业场景的端侧 AI 系统与模型推理加速研究**  
*Industry-Oriented Edge-AI System and On-Device LLM Inference Acceleration on RK3588 (6 TOPS NPU)*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%7C%20Ubuntu%2022.04%20aarch64-orange.svg)](https://github.com/kimkingyu/EdgeAI-Robotic-Arm-Control)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5%20(W4A16)-purple.svg)](https://github.com/QwenLM/Qwen2.5)
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
│            │ (RKLLM W4A16 混合量化加速, 27+ Tokens/s, TTFT < 120ms)     │
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

---

## 🚀 核心关键技术与攻坚成果

### 1. 工业大模型指令微调与端侧任务规划 (Task Planning)
* 针对工业阀芯、法兰、轴承等典型工件规格与流水线装配逻辑，构建工业领域指令微调数据集；
* 基于 Qwen2.5 设计专用工业 Prompt 格式与 Function Calling 解析器，将非结构化自然语言指令秒级转化为严格可执行的底层动作状态机。

### 2. INT4/INT8 极限压缩与非对称量化校准 (PTQ & Calibration)
* **W4A16 混合量化**：针对嵌入式内存带宽瓶颈，采用 `rkllm-toolkit` 对 Qwen2.5 实现权重 4-bit 压缩、激活值 16-bit 浮点运算，将大模型显存占用从 3GB+ 压缩至 **420MB**（**体积缩减 72%**），困惑度损失 < 1.2%；
* **视觉骨干 INT8 量化**：构建 100 张真实场景校准集，采用逐通道非对称 INT8 训练后量化（PTQ），YOLOv8n 体积由 12.2MB 压至 **4.73MB（缩减 61.2%）**，权重内存仅 3.6MB；攻克 `opset 超限`、`校准集路径二次拼接`、`动态形状无法量化` 三处工程硬伤。

### 3. RK3588 3核 NPU 异构并发调度与系统级调优
* **证伪常见误解**：实测发现单实例即便绑定 `NPU_CORE_0_1_2` 也只有 1.03x 加速 —— 因 `RKNNLite.inference()` 是同步阻塞调用，`core_mask` 仅声明"允许用哪些核"，并不会把单个任务拆分并行；
* **正确解法**：为每个物理核心创建独立 RKNNLite 实例，配合任务队列多线程并发喂帧，实测达成 **2.71x 近线性加速（38.5 → 104.5 FPS）**，封装为 `RKNNParallelVisionPool` 后实跑 **112.99 FPS**；
* **瓶颈定位**：NPU 全程满频 1GHz、温度仅 46℃，真正卡点在 CPU 侧的输入拷贝与输出反量化被 `ondemand` 压在 1.2GHz。交付 `scripts/set_performance.sh` 全线锁频后，单帧时延降 **21.5%**，P99 抖动降 **89%**。

### 4. 空间手眼标定与物理控制闭环
* 建立 Eye-to-Hand 相机坐标系到机械臂基座坐标系的高精度齐次变换矩阵，定位误差优于 **3mm**；
* 基于 Linux I2C 总线直接驱动 PCA9685 产生微秒级高精度 PWM 脉冲，控制多自由度机械臂完成毫米级平滑下探与抓取。

---

## 📊 性能评测基准 (Benchmark)

*测试硬件：香橙派 OrangePi 5 Pro (RK3588S, 16GB LPDDR5) / Ubuntu 22.04 LTS (Kernel 5.10.160)*

### 1. 端侧大模型 (Qwen2.5) RKLLM 量化加速对比
| 指标项 | 原始未量化 (FP16) | **W4A16 混合量化 (本项目)** | 优化收益 |
| :--- | :---: | :---: | :---: |
| **模型体积 / 显存驻留** | 1,480 MB | **420 MB** | **显存暴降 72%** |
| **首字响应时延 (TTFT)** | 385 ms | **118.5 ms** | **时延降低 69%** |
| **生成吞吐 (Tokens/s)** | 8.2 tok/s | **27.4 tok/s** | **生成速度翻 3.3 倍** |
| **NPU 算力平均利用率** | 28% | **82% (多核负载均衡)** | 算力充分榨干 |

### 2. 视觉感知引擎板端实测评测

> 以下数据全部由香橙派板端真实跑出（rknn-toolkit2 2.3.2 / librknnrt 2.3.2，300 轮正式测试 + 20 轮预热），  
> 完整测试过程、踩坑记录与复现命令见 **[NPU 推理加速实测报告](docs/benchmarks/NPU_BENCHMARK_REPORT.md)**。

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
