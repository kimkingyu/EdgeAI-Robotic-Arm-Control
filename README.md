# EdgeAI-Robotic-Arm-Control

基于 RK3588 (6 TOPS NPU) 的高性能实时端侧视觉伺服与机械臂闭环控制系统。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588%20%7C%20Linux%20Ubuntu%2022.04-orange.svg)](https://github.com/kimkingyu/EdgeAI-Robotic-Arm-Control)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-brightgreen.svg)](https://github.com/kimkingyu/EdgeAI-Robotic-Arm-Control)
[![Author](https://img.shields.io/badge/Author-kimkingyu-lightgrey.svg)](https://github.com/kimkingyu)

---

## 📌 项目概述

**EdgeAI-Robotic-Arm-Control** 是一套专为 **瑞芯微 RK3588** 边缘计算芯片设计的高性能、全流程端侧视觉伺服与机器人闭环控制系统。

项目结合了 NPU 硬件加速推理、V4L2 零拷贝图像采集机制，以及三级解耦的异步多线程流水线，在 RK3588 边缘端实现了 **50+ FPS** 的高频目标检测、空间位姿映射与机械臂逆运动学（IK）闭环控制，定位误差优于 3mm。

```text
[USB/MIPI 摄像头 (1080P)] 
            │ (V4L2 零拷贝采集)
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        RK3588 C++ 核心控制流水线                        │
│                                                                        │
│  [图像预处理] ──► [线程安全阻塞队列 SafeQueue]                          │
│                             │                                          │
│                             ▼                                          │
│                    [RKNN NPU 硬件推理] (YOLOv8 INT8 量化, <11ms)       │
│                             │                                          │
│                             ▼                                          │
│                    [PnP 空间位姿解算] (手眼标定 Eye-to-Hand 映射)       │
│                             │                                          │
│                             ▼                                          │
│                    [机械臂逆运动学求解 (IK)] ──► [串口/ROS2 驱动总线]   │
└────────────────────────────────────────────────────────────────────────┘
            │
            ▼
[实体机械臂 / Gazebo 仿真环境] (实时闭环视觉伺服对准与抓取)
```

---

## 🚀 核心特性

- **NPU 硬件加速与 INT8 量化**：基于 `RKNN-Toolkit2` 对目标检测模型实施非对称训练后量化（PTQ）与 KL 散度校准，在 RK3588 NPU 上实现单帧 **<11ms** 的低时延推理，模型体积缩减 70%+。
- **异步多线程流水线**：设计“图像采集 - NPU 推理 - 空间映射与逆解”三级解耦架构，结合线程安全队列彻底消除 I/O 阻塞等待。
- **手眼标定与 PnP 空间解算**：基于 Eye-to-Hand 架构与 PnP 算法，实现目标从 2D 像素坐标 $(u, v)$ 向机械臂基座 3D 坐标 $(X_{base}, Y_{base}, Z_{base})$ 的毫米级空间映射。
- **轻量化运动学闭环**：纯 C++ 实现几何/解析法逆运动学（IK）求解，毫秒级计算关节伺服角度并经由串口 UART / ROS2 发布执行。
- **硬件在环（HIL）与数字孪生**：原生适配 ROS2 (Humble) 与 Gazebo 物理仿真环境。

---

## 📊 性能基准测试 (Benchmark)

测试平台：**香橙派 Orange Pi 5 (RK3588S, 16GB 内存)** / Ubuntu 22.04 LTS

| 流水线阶段 | 单线程耗时 | 多线程流水线优化后 | 资源占用 / 负载 |
| :--- | :--- | :--- | :--- |
| **图像采集 (1080P)** | 12.5 ms | **0.5 ms** (DMA-BUF 零拷贝) | CPU < 5% |
| **NPU 推理 (YOLOv8 INT8)** | 11.2 ms | **10.5 ms** | NPU 负载 ~60% |
| **PnP 位姿与空间映射** | 0.4 ms | **0.3 ms** | CPU < 1% |
| **逆运动学求解 (IK)** | 0.2 ms | **0.2 ms** | CPU < 1% |
| **闭环端到端控制时延** | 28.5 ms | **< 15.0 ms** | 跟踪定位精度 < 3mm |
| **系统吞吐量 (FPS)** | ~22 FPS | **55+ FPS** | 芯片工作温度 < 52°C |

---

## 📂 项目结构

```text
EdgeAI-Robotic-Arm-Control/
├── CMakeLists.txt             # CMake 编译构建配置
├── README.md                  # 项目中文说明文档
├── LICENSE                    # MIT 开源协议
├── .gitignore
│
├── include/                   # 核心头文件
│   ├── safe_queue.hpp         # 生产级线程安全阻塞队列
│   ├── camera_v4l2.hpp        # V4L2 低时延摄像头采集驱动
│   ├── rknn_detector.hpp      # RK3588 NPU 推理引擎封装
│   ├── hand_eye_trans.hpp     # 手眼标定与 PnP 空间转换解算器
│   └── kinematics.hpp         # 机械臂正逆运动学解算器
│
├── src/                       # 核心实现源码
│   └── main.cpp               # 三级流水线调度主程序入口
│
├── model/                     # 模型转换与量化工具
│   └── convert_rknn.py        # ONNX 转换为 RKNN INT8 模型的 Python 脚本
│
└── scripts/                   # 一键自动化脚本
    ├── build.sh               # 一键编译脚本
    └── run.sh                 # 一键运行脚本
```

---

## 🛠️ 环境依赖与技术栈

- **运行平台**：RK3588 / RK3588S 开发板（Ubuntu 20.04 / 22.04 aarch64）或 x86 Linux
- **编译工具**：GCC/G++ (>= 9.4.0, C++17 标准), CMake (>= 3.16)
- **AI 运行时**：瑞芯微 `librknnrt.so` (RKNPU2 v1.6.0+), `rknn-toolkit2`
- **基础库**：OpenCV 4.x, POSIX 线程库 (`pthread`)
- **机器人接口**：ROS2 (Humble) / 串口通信协议 (UART)

---

## ⚡ 快速开始

### 1. 模型量化与转换（PC 端运行）
```bash
# 安装 rknn-toolkit2 环境，将 ONNX 导出为 RKNN INT8 模型
cd model
python3 convert_rknn.py --onnx yolov8n.onnx --output yolov8n_int8.rknn --target rk3588
```

### 2. 板端编译
```bash
git clone https://github.com/kimkingyu/EdgeAI-Robotic-Arm-Control.git
cd EdgeAI-Robotic-Arm-Control
chmod +x scripts/*.sh
./scripts/build.sh
```

### 3. 运行控制流水线
```bash
./scripts/run.sh
```

---

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源协议。

---
**开发者**: [kimkingyu](https://github.com/kimkingyu)  
**学校/单位**: 浙江大学机械工程学院 (College of Mechanical Engineering, Zhejiang University)
