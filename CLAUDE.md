# EdgeAI-Robotic-Arm-Control 开发守则与项目核心记忆 (Memory)

本文件兼容 **Claude Code (CLAUDE.md)**、**Cursor (.cursorrules)** 与 **Codex / WorkBuddy** 规范。任何 AI 智能体或开发者介入本项目时，必须严格遵守以下约定。

---

## 🚨 核心铁律：每步必记 (Dev Tracking Discipline)

**严禁做完改动拍拍屁股走人！**
每次完成代码改动、环境配置、编译构建或硬件调试后，必须调用开发追踪器记录：
1. **怎么做的 (How)**：改了哪个文件的哪一行，敲了什么具体命令，修改了什么参数。
2. **是为了什么 (Why)**：解决什么痛点，为什么选这个方案，背后的工程/物理考量。
3. **实测证据 (Evidence)**：终端输出的成功日志、FPS/时延、报错与修复记录。
4. **简历亮点 (Value)**：提炼成可直接用于简历与面试的 STAR 技术点。

> 执行方式：`python C:/Users/admin/.agents/skills/edgeai-arm-dev-tracker/scripts/record_step.py --title "..." --module "..." --how "..." --why "..." --evidence "..." --value "..."`  
> 日志持久化路径：`docs/dev_logs/DEV_PROGRESS_JOURNAL.md`

---

## 🛠️ 项目环境与硬件底座

* **主控平台**：香橙派 Orange Pi 5 Pro (Rockchip RK3588S, 16GB LPDDR5, 6 TOPS NPU)
* **操作系统**：Ubuntu 22.04 LTS (Jammy, Kernel 5.10.160-rockchip-rk3588, aarch64)
* **板端工程路径**：`/home/orangepi/project/EdgeAI-Robotic-Arm-Control`
* **板端远程 SSH**：`ssh orangepi@192.168.0.102` (端口 22, 已配通 ed25519 免密)
* **核心编程语言**：现代 C++ (C++17, GCC 11.4) + Python 3.10
* **视觉与加速引擎**：OpenCV 4.5.4 + 瑞芯微 RKNPU2 (librknnrt.so 0.9.6) + RKLLM (Qwen2.5 W4A16)
* **机械臂构型**：3 自由度 (3-DOF) 连杆 + 1 自由度夹爪，通过 I2C (PCA9685, 总线 7, 地址 0x40) 驱动

---

## ⚡ 常用高频命令

### 1. 板端 C++ 编译与执行
```bash
# 进入板端工程目录
cd /home/orangepi/project/EdgeAI-Robotic-Arm-Control

# 一键编译 Release 版本
bash scripts/build.sh

# 运行核心流水线 (三级多线程并发)
./bin/edge_arm_control
```

### 2. Python 任务规划与闭环模拟
```bash
# 脱机/全真模拟运行
python3 main.py --mock

# 性能压测 Benchmark
python3 tools/benchmark_quant.py
```

### 3. 外设与总线探测
```bash
# 摄像头视频节点检查
ls -l /dev/video*

# I2C 舵机板地址扫描 (PCA9685 默认 0x40)
sudo i2cdetect -y 7

# NPU 驱动状态
dmesg | grep -i rknpu
```

---

## 📐 代码与架构分层规范

1. `include/safe_queue.hpp`：基于条件变量的线程安全阻塞队列，连接各并发线程，禁止裸锁通信。
2. `include/camera_v4l2.hpp`：V4L2 采集驱动，支持 DMA-BUF 零拷贝，未接相机时必须优雅回退合成帧，严禁崩溃。
3. `include/kinematics.hpp` & `src/kinematics.cpp`：3-DOF 几何逆运动学求解，必须包含工作空间工作半径物理硬边界检查 (`r < min_reach || r > max_reach`)。
4. `src/controller/i2c_arm.py`：PCA9685 12-bit PWM 映射计算，未装 `smbus2` 或未挂载硬件时自动进入 Mock 模式。
