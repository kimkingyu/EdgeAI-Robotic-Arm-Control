---
name: edgeai-arm-dev-tracker
description: 面向 EdgeAI-Robotic-Arm-Control 机械臂项目的全流程研发执行追踪与决策记录插件。每当在此项目中执行任何软硬件配置、环境搭建、NPU模型量化、C++编译、外设联调或代码修改时触发。强制要求每做一步，必须详细记录【做了什么】、【怎么做的（具体命令、代码文件修改、参数）】、【为什么做（研发意图、解决的痛点、架构权衡）】、【实测证据（终端日志、Benchmark数据、报错解决）】与【简历亮点映射】。
---

# EdgeAI Arm Dev Tracker (研发追踪与决策记录插件)

## 概述
专为 `EdgeAI-Robotic-Arm-Control`（基于 RK3588 边缘 NPU 的 3 自由度/多轴视觉机械臂控制项目）定制。核心使命是将日常零碎的开发操作，实时沉淀为严谨、结构化、可复现的工业级技术台账，确保每一个动作都有迹可循、有理有据，为毕业课题和求职简历储备无可辩驳的实测证据。

---

## 核心规范：每步必记五大要素

在执行任何开发步骤后，**必须**同步记录以下 5 维信息并追加到项目的 `docs/dev_logs/DEV_PROGRESS_JOURNAL.md`：

1. **步骤标题 (Title)**：清晰定义本步完成的核心动作（如：*修复 kinematics.cpp 链接符号缺失并完成板端构建*）。
2. **怎么做的 (How)**：记录具体执行的终端命令、修改的代码路径、调整的关键宏定义或参数。
3. **是为了什么 (Why)**：阐明背后的工程考量、解决的硬件/软件痛点、为什么不采用替代方案。
4. **实测证据 (Evidence)**：粘贴关键终端输出、启动日志、时延/帧率基准数据或报错排查记录。
5. **简历与课题价值 (Value)**：将本步成果映射到简历“专业技能栏”或面试高频问题中。

---

## 自动化日志记录脚本调用

本技能仅安装在本项目的 `.narrafork/skills/edgeai-arm-dev-tracker/` 中，不安装到用户或 NarraFork 全局技能目录。内置脚本 `scripts/record_step.py`，从项目根目录调用：

```bash
python ".narrafork/skills/edgeai-arm-dev-tracker/scripts/record_step.py" \
  --step "X" \
  --title "任务名称" \
  --module "BSP/NPU/Vision/Kinematics/I2C/Build" \
  --how "具体执行了哪些命令、改动了哪个文件哪行代码" \
  --why "解决什么痛点，为什么必须这样做" \
  --evidence "终端输出的成功日志或Benchmark数据" \
  --value "对应简历中的哪一项技术亮点"
```

### 存储路径约定
- **人类可读日志**：`[工程根目录]/docs/dev_logs/DEV_PROGRESS_JOURNAL.md`
- **机器索引数据**：`[工程根目录]/docs/dev_logs/steps_history.json`

---

## 阶段追踪范例（参考）

- **阶段 1：底层环境与开发底座**
  - 记录 Ubuntu 22.04 镜像选型原因（避免 24.04 Python 3.12 与 rknn prebuilt 轮子断链）；
  - 记录国内镜像源与外设免 sudo 组权限（dialout, video, i2c）配置。
- **阶段 2：端侧模型量化与推理加速**
  - 记录 ONNX 导出、INT8 非对称量化（KL 散度校准）与 W4A16 压缩收益；
  - 记录单核 vs 3核 NPU（`NPU_CORE_0_1_2`）并发调度的帧率对比。
- **阶段 3：C++ 异步并发与零拷贝**
  - 记录三级解耦流水线（V4L2采集 ➔ NPU推理 ➔ 逆解）与 `SafeQueue` 解决的线程阻塞问题；
  - 记录 DMA-BUF 消除内存拷贝将预处理时间压至 0.5ms。
- **阶段 4：手眼标定与运动学闭环**
  - 记录 Eye-to-Hand 齐次矩阵推导与 PnP 空间解算精度；
  - 记录 3-DOF 缩比样机上的几何逆解（IK）极限边界防撞保护。
