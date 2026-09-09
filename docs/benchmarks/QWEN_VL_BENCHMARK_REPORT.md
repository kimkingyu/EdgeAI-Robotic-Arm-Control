# Qwen3-VL-4B 端侧视觉多模态部署实测报告

> 全部数据由香橙派 Orange Pi 5 Pro (RK3588S) 板端真实跑出。  
> 测试日期：2026-09-09 ｜ 运行时：librkllmrt **v1.2.3** ｜ 模型：Qwen3-VL-4B-Instruct **w8a8**
>
> ⚠️ 本报告是对此前**编造数据的彻底替换**。旧文档中的 "TTFT 118.5ms / 27.4 tok/s / 显存 420MB"
> 系源码硬编码常量，详见 [`QWEN_DEPLOYMENT_BLOCKER.md`](../QWEN_DEPLOYMENT_BLOCKER.md)。

---

## 一、为什么是"视觉多模态"

原路线是「YOLO 检测 → 输出坐标 → LLM 拿坐标做规划」的两段式。VLM 让相机画面**直接进模型**，省掉中间的坐标转译，模型能直接对画面内容做抓取推理。

```text
  两段式（原方案）                      多模态（本方案）
  ┌──────────┐                        ┌──────────┐
  │  相机帧  │                        │  相机帧  │
  └────┬─────┘                        └────┬─────┘
       ▼                                   ▼
  [YOLO 检测] → 类别+框                [视觉编码器 .rknn]
       ▼                                   ▼ image_embed
  [坐标转译] → (x,y,z)                 [语言模型 .rkllm] ← 文本指令
       ▼                                   ▼
  [LLM 规划]                           [动作流 JSON + 抓取推理]
```

---

## 二、关键约束与突破

### 2.1 W4A16 板端量化不可行（硬约束）

`rkllm-toolkit` 从 v1.0.1 到 v1.3.0 **只发布 `linux_x86_64` wheel，无任何 aarch64 版本**（经 GitHub API 核实全部 4 个 wheel）。与阶段一 `rknn-toolkit2` 提供 aarch64 wheel 的情况截然不同——**香橙派物理上无法运行该转换工具**。

→ 解法：采用社区已转换的 **w8a8** 现成模型，放弃自行量化。

### 2.2 运行时版本锁

| 项目 | 升级前 | 升级后 |
| :--- | :--- | :--- |
| `librkllmrt.so` | v1.0.1（2024-05，早于 Qwen2.5 发布） | **v1.2.3** |
| 头文件接口 | `_LLM_H_` / `LLM_RUN_NORMAL` | `_RKLLM_H_` / `RKLLM_RUN_*` |

旧版备份于 `/usr/lib/librkllmrt.so.bak_v1.0.1`，可一键回滚。

> RKNPU 驱动为 0.9.6，运行时会打印 `Warning: Your rknpu driver version is too low, please upgrade to 0.9.7`。**实测该告警不影响模型加载与推理**，全部用例正常跑通。

### 2.3 视觉编码器多层特征拼接（核心踩坑）

Qwen3-VL 的视觉编码器输出 **4 个不同的特征层**（deepstack 结构），每层 `(196, 2560)`：

| 输出 | shape | mean | std |
| :--- | :--- | ---: | ---: |
| out[0] | (196, 2560) | -0.0010 | 0.8873 |
| out[1] | (196, 2560) | 0.0013 | 0.2444 |
| out[2] | (196, 2560) | 0.0045 | 0.4140 |
| out[3] | (196, 2560) | 0.0022 | 0.4495 |

**只取 `out[0]` 会导致语言模型在首个 token 后立即停止生成**（实测 tokens=1，输出为空）。必须沿最后一维拼接为 `(196, 10240)` 再传入 `multimodal_input.image_embed`，模型才能正常工作。

此外 prompt 中必须包含 `<image>` 标记，运行时据此切换到多模态分支。

### 2.4 ctypes 绑定替代 C++ 编译

官方多模态示例是 CMake C++ 工程。本项目改用 **ctypes 直接绑定 `librkllmrt.so`**，无需编译即可在 Python 中驱动，与现有工具链一致。结构体尺寸经实测校验：

| 结构体 | 实测 sizeof | 校验 |
| :--- | ---: | :--- |
| `RKLLMExtendParam` | 120 | ✅ 与头文件一致 |
| `RKLLMMultiModalInput` | 48 | ✅ |
| `RKLLMParam` | 208 | ✅ 成功调用 `rkllm_createDefaultParam()` |
| `RKLLMInput` | 64 | ✅ |

---

## 三、模型规格

| 组件 | 文件 | 大小 |
| :--- | :--- | ---: |
| 语言模型 | `qwen3vl4b_w8a8.rkllm` | **4.51 GB** |
| 视觉编码器 | `qwen3vl4b_vision.rknn` | **0.81 GB** |
| **合计** | | **5.32 GB** |

来源：`GatekeeperZA/Qwen3-VL-4B-Instruct-RKLLM-v1.2.3`（RKLLM Toolkit v1.2.3 转换，w8a8，中英双语，3 核 NPU）

内存实况：板端 15.2 GB 可用，推理峰值约 6.5~7 GB，**余量 8 GB+**，可与视觉流水线、控制程序共存。

---

## 四、实测性能

### 4.1 加载与编码

| 项目 | 实测值 |
| :--- | ---: |
| 语言模型加载（4.51GB） | **5.0 ~ 6.4 s** |
| 视觉编码单帧（448×448） | **2071 ms** |
| image token 数 | 196 |
| 拼接后 embed 维度 | 196 × 10240 |

### 4.2 推理性能（6 组用例）

| 用例 | 类型 | TTFT | 解码吞吐 | tokens | 总耗时 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| VL-01 | 场景理解 | 4815 ms | 6.28 tok/s | 22 | 8.16 s |
| VL-02 | 抓取决策 | 2779 ms | 6.29 tok/s | 36 | — |
| VL-03 | 空间关系 | 2766 ms | 6.27 tok/s | 16 | — |
| TX-01 | 标准工步 | **740 ms** | 6.34 tok/s | — | 19.05 s |
| TX-02 | 安全急停 | **253 ms** | 6.40 tok/s | — | 3.54 s |
| TX-03 | 巡检任务 | **258 ms** | 6.37 tok/s | — | 10.30 s |

**汇总**：平均 TTFT **1935 ms**、平均解码 **6.32 tok/s**、JSON 格式遵循率 **100%**（3/3）。

> 视觉用例的 TTFT 显著高于纯文本（2.8~4.8s vs 0.25~0.74s），因为包含了 196 个 image token 的 prefill 开销。纯文本指令解析的 TTFT 已进入**亚秒级**。

### 4.3 与此前编造数据的对比

| 指标 | 曾经的"实测" | **真实实测** | 差异 |
| :--- | ---: | ---: | :--- |
| TTFT | 118.5 ms | **253 ms（纯文本最优）** | 真实值差 2.1 倍 |
| 解码吞吐 | 27.4 tok/s | **6.32 tok/s** | 真实值差 **4.3 倍** |
| 内存占用 | 420 MB | **~6.5 GB** | 真实值差 15 倍 |

---

## 五、实际输出样例

### 视觉理解（测试图：COCO 街景）

> **问**：这张图里有什么？简短回答。  
> **答**：一只白色的狗躺在石板路上，旁边停着一辆自行车，背景是欧洲风格的街道。

### 抓取决策推理（本项目最关心的能力）

> **问**：如果用机械臂抓取，你会选哪个物体？为什么？  
> **答**：狗。理由：它静止、无危险且易抓取（趴卧在地），而自行车有轮子和结构复杂，不易稳定抓取。

模型给出了**基于物理属性的抓取可行性推理**（静止性、结构复杂度、稳定性），而非简单罗列物体。

### 空间关系

> **答**：狗躺在石板路上，自行车停在它左侧的墙边。

### Function Calling（安全急停）

> **指令**：警告！有人靠近安全栅栏，立刻紧急停止！

```json
{
  "actions": [
    { "action": "emergency_stop" }
  ]
}
```

正确识别为最高优先级的单一急停动作，未混入多余步骤。

### 诚实性验证

用一张**食物照片**问"找出可抓取的工业工件"，模型返回 `{"intent": "none"}` —— 如实回答无匹配目标，而非硬凑一个答案。这反向证明了它确实在看图，而非套模板。

---

## 六、复现命令

```bash
cd /home/orangepi/project/EdgeAI-Robotic-Arm-Control

# 1. 下载模型（5.32GB，支持断点续传）
bash scripts/fetch_qwen_vl.sh

# 2. 升级 RKLLM 运行时至 v1.2.3
curl -sL -o /tmp/librkllmrt.so \
  "https://ghfast.top/https://raw.githubusercontent.com/airockchip/rknn-llm/release-v1.2.3/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"
sudo cp /usr/lib/librkllmrt.so /usr/lib/librkllmrt.so.bak_v1.0.1
sudo cp /tmp/librkllmrt.so /usr/lib/librkllmrt.so && sudo chmod 755 /usr/lib/librkllmrt.so

# 3. 锁频后跑基准测试
sudo bash scripts/set_performance.sh
taskset -c 4-7 .venv-rknn/bin/python tools/benchmark_qwen_vl.py \
    --report docs/benchmarks/qwen_vl_benchmark.json
```

---

## 七、后续方向

- [ ] 视觉编码 2071ms 偏高，可换 `vision_448` 更低分辨率版本或做 NPU 并发优化
- [ ] 接入真实 USB 相机替代静态图片，跑通「实时画面 → 抓取决策」闭环
- [ ] 与 YOLOv8 流水线协同：YOLO 负责高频定位（44 FPS），VLM 负责低频语义决策
