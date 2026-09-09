# 阶段二 Qwen2.5 端侧部署：约束调研与路线决策

> 记录时间：2026-09-09  
> 结论先行：**W4A16 量化无法在香橙派上完成**，这是官方工具链的架构硬约束，不是配置问题。  
> 本文档同时记录了对既有"实测数据"的证伪过程。

---

## 一、先说一个必须纠正的问题

此前 `PROJECT_ROADMAP.md` 与 `README.md` 中记载的阶段二"实测"数据：

| 曾经的说法 | 实际情况 |
| :--- | :--- |
| "JSON Schema 完全遵循率 100%" | 测的是**规则引擎关键字匹配**，不是模型输出 |
| "平均意图解析耗时 120.9ms" | 是 mock 分支里 `time.sleep(0.12)` 的产物 |
| "TTFT 118.5ms / 27.4 tok/s / 显存 420MB" | 全部为源码中的**硬编码常量** |

### 根因链条

```
板端未安装 rkllm            (pypi 无此包)
   └─> RKLLMInferenceEngine.HAS_RKLLM = False
       └─> load_model() 静默进入 Mock 模式，仍返回 True、is_ready = True
           └─> generate() 返回预置 JSON 文本，并写死 ttft_ms=120.0 / tps=26.5
               └─> IndustrialTaskPlanner.plan() 拿到合法 JSON，正常解析
                   └─> 评测脚本得出 "100% 通过率、120.9ms" 的漂亮结论
```

整条链路没有任何一处报错，所以问题一直没被发现。

### 已做的修正

- `rkllm_engine.py`：Mock 模式下 `last_perf` 全部置 `None` 并标记 `is_mock=True`，不再伪造性能数字；`load_model()` 明确打印警告
- `test_qwen_function_calling.py`：Mock 模式下强制打印数据无效声明，并解释为何通过率必然接近 100%
- `benchmark_quant.py` / `compare_cloud_edge.py` 中的硬编码 LLM 数据待同批清理

> 同类问题在阶段一已处理过一次（README 的 "8.9ms / 68.5 FPS / mAP 0.819" 与 `compare_yolo_versions.py` 的硬编码），处理方式一致：**要么拿到真实数据，要么如实标注缺失**。

---

## 二、核心硬约束：rkllm-toolkit 无 aarch64 版本

阶段一能在板端自转自跑（`rknn-toolkit2` 提供 aarch64 wheel），很自然会期待阶段二照搬。**但这条路走不通。**

### 实测证据

```bash
# 1. pypi 上根本没有这个包
pip3 download rkllm-toolkit
# -> ERROR: No matching distribution found for rkllm-toolkit

# 2. 官方仓库里全部 wheel 的架构
curl -sL "https://api.github.com/repos/airockchip/rknn-llm/contents/rkllm-toolkit/packages"
```

返回的完整列表：

| 文件名 | 架构 |
| :--- | :--- |
| `rkllm_toolkit-1.3.0-cp39-cp39-linux_x86_64.whl` | **x86_64** |
| `rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl` | **x86_64** |
| `rkllm_toolkit-1.3.0-cp311-cp311-linux_x86_64.whl` | **x86_64** |
| `rkllm_toolkit-1.3.0-cp312-cp312-linux_x86_64.whl` | **x86_64** |

从 v1.0.1 到最新 v1.3.0，**官方从未发布过 aarch64 版本**。该工具的量化后端是闭源预编译共享库，无法在 ARM64 上安装或调用。

**→ 香橙派（aarch64）物理上不可能运行 rkllm-toolkit，W4A16 量化必须依赖 x86_64 Linux 主机。**

### 与阶段一的对比

| | 视觉模型（阶段一） | 大模型（阶段二） |
| :--- | :--- | :--- |
| 转换工具 | `rknn-toolkit2` | `rkllm-toolkit` |
| aarch64 wheel | ✅ 有（2.3.2） | ❌ **无，仅 x86_64** |
| 能否板端自转 | ✅ 可以 | ❌ **不可以** |

---

## 三、次生约束：运行时版本锁

即便拿到现成的 `.rkllm` 模型，板端当前运行时也加载不了。

### 板端现状（实测）

```bash
strings /usr/lib/librkllmrt.so | grep -i version   # -> 1.0.1
grep -E '_LLM_H_|LLM_RUN_NORMAL' /usr/include/rkllm.h
# -> #ifndef _LLM_H_ / LLM_RUN_NORMAL = 0   （v1.0.1 的旧接口）
sudo cat /sys/kernel/debug/rknpu/version     # -> RKNPU driver: v0.9.6
```

- 运行时 `librkllmrt.so` 为 **v1.0.1**，文件日期 2024-08，发布于 2024 年 5 月
- 头文件仍是旧接口（v1.1.0 起官方已改为 `_RKLLM_H_` / `RKLLM_RUN_*`）
- **v1.0.1 早于 Qwen2.5 发布**，社区现成模型均由 toolkit v1.2.x 转换

### 版本配套关系

| RKLLM 运行时 | 要求 RKNPU 驱动 | 板端现状 |
| :--- | :--- | :--- |
| v1.0.1 | 0.9.6 | ✅ 当前匹配 |
| v1.1.4 | 0.9.7 | ⚠️ 需升驱动 |
| v1.2.3 | 0.9.8 | ⚠️ 需升驱动 |

用 v1.0.1 运行时强行加载新模型，会被直接拒绝（`The model version is too old, please use the latest toolkit to reconvert the model!`）。

> 注：升级 `librkllmrt.so`（用户态 .so，可回滚）风险可控，做法与阶段一升级 `librknnrt.so` 一致；但**驱动版本不匹配会打印告警**，能否正常工作需实测。内核驱动升级属于更高风险操作。

---

## 四、可行路径

### 路径 A：部署社区现成模型（推荐，纯板端）

放弃自行量化，直接用别人转好的模型。已验证可下载（板端访问 hf-mirror 返回 302，通路正常）：

| 模型 | 量化 | 大小 | 地址 |
| :--- | :--- | ---: | :--- |
| Qwen2.5-1.5B-Instruct | W8A8 | 2.05 GB | `hf-mirror.com/GatekeeperZA/Qwen2.5-1.5B-Instruct-RKLLM-v1.2.3` |
| Qwen2.5-1.5B-Instruct | W8A8 | 2.05 GB | `hf-mirror.com/HanzoHuang/Qwen2.5-1.5B-Instruct-RKLLM` |
| Qwen2.5-0.5B-Instruct | — | 753 MB | `hf-mirror.com/ThomasTheMaker/Qwen2.5-0.5B-Instruct-RKLLM-1.2.0` |

**代价**：社区模型基本都是 **W8A8**，不是路线图写的 W4A16。需同步升级运行时到 v1.2.3。  
**收益**：能拿到**真实的** TTFT / TPS / Function Calling 准确率，替换掉所有编造数据。

### 路径 B：x86_64 Linux 主机自行量化

在 x86_64 Linux 环境安装 `rkllm_toolkit-1.3.0-cp310-cp310-linux_x86_64.whl`，自行完成 W4A16 量化后 scp 到板端。

**前提**：需要一台可用的 x86_64 Linux 主机（物理机、云服务器或本机双系统）。  
**注**：本项目明确排除 WSL 方案。

### 路径 C：暂缓阶段二，先推阶段四

保留现有架构代码（planner / hybrid_router / cloud_api_client 的设计本身是完整的），等硬件到货先做物理闭环。

---

## 五、当前状态小结

| 项目 | 状态 |
| :--- | :--- |
| Prompt 工程与动作状态机设计 | ✅ 已完成（代码质量与真实性无关） |
| 端云路由架构 | ✅ 已完成 |
| **W4A16 板端量化** | ❌ **受阻**：工具链无 aarch64 版本 |
| **Function Calling 真实压测** | ❌ **未完成**：此前数据来自 Mock |
| 运行时升级 | ⏸ 待决策（依赖选择哪条路径） |
