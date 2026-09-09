# RK3588 NPU 量化与推理加速 —— 完整复现指南

> 从一块刚烧好 Ubuntu 的香橙派开始，一路复现到 **112.99 FPS** 三核并发推理。  
> 全程在板端完成，**不需要 x86 PC**、不需要交叉编译。预计耗时 40~60 分钟（大头是下载与量化）。
>
> 实测数据见 [`benchmarks/NPU_BENCHMARK_REPORT.md`](benchmarks/NPU_BENCHMARK_REPORT.md)。

---

## 0. 前置检查

| 要求 | 说明 |
| :--- | :--- |
| 硬件 | Orange Pi 5 Pro / 5 Plus 等 RK3588(S) 板卡，**内存 ≥ 8GB**（量化过程峰值约 4GB） |
| 系统 | Ubuntu 22.04 aarch64，Python **3.10**（`rknn-toolkit2` 的 wheel 按 cp310 编译） |
| 磁盘 | 空余 ≥ 3GB（venv 约 1GB + 模型与校准集） |
| 网络 | 能访问清华 pip 源；GitHub 若不通则用 `ghfast.top` 镜像（下文已代入） |

```bash
# 一次性确认三件事：架构、Python 版本、剩余磁盘
uname -m && python3 --version && df -h / | tail -1
```

期望输出 `aarch64` + `Python 3.10.x` + 足够空余空间。**Python 不是 3.10 就先别往下走**，`rknn-toolkit2` 官方只发了 cp310 的 aarch64 wheel。

> ⚠️ 本项目 `.gitignore` 屏蔽了 `*.onnx` / `*.rknn` / `weights/` / `.venv/`，所以 clone 下来**不含模型权重和虚拟环境**，需按下文从零构建。这是刻意为之——避免仓库被几百 MB 二进制撑爆。

---

## 1. 安装板端原生工具链

原本以为量化必须在 x86 PC 上做，实测 `rknn-toolkit2` 自 2.x 起已提供 **aarch64 原生 wheel**，所以香橙派可以自转自跑。

```bash
cd /home/orangepi/project/EdgeAI-Robotic-Arm-Control

# venv 模块（Ubuntu 精简镜像常缺，缺了会报 "No module named pip"）
sudo apt-get install -y python3.10-venv python3-dev

# 建独立虚拟环境，避免污染系统 Python
python3 -m venv .venv-rknn
.venv-rknn/bin/python -m pip install -U pip

# 转换工具（含 torch/onnx，约 1GB，清华源约 5~10 分钟）
.venv-rknn/bin/python -m pip install rknn-toolkit2==2.3.2

# 推理运行时（板端跑 .rknn 用这个，比完整版轻量得多）
.venv-rknn/bin/python -m pip install rknn-toolkit-lite2==2.3.0
```

> 💡 `onnxoptimizer` 没有 aarch64 预编译包，会现场编译约 3~5 分钟，卡住不动是正常的，别中断。

**验证：**
```bash
.venv-rknn/bin/python -c "from rknn.api import RKNN; RKNN().release(); print('工具链就绪')"
```

---

## 2. 升级 NPU 运行时（关键，别跳过）

官方镜像自带的 `librknnrt.so` 往往是 **0.9.6（2023 年）** 的老版本，加载 toolkit 2.3.2 转出的模型会直接报版本不匹配。

```bash
# 先看当前版本
strings /usr/lib/librknnrt.so | grep -m1 'librknnrt version'

# 下载 2.3.2 运行时
mkdir -p /tmp/rt && cd /tmp/rt
curl -sL -o librknnrt.so \
  "https://ghfast.top/https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# 备份旧版后替换（备份很重要，出问题能立刻回滚）
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak
sudo cp librknnrt.so /usr/lib/librknnrt.so && sudo chmod 755 /usr/lib/librknnrt.so

# 确认升级成功
strings /usr/lib/librknnrt.so | grep -m1 'librknnrt version'
```

期望看到 `librknnrt version: 2.3.2`。**回滚**：`sudo cp /usr/lib/librknnrt.so.bak /usr/lib/librknnrt.so`

> 📌 本项目板端替换前的原厂 so 备份为 `/usr/lib/librknnrt.so.bak_0.9.6`。注意 `strings` 打印的内部版本串（如 `1.4.0`）与 RKNPU **驱动**版本号（`0.9.6`）不是一回事，别被绕晕——判断是否需要升级，看它是不是明显早于 2.x 即可。

---

## 3. 准备模型与校准集

```bash
cd /home/orangepi/project/EdgeAI-Robotic-Arm-Control
mkdir -p models/weights data/calibration/images

# YOLOv8n / YOLOv11n 官方 ONNX
curl -L --retry 2 -o models/weights/yolov8n.onnx \
  "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx"
curl -L --retry 2 -o models/weights/yolo11n.onnx \
  "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx"

# 校验大小：v8n 约 12MB、v11n 约 10MB。若只有几 KB 说明下到了错误页面
ls -lh models/weights/*.onnx
```

**校准集**（INT8 量化必须用真实图片统计激活值分布，随机噪声会让精度崩掉）：

```bash
curl -L --retry 2 -o /tmp/coco128.zip \
  "https://ghfast.top/https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
unzip -o -q /tmp/coco128.zip -d /tmp/coco128
find /tmp/coco128 -name '*.jpg' | head -100 | xargs -I{} cp {} data/calibration/images/

# 生成清单，必须用绝对路径（原因见第 6 节踩坑记录）
ls $PWD/data/calibration/images/*.jpg | head -100 > data/calibration/dataset.txt
wc -l data/calibration/dataset.txt   # 期望 100
```

---

## 4. ONNX 预处理（绕开两个硬伤）

### 4.1 opset 降级

ultralytics 新版导出的是 opset 20/22，而 RKNN 只吃 **≤ 19**，不处理会直接报 `Unsupport onnx opset 20, need <= 19!`。

```bash
.venv-rknn/bin/python - <<'PY'
import onnx
from onnx import version_converter

# YOLOv8n：标准降级路径走得通
m = onnx.load('models/weights/yolov8n.onnx')
onnx.save(version_converter.convert_version(m, 19), 'models/weights/yolov8n_op19.onnx')
print('yolov8n -> opset 19 OK')
PY
```

### 4.2 YOLOv11n：动态形状固化 + 强制改写 opset

v11 的官方 ONNX 输入是 `[batch,3,height,width]` 符号维，RKNN 量化需要静态图；且它的 opset 22 连 `version_converter` 都不支持转换。

```bash
.venv-rknn/bin/python - <<'PY'
import onnx
from onnx.tools import update_model_dims

m = onnx.load('models/weights/yolo11n.onnx')
# 1) 固化输入输出形状
m = update_model_dims.update_inputs_outputs_dims(
        m, {'images': [1, 3, 640, 640]}, {'output0': [1, 84, 8400]})
# 2) 直接改写 opset 声明（YOLO 用到的算子在 19 中均已覆盖，checker 可验证）
for op in m.opset_import:
    if op.domain in ('', 'ai.onnx'):
        op.version = 19
onnx.checker.check_model(m)
onnx.save(m, 'models/weights/yolo11n_op19.onnx')
print('yolo11n -> 静态形状 + opset 19 OK')
PY
```

---

## 5. 量化、锁频与压测

### 5.1 INT8 量化导出

```bash
# YOLOv8n（约 80 秒）
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx models/weights/yolov8n_op19.onnx \
    --output models/weights/yolov8n_int8.rknn \
    --algorithm normal \
    --report docs/benchmarks/convert_yolov8n_int8.json

# YOLOv11n（约 67 秒，用于横向对比）
.venv-rknn/bin/python tools/export_rknn.py \
    --onnx models/weights/yolo11n_op19.onnx \
    --output models/weights/yolo11n_int8.rknn \
    --algorithm normal \
    --report docs/benchmarks/convert_yolo11n_int8.json
```

期望产物：`yolov8n_int8.rknn` 约 **4.7MB**（12.2MB → 压缩 61.2%）。

> 转换日志默认写在终端，若要留存供第 5.4 节解析算子分布，请重定向：`... > /tmp/conv_v8.log 2>&1`

### 5.2 锁频（不做这步，数据会难看且抖动大）

NPU 其实一直满频 1GHz、温度才 46℃，真正的瓶颈是 CPU 被 `ondemand` 压在 1.2GHz —— 而输入拷贝、归一化、输出反量化全在 CPU 上。

```bash
sudo bash scripts/set_performance.sh          # 锁 performance
# sudo bash scripts/set_performance.sh restore  # 恢复节能
```

期望看到大核 `2400 MHz`、NPU `1000 MHz`。**重启后失效，每次测前都要重跑。**

### 5.3 跑基准测试

```bash
# 单路：对比 单核/双核/三核/自动 四种调度模式的时延
taskset -c 4-7 .venv-rknn/bin/python tools/benchmark_npu.py \
    --rounds 300 --report docs/benchmarks/npu_yolov8n_int8.json

# 三核真并发：测系统吞吐上限
taskset -c 4-7 .venv-rknn/bin/python tools/benchmark_npu_parallel.py \
    --rounds 150 --report docs/benchmarks/npu_parallel_yolov8n.json
```

> `taskset -c 4-7` 把进程绑到 A76 大核。不绑的话调度器可能把它扔到 A55 小核上，数据会明显变差。

**期望结果**（±10% 属正常波动，受散热与后台负载影响）：

| 测试项 | 参考值 |
| :--- | ---: |
| 单路最优（`NPU_CORE_AUTO`） | 22.61 ms / 44.24 FPS / 抖动 ±0.65 ms |
| 三核并发系统吞吐 | 104.50 FPS（2.71x 加速） |
| 生产封装 `RKNNParallelVisionPool` | 112.99 FPS |

### 5.4 汇总对比报告

```bash
# 需要先按 5.1 的提示保留 /tmp/conv_v8.log 与 /tmp/conv_v11.log
.venv-rknn/bin/python tools/compare_yolo_versions.py \
    --v8-log /tmp/conv_v8.log --v11-log /tmp/conv_v11.log
```

该工具**不含任何硬编码数值**，全部结论从构建日志与实测 JSON 现场解析。

### 5.5 验证生产级推理池

```bash
taskset -c 4-7 .venv-rknn/bin/python - <<'PY'
import sys, time, threading; sys.path.insert(0, '.')
import numpy as np
from src.inference.rknn_engine import RKNNParallelVisionPool

pool = RKNNParallelVisionPool('models/weights/yolov8n_int8.rknn', num_cores=3)
assert pool.init_pool()
data = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)

N = 120
t0 = time.perf_counter()
threading.Thread(target=lambda: [pool.submit(data) for _ in range(N)], daemon=True).start()
costs = [r[2] for _ in range(N) if (r := pool.fetch(timeout=15))]
wall = time.perf_counter() - t0
print(f'{len(costs)} 帧 / {wall:.2f}s = {len(costs)/wall:.2f} FPS | 单帧均值 {sum(costs)/len(costs):.2f}ms')
pool.shutdown()
PY
```

---

## 6. 踩坑速查表

复现路上大概率会撞上的问题，都在这儿了：

| 报错 / 现象 | 根因 | 解法 |
| :--- | :--- | :--- |
| `No module named pip`（建 venv 后） | 系统缺 `ensurepip` | `sudo apt install python3.10-venv`，删掉 venv 重建 |
| `Unsupport onnx opset 20, need <= 19!` | ultralytics 新版导出 opset 20/22 | 见 §4.1 降级 |
| `assertInVersionRange` 转换失败 | `version_converter` 不支持 opset 22 | 见 §4.2 直接改写 opset 声明 |
| `The image of .../data/calibration/data/calibration/...` | RKNN 以清单文件所在目录为基准**再拼一次**相对路径 | `dataset.txt` 用**绝对路径** |
| v11 量化报输入维度错误 | 官方 onnx 是动态形状 | 见 §4.2 `update_model_dims` 固化 |
| 加载 .rknn 报版本不匹配 | 运行时还是 0.9.6 | 见 §2 升级 librknnrt |
| 三核比单核快不了多少 | **这不是 bug**：`inference()` 同步阻塞，`core_mask` 不拆分单任务 | 用 `RKNNParallelVisionPool` 多实例并发 |
| 时延忽高忽低、抖动大 | `ondemand` 调频 + 跑在小核 | `set_performance.sh` + `taskset -c 4-7` |
| 量化卡在 `Quantizating` 很久 | 正常，100 张校准图逐层统计约需 60~80s | 耐心等 |

---

## 7. 一键脚本

懒得逐条敲的话，`scripts/` 下已有对应封装：

```bash
sudo bash scripts/set_performance.sh    # 锁频
bash scripts/build.sh                   # C++ 流水线编译
```

完整命令清单也可直接查阅 [`benchmarks/NPU_BENCHMARK_REPORT.md`](benchmarks/NPU_BENCHMARK_REPORT.md) 第七节。
