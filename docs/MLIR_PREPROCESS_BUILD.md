# MLIR 预处理：M1 复现入口

目前跑通了 **MLIR → LLVM IR → AArch64 object → C ABI → C++ / Python**。算子只做 BGR→RGB，尚未实现 resize、融合或像素向量化，也没有加速结论。原生产预处理和 RKNN 环境没改。

版本和源码哈希固定在 [compiler/toolchain.lock.json](compiler/toolchain.lock.json)。只用板端现有 GCC 11.4、CMake 3.22.1、Make 4.3 和系统 Python 3.10；不安装系统包，不执行 install。

## Windows：取源码、同步独立工作区

在仓库根目录的 Git Bash 执行：

```bash
mkdir -p build/mlir-downloads
GODEBUG=http2client=0 py -3.12 -B scripts/mlir/fetch_llvm_source.py \
  --directory build/mlir-downloads --workers 1

ssh orangepi@192.168.0.102 \
  'mkdir -p /home/orangepi/project/edgeai-mlir-work/build/mlir-downloads /home/orangepi/toolchains'
set -o pipefail
tar -cf - compiler scripts/mlir | ssh orangepi@192.168.0.102 \
  'tar -xf - -C /home/orangepi/project/edgeai-mlir-work'
scp build/mlir-downloads/llvm-project-20.1.8.src.tar.xz \
  orangepi@192.168.0.102:/home/orangepi/project/edgeai-mlir-work/build/mlir-downloads/
```

本机曾遇到 HTTP/2 大响应停滞；上面的 `GODEBUG` 只影响这次命令及其 gh 子进程，不改系统代理。下载会保留已完成范围，只有完整大小和官方 SHA256 都相符才接受归档。鉴权失败或内容范围错误不会被重试掩盖。

## 板端：编译工具链和 edge-opt

SSH 登录后执行；不要在工具链目录或它的父目录中启动 bootstrap。

```bash
WORK="$HOME/project/edgeai-mlir-work"
TC="$HOME/toolchains/edgeai-llvm-20.1.8"

/usr/bin/python3 -B "$WORK/scripts/mlir/bootstrap_llvm.py" \
  --archive "$WORK/build/mlir-downloads/llvm-project-20.1.8.src.tar.xz" \
  --root "$TC" --lock "$WORK/compiler/toolchain.lock.json"

cmake -S "$WORK/compiler" -B "$WORK/build/compiler" \
  -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DMLIR_DIR="$TC/build/lib/cmake/mlir" \
  -DLLVM_DIR="$TC/build/lib/cmake/llvm" \
  -DCMAKE_CXX_LINKER_LAUNCHER="/usr/bin/flock;$TC/link.lock"
cmake --build "$WORK/build/compiler" --parallel 2 --target edge-opt
```

- 编译并发 2，C/C++ 链接共享一个 flock；没有把 Ninja 的 job pool 当作 Make 的功能。
- bootstrap 每次保留独立日志和命令 JSON。`--configure-only` 只能得到 configured，不算工具链完成。
- 已有非空目录必须带匹配的所有权 marker。遇到失败解压树，先保存现场，别删除检查或伪造 marker。
- 部署版 Python 3.10 的 tar 过滤曾把相对符号链接按解压根目录解析。脚本保留归档图检查和文件系统边界检查，对符号链接按其所在目录解析；不修改系统 Python，不跳过归档内容。

## 板端：运行真实 AOT 检查

工具链构建结束后再执行。输出目录必须是新目录；重复验证时换一个名字。

```bash
/usr/bin/python3 -B "$WORK/scripts/mlir/run_color_smoke.py" \
  --toolchain "$TC/build" \
  --edge-opt "$WORK/build/compiler/bin/edge-opt" \
  --output "$WORK/build/m1-color-smoke-rerun"
```

入口会保留原始/降级 MLIR、LLVM IR、汇编、对象、库、每条命令日志和摘要；任一步失败就非零退出。缺库不会跳成通过。

C++ 检查真实像素、输入不变、padding/哨兵及容量不足拒绝。Python ctypes 检查 10 组形状/行 padding，每组重复 25 次，并覆盖 16 组非法参数。所有像素计算来自 MLIR，不在 C ABI 包装中另写一份换色循环。

## 已取得的证据

- [M1 摘要](docs/benchmarks/mlir_m1_summary.json)：真实 C++ / Python 日志、产物 SHA256、运行库依赖和边界说明。
- [工具链构建结果](docs/benchmarks/mlir_m1_toolchain.json)：五个工具均为 20.1.8；本次完整构建 14641.61 秒。RSS 字段是最大单个子进程峰值，不是并发总内存峰值。
- [完整烟测命令记录](docs/benchmarks/mlir_m1_color_smoke.json)：13 条真实命令均返回 0。
- [降级 MLIR](docs/benchmarks/mlir_m1_ir/02-llvm.mlir)、[LLVM IR](docs/benchmarks/mlir_m1_ir/03-color.ll)、[ARM 汇编](docs/benchmarks/mlir_m1_ir/04-color.s)。

运行库不依赖 libLLVM/libMLIR。像素内层仍是标量 ldrb/strb；C 接口包装里出现 q 寄存器只是搬运 descriptor，不能拿它宣称像素 SIMD 加速。

## 板端：M2/M3 缩放、融合与目标缓冲复用

`edge-preprocess-gen` 生成 Q11 定点双线性 resize 加 BGR→RGB 的 IR，两个自写 Pass 分别做融合和目标缓冲改写。同一份生成 IR 会构建四种配置，用于消融对比。

```bash
cmake --build "$WORK/build/compiler" --parallel 2 --target edge-opt edge-preprocess-gen

/usr/bin/python3 -B "$WORK/scripts/mlir/build_preprocess.py" \
  --toolchain "$TC/build" --compiler-build "$WORK/build/compiler" \
  --output "$WORK/build/m23-aot-rerun"

"$HOME/project/EdgeAI-Robotic-Arm-Control/.venv-rknn/bin/python" -B \
  "$WORK/compiler/tests/preprocess_aot_checks.py" \
  --manifest "$WORK/build/m23-aot-rerun/manifest.json" \
  --images-dir "$HOME/project/EdgeAI-Robotic-Arm-Control/data/calibration/images" \
  --report "$WORK/reports/m23-correctness-rerun.json"

/usr/bin/python3 -B "$WORK/compiler/tests/preprocess_pass_checks.py" \
  --edge-opt "$WORK/build/compiler/bin/edge-opt" --filecheck "$TC/build/bin/FileCheck" \
  --seed "$WORK/compiler/tests/resize_color_q11.mlir" \
  --checks "$WORK/compiler/tests/preprocess_passes.check" \
  --report "$WORK/reports/m3-pass-legality-rerun.json"
```

数值参考是独立的 `Fraction` 实现，不调用被测代码，也不用 OpenCV 构造期望值：

```bash
"$HOME/project/EdgeAI-Robotic-Arm-Control/.venv-rknn/bin/python" -B \
  "$WORK/compiler/tests/test_preprocess_reference.py"
```

### 已取得的证据

- [M2/M3 摘要](docs/benchmarks/mlir_m23_summary.json)、[Pass 合法性](docs/benchmarks/mlir_m3_pass_legality.json)。
- 6 个输出尺寸 × 4 种配置共 24 个内核，每个内核 36 组输入（合成边界、16 张真实图片、5 个派生尺寸、同尺寸）各重复 3 次，输出与独立参考逐字节相同。
- 每次调用的实测分配/释放：未融合 2/2，仅融合 1/1，仅目标复用 1/1，融合加目标复用 0/0，均无残留。缓冲化 IR 的 alloc/copy 依次为 2/1、1/1、1/0、0/0。
- 216 组与固定版 OpenCV `INTER_LINEAR` 的差分，最大绝对误差 1 个灰度级——这是兼容性上界，不是逐位一致。
- Pass 检查：3 组正例按预期改写；40 组不支持或非法写法被拒绝且 IR 不变。
- 中间 IR 样例：[融合加目标复用](docs/benchmarks/mlir_m23_ir/fused_destination_5x7.mlir)、[其缓冲化结果](docs/benchmarks/mlir_m23_ir/fused_destination_5x7_buffered.mlir)、[未融合缓冲化结果](docs/benchmarks/mlir_m23_ir/unfused_5x7_buffered.mlir)。

分配计数来自链接期包装的 `malloc`/`aligned_alloc`/`free`，只统计单次调用窗口，不是进程 RSS。保护页和只读输入映射用于界定访问范围，不等于完整 sanitizer 验证。

## 板端：M4 分块与向量化消融

十种调度配置从同一份生成 IR 构建，再用同一套正确性检查和统一基准测量：

```bash
/usr/bin/python3 -B "$WORK/scripts/mlir/build_preprocess.py" \
  --toolchain "$TC/build" --compiler-build "$WORK/build/compiler" \
  --output "$WORK/build/m4-rerun"

cd "$WORK/compiler/tests"
taskset -c 4 "$HOME/project/EdgeAI-Robotic-Arm-Control/.venv-rknn/bin/python" -B \
  preprocess_ablation.py --manifest "$WORK/build/m4-rerun/manifest.json" \
  --images-dir "$HOME/project/EdgeAI-Robotic-Arm-Control/data/calibration/images" \
  --report "$WORK/reports/m4-ablation-rerun.json" \
  --profile 640 640 --input 480 640 --warmup 30 --iterations 200
```

### 结果：没有取得加速

640×640、480×640 输入、单核单线程下的 P50：

| 配置 | P50 | 相对 OpenCV |
|---|---|---|
| OpenCV 基线 | 4270.5 us | 1.00 |
| 未融合 | 23208.9 us | 5.44 |
| 融合 | 22414.8 us | 5.25 |
| 仅目标复用 | 21083.4 us | 4.94 |
| 融合＋目标复用 | 19168.7 us | 4.49 |
| 分块 1×16 | 23824.3 us | 5.58 |
| 分块 4×32 | 29443.8 us | 6.89 |
| 分块 8×64 | 28588.0 us | 6.69 |
| 分块 1×128 | 23659.3 us | 5.54 |
| 分块 1×16＋向量化 | 20182.3 us | 4.73 |
| 分块 4×32＋向量化 | 19956.7 us | 4.67 |

448×448 结论一致，最好也只到 3.81 倍慢。计划设定的至少改善 10% 的门槛没有达到，**默认后端仍是 OpenCV**。

固定尺寸分块反而更慢；向量化确实生成了真实的 SIMD 指令（640 profile 下 211 个 q 寄存器引用、240 条 `ld1`/`st1`，对比融合版的 6 个），但仍未追上基线。

### 向量化的语义限制

上游向量化对 `tensor.extract` 会生成 `vector.gather`，其偏移按紧凑布局计算，因而**只对没有行填充的输入成立**。带 5 字节行填充的实测输出最大误差达 249。因此 `--edge-vectorize-preprocess` 默认拒绝这种结果，必须显式加 `allow-packed-source-gather=true` 才能生成，相关内核在清单中标记 `requires_packed_source`。

### 已取得的证据

- [M4 汇总](docs/benchmarks/mlir_m4_summary.json)、[向量化 IR 样例](docs/benchmarks/mlir_m4_ir/tile_1x16_vector_640.mlir)。
- 52 个内核构建并全部通过逐字节正确性、分配审计与保护页检查；8 个因输出尺寸不大于分块尺寸标为不适用，不算通过。
- 每次计时调用都校验完整输出，空转或算错的内核无法显得更快。基准中 MLIR 与参考的不一致次数为 0；OpenCV 的差异均在冻结的 1 个灰度级容差内。
- 温度边界快照 49.9–53.6 °C，属边界读数而非持续监控。

M0 的 OpenCV 基线不能与正确性结果直接换算加速比。分块/向量化未获收益的原因分析、以及是否值得继续优化，属于后续工作。

## M5：可选后端接入

[src/vision/preprocess.py](src/vision/preprocess.py) 给出统一的 `Preprocessor`，OpenCV 为默认，MLIR 为显式可选。[src/pipeline/vlm_grasp_pipeline.py](src/pipeline/vlm_grasp_pipeline.py#L198) 的 640 路径、[src/vision/rknn_detector.py](src/vision/rknn_detector.py#L60) 都改用它，两处不再各写一套像素逻辑。

MLIR 后端失败时**直接报错**，不会退回 OpenCV 再把耗时记在 MLIR 名下：

```bash
python3 tools/run_vlm_grasp.py --cmd "..." --mock          # 默认 OpenCV，无需 MLIR
python3 tools/run_vlm_grasp.py --cmd "..." --mock \
  --preprocess-backend mlir \
  --preprocess-library 640x640=/path/640/libedgeai_preprocess.so \
  --preprocess-library 448x448=/path/448/libedgeai_preprocess.so \
  --preprocess-manifest /path/manifest.json
```

AOT 内核的输出尺寸是编译期固定的，一个 `.so` 只服务一个 profile，所以库按 `HxW=PATH` 分别指定；缺哪个 profile 就报哪个，不会拿 640 的库去凑 448。

### 测试

```bash
taskset -c 4 .venv-rknn/bin/python -B tools/test_mlir_preprocess.py \
  --require-mlir --library build/m4-final/640x640/fused_destination/libedgeai_preprocess.so \
  --manifest build/m4-final/manifest.json
```

22 项，两个 profile 各跑一遍全通过。`--require-mlir` 下缺库是 **failure 而非 skip**（退出码 1）。覆盖：float/灰度/RGBA/空图/负 stride/稀疏列/虚报形状的拒绝、正 stride ROI、调用方缓冲校验、连续调用不串帧、4 线程各持独立输出、输入尺寸变化重建 context 且创建耗时单列、manifest 与 profile 核对。

不装 MLIR 时原链路完整可用：两个 profile 都回落到 OpenCV，detector 正常工作，`preprocess.py` 导入期不加载任何 `.so`。

## M6：真实 RKNN A/B 与复验

```bash
taskset -c 4 .venv-rknn/bin/python -B tools/benchmark_preprocess_rknn.py \
  --model models/weights/yolov8n_int8.rknn \
  --images-dir data/calibration/images --report reports/m6-ab-yolo640.json \
  --mlir-library build/m4-final/640x640/fused_destination/libedgeai_preprocess.so \
  --mlir-manifest build/m4-final/manifest.json \
  --profile 640 640 --input 480 640 --warmup 20 --iterations 100
```

真实 `librknnrt 2.3.2`、真实权重；Mock 推理被显式拒绝。

| 模型 | OpenCV 预处理 P50 | MLIR 预处理 P50 | 比值 | 推理 P50 |
|---|---|---|---|---|
| yolov8n_int8 @ 640×640 | 9544.7 us | 29050.9 us | 3.04× 慢 | ~38 ms |
| qwen3vl4b_vision @ 448×448 | 2096.5 us | 9708.9 us | 4.63× 慢 | ~3.2 s |

### 模型输出确实变了

两个后端的预处理输出最大只差 **1 个灰度级**（YOLO 路径 6.01% 的值不同，Qwen 路径 12.53%），但模型原始输出并不相同：

| 模型 | 张量数 | 逐字节一致 | 最大绝对差 | 最大平均绝对差 | 最多差异元素占比 |
|---|---|---|---|---|---|
| yolov8n_int8 | 8 | 0 | 180.220 | 0.097776 | 2.03% |
| qwen3vl4b_vision | 32 | 0 | 5.070 | 0.036350 | 99.28% |

这个差异做过归因，不是猜的：

- **NPU 是确定的**：同一份输入连续两次推理结果逐字节一致，所以输出差异只能来自输入差异。
- **不是 MLIR 算错**：对独立的 Q11 定点参考，MLIR 内核逐字节一致，偏离参考的是 OpenCV 的舍入实现（最大差 1）。
- 结论是：±1 灰度级的输入差异足以改变模型原始输出。这是更换预处理实现的真实代价，不能因为"只差 1"就当作等价。

因此**不做无缝默认替换**。要替换，前提是与部署所用 OpenCV 版本达到逐字节一致，当前只是近似一致。

### 从零复验

```bash
python3 scripts/mlir/build_preprocess.py --toolchain "$TC/build" \
  --compiler-build "$WORK/build/compiler" --output "$WORK/build/m6-verify"
```

全新目录重建：312 个 IR/汇编产物 SHA256 与原产物**逐字节一致**，0 个不同；52 个内核重新通过正确性检查；A/B 复跑得到 3.06× 慢，结论未变。

### 边界

- 单核单线程、单一输入尺寸 480×640；耗时由逐次计时推导，不是持续吞吐。
- 无标注数据与后处理解码，因此不报 mAP、检测精度保持或实机定位精度；模型输出比较只是原始张量比较。
- Qwen 单次推理约 3.2 秒，样本量 40 次，小于 YOLO 的 100 次。
- 未做多路并发与端到端抓取验证，本期不联调舵机。

证据：[M5/M6 汇总](docs/benchmarks/mlir_m56_summary.json)。
