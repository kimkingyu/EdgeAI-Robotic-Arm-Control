#!/bin/bash
# 下载 Qwen3-VL-4B-Instruct (RKLLM v1.2.3, w8a8) 视觉多模态模型
#
# 该模型由两部分组成，缺一不可：
#   语言模型 .rkllm  (4.51 GB) —— 由 rkllm-toolkit v1.2.3 转换
#   视觉编码器 .rknn (0.81 GB) —— 由 rknn-toolkit2 转换
#
# 支持断点续传，中断后重跑本脚本即可继续。
# 用法：bash scripts/fetch_qwen_vl.sh

set -u

BASE="https://hf-mirror.com/GatekeeperZA/Qwen3-VL-4B-Instruct-RKLLM-v1.2.3/resolve/main"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/weights"
mkdir -p "$DEST"

fetch() {
    local url="$1" out="$2" expect="$3"
    echo ">>> 下载 $(basename "$out") (期望约 $expect)"
    curl -L -C - --retry 5 --retry-delay 3 --progress-bar -o "$out" "$url"
    local sz
    sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
    printf "    实际大小: %.2f GB\n" "$(echo "$sz" | awk '{print $1/1073741824}')"
    if [ "$sz" -lt 100000000 ]; then
        echo "    [警告] 文件过小，可能下载失败"
        return 1
    fi
}

fetch "$BASE/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm" \
      "$DEST/qwen3vl4b_w8a8.rkllm" "4.51 GB" || exit 1

fetch "$BASE/qwen3-vl-4b-vision_rk3588.rknn" \
      "$DEST/qwen3vl4b_vision.rknn" "0.81 GB" || exit 1

echo ""
echo "===== 下载完成 ====="
ls -lh "$DEST"/qwen3vl4b*
