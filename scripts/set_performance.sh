#!/bin/bash
# RK3588 边缘推理性能锁频脚本
#
# 背景：Ubuntu 默认 ondemand 调速下 CPU 停在 1.2GHz，而 rknn 推理的输入拷贝、
#       归一化与输出反量化均在 CPU 侧执行，会把 NPU 饿着。
# 实测：锁定 performance 后单帧 29.60ms -> 23.23ms（-21.5%），
#       P99 抖动 ±5.94ms -> ±1.49ms（-75%），吞吐 33.79 -> 43.04 FPS。
#
# 用法：sudo bash scripts/set_performance.sh [restore]

set -u

MODE="${1:-performance}"
if [ "$MODE" = "restore" ]; then
    CPU_GOV="ondemand"
    DEV_GOV="simple_ondemand"
    NPU_GOV="rknpu_ondemand"
    echo ">>> 恢复默认节能调速策略 ..."
else
    CPU_GOV="performance"
    DEV_GOV="performance"
    NPU_GOV="performance"
    echo ">>> 锁定高性能调速策略 ..."
fi

if [ "$(id -u)" != "0" ]; then
    echo "[错误] 需要 root 权限，请使用: sudo bash $0 $MODE"
    exit 1
fi

# 1. CPU 各 policy 集群（policy0=A55 小核, policy4/policy6=A76 大核）
for gov in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    [ -w "$gov" ] && echo "$CPU_GOV" > "$gov" 2>/dev/null
done

# 2. NPU / GPU / DDR 控制器
for dev in /sys/class/devfreq/*/; do
    gov_file="${dev}governor"
    [ -w "$gov_file" ] || continue
    name=$(basename "$dev")
    case "$name" in
        *npu*) echo "$NPU_GOV" > "$gov_file" 2>/dev/null || echo "$DEV_GOV" > "$gov_file" 2>/dev/null ;;
        *)     echo "$DEV_GOV" > "$gov_file" 2>/dev/null ;;
    esac
done

echo ""
echo "================= 当前频率状态 ================="
for p in /sys/devices/system/cpu/cpufreq/policy*/; do
    name=$(basename "$p")
    cur=$(cat "${p}scaling_cur_freq" 2>/dev/null)
    gov=$(cat "${p}scaling_governor" 2>/dev/null)
    printf "  CPU %-10s %6s MHz   [%s]\n" "$name" "$((cur / 1000))" "$gov"
done
for dev in /sys/class/devfreq/*/; do
    name=$(basename "$dev")
    cur=$(cat "${dev}cur_freq" 2>/dev/null) || continue
    gov=$(cat "${dev}governor" 2>/dev/null)
    printf "  DEV %-14s %4s MHz   [%s]\n" "${name:0:13}" "$((cur / 1000000))" "$gov"
done

TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
[ -n "$TEMP" ] && printf "  SoC 温度        %s °C\n" "$((TEMP / 1000))"
echo "================================================"
echo ""
echo "提示：推理进程建议绑定 A76 大核运行 ->  taskset -c 4-7 python3 xxx.py"
