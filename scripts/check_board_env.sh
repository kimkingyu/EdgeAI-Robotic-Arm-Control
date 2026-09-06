#!/usr/bin/env bash
# ==============================================================================
# RK3588 / OrangePi 5 Pro 环境自检脚本
# 用法: bash scripts/check_board_env.sh
# ==============================================================================

echo "========================================================"
echo "  RK3588 EdgeAI 机械臂开发环境体检"
echo "========================================================"

# 1. 架构检测
ARCH=$(uname -m)
echo "[1] 系统架构: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    echo "    [警告] 当前非 aarch64 架构，RK3588 NPU 运行时需要 aarch64 环境！"
fi

# 2. 内核与系统发行版
echo "[2] 系统版本与内核:"
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID" | sed 's/^/    /'
echo "    内核版本: $(uname -r)"

# 3. NPU 驱动状态
echo "[3] RKNPU 驱动状态:"
if dmesg | grep -iq "rknpu"; then
    dmesg | grep -i "rknpu" | tail -n 5 | sed 's/^/    /'
    echo "    [OK] 检测到 RKNPU 驱动日志"
else
    echo "    [提示] 未在 dmesg 找到 rknpu 日志，请检查驱动模块是否加载"
fi

# 检查设备节点
if [ -e "/dev/rknpu" ] || [ -e "/dev/dri/renderD128" ]; then
    echo "    [OK] NPU/GPU 相关渲染与算力节点正常"
    ls -l /dev/rknpu* /dev/dri/render* 2>/dev/null | sed 's/^/    /'
else
    echo "    [警告] 未检测到 /dev/rknpu 节点"
fi

# 4. 摄像头设备
echo "[4] 摄像头视频节点 (/dev/video*):"
if ls /dev/video* 1> /dev/null 2>&1; then
    ls -l /dev/video* | sed 's/^/    /'
else
    echo "    [提示] 未找到 /dev/video* 设备，请插入 USB/CSI 相机"
fi

# 5. 串口设备 (机械臂通信)
echo "[5] 串口设备 (/dev/ttyUSB* /dev/ttyACM*):"
if ls /dev/ttyUSB* /dev/ttyACM* 1> /dev/null 2>&1; then
    ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | sed 's/^/    /'
else
    echo "    [提示] 暂未检测到 USB 转串口设备，连接机械臂主控后重新检查"
fi

# 6. Python 环境
echo "[6] Python 环境:"
if command -v python3 &> /dev/null; then
    echo "    Python3 版本: $(python3 --version)"
    echo "    Pip3 版本: $(python3 -m pip --version 2>/dev/null || echo '未安装 pip3')"
else
    echo "    [错误] 未找到 python3"
fi

echo "========================================================"
echo "体检完成。请将以上信息对照并按需配置各外设权限 (如 sudo usermod -aG dialout,video $USER)"
echo "========================================================"
