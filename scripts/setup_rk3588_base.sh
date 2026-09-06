#!/usr/bin/env bash
# ==============================================================================
# 香橙派 5 Pro (RK3588S) Ubuntu 22.04 基础开发环境初始化脚本
# ==============================================================================
set -e

echo "=========================================================="
echo "  开始配置香橙派 5 Pro 基础系统环境"
echo "=========================================================="

# 1. 设置时区为上海
echo "[1/5] 校准时区为 Asia/Shanghai..."
sudo timedatectl set-timezone Asia/Shanghai

# 2. 换国内 Ubuntu Ports 清华源 (针对 arm64 架构)
echo "[2/5] 备份并替换 apt 国内镜像源 (清华源)..."
if [ ! -f /etc/apt/sources.list.bak ]; then
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
fi

sudo tee /etc/apt/sources.list > /dev/null << 'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse
EOF

sudo apt update -y

# 3. 安装常用编译与系统管理工具
echo "[3/5] 安装基础编译依赖与开发工具链..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    curl \
    wget \
    htop \
    net-tools \
    python3-dev \
    python3-pip \
    python3-setuptools

# 4. 配置 pip 国内镜像源
echo "[4/5] 配置 pip 清华源..."
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 5. 配置用户外设免 sudo 权限 (dialout: 串口, video: 摄像头, i2c: 舵机, gpio: 引脚)
echo "[5/5] 配置硬件外设免 sudo 访问权限..."
sudo usermod -aG dialout,video,i2c,gpio orangepi

echo "=========================================================="
echo "基础系统环境配置完成！"
echo "时区已校准、apt/pip 已加速、编译工具链已就绪、外设权限已开通。"
echo "=========================================================="
