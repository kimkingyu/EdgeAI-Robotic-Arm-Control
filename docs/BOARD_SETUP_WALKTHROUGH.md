# OrangePi 5 Pro (RK3588S) 实战配置通关秘籍

> 专门为你准备的手动实战闯关指南。在 VS Code 终端连上板子（`ssh orangepi@192.168.0.102`），一行行敲下去，亲自体验把一块裸板调教成顶级边缘 AI 机械臂主控的全过程！

---

## 关卡 0：连接板子并验明正身

在电脑的 VS Code 终端里连上香橙派：
```bash
ssh orangepi@192.168.0.102
```
输入你的密码进入系统。

### 1.1 查看板子架构与内核
```bash
uname -a && lsb_release -a
```
* **预期看到**：`aarch64`（ARM64 架构）、`Ubuntu 22.04.5 LTS (Jammy)`、内核 `5.10.160-rockchip-rk3588`。
* **为什么**：RK3588 必须在 aarch64 下跑，22.04 自带 Python 3.10，是目前 RKNN 驱动最稳定的黄金组合。

### 1.2 检查板载 6 TOPS NPU 驱动
```bash
dmesg | grep -i rknpu
```
* **预期看到**：`[drm] Initialized rknpu 0.9.6`。
* **为什么**：证明瑞芯微官方 NPU 内核模块已经挂载，6 TOPS 硬件算力已经通电待命。

---

## 关卡 1：校准时区与国内源加速

刚刷好的系统默认是国外源和 UTC 时间，不改源下载只有几十 KB/s 还容易断连。

### 2.1 校准时区到北京时间
```bash
sudo timedatectl set-timezone Asia/Shanghai
date
```
* **为什么**：时间不对会导致之后拉 GitHub 代码或者 `apt` 装包时疯狂弹 SSL 证书过期错误。

### 2.2 替换为华为云 / 清华云 arm64 源
```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo tee /etc/apt/sources.list > /dev/null << 'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse
EOF
```
刷新软件源缓存：
```bash
sudo apt update
```
* **感受一下**：下载速度直接飙到几 MB/s 到十几 MB/s，顺滑起飞。

### 2.3 配置 pip 国内镜像源
```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```
* 以后用 `pip3 install` 安装 Python 库，全部从清华源秒级下载。

---

## 关卡 2：安装机器人开发核心工具链

这是编译 C++ 驱动、OpenCV、机械臂通信库必不可少的系统底座：

```bash
sudo apt install -y build-essential cmake git pkg-config htop net-tools python3-dev python3-pip i2c-tools
```

* `build-essential`：包含 gcc、g++、make 编译器。
* `cmake`：跨平台编译工具（后面编译 NPU C++ 推理引擎必用）。
* `i2c-tools`：用来直接扫描和调试机械臂舵机板硬件的工具。

---

## 关卡 3：外设免 sudo 提权（彻底告别 Permission Denied）

Linux 下串口、摄像头、I2C 引脚默认属于 `root` 用户。普通用户不提权去读写设备节点就会被系统打脸拒掉。

把当前用户 `orangepi` 加入硬件用户组：
```bash
sudo usermod -aG dialout,video,i2c orangepi
```
* `dialout`：免 sudo 读写 USB 串口（`/dev/ttyUSB0`）。
* `video`：免 sudo 抓取摄像头视频流（`/dev/video*`）。
* `i2c`：免 sudo 控制 PCA9685 舵机板（`/dev/i2c-*`）。

---

## 关卡 4：把香橙派挂载为 Windows 的 Z: 盘（Samba 共享）

不用再拿 U 盘拷文件，直接在 Windows 资源管理器里像本地磁盘一样拖拽文件：

### 4.1 在板子上安装 Samba
```bash
sudo apt install -y samba
```

### 4.2 配置共享主目录
编辑配置文件：
```bash
sudo nano /etc/samba/smb.conf
```
按快捷键 `Alt + /` 移到文件最末尾，粘贴以下内容：
```ini
[orangepi]
   comment = OrangePi 5 Pro Home
   path = /home/orangepi
   browseable = yes
   writable = yes
   create mask = 0775
   directory mask = 0775
   valid users = orangepi
```
按 `Ctrl + O` 回车保存，按 `Ctrl + X` 退出。

### 4.3 设定共享密码并重启服务
```bash
sudo smbpasswd -a orangepi
```
（输入你的密码，比如 `4538`，输两遍）。

重启生效：
```bash
sudo systemctl restart smbd
```

### 4.4 回到 Windows 电脑挂载
1. 按键盘快捷键 `Win + R`。
2. 输入 `\\192.168.0.102` 按回车。
3. 输入账号 `orangepi`、密码 `4538`。
4. 看到 `orangepi` 文件夹后，**右键 -> 映射网络驱动器 -> 选 Z: 盘**！
5. 打开“此电脑”，`Z:` 盘就在那里，直接往里拖文件试试！

---

## 关卡 5：实机 I2C 舵机板探测与调试

你的机械臂是用 I2C 舵机驱动板（常见为 PCA9685）控制的：

### 5.1 查看当前系统已经开启的 I2C 总线
```bash
ls -l /dev/i2c*
```
* 会看到比如 `/dev/i2c-0`, `/dev/i2c-1`, `/dev/i2c-7` 等设备节点。

### 5.2 扫描总线上的设备（接上舵机板后）
香橙派 40-Pin 引脚上的 I2C 通常是 7 号或 2 号总线。接好 4 根线（VCC 5V、GND、SDA、SCL）后敲：
```bash
sudo i2cdetect -y 7
```
* **如果接线正确**：屏幕矩阵的 `40` 位置会出现数字 `40`（或者 `UU`），表示检测到了地址为 `0x40` 的 PCA9685 舵机驱动芯片！

---

## 关卡 6：板载 NPU 运行库（rknn-toolkit-lite2）

这是机械臂跑 YOLOv8/YOLOv5 视觉目标检测的心脏。

### 6.1 拉取瑞芯微官方 RKNPU2 核心库
```bash
cd /home/orangepi
git clone https://github.com/airockchip/rknpu2.git --depth=1
```

### 6.2 安装 NPU 板端动态链接库
```bash
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
```

### 6.3 安装 Python 推理包（针对 Python 3.10）
```bash
cd rknpu2/rknn_toolkit_lite2/packages/
ls -l
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

### 6.4 验证 NPU Python 接口
```bash
python3 -c "from rknnlite.api import RKNNLite; rknn = RKNNLite(); print('RKNN-Lite NPU 环境安装大获全胜！')"
```
只要看到打印出这一行，恭喜你，你的香橙派已经具备完整的端侧 6 TOPS AI 推理能力！

---

## 关卡 7：运行你的开源机械臂控制项目

代码已经同步在板子的 `/home/orangepi/EdgeAI-Robotic-Arm-Control`：

```bash
cd /home/orangepi/EdgeAI-Robotic-Arm-Control

# 1. 运行自检脚本
bash scripts/check_board_env.sh

# 2. 跑一次全真模拟抓取闭环
python3 main.py --mock
```

看到各关节指令和抓取动作一步步在终端执行完毕，整个系统的从感知到控制链路就彻底成型了！
