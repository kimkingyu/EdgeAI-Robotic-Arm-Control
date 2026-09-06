#!/usr/bin/env bash
# 一键开启香橙派 Samba 共享，让 Windows 资源管理器直接像本地文件夹一样打开
set -e

echo "[1/3] 安装 samba 服务..."
sudo apt update -y
sudo apt install -y samba

echo "[2/3] 配置共享目录 /home/orangepi ..."
SHARE_CONF="
[OrangePi-Home]
   comment = OrangePi 5 Pro Home Directory
   path = /home/orangepi
   browseable = yes
   writable = yes
   create mask = 0775
   directory mask = 0775
   valid users = orangepi
"

# 检查是否已追加配置
if ! grep -q "\[OrangePi-Home\]" /etc/samba/smb.conf; then
    echo "$SHARE_CONF" | sudo tee -a /etc/samba/smb.conf
fi

echo "[3/3] 设置共享密码 (默认密码: orangepi)..."
(echo "orangepi"; echo "orangepi") | sudo smbpasswd -a orangepi

sudo systemctl restart smbd
echo "=========================================================="
echo "Samba 共享配置成功！"
echo "在 Windows 电脑上按 Win + R，输入: \\\\$(hostname -I | awk '{print $1}')"
echo "用户名: orangepi，密码: orangepi"
echo "就可以直接把板子当 Windows 文件夹使用！"
echo "=========================================================="
