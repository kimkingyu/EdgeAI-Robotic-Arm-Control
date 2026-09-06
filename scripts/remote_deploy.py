#!/usr/bin/env python3
"""
一键将本地工程同步到香橙派 5 Pro 并远程执行
用法:
  python scripts/remote_deploy.py --ip 192.168.x.x
  python scripts/remote_deploy.py --ip 192.168.x.x --run
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="香橙派 5 Pro 远程代码部署与运行工具")
    parser.add_argument("--ip", required=True, help="香橙派板子的局域网 IP 地址")
    parser.add_argument("--user", default="orangepi", help="登录用户名 (默认 orangepi)")
    parser.add_argument("--remote-dir", default="/home/orangepi/EdgeAI-Robotic-Arm-Control", help="板端目标工程路径")
    parser.add_argument("--run", action="store_true", help="同步完成后在板端执行自检或主程序")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print("=" * 60)
    print(f"  正在部署代码到香橙派: {args.user}@{args.ip}:{args.remote_dir}")
    print("=" * 60)

    # 1. 检查 SSH 连通
    ssh_target = f"{args.user}@{args.ip}"
    print("[1/3] 测试 SSH 连通性...")
    ssh_test_cmd = ["ssh", "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no", ssh_target, "echo 'SSH_OK'"]
    try:
        ret = subprocess.run(ssh_test_cmd, capture_output=True, text=True)
        if "SSH_OK" not in ret.stdout:
            print(f"[错误] 无法通过 SSH 连接到 {ssh_target}。")
            print("提示: 请确认板子已开机连网，且 IP 地址正确。如果是初次连接，在终端手动执行一次: ssh orangepi@{args.ip}")
            sys.exit(1)
        print("  -> SSH 连通正常！")
    except FileNotFoundError:
        print("[警告] 本地未找到 ssh 命令，请确保系统已安装 OpenSSH 客户端。")
        sys.exit(1)

    # 2. 在板端创建目标目录
    print("[2/3] 创建远程目录...")
    mkdir_cmd = ["ssh", ssh_target, f"mkdir -p {args.remote_dir}"]
    subprocess.run(mkdir_cmd, check=True)

    # 3. 同步文件
    print("[3/3] 上传项目核心文件 (src, configs, scripts, main.py)...")
    items_to_sync = ["src", "configs", "scripts", "main.py", "requirements.txt", "README.md"]
    for item in items_to_sync:
        local_path = project_root / item
        if local_path.exists():
            scp_cmd = ["scp", "-r", str(local_path), f"{ssh_target}:{args.remote_dir}/"]
            subprocess.run(scp_cmd, check=True)
    print("  -> 部署完成！")

    if args.run:
        print("\n[远程执行] 正在板端运行环境体检脚本...")
        run_cmd = ["ssh", "-t", ssh_target, f"cd {args.remote_dir} && bash scripts/check_board_env.sh"]
        subprocess.run(run_cmd)


if __name__ == "__main__":
    main()
