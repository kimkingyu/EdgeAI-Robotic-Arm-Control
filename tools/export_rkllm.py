#!/usr/bin/env python3
"""
面向 RK3588 的 Qwen2.5 大语言模型 W4A16 极限压缩与转换工具
使用方法:
  python tools/export_rkllm.py --model-dir ./Qwen2.5-0.5B-Instruct --output ./models/weights/qwen2.5_w4a16.rkllm
"""
import argparse
import sys
import os

try:
    from rkllm.api import RKLLM
    HAS_RKLLM_TOOLKIT = True
except ImportError:
    HAS_RKLLM_TOOLKIT = False


def main():
    parser = argparse.ArgumentParser(description="Qwen 大模型 RKLLM W4A16 极限压缩转换工具")
    parser.add_argument("--model-dir", default="./Qwen2.5-0.5B-Instruct", help="开源大模型原始权重目录")
    parser.add_argument("--output", default="models/weights/qwen2.5_0.5b_w4a16.rkllm", help="导出的 .rkllm 文件路径")
    parser.add_argument("--quant-type", default="w4a16", choices=["w4a16", "w8a16"], help="权重量化类型 (W4A16 压缩比更高)")
    parser.add_argument("--target-platform", default="rk3588", help="目标芯片平台 (rk3588)")
    parser.add_argument("--max-context", type=int, default=1024, help="最大支持上下文长度")
    args = parser.parse_args()

    print("=" * 60)
    print("  工业大模型端侧部署: Qwen2.5 RKLLM W4A16 极限压缩转换")
    print(f"  量化精度: {args.quant_type.upper()} | 目标平台: {args.target_platform}")
    print("=" * 60)

    if not HAS_RKLLM_TOOLKIT:
        print("[警告] 本机环境未检测到 rkllm-toolkit 安装包。")
        print("提示: 模型转换通常在 x86 PC (Ubuntu 20.04/22.04) 上完成，安装命令:")
        print("      pip3 install rkllm-toolkit")
        print("\n生成模拟量化配置文件骨架...")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output + ".meta", "w", encoding="utf-8") as f:
            f.write(f"model_type: qwen2\nquant_type: {args.quant_type}\ntarget: {args.target_platform}\nmax_context: {args.max_context}\n")
        print(f"元数据已写入: {args.output}.meta")
        return

    rkllm = RKLLM()
    # 1. 载入原始 HuggingFace 格式模型
    print(f"[1/3] 解析模型结构: {args.model_dir} ...")
    ret = rkllm.load_huggingface(model_dir=args.model_dir, model_type="qwen2")
    if ret != 0:
        print("[错误] 载入 HuggingFace 模型失败！")
        sys.exit(1)

    # 2. 执行 W4A16 / W8A16 混合量化与计算图编译
    print(f"[2/3] 执行 {args.quant_type.upper()} 硬件级量化与核融合...")
    dataset_path = "data/calibration/industrial_qa_calib.jsonl" if os.path.exists("data/calibration/industrial_qa_calib.jsonl") else None
    
    ret = rkllm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=args.quant_type,
        quantized_algorithm="normal",
        target_platform=args.target_platform,
        num_npu_core=3,
        dataset=dataset_path
    )
    if ret != 0:
        print("[错误] 构建编译失败！")
        sys.exit(1)

    # 3. 导出 .rkllm 模型权重
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"[3/3] 导出端侧部署模型至: {args.output} ...")
    ret = rkllm.export_rkllm(args.output)
    if ret == 0:
        print("量化转换圆满成功！可在 RK3588 上全速跑动！")
    else:
        print("[错误] 导出失败！")


if __name__ == "__main__":
    main()
