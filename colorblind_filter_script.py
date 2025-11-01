#!/usr/bin/env python3
"""
色盲滤镜转换脚本
基于ColorBlindTransform模块实现的专业色盲模拟工具.

使用方法:
    python colorblind_filter_script.py --input image.jpg --output result.jpg --mode protanopia
    python colorblind_filter_script.py --input image.jpg --mode deuteranopia --output deuteranopia_result.jpg
    python colorblind_filter_script.py --input image.jpg --mode tritanopia
    python colorblind_filter_script.py --input image.jpg --mode achromatopsia
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ColorBlindTransform(nn.Module):
    """色盲滤镜层，基于1×1卷积实现."""

    def __init__(self, mode="protanopia"):
        super().__init__()
        self.mode = mode
        # 1x1 卷积，相当于对通道做线性变换
        self.transform = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_matrix()

    def _init_matrix(self):
        # 常见色盲模式转换矩阵
        matrices = {
            "protanopia": torch.tensor([[0.567, 0.433, 0.000], [0.558, 0.442, 0.000], [0.000, 0.242, 0.758]]),
            "deuteranopia": torch.tensor([[0.625, 0.375, 0.000], [0.700, 0.300, 0.000], [0.000, 0.300, 0.700]]),
            "tritanopia": torch.tensor([[0.950, 0.050, 0.000], [0.000, 0.433, 0.567], [0.000, 0.475, 0.525]]),
            "achromatopsia": torch.tensor([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]]),
        }

        M = matrices.get(self.mode, matrices["protanopia"])
        # Conv2d 权重 shape = (out_channels, in_channels, 1, 1)
        self.transform.weight.data = M.view(3, 3, 1, 1)
        self.transform.weight.requires_grad_(False)

    def forward(self, x):
        return self.transform(x)


def load_image(image_path):
    """
    加载图像并转换为PyTorch张量格式.

    Args:
        image_path (str): 图像文件路径

    Returns:
        torch.Tensor: 形状为 (1, 3, H, W) 的张量，值范围 [0, 1]
    """
    try:
        # 使用PIL加载图像
        image = Image.open(image_path).convert("RGB")
        # 转换为numpy数组
        image_np = np.array(image, dtype=np.float32) / 255.0
        # 转换为PyTorch张量并添加batch维度
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise ValueError(f"无法加载图像 {image_path}: {e}")


def save_image(tensor, output_path):
    """
    将PyTorch张量保存为图像文件.

    Args:
        tensor (torch.Tensor): 形状为 (1, 3, H, W) 的张量，值范围 [0, 1]
        output_path (str): 输出文件路径
    """
    try:
        # 移除batch维度并转换为numpy
        image_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
        # 确保值在[0, 1]范围内
        image_np = np.clip(image_np, 0, 1)
        # 转换为PIL图像并保存
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil.save(output_path)
        print(f"✅ 处理后的图像已保存至: {output_path}")
    except Exception as e:
        raise ValueError(f"保存图像失败 {output_path}: {e}")


def apply_colorblind_filter(input_path, output_path, mode="protanopia"):
    """
    对图像应用色盲滤镜.

    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        mode (str): 色盲类型 ('protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia')
    """
    # 检查输入文件是否存在
    if not Path(input_path).exists():
        raise FileNotFoundError(f"输入图像不存在: {input_path}")

    # 创建输出目录
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 正在处理图像: {input_path}")
    print(f"🎨 应用色盲类型: {mode}")

    # 加载图像
    image_tensor = load_image(input_path)

    # 创建色盲转换器
    colorblind_transform = ColorBlindTransform(mode=mode)

    # 应用转换
    with torch.no_grad():
        transformed_tensor = colorblind_transform(image_tensor)

    # 保存结果
    save_image(transformed_tensor, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="色盲滤镜转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的色盲类型:
  protanopia     红色盲 (红色锥体缺失)
  deuteranopia   绿色盲 (绿色锥体缺失)  
  tritanopia     蓝色盲 (蓝色锥体缺失)
  achromatopsia  全色盲 (所有锥体缺失)

示例用法:
  python colorblind_filter_script.py --input photo.jpg --mode protanopia
  python colorblind_filter_script.py --input photo.jpg --output result.png --mode deuteranopia
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出图像路径 (默认为输入文件名_色盲类型.jpg)")
    parser.add_argument(
        "--mode",
        "-m",
        default="protanopia",
        choices=["protanopia", "deuteranopia", "tritanopia", "achromatopsia"],
        help="色盲类型 (默认: protanopia)",
    )

    args = parser.parse_args()

    # 生成默认输出路径
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_{args.mode}{input_path.suffix}")

    try:
        apply_colorblind_filter(args.input, args.output, args.mode)
        print("🎉 处理完成!")

    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
