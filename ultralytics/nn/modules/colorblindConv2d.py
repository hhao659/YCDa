import torch
import torch.nn as nn


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
