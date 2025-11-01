import torch
import torch.nn as nn

# class IAChannelAttention(nn.Module):
#    """
#    通道注意力模块
#    使用全局平均池化和全局最大池化，结合MLP来学习通道权重
#    """
#    def __init__(self, channels, reduction=16):
#        super(IAChannelAttention, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#        # 共享MLP
#        self.mlp = nn.Sequential(
#            nn.Conv2d(channels, channels // reduction, 1, bias=False),
#            nn.GELU(),
#            nn.Conv2d(channels // reduction, channels, 1, bias=False)
#        )
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        avg_out = self.mlp(self.avg_pool(x))
#        max_out = self.mlp(self.max_pool(x))
#        attention = self.sigmoid(avg_out + max_out)
#        return x * attention


class IAChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4, hidden_dim=128):
        super().__init__()
        self.channels = channels
        self.compress = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
        self.extractor = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels // 4, 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(40),
        )
        self.SpatialMLP = nn.Sequential(
            nn.Linear(40 * 40, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, channels)
        )
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _H, _W = x.shape
        spatial_feat = self.extractor(x)  # (B,1,H,W)
        spatial_flat = spatial_feat.view(B, -1)  # (B, 8)
        sap = self.SpatialMLP(spatial_flat).view(B, C, 1, 1)  # (B,C,1,1)

        v = x.view(B, C, -1).var(dim=-1, keepdim=True)
        var = v.view(B, C)
        # Channel Self-Attention
        q, k, v = self.qkv(var).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (C**0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        var = self.proj(out).view(B, C, 1, 1)

        fused = self.compress(torch.cat([var, sap], dim=1))

        score = self.mlp(fused)
        attention = self.sigmoid(score)
        return x * attention


# 最新版
# class IAChannelAttention(nn.Module):
#    def __init__(self, channels, reduction=4):
#        super().__init__()
#        self.channels = channels
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.CompressMlp = nn.Sequential(
#            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#            nn.BatchNorm2d(channels),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mlp = nn.Sequential(
#            nn.Conv2d(channels, channels // reduction, 1, bias=True),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(channels // reduction, channels, 1, bias=True),
#        )
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        B, C, H, W = x.shape
#        v = x.view(B, C, -1).var(dim=-1, keepdim=True)  # (C)
#        var = v.view(B, C, 1, 1)
#        avg = self.avg_pool(x)
#        out = torch.cat([var, avg], dim=1)
#
#        fustion = self.CompressMlp(out)
#        score = self.mlp(fustion)
#        attention = self.sigmoid(score)
#
##        return x + x * attention  # 新版直接乘不加
#        return x * attention

# class IAChannelAttention(nn.Module):
#    def __init__(self, channels, reduction=4, hidden_dim=128):
#        super().__init__()
#        self.channels = channels
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        # spatial information
#        self.extractor = nn.Sequential(
#            nn.Conv2d(channels, channels // 4, 1),
#            nn.Conv2d(channels // 4, 4, 3, stride=2, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(4, 1, 3, stride=2, padding=1),
#            nn.ReLU(inplace=True),
#            nn.AdaptiveAvgPool2d(40)
#        )
#        self.SpatialMLP = nn.Sequential(
#            nn.Linear(40 * 40, hidden_dim),
#            nn.ReLU(inplace=True),
#            nn.Linear(hidden_dim, channels)
#        )
#
#        self.CompressMlp = nn.Sequential(
#            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
#            nn.BatchNorm2d(channels),
#            nn.ReLU(inplace=True)
#        )
#
#        # MLP 仅做轻微微调
#        self.mlp = nn.Sequential(
#            nn.Conv2d(channels, channels // reduction, 1, bias=True),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(channels // reduction, channels, 1, bias=True),
#        )
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        B, C, H, W = x.shape
##        avg_out = torch.mean(x, dim=1, keepdim=True)
#        spatial_feat = self.extractor(x)         # (B,1,H,W)
#        spatial_flat = spatial_feat.view(B, -1)        # (B, 8)
#        sap = self.SpatialMLP(spatial_flat).view(B, C, 1, 1)  # (B,C,1,1)
#
#        v = x.view(B, C, -1).var(dim=-1, keepdim=True)  # (C)
#        var = v.view(B, C, 1, 1)
#
#        avg = self.avg_pool(x)
#        out = torch.cat([var, avg, sap], dim=1)
#
#        fustion = self.CompressMlp(out)
#        score = self.mlp(fustion)
#        attention = self.sigmoid(score)
#
#        return x * attention

# class SpatialAttention(nn.Module):
#    """
#    方差驱动的空间注意力模块（使用 gather 实现）
#    - 选取方差最大的 top_ratio 通道
#    - 对这些通道求平均 -> Conv2d(1,1,kernel_size)
#    """
#    def __init__(self, kernel_size=7, top_ratio=0.2):
#        super(SpatialAttention, self).__init__()
#        self.top_ratio = top_ratio
#        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        B, C, H, W = x.shape
#
#        var = x.view(B, C, -1).var(dim=-1)  # (B, C)
#
#        k = max(1, int(C * self.top_ratio))
#        topk_idx = torch.topk(var, k, dim=1).indices  # (B, k)
#
#        topk_idx_expanded = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, k, H, W)
#        x_selected = torch.gather(x, dim=1, index=topk_idx_expanded)  # (B, k, H, W)
#
#        pooled = x_selected.mean(dim=1, keepdim=True)  # (B, 1, H, W)
#
#        attention = 0.5 + self.sigmoid(self.conv(pooled))  # (B, 1, H, W)
#
#        out = x * attention
#        return out

# class SpatialAttention(nn.Module):
#    """
#    空间注意力模块
#    使用通道维度的平均和最大值来生成空间注意力图
#    """
#    def __init__(self, kernel_size=7):
#        super(SpatialAttention, self).__init__()
#        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        avg_out = torch.mean(x, dim=1, keepdim=True)
#        max_out, _ = torch.max(x, dim=1, keepdim=True)
#        attention_map = torch.cat([avg_out, max_out], dim=1)
#        attention = self.sigmoid(self.conv(attention_map))
#        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):  # , kernel_size=7
        super().__init__()
        self.channel_attention = IAChannelAttention(channels, reduction)

    #        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        #        x = self.spatial_attention(x)
        return x


# from torch.cuda.amp import autocast
# class TopKChannelSelector(nn.Module):
#    """
#    使用 Top-K + STE 的通道硬选择
#    输入为通道注意力权重 (B, C, 1, 1)
#    """
#    def __init__(self, k_ratio=0.6667):  # 默认选 2/3 通道
#        super().__init__()
#        self.k_ratio = k_ratio
#
#    def forward(self, scores):  # scores from CA: (B,C,1,1)
#        B, C, _, _ = scores.shape
#        k = max(1, int(C * self.k_ratio))  # 选前 k 个通道
#
#        flat_scores = scores.view(B, C)
#
#        # Top-K indices
#        _, idx = torch.topk(flat_scores, k, dim=1)
#
#        # 构造 hard one-hot mask（不可微）
#        hard_mask = torch.zeros_like(flat_scores)
#        hard_mask.scatter_(1, idx, 1.0)
#
#        hard_mask = hard_mask.view(B, C, 1, 1)
#
#        # STE: forward 采用 hard mask，backward 采用 soft scores
#        soft_mask = scores.sigmoid()
#        mask = hard_mask + (soft_mask - soft_mask.detach())
#
#        return mask  # (B,C,1,1)
#
#
# class ChannelAttention(nn.Module):
#    """
#    通道注意力 + Top-K STE Channel Selector
#    """
#    def __init__(self, channels, reduction=16, k_ratio=0.6667):
#        super(ChannelAttention, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#        self.mlp = nn.Sequential(
#            nn.Conv2d(channels, channels // reduction, 1, bias=False),
#            nn.GELU(),
#            nn.Conv2d(channels // reduction, channels, 1, bias=False)
#        )
#
#        # ⚡ 加入 Top-K 通道选取
#        self.selector = TopKChannelSelector(k_ratio)
#
#    def forward(self, x):
#        avg_out = self.mlp(self.avg_pool(x))
#        max_out = self.mlp(self.max_pool(x))
#        scores = avg_out + max_out  # 尚未 sigmoid
#
#        # ✅ Top-K 通道选择 + 可微近似
#        mask = self.selector(scores)
#
#        return x * mask
#
#
# class SpatialAttention(nn.Module):
#    def __init__(self, kernel_size=7):
#        super(SpatialAttention, self).__init__()
#        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        avg_out = torch.mean(x, dim=1, keepdim=True)
#        max_out, _ = torch.max(x, dim=1, keepdim=True)
#        attention_map = torch.cat([avg_out, max_out], dim=1)
#        attention = self.sigmoid(self.conv(attention_map))
#        return x * attention
#
#
# class CBAM(nn.Module):
#    """
#    ⚡增强版 CBAM：通道注意力 + Top-K Mask + 空间注意力
#    """
#    def __init__(self, channels, reduction=16, kernel_size=7, k_ratio=0.6667):
#        super(CBAM, self).__init__()
#        self.channel_attention = ChannelAttention(channels, reduction, k_ratio)
#        self.spatial_attention = SpatialAttention(kernel_size)
#
#    def forward(self, x):
#        x = self.channel_attention(x)
#        x = self.spatial_attention(x)
#        return x

# class CBAM(nn.Module):
#    """Channel attention with Gumbel-Softmax for training and top-k hard selection for inference.
#
#    Args:
#        channels (int): Input channels (e.g., 192).
#        k (int): Number of channels to select (e.g., 128).
#        reduction (int): Reduction ratio for MLP.
#        temperature (float): Gumbel-Softmax temperature.
#    """
#    def __init__(self, channels, k=128, reduction=16, temperature=1.0):
#        super(CBAM, self).__init__()
#        self.channels = channels
#        self.k = k
#        self.temperature = temperature
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.max_pool = nn.AdaptiveMaxPool2d(1)
#        self.mlp = nn.Sequential(
#            nn.Conv2d(channels, channels // reduction, 1, bias=False),
#            nn.GELU(),
#            nn.Conv2d(channels // reduction, channels, 1, bias=False)
#        )
#        self.sigmoid = nn.Sigmoid()
#
#    def gumbel_softmax(self, logits, temperature, hard=False):
#        with autocast(enabled=False):  # Disable autocast for Gumbel-Softmax to ensure FP32 stability
#            gumbels = -torch.empty_like(logits).exponential_().log()
#            logits = (logits + gumbels) / temperature
#            y_soft = F.softmax(logits, dim=1)
#            if hard:
#                _, indices = torch.topk(logits, self.k, dim=1)
#                y_hard = torch.zeros_like(logits).scatter_(1, indices, 1.0)
#                y = y_hard - y_soft.detach() + y_soft
#            else:
#                y = y_soft
#            return y
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        avg_out = self.mlp(self.avg_pool(x))
#        max_out = self.mlp(self.max_pool(x))
#        scores = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
#
#        with autocast(enabled=False):  # Ensure FP32 for top-k and scatter operations
#            if self.training:
#                weights = self.gumbel_softmax(scores.squeeze(-1).squeeze(-1), self.temperature, hard=True)
#            else:
#                _, indices = torch.topk(scores.squeeze(-1).squeeze(-1), self.k, dim=1)
#                weights = torch.zeros(b, c, device=x.device, dtype=x.dtype).scatter_(1, indices, 1.0)
#
#        weights = weights.view(b, c, 1, 1)
#        return x * weights
