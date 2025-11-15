import torch
import torch.nn as nn
import torch.nn.functional as F

class ICA(nn.Module):
   def __init__(self, channels, reduction=4):
       super().__init__()
       self.channels = channels
       self.avg_pool = nn.AdaptiveAvgPool2d(1)
       self.CompressMlp = nn.Sequential(
           nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
           nn.BatchNorm2d(channels),
           nn.ReLU(inplace=True)
       )

       self.mlp = nn.Sequential(
           nn.Conv2d(channels, channels // reduction, 1, bias=True),
           nn.ReLU(inplace=True),
           nn.Conv2d(channels // reduction, channels, 1, bias=True),
       )
       self.sigmoid = nn.Sigmoid()

   def forward(self, x):
       B, C, H, W = x.shape
       v = x.view(B, C, -1).var(dim=-1, keepdim=True)  # (C)
       var = v.view(B, C, 1, 1)
       avg = self.avg_pool(x)
       out = torch.cat([var, avg], dim=1)
       
       fustion = self.CompressMlp(out)
       score = self.mlp(fustion)
       attention = self.sigmoid(score)

       return x * attention
