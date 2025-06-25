# models/uiqa.py
import torch
import torch.nn as nn


class UIQA(nn.Module):
    def __init__(self, channels):
        super(UIQA, self).__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)
