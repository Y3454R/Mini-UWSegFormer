# models/maa.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAA(nn.Module):
    def __init__(self, channel_list, out_channels=128):
        super(MAA, self).__init__()
        self.conv = nn.Conv2d(sum(channel_list), out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        base_size = features[0].shape[2:]
        upsampled = [
            F.interpolate(f, size=base_size, mode="bilinear", align_corners=False)
            for f in features
        ]
        fused = torch.cat(upsampled, dim=1)
        return self.conv(fused)
