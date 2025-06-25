import torch
import torch.nn as nn
import torch.nn.functional as F
from models.uiqa import UIQA
from models.maa import MAA


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.stage1 = nn.Sequential(BasicBlock(in_channels, 32), nn.MaxPool2d(2))  # 1/2
        self.stage2 = nn.Sequential(BasicBlock(32, 64), nn.MaxPool2d(2))  # 1/4
        self.stage3 = nn.Sequential(BasicBlock(64, 128), nn.MaxPool2d(2))  # 1/8
        self.stage4 = nn.Sequential(BasicBlock(128, 256), nn.MaxPool2d(2))  # 1/16

    def forward(self, x):
        f1 = self.stage1(x)  # 32 x H/2
        f2 = self.stage2(f1)  # 64 x H/4
        f3 = self.stage3(f2)  # 128 x H/8
        f4 = self.stage4(f3)  # 256 x H/16
        return [f1, f2, f3, f4]


class Decoder(nn.Module):
    def __init__(self, feature_channels, out_channels):
        super(Decoder, self).__init__()
        self.maa = MAA(feature_channels, out_channels)
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        x = self.maa(features)
        x = self.final_conv(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.seg = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x, input_shape):
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return self.seg(x)


class UWSegFormer(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):
        super(UWSegFormer, self).__init__()
        self.encoder = Encoder(in_channels)
        self.uiqa_blocks = nn.ModuleList([UIQA(32), UIQA(64), UIQA(128), UIQA(256)])
        self.decoder = Decoder([32, 64, 128, 256], out_channels=128)
        self.seg_head = SegmentationHead(128, num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        features = self.encoder(x)
        features = [uiqa(f) for uiqa, f in zip(self.uiqa_blocks, features)]
        x = self.decoder(features)
        out = self.seg_head(x, input_shape)
        return out
