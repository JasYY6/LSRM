import torch
import torch.nn as nn
import torch.nn.functional as F


class MRIBlock(nn.Module):
    def __init__(self, channel):
        super(MRIBlock, self).__init__()
        self.channel = channel
        # 卷积
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size=1)

        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        self.bn_res1 = nn.BatchNorm2d(channel * 2)
        self.relu_res1 = nn.ReLU(inplace=True)
        self.conv_res1 = nn.Conv2d(channel * 2, channel, kernel_size=1)

        # 空洞卷积
        self.bn3 = nn.BatchNorm2d(channel * 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channel * 2, channel, kernel_size=1)

        self.bn4 = nn.BatchNorm2d(channel)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x = torch.cat((x1, x2), dim=1)

        # 普通卷积
        x_res = self.conv_res1(self.relu_res1(self.bn_res1(x)))
        x3_temp = self.conv1(self.relu1(self.bn1(x)))  # 2channel->channel
        x3 = self.conv2(self.relu2(self.bn2(x3_temp)))  # channel->channel

        x_temp = torch.cat((x2, x3), dim=1)
        # 空洞卷积
        x4_temp = self.conv3(self.relu3(self.bn3(x_temp)))
        x4 = self.conv4(self.relu4(self.bn4(x4_temp)))
        x4 += x_res

        return (x3, x4)


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):  # 512 256
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class MRINet(nn.Module):
    def __init__(self, nblocks, basic_channel, ratio):  # [6 12 24 16]   64   2
        super(MRINet, self).__init__()

        self.basic_conv = nn.Sequential(  # 初始化卷积层
            nn.Conv2d(1, basic_channel, kernel_size=7, stride=2, padding=(5, 3)),
            nn.BatchNorm2d(basic_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.mri1 = self._make_layers(nblocks[0], basic_channel)
        self.trans1 = Transition(basic_channel, basic_channel * (ratio ** 1))  # 64 128

        self.mri2 = self._make_layers(nblocks[1], basic_channel * (ratio ** 1))
        self.trans2 = Transition(basic_channel * (ratio ** 1), basic_channel * (ratio ** 2))  # 128 256

        self.mri3 = self._make_layers(nblocks[2], basic_channel * (ratio ** 2))
        self.trans3 = Transition(basic_channel * (ratio ** 2), basic_channel * (ratio ** 3))  # 256 512

        self.mri4 = self._make_layers(nblocks[3], basic_channel * (ratio ** 3))

    def _make_layers(self, nblock, channel):
        layers = []
        for i in range(nblock):
            layers.append(MRIBlock(channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x(1,1,1001,64)
        B, C, T, F = x.shape  # (1,1,1001,64)
        target_T = 1024
        target_F = 64
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        # x(1,1,1024,64)
        out = self.basic_conv(x)
        # out(1,64,257,16)
        out = (out, out)
        out = self.mri1(out)
        out1 = self.trans1(out[0])
        out2 = self.trans1(out[1])

        out = (out1, out2)
        out = self.mri2(out)
        out1 = self.trans2(out[0])
        out2 = self.trans2(out[1])

        out = (out1, out2)
        out = self.mri3(out)
        out1 = self.trans3(out[0])
        out2 = self.trans3(out[1])

        out = (out1, out2)
        out = self.mri4(out)

        return out[1]
