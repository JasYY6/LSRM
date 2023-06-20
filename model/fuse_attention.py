import torch
from torch import nn


class Fuse_Attention(nn.Module):
    def __init__(self, features, M, radio, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.输入通道数
            M: the number of branchs.分支数
            radio: the radio for compute d, the length of z. 计算z的长度
            stride: stride, default 1. 步长，默认1
            L: the minimum dim of the vector z in paper, default 32.论文中z的最小尺寸，默认为32
        """
        super(Fuse_Attention, self).__init__()
        d = max(int(features / radio), L)
        self.M = M  # 分支数
        self.features = features

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=False)
        )
        self.conv1ds = nn.ModuleList([])
        for i in range(M):
            self.conv1ds.append(
                nn.Conv1d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1),
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, output_resnet, output_transformer):

        output_fuse = output_resnet + output_transformer
        output_fuse = output_fuse.mean(-1).mean(-1)

        output_fuse = output_fuse.unsqueeze_(dim=2)
        output_fuse = self.conv1d(output_fuse)

        global attentions
        for i, conv1d in enumerate(self.conv1ds):
            attention = conv1d(output_fuse).unsqueeze_(dim=1).squeeze_(dim=3)
            if i == 0:
                attentions = attention
            else:
                attentions = torch.cat([attentions, attention], dim=1)

        attentions = self.softmax(attentions)
        attentions = attentions.unsqueeze(-1).unsqueeze(-1)
        output_resnet = output_resnet.unsqueeze(1)
        output_transformer = output_transformer.unsqueeze(1)
        output_attention = torch.cat((output_resnet, output_transformer), dim=1)
        output_attention = (output_attention * attentions).sum(dim=1)
        return output_attention
