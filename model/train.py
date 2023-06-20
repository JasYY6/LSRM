import matplotlib.pyplot as plt
import logging
import pdb
import math
import random
from numpy.core.fromnumeric import clip, reshape
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from torch.autograd import Variable as V
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from itertools import repeat
from typing import List

from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, to_2tuple
from utils import do_mixup, interpolate
from model.fuse_attention import Fuse_Attention
from model.mrinet import MRINet
from model.swin_transformer import BasicLayer
from model.swin_transformer import PatchMerging

class Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(self,
                 spec_size=256,  # 输入谱图大小
                 patch_size=4,  # 每个切片的大小
                 patch_stride=(4, 4),  # 频率和时间轴的步长
                 in_chans=1,  # 输入图像通道数，默认是1
                 num_classes=527,  # 分类的数量
                 embed_dim=96,  # 嵌入补丁的维数
                 depths=[2, 2, 6, 2],  # 每层的深度
                 num_heads=[4, 8, 16, 32],  # 每层的注意力头数
                 window_size=8,  # 窗口大小，默认是8
                 mlp_ratio=4.,  # 多层感知机的隐层尺寸与嵌入尺寸的比例
                 qkv_bias=True,  # 是否给qkv向量添加可学习的偏差，默认是TRUE
                 qk_scale=None,  # 默认为none，即不设置
                 drop_rate=0.,  # 丢弃速率
                 attn_drop_rate=0.,  # 注意力丢弃率
                 drop_path_rate=0.1,  # 随机深度速率
                 norm_layer=nn.LayerNorm,  # 规范化层，默认使用nn.LayerNorm
                 ape=False,  # 如果为True，将绝对位置嵌入添加到补丁嵌入。默认值：False
                 patch_norm=True,  # 如果为True，则在嵌片后添加规格化。默认值：True
                 use_checkpoint=False,  # 是否从保存点开始
                 norm_before_mlp='ln',
                 config=None, **kwargs):  # config.py

        super(Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = None
        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins  # 输入谱图大小/梅尔滤波器数
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio 下采样率

        # Spectrogram extractor 语谱图提取器
        # n_fft为FFT的长度，默认为400，会生成n_fft // 2 + 1个bins
        # win_length为窗口的长度，默认为n_fft
        # hop_length为相邻两个滑动窗口帧之间的距离，即帧移，默认为win_length // 2
        # pad为对输入信号两边的补零长度，默认为0
        # window_fn为窗函数，默认为torch.hann_window
        # power为语谱图的幂指数，默认为2.0，值必须大于0，取1代表能量，取2代表功率。
        # normalized为是否对语谱图进行归一化，默认为False
        # wkwargs为窗函数的参数，默认为None
        # center为是否对输入tensor在两端进行padding，使得第t帧是以t × hop_length为中心的，默认为True，如果为False，则第t帧以t × hop_length开始。
        # pad_mode为补零方式，默认为reflect
        # onesided为是否只计算一侧的语谱图，默认为True，即返回单侧的语谱图，如果为False，则返回全部的语谱图。
        # return_complex不再使用了
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size,
                                                 hop_length=config.hop_size,
                                                 win_length=config.window_size,
                                                 window=window,
                                                 center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor 对数梅尔语谱图提取器
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate,
                                                 n_fft=config.window_size,
                                                 n_mels=config.mel_bins,
                                                 fmin=config.fmin,
                                                 fmax=config.fmax,
                                                 ref=ref,
                                                 amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter SpecAugment是对数梅尔语谱图层面上的数据增强方法
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)  # 2 2

        # 对输入batch的每一个特征通道进行normalize
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)

        # split spctrogram into non-overlapping patches  将谱图拆分为非重叠补丁
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size,  # 输入谱图大小
            patch_size=self.patch_size,  # patch-size?
            in_chans=self.in_chans,  # 输入图像通道数，默认是1
            embed_dim=self.embed_dim,  # 嵌入补丁的维数
            norm_layer=self.norm_layer,  # 规范化层，默认使用nn.LayerNorm
            patch_stride=patch_stride)  # 频率和时间轴的步长(4,4)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding 绝对位置嵌入，默认是false
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # dropout机制，使用torch.nn.Dropout
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth 随机深度
        dpr = [x.item() for x in
               torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers 构建各层
        # 输入 x(1,4096,96)
        # 输出 x(1,64,768) attn(1,32,64,64)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),  # 输入维数（嵌入补丁维数*2）^层数
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                # 输入分辨率
                depth=self.depths[i_layer],  # 块的数量[2, 2, 6, 2]
                num_heads=self.num_heads[i_layer],  # 注意力的头数[4, 8, 16, 32]
                window_size=self.window_size,  # 窗口大小
                mlp_ratio=self.mlp_ratio,  # mlp隐藏尺寸与嵌入尺寸的比率
                qkv_bias=self.qkv_bias,  # 是否给qkv向量添加可学习的偏差，默认是TRUE
                qk_scale=self.qk_scale,  # 默认为none，即不设置
                drop=self.drop_rate,  # 丢弃速率?
                attn_drop=self.attn_drop_rate,  # 注意力丢弃率?
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,  # 规范化层
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # 下采样
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp)  # 值为'ln'
            self.layers.append(layer)

        # 规范化层，默认使用nn.LayerNorm
        self.norm = self.norm_layer(self.num_features)
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 最大池化层
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # 频域特征
        # self.frequency_layer = ResNet34()  # 创建ResNet残差网络结构的模型的实例  (1)  Resnet34
        self.frequency_layer = MRINet(nblocks=[6, 8, 12, 10], basic_channel=64, ratio=2)
        self.norm_frequency = self.norm_layer(32)
        self.time_to_frequency = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, padding=1, stride=1)

        # 注意力
        self.fuse_attention = Fuse_Attention(features=512, M=2, radio=16)

        # 语义标记模块(1,64,768)
        if self.config.enable_tscam:
            SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio  # SF=2
            # 卷积层
            # self.tscam_conv = nn.Conv2d(
            #     in_channels=self.num_features,
            #     out_channels=self.num_classes,
            #     kernel_size=(SF, 3),
            #     padding=(0, 1)
            # )
            self.tscam_conv1 = nn.Conv2d(
                in_channels=512,
                out_channels=128,
                kernel_size=(2, 3),
                padding=(0, 1)
            )
            self.tscam_conv2 = nn.Conv2d(
                in_channels=128,
                out_channels=self.num_classes,
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=2
            )
            # 线性层
            #self.head = nn.Linear(num_classes, num_classes)
        else:
            # MLP
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.MLP = nn.Sequential(
                nn.Linear(32768, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.ReLU()
            )

        # 将一个函数fn（_init_weights）递归地应用到模块自身以及该模块的每一个子模块(即在函数.children()中返回的子模块).该方法通常用来初始化一个模型中的参数
        # 返回的是初始化参数后的自身
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 如果当前层是全连接层
            trunc_normal_(m.weight, std=.02)  # 使用标准差为0.02的正态分布初始化m的权重
            if isinstance(m, nn.Linear) and m.bias is not None:  # 如果m的偏置不是空则将其填充为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # 如果当前层是规范化层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size, :]
        return tx

    # 将wav转成img，（1,1,1001,64）到（1,1,256,256），以适应swin-transformer的输入
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape  # (1,1,1001,64)
        target_T = int(self.spec_size * self.freq_ratio)  # 256*(256/64)=1024
        target_F = self.spec_size // self.freq_ratio  # 256/(256/64)=64
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        # nn.functional.interpolate上采样填充数据
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        # reshape
        # x(1,1,1024,64)
        x = x.permute(0, 1, 3, 2).contiguous()
        # x(1,1,64,1024)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        # x(1,1,64,4,256)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        # x(1,1,4,64,256)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        # (1,1,256,256)
        return x

    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()  # B C F T
        x = x[:, :, :, cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats=(1, 1, 4, 1))
        return x

    def forward_features(self, x, output_frequency):
        # x(1,1,256,256)
        # output_resnet(1,512,32,2)

        frames_num = x.shape[2]
        x = self.patch_embed(x)  # 卷积后展平

        # x(1,4096,96)
        if self.ape:  # 默认为false，采用补丁嵌入
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)  # dropout
        # x(1,4096,96)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
        output_time = x
        # x(1,64,768)
        # attn(1,32,64,64)
        # output_resnet(1,768,32,2)

        # 语义标记模块
        if self.config.enable_tscam:

            # for output_time
            output_time = self.norm(output_time)
            B, N, C = output_time.shape  # output_time(1,64,768)
            # 256/(2^(4-1))/4
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]  # 8
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]  # 8
            output_time = output_time.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)  # 调整维度分布
            # output_time(1,768,8,8)
            B, C, F, T = output_time.shape  # output_time(1,768,8,8)
            # group 2D CNN
            c_freq_bin = F // self.freq_ratio  # 8/(256/64)=2
            output_time = output_time.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            # output_time(1,768,4,2,8)
            output_time = output_time.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            # output_time(1,768,2,32)

            # get latent_output 对output_time做reshape
            latent_output = self.avgpool(torch.flatten(output_time, 2))  # output_time(1,768,64)做平均池化
            latent_output = torch.flatten(latent_output, 1)

            # for mssnet
            output_frequency = rearrange(output_frequency, 'b c w h -> b c h w')
            output_frequency = self.norm_frequency(output_frequency)
            # output_frequency(1,512,2,32)
            output_time = self.time_to_frequency(output_time)

            # 分支注意力
            output_attention = self.fuse_attention(output_frequency, output_time)

            # 语义分类
            output_attention = self.tscam_conv1(output_attention)
            # output_attention(1,128,1,32)
            output_attention = self.tscam_conv2(output_attention)
            # output_attention(1,4,1,16)
            output_attention = torch.flatten(output_attention, 2)  # B, C, T
            # output_attention(1,4,16)


            # y = output_attention.clone().cpu()
            # y = y[0]
            # plt.subplots(figsize=(16, 4))
            # axes = plt.gca()
            # plt.axis('off')
            # axes.imshow(y.detach().numpy(), aspect='auto', origin='lower')
            # plt.savefig(r"C:\Users\11638\Desktop\论文图\材料\语义图\3语义图.png", bbox_inches="tight", pad_inches=0)


            if self.config.htsat_attn_heatmap:
                fpx = interpolate(torch.sigmoid(output_attention).permute(0, 2, 1).contiguous() * attn,
                                  8 * self.patch_stride[1])
            else:
                fpx = interpolate(torch.sigmoid(output_attention).permute(0, 2, 1).contiguous(),
                                  8 * self.patch_stride[1])

            output_attention = self.avgpool(output_attention)
            # output_attention(1,4,1)
            output_attention = torch.flatten(output_attention, 1)
            # output_attention(1,4)


            # y = output_attention.clone().cpu()
            # y = y[0]
            # y = y.unsqueeze(dim=0)
            # plt.subplots(figsize=(16, 4))
            # axes = plt.gca()
            # plt.axis('off')
            # axes.imshow(y.detach().numpy(), aspect='auto', origin='lower')
            # plt.savefig(r"C:\Users\11638\Desktop\论文图\材料\语义图\4标签映射图.png", bbox_inches="tight", pad_inches=0)



            if self.config.loss_type == "clip_ce":
                output_dict = {
                    'framewise_output': fpx,  # already sigmoided
                    'clipwise_output': output_attention,
                    'latent_output': latent_output
                }
            else:
                output_dict = {
                    'framewise_output': fpx,  # already sigmoided
                    'clipwise_output': torch.sigmoid(output_attention),
                    'latent_output': latent_output
                }

        else:
            output_time = self.norm(output_time)  # B N C
            # output_time = self.norm_block(output_time)  # B N C

            B, N, C = output_time.shape
            fpx = output_time.permute(0, 2, 1).contiguous().reshape(B, C,
                                                                           frames_num // (2 ** (len(self.depths) + 1)),
                                                                           frames_num // (2 ** (len(self.depths) + 1)))
            B, C, F, T = fpx.shape
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim=2)
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            output_time = self.avgpool(output_time.transpose(1, 2))  # B C 1

            output_time = torch.flatten(output_time, 1)
            # output_time(1,32768)
            if self.num_classes > 0:
                # output_time = self.head(output_time)
                fpx = self.head(fpx)
                output_time = self.MLP(output_time)

            output_dict = {
                'framewise_output': torch.sigmoid(fpx),
                'clipwise_output': torch.sigmoid(output_time)}

        return output_dict

    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False):  # out_feat_keys: List[str] = None):
        # x(1,320000)
        x = self.spectrogram_extractor(x)  # 输出格式(batch_size, 1, time_steps, freq_bins)
        # x(1,1,1001,513)
        x = self.logmel_extractor(x)  # 输出格式(batch_size, 1, time_steps, mel_bins)

        # y = x.clone().cpu()
        # y = y[0].squeeze(0)
        # y = y.transpose(0, 1)  # 1 3维交换
        # plt.subplots(figsize=(10, 5))
        # axes = plt.gca()
        # plt.axis('off')
        # axes.imshow(y, aspect='auto', origin='lower')
        # plt.savefig(r"C:\Users\11638\Desktop\论文图\材料\语义图\1梅尔语谱图.png", bbox_inches="tight", pad_inches=0)



        # x(1,1,1001,64)
        x = x.transpose(1, 3)  # 1 3维交换
        # x(1,64,1001,1)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # x(1,1,1001,64)
        # 数据增强
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # y = x.clone().cpu()
        # y = y[0].squeeze(0)
        # y = y.transpose(0, 1)  # 1 3维交换
        # plt.subplots(figsize=(10, 5))
        # axes = plt.gca()
        # plt.axis('off')
        # axes.imshow(y.detach().numpy(), aspect='auto', origin='lower')
        # plt.savefig(r"C:\Users\11638\Desktop\论文图\材料\语义图\2增强梅尔语谱图.png", bbox_inches="tight", pad_inches=0)


        # x(1,1,1001,64)
        output_frequency = self.frequency_layer(x)
        # output_resnet(1,768,32,2)

        x = self.reshape_wav2img(x)
        # x(1,1,256,256)

        output_dict = self.forward_features(x, output_frequency)

        return output_dict
