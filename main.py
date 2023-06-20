import os
from re import A, S
import sys
import librosa
import numpy as np
import argparse
import h5py
import math
import time
import logging
import pickle
import random
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from utils import create_folder, dump_config, process_idc, prepprocess_audio, init_hier_head

import config
from sed_model import SEDWrapper, Ensemble_SEDWrapper
from data_generator import SEDDataset, DESED_Dataset, ESC_Dataset, SCV2_Dataset

from model.train import Swin_Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")


# 准备数据
class data_prep(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=config.num_workers,  # 多线程
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=eval_sampler
        )
        return eval_loader

    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=config.num_workers,
            batch_size=config.batch_size // self.device_num,
            shuffle=False,
            sampler=test_sampler
        )
        return test_loader


def save_idc():
    train_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", config.index_type + ".h5")
    eval_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", "eval.h5")
    process_idc(train_index_path, config.classes_num, config.index_type + "_idc.npy")
    process_idc(eval_index_path, config.classes_num, "eval_idc.npy")


# 平均权重，提高优化器稳定性（在优化的末期取k个优化轨迹上的checkpoints，平均他们的权重，得到最终的网络权重）
def weight_average():
    model_ckpt = []
    model_files = os.listdir(config.wa_folder)
    wa_ckpt = {
        "state_dict": {}
    }

    for model_file in model_files:
        model_file = os.path.join(config.wa_folder, model_file)
        model_ckpt.append(torch.load(model_file, map_location="cpu")["state_dict"])
    keys = model_ckpt[0].keys()
    for key in keys:
        model_ckpt_key = torch.cat([d[key].float().unsqueeze(0) for d in model_ckpt])
        model_ckpt_key = torch.mean(model_ckpt_key, dim=0)
        assert model_ckpt_key.shape == model_ckpt[0][
            key].shape, "the shape is unmatched " + model_ckpt_key.shape + " " + model_ckpt[0][key].shape
        wa_ckpt["state_dict"][key] = model_ckpt_key
    torch.save(wa_ckpt, config.wa_model_path)


def test():
    device_num = torch.cuda.device_count()
    print("each batch size:", config.batch_size // device_num)
    # dataset file pathes
    if config.fl_local:
        fl_npy = np.load(config.fl_dataset, allow_pickle=True)
        # import dataset SEDDataset
        eval_dataset = DESED_Dataset(
            dataset=fl_npy,
            config=config
        )
    else:
        if config.dataset_type == "audioset":
            eval_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", "eval.h5")
            eval_idc = np.load("eval_idc.npy", allow_pickle=True)
            eval_dataset = SEDDataset(
                index_path=eval_index_path,
                idc=eval_idc,
                config=config,
                eval_mode=True
            )
        elif config.dataset_type == "esc-50":
            full_dataset = np.load(os.path.join(config.dataset_path, "esc-50/esc-50-data.npy"), allow_pickle=True)
            eval_dataset = ESC_Dataset(
                dataset=full_dataset,
                config=config,
                eval_mode=True
            )
        elif config.dataset_type == "scv2":
            test_set = np.load(os.path.join(config.dataset_path, "scv2_test.npy"), allow_pickle=True)
            eval_dataset = SCV2_Dataset(
                dataset=test_set,
                config=config,
                eval_mode=True
            )
        elif config.dataset_type == "LungSound":
            full_dataset = np.load("LungSound/LungSound_data_test.npy", allow_pickle=True)
            eval_dataset = ESC_Dataset(
                dataset=full_dataset,
                config=config,
                eval_mode=True,
                test_mode=True
            )
        # import dataset SEDDataset

    audioset_data = data_prep(eval_dataset, eval_dataset, device_num)
    trainer = pl.Trainer(
        deterministic=True,
        gpus=device_num,
        max_epochs=config.max_epoch,
        auto_lr_find=True,
        sync_batchnorm=True,
        checkpoint_callback=False,
        accelerator="ddp" if device_num > 1 else None,
        num_sanity_val_steps=0,
        resume_from_checkpoint=config.test_checkpoint,
        replace_sampler_ddp=False,
        gradient_clip_val=1.0
    )
    sed_model = Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )

    model = SEDWrapper.load_from_checkpoint(
        checkpoint_path=config.test_checkpoint,
        sed_model=sed_model,
        config=config,
        dataset=eval_dataset
    )

    # if config.test_checkpoint is not None:
    #     ckpt = torch.load(config.test_checkpoint, map_location="cpu")
    #     ckpt["state_dict"].pop("sed_model.head.weight")
    #     ckpt["state_dict"].pop("sed_model.head.bias")
    #     model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer.test(model, datamodule=audioset_data)


def train():
    device_num = torch.cuda.device_count()
    print(device_num)
    print("each batch size:", config.batch_size // device_num)  # //除法，向下取整

    # 加载数据集
    if config.dataset_type == "audioset":
        train_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", config.index_type + ".h5")
        eval_index_path = os.path.join(config.dataset_path, "hdf5s", "indexes", "eval.h5")
        train_idc = np.load(config.index_type + "_idc.npy", allow_pickle=True)
        eval_idc = np.load("eval_idc.npy", allow_pickle=True)
    elif config.dataset_type == "esc-50":
        full_dataset = np.load(os.path.join(config.dataset_path, "esc-50/esc-50-data.npy"), allow_pickle=True)
    elif config.dataset_type == "scv2":
        train_set = np.load(os.path.join(config.dataset_path, "scv2_train.npy"), allow_pickle=True)
        test_set = np.load(os.path.join(config.dataset_path, "scv2_test.npy"), allow_pickle=True)
    elif config.dataset_type == "LungSound":
        full_dataset = np.load(os.path.join(config.dataset_path, "LungSound/LungSound_data.npy"), allow_pickle=True)

    # 创建输出目录
    exp_dir = os.path.join(config.workspace, "results", config.exp_name)
    checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
    if not config.debug:
        create_folder(os.path.join(config.workspace, "results"))
        create_folder(exp_dir)
        create_folder(checkpoint_dir)
        dump_config(config, os.path.join(exp_dir, config.exp_name), False)  # 保存config文件

    # 导入并配置数据集
    if config.dataset_type == "audioset":
        print("Using Audioset")
        dataset = SEDDataset(
            index_path=train_index_path,
            idc=train_idc,
            config=config
        )
        eval_dataset = SEDDataset(
            index_path=eval_index_path,
            idc=eval_idc,
            config=config,
            eval_mode=True
        )
    elif config.dataset_type == "esc-50":
        print("Using ESC")
        # 训练集
        dataset = ESC_Dataset(
            dataset=full_dataset,
            config=config,
            eval_mode=False
        )
        # 验证集
        eval_dataset = ESC_Dataset(
            dataset=full_dataset,
            config=config,
            eval_mode=True
        )
    elif config.dataset_type == "scv2":
        print("Using SCV2")
        dataset = SCV2_Dataset(
            dataset=train_set,
            config=config,
            eval_mode=False
        )
        eval_dataset = SCV2_Dataset(
            dataset=test_set,
            config=config,
            eval_mode=True
        )
    elif config.dataset_type == "LungSound":
        print("Using LungSound")
        # 训练集
        dataset = ESC_Dataset(
            dataset=full_dataset,
            config=config,
            eval_mode=False
        )
        # 验证集
        eval_dataset = ESC_Dataset(
            dataset=full_dataset,
            config=config,
            eval_mode=True
        )

    # 最终数据
    audioset_data = data_prep(dataset, eval_dataset, device_num)

    if config.dataset_type == "audioset":
        checkpoint_callback = ModelCheckpoint(
            monitor="mAP",
            filename='l-{epoch:d}-{mAP:.3f}-{mAUC:.3f}',
            save_top_k=30,
            mode="max",
            save_last=True
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="acc",
            filename='l-{epoch:d}-{acc:.4f}',
            save_top_k=30,
            mode="max",
            save_last=True
        )

    # 配置训练器
    trainer = pl.Trainer(
        deterministic=True,  # 设置PyTorch操作必须使用确定性算法
        default_root_dir=checkpoint_dir,  # 模型保存和日志记录默认根路径
        gpus=device_num,  # 使用的GPU数量
        val_check_interval=1.0,  # 检查验证集的频率。使用float在训练纪元内检查，使用int检查每n个步骤（批次）
        max_epochs=config.max_epoch,  # 最多训练轮数
        auto_lr_find=True,  # 自动搜索最佳学习率并存储到self.lr或self.learing_rate,str参数表示学习率参数的属性名
        sync_batchnorm=True,  # 多GPU参数，用于将子卡同步注册到主卡
        callbacks=[checkpoint_callback],  # 添加回调函数
        accelerator="ddp" if device_num > 1 else None,  # 根据加速器类型传递不同的训练策略（如CPU、GPU等）
        num_sanity_val_steps=0,  # 开始训练前加载n个验证数据进行测试，k=-1时加载所有验证数据
        resume_from_checkpoint=None,  # 恢复训练时的检查点路径
        replace_sampler_ddp=False,
        # 显式启用或禁用采样器替换。如果未指定，则在使用DDP时将自动切换。默认情况下，它将为序列采样器添加``shuffle=True``，为val\/test采样器增加``shaffle=False``。如果要自定义它，可以设置``replace_sampler_ddp=False``并添加自己的分布式采样器。
        gradient_clip_val=1.0  # 要剪裁渐变的值。传递``gradient_clip_val=None``将禁用渐变剪裁。如果使用自动混合精度（AMP），梯度将在之前取消缩放。
    )
    # 配置模型参数
    sed_model = Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )
    print("***********************************************************************")
    total = sum([param.nelement() for param in sed_model.parameters()])
    print("Number of parameter: % .2fM" % (total / 1e6))
    print("***********************************************************************")
    # 加载模型
    model = SEDWrapper(
        sed_model=sed_model,
        config=config,
        dataset=dataset
    )

    # 知识迁移
    if config.resume_checkpoint is not None:  # 加载AudioSet预训练模型，效果最好
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        # finetune on the esc and spv2 dataset
        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif config.swin_pretrain_path is not None:  # 加载swin-transformer在ImageNet上的预训练模型，效果一般
        ckpt = torch.load(config.swin_pretrain_path, map_location="cpu")
        # load pretrain model
        ckpt = ckpt["model"]
        found_parameters = []
        unfound_parameters = []
        model_params = dict(model.state_dict())

        for key in model_params:
            m_key = key.replace("sed_model.", "")
            if m_key in ckpt:
                if m_key == "patch_embed.proj.weight":
                    ckpt[m_key] = torch.mean(ckpt[m_key], dim=1, keepdim=True)
                if m_key == "head.weight" or m_key == "head.bias":
                    ckpt.pop(m_key)
                    unfound_parameters.append(key)
                    continue
                assert model_params[key].shape == ckpt[m_key].shape, "%s is not match, %s vs. %s" % (
                    key, str(model_params[key].shape), str(ckpt[m_key].shape))
                found_parameters.append(key)
                ckpt[key] = ckpt.pop(m_key)
            else:
                unfound_parameters.append(key)
        print("pretrain param num: %d \t wrapper param num: %d" % (len(found_parameters), len(ckpt.keys())))
        print("unfound parameters: ", unfound_parameters)
        model.load_state_dict(ckpt, strict=False)
        model_params = dict(model.named_parameters())

    # 训练
    trainer.fit(model, audioset_data)


def main():
    # argparse是一个Python模块：命令行选项、参数和子命令解析器。argparse模块的作用是用于解析命令行参数。
    parser = argparse.ArgumentParser(description="HTS-AT")
    subparsers = parser.add_subparsers(dest="mode")  # 创建子命令，dest存储输入属性的代名称；默认情况下 None 并且不存储任何值
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    parser_esm_test = subparsers.add_parser("esm_test")
    parser_saveidc = subparsers.add_parser("save_idc")
    parser_wa = subparsers.add_parser("weight_average")
    args = parser.parse_args()  # parse_args() 返回的对象将仅包含由命令行选择的主解析器和子解析器的属性(而不包含任何其他子解析器)。即仅包含命令行输入的指令

    # default settings
    logging.basicConfig(level=logging.INFO)  # 输出日志
    pl.utilities.seed.seed_everything(seed=config.random_seed)  # 随机种子

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "save_idc":
        save_idc()
    elif args.mode == "weight_average":
        weight_average()
    else:
        raise Exception("Error Mode!")


if __name__ == '__main__':
    main()
