from datasets.dataset import TemporalDataset
from network.autoencoders.autoencoder_kl import AutoencoderKL
from network.autoencoder_kl_opensora import AutoencoderKLOpenSora
from network.transformers.transformer_3d import Transformer3DModel



import os
import torch
import numpy as np
import collections
from torch import nn
from tqdm import tqdm
from safetensors import safe_open
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import argparse

def freeze_module(module, trainable=False):
    """Freeze parameters of given module."""
    module.eval() if not trainable else module.train()
    for param in module.parameters():
        param.requires_grad = trainable
    return module

def get_param_groups(model):
    """Separate parameters into groups."""
    memo, groups, lr_scale_getter = set(), collections.OrderedDict(), None
    norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm, nn.LayerNorm)
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or param in memo:
                continue
            memo.add(param)
            attrs = collections.OrderedDict()
            if lr_scale_getter:
                attrs["lr_scale"] = lr_scale_getter(f"{module_name}.{param_name}")
            if hasattr(param, "lr_scale"):
                attrs["lr_scale"] = param.lr_scale
            if getattr(param, "no_weight_decay", False) or isinstance(module, norm_types):
                attrs["weight_decay"] = 0
            group_name = "/".join(["%s:%s" % (v[0], v[1]) for v in list(attrs.items())])
            groups[group_name] = groups.get(group_name, {**attrs, **{"params": []}})
            groups[group_name]["params"].append(param)
    return list(groups.values())

class ConstantLR(object):
    """Constant LR scheduler."""

    def __init__(self, **kwargs):
        self._lr_max = kwargs.pop("lr_max")
        self._lr_min = kwargs.pop("lr_min", 0)
        self._warmup_steps = kwargs.pop("warmup_steps", 0)
        self._warmup_factor = kwargs.pop("warmup_factor", 0)
        self._step_count = 0
        self._last_decay = 1.0

    def step(self):
        self._step_count += 1

    def get_lr(self):
        if self._step_count < self._warmup_steps:
            alpha = (self._step_count + 1.0) / self._warmup_steps
            return self._lr_max * (alpha + (1.0 - alpha) * self._warmup_factor)
        return self._lr_min + (self._lr_max - self._lr_min) * self.get_decay()

    def get_decay(self):
        return self._last_decay

def main(args):

    ARmodel = Transformer3DModel()
    writer = SummaryWriter(log_dir=args.output_dir)

    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    ARmodel.load_state_dict(state_dict, strict=False)
    ckpt_lvl = 2
    [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in ARmodel.video_encoder.blocks]
    [setattr(blk, "mlp_checkpointing", ckpt_lvl > 1) for blk in ARmodel.image_encoder.blocks]
    [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in ARmodel.image_decoder.blocks]
    
    ARmodel = ARmodel.to(args.device1)
    ARmodel.train()  
    freeze_module(ARmodel.label_embed.norm)
    freeze_module(ARmodel.motion_embed) if ARmodel.motion_embed else None

    vae = AutoencoderKLOpenSora.from_pretrained(args.video_vae_path)
    vae = vae.to(args.device2).eval()

    vae_image = AutoencoderKL.from_pretrained(args.image_vae_path)
    vae_image = vae_image.to(args.device2).eval()

    dataset = TemporalDataset(list_path=args.patient_list_file)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    param_groups = filter(lambda p: p.requires_grad, ARmodel.parameters())
    optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=0.02, betas=(0.9, 0.95))
    loss_scaler = torch.amp.GradScaler("cuda", enabled=False)
    scheduler = ConstantLR(lr_max=1e-4, lr_min=0.0, warmup_steps=250, warmup_factor=0.001)
    total_step = 0
    epochs = 0
    while True: 
        loss_list = []
        for contrast, gray, class_id in tqdm(train_dataloader):
            lr = scheduler.get_lr()
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)

            with torch.no_grad():
                x = contrast.to(args.device2)
                inputs = {}
                x = vae.encode(x).latent_dist.parameters

                gray = gray.to(args.device2)
                gray_latents = vae_image.encode(gray).latent_dist.parameters
                gray_latents = vae_image.scale_(vae_image.latent_dist(gray_latents).sample()).to(args.device1, non_blocking=True).unsqueeze(dim=2)
                inputs["x"] = torch.cat([gray_latents, vae.scale_(vae.latent_dist(x).sample()).to(args.device1, non_blocking=True)], dim=2)
            
            inputs["aspect_ratio"] = 1
            class_id = class_id.to(args.device1)
            optimizer.zero_grad()
            inputs["c"] = class_id
            losses = ARmodel.train_total(inputs)

            loss_scaler.scale(losses).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()

            scheduler.step()
            total_step += 1

            loss_list.append(losses.item())

            if total_step % 1000 == 0:
                checkpoint_path = os.path.join(args.output_dir, f"model_step_{total_step}.pth")
                torch.save({
                    'iteration': total_step,
                    'model_state_dict': ARmodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses.item(),
                }, checkpoint_path)
                print(f"Model saved at iteration {total_step} to {checkpoint_path}")
        
        print(f"Epoch: {epochs}, Loss: {np.mean(loss_list)}")
        writer.add_scalar('Loss/train', np.mean(loss_list), epochs)
        epochs += 1

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description='Traning Scripts for Stage2')

    # 添加参数
    parser.add_argument('--output_dir')
    parser.add_argument('--pretrained_path', help="Path of the pretrained first-stage model")
    parser.add_argument('--video_vae_path', help="Path of the pretrained video-VAE model")
    parser.add_argument('--image_vae_path', help="Path of the pretrained image-VAE model")
    parser.add_argument('--patient_list_file', help="File contain the list of patient")
    parser.add_argument('--device1')
    parser.add_argument('--device2')
    main(args)
