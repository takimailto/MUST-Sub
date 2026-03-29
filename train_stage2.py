from datasets.dataset import TemporalDataset
from network.autoencoders.autoencoder_kl import AutoencoderKL
from network.autoencoder_kl_opensora import AutoencoderKLOpenSora
from network.transformers.transformer_3d import Transformer3DModel

import os
import torch
import numpy as np
import collections
import random
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_module(module, trainable=False):
    module.eval() if not trainable else module.train()
    for param in module.parameters():
        param.requires_grad = trainable
    return module


class ConstantLR(object):
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


@torch.inference_mode()
def validate_model(model, vae_image, vae, val_dataloader, device1, device2):
    model.eval()
    all_preds = []
    all_labels = []

    for contrast, gray, class_id in tqdm(val_dataloader, desc="Validating"):
        with torch.no_grad():
            x = contrast.to(device2)
            x = vae.encode(x).latent_dist.parameters

            gray = gray.to(device2)
            gray_latents = vae_image.encode(gray).latent_dist.parameters
            gray_latents = vae_image.scale_(vae_image.latent_dist(gray_latents).sample()).to(device1, non_blocking=True).unsqueeze(dim=2)
            x_latents = vae.scale_(vae.latent_dist(x).sample()).to(device1, non_blocking=True)
            inputs = {"x": torch.cat([gray_latents, x_latents], dim=2)}

        logits, _ = model.predict_total(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(class_id.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    model.train()
    return macro_f1


def main(args):
    set_seed(42)
    writer = SummaryWriter(log_dir=args.output_dir)

    # ================= Model =================
    ARmodel = Transformer3DModel()
    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    ARmodel.load_state_dict(checkpoint['model_state_dict'], strict=False)

    ckpt_lvl = 2
    [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in ARmodel.video_encoder.blocks]
    [setattr(blk, "mlp_checkpointing", ckpt_lvl > 1) for blk in ARmodel.image_encoder.blocks]
    [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in ARmodel.image_decoder.blocks]

    ARmodel = ARmodel.to(args.device1)
    ARmodel.train()
    freeze_module(ARmodel.label_embed.norm)
    if ARmodel.motion_embed:
        freeze_module(ARmodel.motion_embed)

    # ================= VAE =================
    vae = AutoencoderKLOpenSora.from_pretrained(args.video_vae_path).to(args.device2).eval()
    vae_image = AutoencoderKL.from_pretrained(args.image_vae_path).to(args.device2).eval()

    # ================= Dataset =================
    train_dataset = TemporalDataset(list_path=args.train_list_file)
    val_dataset = TemporalDataset(list_path=args.val_list_file)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # ================= Optimizer =================
    param_groups = filter(lambda p: p.requires_grad, ARmodel.parameters())
    optimizer = torch.optim.Adam(
        param_groups,
        lr=1e-5,
        weight_decay=0.01
    )

    loss_scaler = torch.amp.GradScaler("cuda", enabled=False)
    scheduler = ConstantLR(lr_max=1e-5, lr_min=1e-5, warmup_steps=250, warmup_factor=0.001)

    # ================= Training Loop =================
    total_step = 0
    best_macro_f1 = 0.0
    early_stop_patience = 5
    early_stop_counter = 0

    for epoch in range(100):
        loss_list = []
        ARmodel.train()

        for contrast, gray, class_id in tqdm(train_dataloader):
            lr = scheduler.get_lr()
            for g in optimizer.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)

            with torch.no_grad():
                x = contrast.to(args.device2)
                x = vae.encode(x).latent_dist.parameters

                gray = gray.to(args.device2)
                gray_latents = vae_image.encode(gray).latent_dist.parameters
                gray_latents = vae_image.scale_(vae_image.latent_dist(gray_latents).sample()).to(args.device1, non_blocking=True).unsqueeze(2)
                x_latents = vae.scale_(vae.latent_dist(x).sample()).to(args.device1, non_blocking=True)
                inputs = {"x": torch.cat([gray_latents, x_latents], dim=2)}

            inputs["c"] = class_id.to(args.device1)
            optimizer.zero_grad()
            loss = ARmodel.train_total(inputs)

            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            scheduler.step()

            total_step += 1
            loss_list.append(loss.item())

            if total_step % 1000 == 0:
                torch.save({
                    'iteration': total_step,
                    'model_state_dict': ARmodel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(args.output_dir, f"model_step_{total_step}.pth"))

        # Epoch loss
        epoch_loss = np.mean(loss_list)
        print(f"Epoch {epoch+1}/100 | Train Loss: {epoch_loss:.6f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation
        val_macro_f1 = validate_model(ARmodel, vae_image, vae, val_dataloader, args.device1, args.device2)
        print(f"Epoch {epoch+1} | Val Macro-F1: {val_macro_f1:.4f}")
        writer.add_scalar('Metric/val_macro_f1', val_macro_f1, epoch)

        # Best model & Early Stop
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            early_stop_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'best_macro_f1': best_macro_f1,
                'model_state_dict': ARmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"✅ New best model saved (Macro-F1: {best_macro_f1:.4f})")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stop triggered. Training finished.")
                break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pretrained_path', required=True)
    parser.add_argument('--video_vae_path', required=True)
    parser.add_argument('--image_vae_path', required=True)
    parser.add_argument('--train_list_file', required=True)
    parser.add_argument('--val_list_file', required=True)
    parser.add_argument('--device1', default="cuda:0")
    parser.add_argument('--device2', default="cuda:1")
    args = parser.parse_args()
    main(args)