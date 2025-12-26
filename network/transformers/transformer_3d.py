# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from network.transformers.vision_transformer import VisionTransformer
from network.transformers.diffusion_mlp import DiffusionMLP
from network.transformers.embeddings import MaskEmbed, MotionEmbed, TextEmbed, LabelEmbed
from network.transformers.embeddings import PosEmbed, VideoPosEmbed, RotaryEmbed3D
from network.schedulers.scheduling_ddpm import DDPMScheduler
from network.transformers.normalization import AdaLayerNorm

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout=0.1):
        super().__init__()
        layers = []
        if hidden_dims:
            dims = [input_dim] + hidden_dims
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_dims[-1], num_classes))
        else:
            layers.append(nn.Linear(input_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Transformer3DModel(nn.Module):
    def __init__(
        self,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder = VisionTransformer(
            depth=16, embed_dim=1024, num_heads=16, patch_size=4, image_dim=4, image_size=(32,32)
        )  # load pretrained model
        video_mixer_rank = 24
        self.video_encoder.mixer = AdaLayerNorm(self.video_encoder.embed_dim, video_mixer_rank, eps=None)
        self.image_encoder = VisionTransformer(
            depth=32, embed_dim=1024, num_heads=16, patch_size=2, image_dim=4, image_size=(32,32)
        )
        image_base_size = [16,16]
        self.image_encoder.pos_embed = PosEmbed(self.image_encoder.embed_dim, image_base_size)
        # load pretrained model
        self.image_decoder = DiffusionMLP(
            depth=3, embed_dim=1280, cond_dim=self.image_encoder.embed_dim, patch_size=2, image_dim=4
        )  # load pretrained model
        self.mask_embed = MaskEmbed(self.image_encoder.embed_dim)  # load pretrained model
        self.label_embed = LabelEmbed(self.image_encoder.embed_dim, num_classes=3)  # NEW!
        video_base_size = [30,8,8]
        self.video_pos_embed = VideoPosEmbed(self.video_encoder.embed_dim, video_base_size)  # load pretrained model
        self.image_pos_embed = None
        self.motion_embed = None
        self.loss_repeat = 4

        self.noise_scheduler = DDPMScheduler(
            beta_schedule = "squaredcos_cap_v2",
            num_train_timesteps = 1000,
            #set_alpha_to_one =  False,
            steps_offset = 1,
            clip_sample = False
        )

        self.sample_scheduler = self.noise_scheduler
        self.learnable_cls_temporal = nn.Parameter(torch.randn(1, 1, 1024))
        self.cls = SimpleClassifier(1024, 3, [512, 256])


    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        return {"loss": loss.sum()}

    def train_video(self, inputs):
        """Train a batch of videos."""
        # 3D temporal autoregressive modeling (TAM).
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None  # B C T H W
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)  # B, T
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1])  # B T-1 hw C 
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1)  # B 1 hw C
        c, pos = self.video_pos_embed(torch.cat([bov, c], dim=1)), None  # B T hw C, None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        attn_mask = self.mask_embed.get_attn_mask(c, inputs["c"]) if latent_length > 1 else None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)  # B Thw C, B k C, B Thw C
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)  # B T hw C, [B 1 hw C, B T-1 hw C]
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)  # B T hw C
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        z = self.image_encoder(self.mask_embed(z), c.reshape(bs, -1, c.size(-1)), pos=pos)
        # 1D token-wise diffusion modeling (MLP).
        video_shape = (latent_length, z.size(1)) if latent_length > 1 else None
        return self.get_losses(z, x, video_shape=video_shape)
    
    def train_temporal(self, inputs):
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None  # B C T H W
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)  # B, T
        c = self.video_encoder.patch_embed(inputs["x"])  # B T hw C 
        c = self.video_pos_embed(c)  # B T+1 hw C
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        c = self.video_encoder(torch.cat((c.flatten(1, 2), cls_tokens), dim=1), None, pos=None)[:,-1]  # B Thw+1 C, B k C, B C
        logits = self.cls(c)
        loss = F.cross_entropy(logits, inputs["c"])
        return loss
    
    @torch.inference_mode()
    def predict_temporal(self, inputs):
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None  # B C T H W
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)  # B, T
        c = self.video_encoder.patch_embed(inputs["x"])  # B T hw C 
        c, pos = self.video_pos_embed(c), None  # B T+1 hw C, None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        c = self.video_encoder(torch.cat((c.flatten(1, 2), cls_tokens), dim=1), None, pos=pos)[:,-1]  # B Thw+1 C, B k C, B C
        logits = self.cls(c)
        return logits
    
    def train_spatial(self, inputs):
        x = inputs["x"].transpose(1, 2).flatten(0, 1)  # BT C H W
        z, bs = self.image_encoder.patch_embed(x), x.shape[0]  # BT C hw, BT
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        z = self.image_encoder.forward_image(z, cls_tokens, pos=None)
        logits = self.cls(z)
        target = inputs["c"].repeat_interleave(bs//inputs["c"].shape[0])
        loss = F.cross_entropy(logits, target)
        return loss
    
    @torch.inference_mode()
    def predict_spatial(self, inputs):
        x = inputs["x"].transpose(1, 2).flatten(0, 1)  # BT C H W
        z, bs = self.image_encoder.patch_embed(x), x.shape[0]  # BT C hw, BT
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        z = self.image_encoder.forward_image(z, cls_tokens, pos=None)
        logits = self.cls(z)
        return logits

    def train_total(self, inputs):
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None  # B C T H W
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)  # B, T
        c = self.video_encoder.patch_embed(inputs["x"])  # B T-1 hw C 
        c, pos = self.video_pos_embed(c), None  # B T hw C, None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        c = self.video_encoder(c.flatten(1, 2), None, pos=pos)  # B Thw C, B Thw C
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)  # B T hw C, [B 1 hw C, B T-1 hw C]
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)  # B T hw C
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1) # BT C H W
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length # BT C hw, BT
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        z = self.image_encoder.forward_image(z, torch.cat([cls_tokens, c.reshape(bs, -1, c.size(-1))], dim=1), pos=pos)
        logits = self.cls(z)
        target = inputs["c"].repeat_interleave(bs//inputs["c"].shape[0])
        loss = F.cross_entropy(logits, target)
        return loss
    
    @torch.inference_mode()
    def predict_total(self, inputs):
        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None  # B C T H W
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)  # B, T
        c = self.video_encoder.patch_embed(inputs["x"])  # B T-1 hw C 
        c, pos = self.video_pos_embed(c), None  # B T hw C, None
        if self.image_pos_embed:
            pos = self.video_pos_embed.get_pos(c.size(1), bs, self.video_encoder.patch_embed.hw)
        c = self.video_encoder(c.flatten(1, 2), None, pos=pos)  # B Thw C, B Thw C
        if not isinstance(self.video_encoder.mixer, torch.nn.Identity) and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)  # B T hw C, [B 1 hw C, B T-1 hw C]
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)  # B T hw C
        # 2D masked autoregressive modeling (MAM).
        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1) # BT C H W
        z, bs = self.image_encoder.patch_embed(x), bs * latent_length # BT C hw, BT
        if self.image_pos_embed:
            pos = self.image_pos_embed.get_pos(1, bs, self.image_encoder.patch_embed.hw)
        cls_tokens = self.learnable_cls_temporal.expand(bs, -1, -1)
        z = self.image_encoder.forward_image(z, torch.cat([cls_tokens, c.reshape(bs, -1, c.size(-1))], dim=1), pos=pos)
        logits = self.cls(z)
        return logits, z

    def forward(self, inputs):
        """Define the computation performed at every call."""
        return self.train_video(inputs)['loss']
    
    def inference(self, inputs):
        inputs["latents"] = inputs.pop("latents", [])
        inputs["c"], dtype, device = inputs.get("c", []), self.dtype, self.device
        batch_size = inputs.get("batch_size", 1)
        image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
        inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
        self.generate_video(inputs)

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1)
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        if self.image_pos_embed:
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length)
        [setattr(blk.attn, "cache_kv", max_latent_length > 1) for blk in self.video_encoder.blocks]
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            pos = time_pos[states["t"]] if time_pos else None
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c
            c = torch.cat([c] * 2) if guidance_scale > 1 else c
            c = states["c"] = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos)
            if not isinstance(self.video_encoder.mixer, torch.nn.Identity):
                states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
                states["*"] = states["*"] if states["t"] else states["c"]
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
        [setattr(blk.attn, "cache_kv", False) for blk in self.video_encoder.blocks]

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """Generate a batch of frames."""
        guidance_scale = inputs.get("guidance_scale", 1)
        min_guidance_scale = inputs.get("min_guidance_scale", guidance_scale)
        max_guidance_scale = inputs.get("max_guidance_scale", guidance_scale)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        guidance_end = max_guidance_scale if states["t"] else guidance_scale
        guidance_start = max_guidance_scale if states["t"] else min_guidance_scale
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            guidance_level = (i + 1) / len(all_num_preds)
            guidance_scale = (guidance_end - guidance_start) * guidance_level + guidance_start
            z = self.mask_embed(self.image_encoder.patch_embed(x))
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = torch.cat([z] * 2) if guidance_scale > 1 else z
            z = self.image_encoder(z, c, prev_ids, pos=pos)
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)
            states["noise"].normal_(generator=generator)
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids)
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))

    @torch.no_grad()
    def denoise(self, z, x, guidance_scale=1, generator=None, pred_ids=None) -> torch.Tensor:
        """Run diffusion denoising process."""
        self.sample_scheduler._step_index = None  # Reset counter.
        for t in self.sample_scheduler.timesteps:
            x_pack = torch.cat([x] * 2) if guidance_scale > 1 else x
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            noise_pred = self.image_decoder(x_pack, timestep, z, pred_ids)
            if guidance_scale > 1:
                cond, uncond = noise_pred.chunk(2)
                noise_pred = uncond.add_(cond.sub_(uncond).mul_(guidance_scale))
            noise_pred = self.image_encoder.patch_embed.unpatchify(noise_pred)
            x = self.sample_scheduler.step(noise_pred, t, x, generator=generator).prev_sample
        return self.image_encoder.patch_embed.patchify(x)
