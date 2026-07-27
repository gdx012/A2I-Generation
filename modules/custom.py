import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import ClapAudioModelWithProjection, AutoProcessor, ClapModel, AutoTokenizer, ClapProcessor
from torchvision.transforms import v2
import cv2
import numpy as np
import torchaudio
import librosa
from .audio2text import AudioToPseudoTextSequence


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x, cond):
        """
        x: [B, N, C]
        cond: [B, cond_dim]
        """
        gamma = self.gamma(cond)[:, None, :]
        beta = self.beta(cond)[:, None, :]
        return x * (1 + gamma) + beta


# Modified from https://github.com/tencent-ailab/IP-Adapter
class IPAttnProcessor(nn.Module):
    def __init__(self, num_tokens, hidden_size, cross_attention_dim=None, scale=1.,
                 k_weight=None, v_weight=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.film_audio = FiLM(hidden_size, hidden_size)
        self.film_text = FiLM(hidden_size, hidden_size)
        if k_weight is not None: self.to_k_ip.weight.data = k_weight.data
        if v_weight is not None: self.to_v_ip.weight.data = v_weight.data

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        h_audio = self.film_audio(ip_hidden_states, ip_hidden_states.mean(dim=1))
        h_text = self.film_text(hidden_states, hidden_states.mean(dim=1))
        hidden_states = hidden_states + 0.5 * h_audio + (1 - 0.5) * h_text

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StableDiffusion(nn.Module):
    def __init__(self,
                 sd_path,
                 pt_path,
                 train=True,
                 dataset=None,
                 cfg=False,
                 device=None):
        super(StableDiffusion, self).__init__()
        self.cfg = cfg
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder='text_encoder').to(self.device)
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder='vae').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder='unet').to(self.device)
        self.set_adapter()
        self.audio_extractor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        if train:
            self.scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder='scheduler')
        else:
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_path, subfolder='scheduler')

        self.CLAP = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)

        self.audio_to_seq_model = AudioToPseudoTextSequence(
            target_seq_len=77,
            target_embed_dim=768,  # CLIP Text Encoder 的输出维度
            num_decoder_layers=12, # 可以调整层数
            nhead=12,              # 注意力头数，应能被 target_embed_dim 整除
            dim_feedforward_factor=4
        )
        
        # Stage 1, start with pretrained audio-to-text model
        # self.audio_to_seq_model.load_state_dict(torch.load(os.path.join(pt_path, "audio_to_text_model_epoch_10.pth")))
        
        # Stage 2, start with finetuned audio-to-text model

        weight = torch.load(os.path.join(pt_path, f'{dataset}_projector_epoch_70.pt'), map_location='cpu')
        self.audio_to_seq_model.load_state_dict(weight, strict=True)
        self.audio_to_seq_model = self.audio_to_seq_model.to(self.device) if self.device is not None else self.audio_to_seq_model
        
        # UNET
        weight = torch.load(os.path.join(pt_path, f'{dataset}_unet_epoch_70.pt'), map_location='cpu')
        self.unet.load_state_dict(weight)
        self.unet = self.unet.to(self.device)
        # grad setting
        if train:
            self.requires_grad_(False)
            self.eval()
            self.audio_to_seq_model.requires_grad_(True)
            self.audio_to_seq_model.train()
            for n, p in self.unet.named_modules():
                if isinstance(p, IPAttnProcessor):
                    p.requires_grad_(True)
                    p.train()
        else:
            self.requires_grad_(False)
            self.eval()


    def set_adapter(self):
        unet = self.unet
        state_dict = self.unet.state_dict()
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                k_weight = state_dict[name.replace('processor', 'to_k.weight')].detach().clone()
                v_weight = state_dict[name.replace('processor', 'to_v.weight')].detach().clone()
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_tokens=77,
                    scale=1.,
                    k_weight=k_weight, v_weight=v_weight
                ).to(self.device)
                print(f'[Adapter] {name} is set to IPAttnProcessor.')
        unet.set_attn_processor(attn_procs)
        print('[Adapter] IPAdapters are set.')




    def infer(self, audio, steps=25, device='cuda', prompt='', negative_prompt='', guidance_scale=8.5):
        
        self.eval()
        size = 512
        audio = audio.to(self.device)
        
        with torch.no_grad():
            audio_outputs = self.CLAP.audio_model(input_features = audio.input_features.squeeze(1), is_longer = audio.is_longer.squeeze(1), output_hidden_states=True)
            hidden_coeff = self.CLAP.audio_projection(audio_outputs.pooler_output)
            audio_features = [audio_outputs.hidden_states[i+1] for i in range(len(audio_outputs.hidden_states)-1)]
        feat_vec = self.audio_to_seq_model(audio_features) # [B, 77, 768]
        
        input = self.tokenizer([prompt] * hidden_coeff.shape[0], max_length=77, padding='max_length', truncation=True,
                               return_tensors="pt").input_ids.to('cuda')
        text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]

        pos_hidden_states = torch.cat([text_vec, feat_vec], dim=1)

        latent = torch.randn(1, self.unet.in_channels, size // 8, size // 8,
                             device=device) * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(steps)

        if self.cfg:
            latent = torch.cat([latent] * 2, dim=0)

            hidden_coeff = torch.zeros_like(hidden_coeff).to(self.device)

            input = self.tokenizer([''] * hidden_coeff.shape[0], max_length=77, padding='max_length',
                                   truncation=True, return_tensors="pt").input_ids.to(self.device)
            text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]
            audio_features_zeros = [torch.zeros_like(feature).to(self.device) for feature in audio_features]
            zero_input = self.audio_to_seq_model(audio_features_zeros) # [B, 77, 768]

            zero_hidden_states = torch.cat([text_vec, zero_input], dim=1)
            hidden_states = torch.cat([pos_hidden_states, zero_hidden_states], dim=0)
            for ts in (self.scheduler.timesteps):
                t = torch.tensor([ts], dtype=torch.long, device=device)
                latent = self.scheduler.scale_model_input(latent, t)
                with torch.no_grad():
                    pos_output, zero_output = self.unet(latent, t, encoder_hidden_states=hidden_states).sample.chunk(2)
                output = zero_output + guidance_scale * (pos_output - zero_output)
                latent = self.scheduler.step(output, t, latent).prev_sample
        else:
            for ts in (self.scheduler.timesteps):
                t = torch.tensor([ts], dtype=torch.long, device=device)
                latent = self.scheduler.scale_model_input(latent, t)
                with torch.no_grad():
                    output = self.unet(latent, t, encoder_hidden_states=feat_vec).sample
                latent = self.scheduler.step(output, t, latent).prev_sample

        latent /= 0.18215
        with torch.no_grad():
            image = self.vae.decode(latent).sample
        image = (image / 2 + .5).clamp(0, 1)  # [B, C, H, W]
        image = image.permute(0, 2, 3, 1)

        return image

    def forward(self, audio, image, prompt=''):
        audio = audio.to(self.device)
        with torch.no_grad():
            audio_outputs = self.CLAP.audio_model(input_features = audio.input_features.squeeze(1), is_longer = audio.is_longer.squeeze(1), output_hidden_states=True)
            hidden_coeff = self.CLAP.audio_projection(audio_outputs.pooler_output)
            audio_features = [audio_outputs.hidden_states[i+1] for i in range(len(audio_outputs.hidden_states)-1)]

        image = image.to(self.device)
        if random.random() <= 0.1:
            hidden_coeff = torch.zeros_like(hidden_coeff).to(self.device)
            audio_features = [torch.zeros_like(feature).to(self.device) for feature in audio_features]
        prompt = [''] * hidden_coeff.shape[0]
        
        
        # Text Encoder
        input = self.tokenizer(prompt , max_length=77, padding='max_length', truncation=True,
                            return_tensors="pt").input_ids.to(self.device)
        
        text_vec = self.text_encoder(input_ids=input).last_hidden_state  # [B, 77, 768]
        feat_vec = self.audio_to_seq_model(audio_features) # [B, 77, 768]
        
        # Stable Diffusion
        latent = self.vae.encode(image).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latent)

        encoder_hidden_states = torch.cat([text_vec, feat_vec], dim=1)

        bsz = latent.shape[0]
        time_step = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,),
                                  device=latent.device).long()
        noise_latent = self.scheduler.add_noise(latent, noise, time_step)
        unet_output = self.unet(noise_latent, time_step, encoder_hidden_states=encoder_hidden_states).sample

        loss = F.mse_loss(unet_output, noise, reduction="mean")

        output = {}
        output['sd_loss'] = loss


        return output
