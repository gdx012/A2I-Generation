import os
import argparse
import random
from tqdm.auto import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, ClapModel, AutoTokenizer, ClapProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
import xformers
import json
import librosa
import logging
import csv
import time
from typing import Union, Optional, Dict
from PIL import Image
from accelerate.logging import get_logger
from diffusers import DiffusionPipeline, AutoencoderKL
import argparse
from typing import List, Optional
import re

TARGET_DTYPE = torch.float32
logger = get_logger(__name__)

TEMPLATES = [
    "A photo of {}.",
    "A close-up photo of {}.",
    "A candid shot of {}.",
    "A cinematic photo of {}.",
    "A high-resolution photo of {}.",
    "A realistic photo of {}.",
    "A studio shot of {}.",
    "A realistic outdoor photo of {}.",
    "A textural close-up photo of {}.",
    "A clear documentary-style photo of {}.",
    "A soft natural-light photo of {}.",
    "A high-clarity product photo of {}.",
    "A natural candid environmental photo of {}.",
    "A cinematic-style landscape photo of {}.",
    "A realistic wilderness scene photo of {}.",
]

STYLE_SUFFIXES = [
    ", photorealistic, 4k, ultra-detailed",
    ", cinematic lighting, dramatic shadows",
    ", vibrant colors, high contrast",
    ", sharp focus, natural texture, 8k resolution",
    ", soft ambient light, subtle highlights, lifelike tones",
    ", documentary-style framing, clear details, neutral color palette",
    ", natural light, even exposure, fine-grained texture",
    ", natural light, concrete texture, vibrant yet authentic tones",
    ""  # allow no suffix
]

def choose_prompt(subject: str,
                  templates: List[str] = TEMPLATES,
                  styles: Optional[List[str]] = STYLE_SUFFIXES,
                  seed: Optional[int] = None) -> str:
    """
    从模板随机生成 prompt，并可随机附加风格后缀。
    """
    if seed is not None:
        random.seed(seed)
    tpl = random.choice(templates)
    suffix = random.choice(styles) if styles else ""
    prompt = tpl.format(subject) + suffix
    return prompt


def seed_everything(random_seed: int = 42):
     """
     Set random seed for reproducibility.
     """
     torch.manual_seed(random_seed)
     torch.cuda.manual_seed(random_seed)
     torch.cuda.manual_seed_all(random_seed)
     torch.backends.cudnn.benchmark=False
     torch.backends.cudnn.deterministic=True
     np.random.seed(random_seed)
     random.seed(random_seed)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Audio2Text training script.")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="Directory to save results.")
    parser.add_argument("--clap_model_path", type=str, default="laion/clap-htsat-fused",
                        help="Path to CLAP model/processor.")
    parser.add_argument("--sd_path", type=str, default='CompVis/stable-diffusion-v1-4',
                        help="Path to SDXL pipeline (used to obtain text encoders/tokenizers).")
    parser.add_argument("--response_json", type=str, default="./data/A2I_Set/caption.json",
                        help="Path to response JSON mapping.")
    parser.add_argument("--audio_dir", type=str, default="./data/A2I_Set/A2I_audio",
                        help="Directory for train audio.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Weight for MSE loss.")
    parser.add_argument("--contrast_weight", type=float, default=0.2, help="Weight for contrastive loss.")
    parser.add_argument("--contrast_temperature", type=float, default=0.07, help="Temperature for contrastive loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=True, help="Path to checkpoint file (.pth) to resume training from")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

class LearnedPositionalEncoding(nn.Module):
    """简单的可学习位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # self.pos_embedding shape: (1, max_len, d_model)
        # 截取或填充到 x 的序列长度
        return x + self.pos_embedding[:, :x.size(1), :]

class AudioToPseudoTextSequence(nn.Module):
    def __init__(self,
                 target_seq_len=77,
                 target_embed_dim=2048, # 最终输出的伪文本嵌入维度
                 num_decoder_layers=6,  # Transformer Decoder层数
                 nhead=8,               # 多头注意力头数
                 dim_feedforward_factor=4, # Transformer FFN中间层维度因子
                 dropout=0.01):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_embed_dim = target_embed_dim

        # 1. MLP配置：{名称: (输入通道数C, 原始H, 原始W)}
        self.mlp_configs = {
            "feat1": (192, 32, 32),  # 32×32×192 → 目标32×768
            "feat2": (384, 16, 16),  # 16×16×384 → 目标16×768
            "feat3": (768, 8, 8),    # 8×8×768 → 目标8×768
            "feat4": (768, 8, 8)     # 8×8×768 → 目标8×768
        }
        
        # 2. 批量创建MLP：压缩W×C → 768（保留H维度）
        self.mlp_layers = nn.ModuleDict()
        for mlp_name, (in_dim, H, W) in self.mlp_configs.items():
            # MLP输入维度：W×C（如32×192=6144），输出：768
            self.mlp_layers[mlp_name] = nn.Sequential(
                nn.Linear(W * in_dim, target_embed_dim),
                nn.LeakyReLU(inplace=False),
                nn.LayerNorm(target_embed_dim)
            )
        
        # 2. 可学习的 "查询" 向量，它们将被转换为最终的输出序列。
        # 这些查询作为目标序列中每个位置的初始输入。
        self.positional_queries = nn.Parameter(torch.randn(1, target_seq_len, target_embed_dim))
        # nn.Parameter 使其成为可学习的张量。
        # 也可以选择使用固定的或可学习的显式位置编码加到查询上，
        # 但可学习查询本身已经具有位置特异性。
        # self.query_pos_encoder = LearnedPositionalEncoding(target_embed_dim, target_seq_len)


        # 3. Transformer Decoder 层
        # TransformerDecoderLayer 期望 (seq, batch, feature) 如果 batch_first=False (默认)
        # 或者 (batch, seq, feature) 如果 batch_first=True
        # 替换原始TransformerDecoderLayer为xformers优化版本
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=target_embed_dim,
            nhead=nhead,
            dim_feedforward=target_embed_dim * dim_feedforward_factor,
            dropout=dropout,
            batch_first=True,
            activation="gelu",  # 使用GELU激活，更适合GPU
            norm_first=True,  # 前置归一化，更稳定且更快
        )
        # 使用xformers优化注意力
        try:
            from xformers.ops import memory_efficient_attention
            decoder_layer.self_attn = memory_efficient_attention(
                decoder_layer.self_attn, 
                attn_bias=None,
                p=0.0
            )
        except:
            pass
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        
        # 初始化权重 (可选但推荐)
        self._init_weights()

    def _init_weights(self):
        # 简单的高斯初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_vec):
        feat = []

        for i, feat_name in enumerate(self.mlp_configs.keys()):
            feat_reshaped = self._reshape_feat(audio_vec[i], self.mlp_configs[feat_name])  # (B, H, W×C)
            feat_mapped = self.mlp_layers[feat_name](feat_reshaped)  # [B,N,768]
            feat.append(feat_mapped)
        
        audio_memory = torch.cat(feat, dim=1).contiguous()  # (B, total_seq_len, target_embed_dim)，total_seq_len=32+16+8+8=64
        batch_size = audio_memory.shape[0]

        # 1. 将音频向量投影为 Decoder 的 "memory"
        # (B, audio_dim) -> (B, target_embed_dim)
        
        
        # 2. 将 audio_memory unsqueeze 以适应 Transformer Decoder 的 memory 输入格式
        # (B, target_embed_dim) -> (B, 1, target_embed_dim)
        # memory 序列长度为 1，表示全局上下文

        # 3. 扩展可学习的 positional_queries 以匹配当前的 batch_size
        # (1, target_seq_len, target_embed_dim) -> (B, target_seq_len, target_embed_dim)
        queries = self.positional_queries.expand(batch_size, -1, -1)
        # 如果使用显式位置编码:
        # queries = self.query_pos_encoder(queries)


        # 4. 通过 Transformer Decoder
        # tgt (target): 是我们希望解码器细化的序列，这里是 `queries`
        # memory: 是编码器提供的上下文，这里是 `audio_memory`
        # 输出形状: (B, target_seq_len, target_embed_dim)
        output_sequence = self.transformer_decoder(tgt=queries, memory=audio_memory)
        
        return output_sequence
    
    
    def _reshape_feat(self, feat, config):
        """
        辅助函数：将 [B,C,H,W] → [B,H, W×C]
        Args:
            feat: 输入特征图 [B,C,H,W]
            config: (in_dim, H, W)
        Returns:
            reshaped_feat: [B,H, W×C]
        """
        B, C, H, W = feat.shape
        # 步骤1：交换C和H维度 → [B,H,C,W]
        feat = feat.transpose(1, 2)
        # 步骤2：展平C×W → [B,H, C×W]
        feat = feat.flatten(2)
        return feat



def encode_audio(audio_extractor, CLAP, audio_path, sr=None, device="cuda"):
    """
    修复：CLAP 输出是 (1, 512)，不是 (1,77,512)
    """
    
    try:
        if isinstance(audio_path, str):
            samples, samplerate = torchaudio.load(audio_path)
            samples = samples[0]  # 单通道
        elif isinstance(audio_path, torch.Tensor):
            samplerate = sr
            samples = audio_path
        else:
            raise ValueError("audio_path must be str or np.ndarray")
        
        if sr != 48000:
            # 重采样到 48000
            resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=48000).to(device)
            samples = resampler(samples).cpu().numpy()
            sr = 48000
        
        # 提取音频特征
        inputs = audio_extractor(
            audios=samples.cpu().numpy(), 
            sampling_rate=48000, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        with torch.no_grad():
            audio_outputs = CLAP.audio_model(**inputs, output_hidden_states=True) # [torch.Size([1, 96, 64, 64]), torch.Size([1, 192, 32, 32]), torch.Size([1, 384, 16, 16]), torch.Size([1, 768, 8, 8]), torch.Size([1, 768, 8, 8])]
            audio_features = [audio_outputs.hidden_states[i+1].squeeze(0) for i in range(len(audio_outputs.hidden_states)-1)]
        return audio_features
    except Exception as e:
        logger.error(f"音频编码失败: {audio_path}, 错误: {str(e)}")
        return None



def encode_text(tokenizer, text_encoder, raw_text, device="cuda"):
    try:
        # 处理文本（拼接标签）
        # 编码文本
        inputs = tokenizer(
            raw_text, 
            padding="max_length", 
            max_length=77, 
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            text_outputs = text_encoder(**inputs)
            text_vec = text_outputs.last_hidden_state  # (1, 77, 768)
        
        return text_vec.squeeze(0)  # (77, 768)
    except Exception as e:
        logger.error(f"文本编码失败: {raw_text}, 错误: {str(e)}")
        return None

def reshape_t(t):
    return t.reshape((t.shape[0], 1, 1, 1))

def preprocess_image_for_sdxl(image_path, target_size=1024):
    # Load image
    image = Image.open(image_path).convert("RGB")
    # Resize and center crop
    transform = T.Compose([
        T.Resize(target_size, interpolation=Image.BICUBIC),
        #T.RandomHorizontalFlip(p=0.5),  # Add random horizontal flip
        T.CenterCrop(target_size),
        T.ToTensor(),  # Converts to [0, 1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Now in [-1, 1]
    ])
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


# 定义对比损失计算函数
# 2.4 优化损失计算（向量化，避免循环）
def compute_info_nce_loss(
    query_embeds: torch.Tensor,
    key_embeds: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    temperature: float = 0.07,
    seq_level: bool = True
) -> torch.Tensor:
    # 向量化实现，移除for循环
    B, L, D = query_embeds.shape
    
    # 归一化
    
    if seq_level:
        query_embeds = F.normalize(query_embeds, dim=-1)
        key_embeds = F.normalize(key_embeds, dim=-1)
        query_flat = query_embeds.reshape(-1, D)
        key_flat = key_embeds.reshape(-1, D)
        sim_matrix = torch.matmul(query_flat, key_flat.t()) / temperature
        
        # 向量化掩码扩展
        pos_mask_expand = pos_mask.unsqueeze(1).repeat(1, L, 1).unsqueeze(2).repeat(1, 1, L, 1).reshape(B*L, B*L)
        neg_mask_expand = neg_mask.unsqueeze(1).repeat(1, L, 1).unsqueeze(2).repeat(1, 1, L, 1).reshape(B*L, B*L)
    else:
        
        query_cls = query_embeds.mean(dim=1)
        key_cls = key_embeds.mean(dim=1)
        query_embeds = F.normalize(query_cls, dim=-1)
        key_embeds = F.normalize(key_cls, dim=-1)
        sim = torch.matmul(query_embeds, key_embeds.T) / temperature
        exp_sim = torch.exp(sim)
        pos = exp_sim * pos_mask   
        numerator = pos.sum(dim=1)
        denominator = exp_sim.sum(dim=1)
        loss = -torch.log(numerator / denominator)
        loss = loss.mean()
    
    return loss



# 定义一个新的 Dataset，从中选择几个键
class SelectedKeysDataset(Dataset):
    def __init__(self, pic_list, device="cuda"):
        super().__init__()
        self.dataset = pic_list
        self.max_samples = int(10 * 48000)
        self.device = device
        # 构造一个字典，用于快速查找每个标签对应的样本索引
        self.label_to_indices = {}   # 修复：标签到样本索引的映射
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_path = sample.get('audio')
        waveform, sr = torchaudio.load(audio_path)
        audio = waveform.mean(dim=0).numpy()
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        sample['audio_len'] = audio.shape[0]
        
        if sample['audio_len'] < self.max_samples:
            audio = np.pad(audio, pad_width = (self.max_samples-sample['audio_len'],0), mode='constant')
        else:
            audio = np.array(audio[:self.max_samples])
        
        raw_text = " and ".join(sample['text'])
        choose_prompted = choose_prompt(raw_text, seed=42)  # 使用固定种子确保每次生成相同的 prompt
        

        return {'audio': audio, 'sampling_rate': sr,  'audio_len': sample['audio_len'], 'text': choose_prompted, 'raw_text': raw_text, 'audio_path': audio_path, 'sample_idx': idx}
    
    def build_contrast_masks(self, batch_indices: List[int]) -> (torch.Tensor, torch.Tensor):
        """
        为batch构建正负样本掩码
        Args:
            batch_indices: batch内样本的原始索引列表
        Returns:
            pos_mask: (B, B) 正样本掩码，1表示正样本
            neg_mask: (B, B) 负样本掩码，1表示负样本
        """
        B = len(batch_indices)
        pos_mask = torch.zeros((B, B), dtype=torch.bool, device=self.device)
        neg_mask = torch.zeros((B, B), dtype=torch.bool, device=self.device)
        
        # 遍历每个样本
        for i, idx_i in enumerate(batch_indices):
            # 获取样本i的标签集合
            labels_i = set(self.dataset[idx_i]['text'])
            
            # 遍历batch内其他样本
            for j, idx_j in enumerate(batch_indices):
                if i == j:
                    pos_mask[i, j] = 1  # 自己是正样本
                    continue
                
                # 获取样本j的标签集合
                labels_j = set(self.dataset[idx_j]['text'])
                
                # 判断是否为正样本（有共享标签）
                if len(labels_i & labels_j) > 0:
                    pos_mask[i, j] = 1
                else:
                    neg_mask[i, j] = 1
        
        return pos_mask, neg_mask

def custom_collate(batch):
    """过滤掉None样本"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)

    # 设置随机种子
    seed_everything(args.seed)

    mp = args.mixed_precision
    amp = None if mp == "no" else mp
    accelerator = Accelerator(mixed_precision=amp)
    # 设备设置
    device = accelerator.device
    logger.info(f"使用设备: {device}")
    CLAP = ClapModel.from_pretrained(args.clap_model_path).eval()
    audio_extractor = ClapProcessor.from_pretrained(args.clap_model_path)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder='text_encoder').eval()
    with open(args.response_json, "r") as f:
        data = json.load(f)
    keys = list(data.keys())

    audio_list = set()
    for file in os.listdir(args.audio_dir):
        if file.endswith(".flac") or file.endswith(".wav"):
            audio_list.add(file.split('.')[0])

    pic_list = []
    for key in keys:       
        try:
            name = key.replace('t2i_caption_',"")
            if name not in audio_list:
                continue
            audio_path = f"{args.audio_dir}/{name}.flac" if os.path.exists(f"{args.audio_dir}/{name}.flac") else f"{args.audio_dir}/{name}.wav"
            objects = data[key]['[Main Subject]']
            labels = re.findall(r'<([^>]+)>', objects)
            labels = set(labels)
            pic_list.append({'audio': audio_path, 'audio_len': None, 'text': labels})
        
        except Exception as e:
            logger.warning(f"样本处理失败: {name}, 错误: {str(e)}")  # 记录失败样本
            pass

    new_dataset = SelectedKeysDataset(
        pic_list, 
        device=device
    )
    
    dataloader = DataLoader(
        new_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )

     # 初始化模型


    audio_to_seq_model = AudioToPseudoTextSequence(
            target_seq_len=77,
            target_embed_dim=768,  # CLIP Text Encoder 的输出维度
            num_decoder_layers=12, # 可以调整层数
            nhead=12,              # 注意力头数，应能被 target_embed_dim 整除
            dim_feedforward_factor=4
        )
    
    num_params = sum(p.numel() for p in audio_to_seq_model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params / 1e6:.2f} M")
    print("-" * 50)


    if args.resume_from_checkpoint is not None:
        # 尝试加载最新的权重文件
        latest = 0
        for name in os.listdir(args.result_dir):
            if name.startswith("audio_to_text_model_epoch_") and name.endswith(".pth"):
                latest = max(latest, int(name[len("audio_to_text_model_epoch_"):-len(".pth")]))
        if latest > 0:
            checkpoint_name = "audio_to_text_model_epoch_{:02d}.pth".format(latest)
            checkpoint_path = os.path.join(args.result_dir, checkpoint_name)
            audio_to_seq_model.load_state_dict(torch.load(checkpoint_path))
            print(f"已加载模型检查点: {checkpoint_path}")
            
    
    optimizer = torch.optim.AdamW(audio_to_seq_model.parameters(), lr=3e-5)
    audio_to_seq_model, optimizer, dataloader, CLAP, text_encoder = accelerator.prepare(
        audio_to_seq_model, optimizer, dataloader, CLAP, text_encoder
    )
    num_epochs = 10
    global_step = 0
    for epoch in range(num_epochs):
        audio_to_seq_model.train()
        epoch_loss = 0.0
        epoch_contrast_loss = 0.0
        epoch_mse_loss = 0.0
        valid_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            # 提取音频特征
            
            audio_inputs = encode_audio(
                audio_extractor, 
                accelerator.unwrap_model(CLAP), 
                batch['audio'].to(accelerator.device), 
                sr=48000, 
                device=accelerator.device
            )
            text_targets = encode_text(tokenizer, accelerator.unwrap_model(text_encoder), batch["text"], device=accelerator.device)
            
            
            batch_indices = batch['sample_idx']

            # contrast loss mask
            # ========== 1. 构建对比损失掩码 ==========
            pos_mask, neg_mask = new_dataset.build_contrast_masks(batch_indices)
            pos_mask = pos_mask.to(accelerator.device)
            neg_mask = neg_mask.to(accelerator.device)

            # ========== 2. 前向传播 ==========
            optimizer.zero_grad()
            output_sequences = audio_to_seq_model(audio_inputs)              # (B, 77, 768)


            # ========== 3. 损失计算 ==========
            # MSE损失（序列重建损失）
            mse_loss = F.mse_loss(output_sequences, text_targets)

            # 对比损失（InfoNCE）
            contrast_loss = compute_info_nce_loss(
                query_embeds=output_sequences,
                key_embeds=text_targets,
                pos_mask=pos_mask,
                neg_mask=neg_mask,
                temperature=args.contrast_temperature,
                seq_level=False
            )

            total_loss = args.mse_weight * mse_loss + args.contrast_weight * contrast_loss

            accelerator.backward(total_loss)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_contrast_loss += contrast_loss.item()
            epoch_mse_loss += mse_loss.item()
            global_step += 1
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_contrast_loss = epoch_contrast_loss / len(dataloader)
        avg_mse_loss = epoch_mse_loss / len(dataloader)

        logger.info(f"Epoch {epoch+1} 完成 | 总损失: {avg_epoch_loss:.4f} | 对比损失: {avg_contrast_loss:.4f} | MSE损失: {avg_mse_loss:.4f}")
        
        # 保存模型检查点
        if accelerator.is_main_process:
            save_path = os.path.join("audio_to_text_model_epoch_{:02d}.pth".format(epoch+1))
            torch.save(accelerator.unwrap_model(audio_to_seq_model).state_dict(), save_path)
            print(f"模型检查点已保存到 {save_path}")
    
    logger.info("训练完成！")
if __name__ == "__main__":
    args = parse_args()
    main(args)

