import argparse
import math
import os.path
import warnings

import torch
from accelerate import Accelerator
from diffusers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import StableDiffusion
from dataloader.full_dataset import Fulldataset
warnings.filterwarnings("ignore")


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok = True)


def parse_args():
    ps = argparse.ArgumentParser()

    # regular
    ps.add_argument('--epochs', type=int, default=200)
    ps.add_argument('--start_epoch', type=int, default=0)
    ps.add_argument('--batch_size', type=int, default=2)
    ps.add_argument('--lr', type=float, default=5e-5)
    ps.add_argument('--scheduler', type=str, default='cosine')
    ps.add_argument('--gradient_accumulation_steps', type=int, default=8)
    ps.add_argument('--mixed_precision', type=str, default='no')
    ps.add_argument('--output_dir', type=str, default='./results/output_lr_5e5_FiLM')
    
    # functional params
    ps.add_argument('--cfg', action='store_true')

    # model dirs
    ps.add_argument('--pt_path', type=str, help='Path to the pretrained weights. (Must be a directory)', default='./results/output_lr_5e5_FiLM')
    ps.add_argument('--sd_path', type=str, help='Path to Stable Diffusion (local/huggingface)', default='CompVis/stable-diffusion-v1-4')

    # dataloader params
    ps.add_argument('--dataloader_num_workers', type=int, default=2)
    ps.add_argument('--dataset_name', type=str, default='Fulldataset')
    ps.add_argument('--data_set', type=str, default='train')
    ps.add_argument('--input_length', type=int, default=10)
    ps.add_argument('--sr', type=int, default=48000)
    ps.add_argument('--training', default=True)
    ps.add_argument('--output_size', default=512)
    ps.add_argument('--only_real', action='store_true', help='If set, only real images will be used for training.')
    ps.add_argument('--only_gen', action='store_true', help='If set, only generated images will be used for training.')
    arg = ps.parse_args()

    make_if_not_exist(arg.output_dir)
    return arg


args = parse_args()

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision
)


dataset = Fulldataset(args)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.dataloader_num_workers)

dtype = torch.float32
if accelerator.mixed_precision == 'fp16':
    dtype = torch.float16

model = StableDiffusion(pt_path=args.pt_path, sd_path=args.sd_path, dataset=args.dataset_name, device = accelerator.device).to(
    accelerator.device)








model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
total_training_steps = args.epochs * math.ceil(len(dataloader) / args.gradient_accumulation_steps / accelerator.num_processes)
warmup_steps = 0
scheduler = get_scheduler(
    args.scheduler, optimizer=optimizer,
    num_warmup_steps=warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=total_training_steps * args.gradient_accumulation_steps
)
optimizer, dataloader, scheduler = accelerator.prepare(
    optimizer, dataloader, scheduler
)

total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)

global_step, accuracy = 0, []
for epoch in range(args.start_epoch, args.epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            image = batch['pixel_values']
            audio = batch['audio_values']
            text = batch['raw_text']
            output = accelerator.unwrap_model(model)(audio, image)
            loss = output['sd_loss']

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), 1.)
                progress_bar.update(1)
                global_step += 1

            optimizer.step()
            scheduler.step()

        losses = {k: v.detach().item() for k, v in output.items() if 'loss' in k}
        logs = {'lr': scheduler.get_last_lr()[0]}
        logs.update(losses)
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if (epoch + 1) % 5 == 0:  # 推荐：第5、10、15...轮保存（符合"每5轮"的直觉）
            # 路径加入epoch编号，避免覆盖不同轮次的权重
            projector_path = os.path.join(args.output_dir, f'{args.dataset_name}_projector_epoch_{epoch+1}_gen.pt')
            unet_path = os.path.join(args.output_dir, f'{args.dataset_name}_unet_epoch_{epoch+1}_gen.pt')


            # 保存权重
            torch.save(accelerator.unwrap_model(model).audio_to_seq_model.state_dict(), projector_path)
            torch.save(accelerator.unwrap_model(model).unet.state_dict(), unet_path)
            # （可选）打印保存日志，方便调试
            print(f"已保存第 {epoch+1} 轮权重：")
            print(f"- PROJECTOR权重路径：{projector_path}")
            print(f"- UNet权重路径：{unet_path}")
accelerator.end_training()
