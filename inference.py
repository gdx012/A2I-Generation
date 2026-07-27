import argparse
import os.path
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import StableDiffusion
from dataloader.full_dataset import Fulldataset
from dataloader.landscapeSet import landscapeSet
from dataloader.audio_dataset import AudioDataset
warnings.filterwarnings('ignore')


def parse_args():
    ps = argparse.ArgumentParser()

    # functional params
    ps.add_argument('--prompt', type=str, default='')
    ps.add_argument('--negative_prompt', type=str, default='')
    ps.add_argument('--cfg', action='store_true')

    # model dirs
    ps.add_argument('--pt_path', type=str, help='Path to the pretrained weights. (Must be a directory)', default='./results/output_lr_5e5_FiLM')
    ps.add_argument('--sd_path', type=str, help='Path to Stable Diffusion (local/huggingface)', default='CompVis/stable-diffusion-v1-4')

    # others
    ps.add_argument('--batch_size', type=int, default=1)
    ps.add_argument('--data_set', type=str, default='test')
    ps.add_argument('--output_dir', type=str, default='./results/output_lr_5e5_FiLM/epoch_70')
    ps.add_argument('--input_length', type=int, default=10)
    ps.add_argument('--sr', type=int, default=48000)
    ps.add_argument('--mixed_precision', default='no')
    ps.add_argument('--dataset_name', type=str, default='Fulldataset')
    ps.add_argument('--training', default=False)
    ps.add_argument('--eval_set', type=str, default='A2I', choices=['A2I', 'landscape'])
    args = ps.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    return args


args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for fn in os.listdir(args.output_dir):
    file = os.path.join(args.output_dir, fn)
    os.remove(file)

if args.eval_set == 'A2I':
    dataset = AudioDataset(args)
elif args.eval_set == 'landscape':
    dataset = landscapeSet(args) #landscape

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

model = StableDiffusion(pt_path=args.pt_path, sd_path=args.sd_path, train=False, dataset=args.dataset_name,
                        cfg=args.cfg, device = device)
generated = []
sr, length = args.sr, args.input_length
for batch in tqdm(dataloader):
    audio = batch['audio_values']
    ytid = batch['ytid'][0]
    
    i = 0
    with torch.no_grad():
        image = model.infer(audio, prompt=args.prompt, negative_prompt=args.negative_prompt)
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
    
    while(True):
        if not os.path.exists(f'{args.output_dir}/{ytid}_{i}.png'):
            cv2.imwrite(f'{args.output_dir}/{ytid}_{i}.png', image)
            break
        
        i += 1
