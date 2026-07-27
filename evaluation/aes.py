import os, json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import clip
from PIL import Image
from tqdm import tqdm
import csv
# from imagebind import data as d
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType
from moviepy import VideoFileClip
import tempfile
from typing import List
import argparse
from glob import glob

def get_all_video_paths(root_dir: str) -> List[str]:
    """
    遍历根文件夹下所有子文件夹，获取所有视频文件的绝对路径
    :param root_dir: 根文件夹路径（如 "./video_root"）
    :return: 所有视频文件路径的列表
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv")  # 支持的视频格式
    video_paths = []

    # 遍历根文件夹及其所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 判断文件是否为视频（通过后缀名匹配）
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)  # 拼接视频绝对路径
                video_paths.append(video_path)
    
    if not video_paths:
        raise FileNotFoundError(f"在 {root_dir} 及其子文件夹中未找到任何视频文件")
    
    return video_paths

def get_all_audio_paths(root_dir: str) -> List[str]:
    """
    遍历根文件夹下所有子文件夹，获取所有视频文件的绝对路径
    :param root_dir: 根文件夹路径（如 "./video_root"）
    :return: 所有视频文件路径的列表
    """
    
    audio_paths = []

    for name in os.listdir(root_dir):
        name = name.split('.')[0].replace("_0.png","") 
        audio_list = glob(f"data/A2I_Set/eval_audio_all/{name}*")
        audio_path = audio_list[0]
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)
    
    if not audio_paths:
        raise FileNotFoundError(f"在 {root_dir} 及其子文件夹中未找到任何视频文件")
    
    return audio_paths



imagebind_score = []
aes_score = []



p = argparse.ArgumentParser()
p.add_argument('-d', '--dataset', type=str)
p.add_argument('-m', '--model_name', type=str, default='proposed')
p.add_argument('-i', '--image_path', type=str, default=None)
p.add_argument('-a', '--audio_path', type=str, default=None)
p.add_argument('--clip', type=str, default='openai/clip-vit-base-patch32')
args = p.parse_args()

image_path = args.image_path
dataset = args.dataset
if dataset == 'landscape':
    aup = get_all_video_paths("data/landscape/test")
if dataset == 'A2I':
    aup = get_all_audio_paths(image_path)
imp = os.listdir(image_path)


# https://github.com/christophschuhmann/improved-aesthetic-predictor
class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
        nn.Linear(self.input_size, 1024),
        # nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        # nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        # nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        # nn.ReLU(),
        nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]

        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_av = MLP(768).cuda()
s = torch.load("evaluation/sac+logos+ava1-l14-linearMSE.pth")
# load the model you trained previously or the model available in this repo
model_av.load_state_dict(s)
model_av.to("cuda")
model_av.eval()

model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


for fn in (pbar := tqdm(imp)):
    if not fn.endswith('jpg') and not fn.endswith('png'):
        continue
    if dataset == 'landscape':
        ytid = fn[:-4]
        last_underscore_index = ytid.rfind('_')
        name = ytid[:last_underscore_index].replace(".mp4","")
        img = os.path.join(image_path, f'{ytid}.png')
    if dataset == 'A2I':
        img = os.path.join(image_path, fn)
        
    pil_image = Image.open(img)   
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model2.encode_image(image)        
    im_emb_arr = normalized(image_features.cpu().detach().numpy() ) 
    prediction = model_av(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    
    aes_score.append(prediction.item())

    pbar.set_description(f'score_mean = {sum(aes_score) / len(aes_score):.4f}')
print(f"aes_score:{sum(aes_score)/len(aes_score)}")