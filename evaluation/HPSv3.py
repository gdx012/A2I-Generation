import argparse
import os
from datetime import datetime
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from typing import List
import json
from hpsv3 import HPSv3RewardInferencer

# Initialize the model
inferencer = HPSv3RewardInferencer(device='cuda', checkpoint_path = "./HPSv3/HPSv3.safetensors")

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

p = argparse.ArgumentParser()
p.add_argument('-d', '--dataset', type=str)
p.add_argument('-m', '--model_name', type=str, default='proposed')
p.add_argument('--pred', type=str, default=None)
p.add_argument('--gt', type=str, default=None)
args = p.parse_args()

dataset = args.dataset
model_name = args.model_name
image_path = args.pred
gt_path = args.gt

ret = []
imp = os.listdir(image_path)
imp = list(filter(lambda x: 'ipynb' not in x, imp))
if dataset == 'landscape':
    aup = get_all_video_paths(gt_path)
with open("/data/caption.json", "r") as f:
    data = json.load(f)
keys = list(data.keys())
for fn in (pbar := tqdm(imp)):
    if not fn.endswith('jpg') and not fn.endswith('png'):
        continue
    if dataset == 'A2I':

        img = os.path.join(image_path, fn)
        name = fn.replace("_0.png","").replace("onec_re_","") 
        objects = data["t2i_caption_" + name.split('|')[0].replace(".wav","").replace(".flac","")]['[Main Subject]']
        labels = re.findall(r'<([^>]+)>', objects)
        text = "An image related to " + " and ".join(labels)
        
    if dataset == 'landscape':
        ytid = fn[:-4]
        last_underscore_index = ytid.rfind('_')
        name = ytid[:last_underscore_index].replace(".mp4","")
        img = os.path.join(image_path, f'{ytid}.png')
        i = 0
        audio_path = [aup[i] for i in range(len(aup)) if aup[i].split('/')[-1] == name + '.mp4']
        text = ["An image related to " + audio_path[0].split('/')[-2]]
    # print(text)
    rewards = inferencer.reward(prompts=[text], image_paths=[img])
    r = rewards[0][0].item()
    
    ret.append(r)
    pbar.set_description(f'score_mean = {sum(ret) / len(ret):.4f}')

print(f'Final Result: HPSv3 = {sum(ret) / len(ret):.4f}')