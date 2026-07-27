import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from typing import List
import json
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

p = argparse.ArgumentParser()
p.add_argument('-d', '--dataset', type=str)
p.add_argument('-m', '--model_name', type=str, default='proposed')
p.add_argument('--pred', type=str, default=None)
p.add_argument('--gt', type=str, default=None)
p.add_argument('--clip', type=str, default='openai/clip-vit-base-patch32')
args = p.parse_args()

vision_model = CLIPModel.from_pretrained(args.clip).cuda()
processor = CLIPProcessor.from_pretrained(args.clip)
tokenizer = AutoTokenizer.from_pretrained(args.clip)
dataset = args.dataset
model_name = args.model_name
image_path = args.pred
gt_path = args.gt

ret, zret = [], []
imp = os.listdir(image_path)
imp = list(filter(lambda x: 'ipynb' not in x, imp))
if dataset == 'landscape':
    aup = get_all_video_paths(gt_path)
with open("data/A2I_Set/caption.json", "r") as f:
    data = json.load(f)
keys = list(data.keys())
for fn in (pbar := tqdm(imp)):
    try:
        if not fn.endswith('jpg') and not fn.endswith('png'):
            continue
        if dataset == 'A2I':

            img = os.path.join(image_path, fn)
            name = fn.replace("_0.png","").replace("onec_re_","") 
            audio_list = glob(f"data/A2I_Set/eval_audio_all/{name}*")
            audio_path = audio_list[0]
            classes = os.path.basename(audio_path).split('.')[0].split('|')[1:]
            text = ['An image related to ' + ' and '.join(classes)]

        if dataset == 'landscape':
            ytid = fn[:-4]
            last_underscore_index = ytid.rfind('_')
            name = ytid[:last_underscore_index].replace(".mp4","")
            img = os.path.join(image_path, f'{ytid}.png')
            i = 0
            audio_path = [aup[i] for i in range(len(aup)) if aup[i].split('/')[-1] == name + '.mp4']
            text = ['An image related to ' + audio_path[0].split('/')[-2]]
        
        # print(text)

        gt = tokenizer(text[:77], padding=True, return_tensors="pt").to('cuda')
        image = Image.open(img)
        image = processor(images=image, return_tensors='pt').pixel_values.cuda()

        with torch.no_grad():
            image_features = vision_model.get_image_features(image)
            gt_features = vision_model.get_text_features(**gt)

        sim = F.cosine_similarity(gt_features, image_features)
        r = sim.cpu().item()
        ret.append(r)
        pbar.set_description(f'score_mean = {sum(ret) / len(ret):.4f}')
    except RuntimeError as e:
        print(e)
        print(text)

print(f'Final Result: tis = {sum(ret) / len(ret):.4f}')