import argparse
import os
from datetime import datetime

import librosa.display
import numpy as np
import torch
import torch.nn.functional as F
import wav2clip
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from typing import List
from moviepy import VideoFileClip
import tempfile
from glob import glob
p = argparse.ArgumentParser()
p.add_argument('-d', '--dataset', type=str)
p.add_argument('-m', '--model_name', type=str, default='proposed')
p.add_argument('-i', '--image_path', type=str, default=None)
p.add_argument('-a', '--audio_path', type=str, default=None)
p.add_argument('--clip', type=str, default='openai/clip-vit-base-patch32')
args = p.parse_args()


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

audio_model = wav2clip.get_model()

vision_model = CLIPModel.from_pretrained(args.clip).cuda()
processor = CLIPProcessor.from_pretrained(args.clip)

dataset = args.dataset
model_name = args.model_name
image_path = args.image_path
audio_path = args.audio_path

ret, zret = [], []
imp = os.listdir(image_path)
imp = list(filter(lambda x: 'ipynb' not in x, imp))
if dataset == 'landscape':
    aup = get_all_video_paths(audio_path)

for fn in (pbar := tqdm(imp)):
    # import ipdb; ipdb.set_trace()
    if dataset == 'landscape':
        ytid = fn[:-4]
        last_underscore_index = ytid.rfind('_')
        name = ytid[:last_underscore_index].replace(".mp4","")
        img = os.path.join(image_path, f'{ytid}.png')
        i = 0
        audio_path = [aup[i] for i in range(len(aup)) if aup[i].split('/')[-1] == name + '.mp4']
        # 创建临时文件存储提取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_path = temp_wav.name
        # 使用moviepy提取音频并保存为临时WAV文件
            with VideoFileClip(audio_path[0]) as video:
                audio = video.audio
                audio.write_audiofile(temp_path, codec='pcm_s16le', logger=None)  # 保存为PCM 16位WAV
        track, _ = librosa.load(temp_path, sr=16000)
        os.remove(temp_path)
    if dataset == 'A2I':
        if not fn.endswith('jpg') and not fn.endswith('png'):
            continue
        img = os.path.join(image_path, fn)
        name = fn.replace("_0.png","") 
        audio_list = glob(f"data/A2I_Set/eval_audio_all/{name}*")
        audio_path = audio_list[0]
        track, _ = librosa.load(audio_path, sr=16000)
    image = Image.open(img)
    image = processor(images=image, return_tensors='pt').pixel_values.cuda()

    with torch.no_grad():
        image_features = vision_model.get_image_features(image)
        audio_features = wav2clip.embed_audio(track, audio_model)
        audio_features = torch.tensor(audio_features, device='cuda')

    sim = F.cosine_similarity(audio_features, image_features)
    r = sim.cpu().item()
    ret.append(r)

    pbar.set_description(f'score_mean = {sum(ret) / len(ret):.4f}')

print(f'Final Result: ais = {sum(ret) / len(ret):.4f}')
