import os
import random
import torchaudio
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from typing import List
from moviepy import VideoFileClip
import tempfile
from transformers import ClapAudioModelWithProjection, AutoProcessor, ClapModel, AutoTokenizer, ClapProcessor


def get_all_video_paths(root_dir: str) -> List[str]:
    """
    遍历根文件夹下所有子文件夹，获取所有视频文件的绝对路径
    :param root_dir: 根文件夹路径（如 "./video_root"）
    :return: 所有视频文件路径的列表
    """
    video_extensions = (".wav", ".mp3", ".flac", ".mp4")  # 支持的视频格式
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

class landscapeSet(Dataset):
    def __init__(self, args):
        super(landscapeSet, self).__init__()
        image_lst = "image/"
        audio_lst = "audio/"
        balance_dataset = []
        self.image_path = list()
        self.audio_path = list()
        
        audio_path  = get_all_video_paths("./data/landscape/test")
        self.audio_path = [element for element in audio_path for _ in range(20)]
        self.audio_extractor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

        
        self.image_resolution = (512, 512)
        self.bsz = args.batch_size
        self.size = 512
        self.data_set = args.data_set
        self.input_length, self.sr = args.input_length, args.sr
        self.audio_only = not args.training
        self.cfg = args.cfg if hasattr(args, 'cfg') else False
        
        self.num_samples = len(self.audio_path)
        print(f"数据预加载完成！共加载 {self.num_samples} 个样本到内存")
        self._length = self.num_samples
    def __len__(self):
        return self._length

    def process_audio(self, aud, ):
        wav, sr = torchaudio.load(aud)
        audio = wav.mean(dim=0).numpy()
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            # sample['audio'][1] = 48000
            sr = 48000
        if audio.shape[0] > sr * 10:
            start = random.randint(0, audio.shape[0] - sr * 10 - 1)
            end = start + sr * 10
            audio = audio[start:end]
            audio = np.array(audio)
        else:
            audio = np.pad(audio, pad_width = (480000-audio.shape[0],0), mode='constant')

        return audio

    def process_audio(self, aud, ):
        wav, sr = librosa.load(aud, sr=self.sr)
        if wav.shape[0] > sr * self.input_length:
            start = random.randint(0, wav.shape[0] - sr * self.input_length - 1)
            end = start + sr * self.input_length
            wav = wav[start:end]
        else:
            wav = np.pad(wav, (0, sr * self.input_length - wav.shape[0]), mode='reflect')
        wav = torch.from_numpy(wav)
        return wav

    def __getitem__(self, i):
        result = {}
        audio_path = self.audio_path[i]

        # 创建临时文件存储提取的音频
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_path = temp_wav.name
        # 使用moviepy提取音频并保存为临时WAV文件
            with VideoFileClip(audio_path) as video:
                audio = video.audio
                audio.write_audiofile(temp_path, codec='pcm_s16le')  # 保存为PCM 16位WAV
        process_audio = self.process_audio(audio_path)
        audio = self.audio_extractor(audios = process_audio, sampling_rate = 48000, return_tensors="pt", padding=True)
        result["audio_values"] = audio
        result['ytid'] = os.path.basename(audio_path)
        os.remove(temp_path)
        return result
