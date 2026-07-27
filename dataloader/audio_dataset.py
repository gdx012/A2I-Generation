import os
import random
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import ClapAudioModelWithProjection, AutoProcessor, ClapModel, AutoTokenizer, ClapProcessor
import torchaudio
from glob import glob

class AudioDataset(Dataset):
    def __init__(self, args):
        super(AudioDataset, self).__init__()
        image_lst = "image/"
        audio_lst = "audio/"
        balance_dataset = []
        self.image_path = list()
        self.audio_path = list()
        
        for name in os.listdir("./data/A2I_Set/eval_audio_all"):
            audio_path = os.path.join("./data/A2I_Set/eval_audio_all", name)
            if os.path.exists(audio_path):
                self.audio_path.append(audio_path)
        
        self.image_resolution = (512, 512)
        self.bsz = args.batch_size
        self.size = 512
        self.data_set = args.data_set
        self.input_length, self.sr = args.input_length, args.sr
        self.audio_only = not args.training
        self.cfg = args.cfg if hasattr(args, 'cfg') else False
        self.audio_extractor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.num_samples = len(self.audio_path)
        print(f"数据预加载完成！共加载 {self.num_samples} 个样本到内存")
        self._length = self.num_samples
    def __len__(self):
        return self._length

    def process_image(self, image, resolution=(512, 512)):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            v2.Resize(resolution),
            v2.RandomHorizontalFlip(),
        ])

        sd_image = transforms(image)
        return sd_image

    def process_audio(self, aud, ):
        wav, sr = torchaudio.load(aud)
        audio = wav.mean(dim=0).numpy()
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        if audio.shape[0] > sr * 10:
            start = random.randint(0, audio.shape[0] - sr * 10 - 1)
            end = start + sr * 10
            audio = audio[start:end]
            audio = np.array(audio)
        else:
            audio = np.pad(audio, pad_width = (480000-audio.shape[0],0), mode='constant')

        return audio

    def __getitem__(self, i):
        result = {}
        audio_path = self.audio_path[i]

        process_audio = self.process_audio(audio_path)
        audio = self.audio_extractor(audios = process_audio, sampling_rate = 48000, return_tensors="pt", padding=True)
        result["audio_values"] = audio
        result['ytid'] = os.path.basename(audio_path)

        return result
