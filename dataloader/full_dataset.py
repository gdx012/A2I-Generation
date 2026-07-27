import os
import random
import numpy as np
import cv2
import torchaudio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm
import json
import librosa
import pandas as pd
from transformers import ClapAudioModelWithProjection, AutoProcessor, ClapModel, AutoTokenizer, ClapProcessor



class Fulldataset(Dataset):
    def __init__(self, args):
        super(Fulldataset, self).__init__()
        image_lst = "image/"
        audio_lst = "audio/"
        balance_dataset = []
        self.image_path = list()
        self.audio_path = list()
        
        self.only_real = args.only_real if hasattr(args, 'only_real') else False
        self.only_gen = args.only_gen if hasattr(args, 'only_gen') else False

        
        for data in os.listdir("./data/A2I_Set/A2I_real"):       
            img_path = os.path.join("./data/A2I_Set/A2I_real", data)
            if self.only_gen:
                continue
            if not os.path.exists(img_path):
                continue
            self.image_path.append(img_path)
            name = img_path.split("/")[-1].split(".")[0]
            audio_path = f"./data/A2I_Set/A2I_audio/{name}.flac" if os.path.exists(f"./data/A2I_Set/A2I_audio/{name}.flac") else f"./data/A2I_Set/A2I_audio/{name}.wav"
            self.audio_path.append(audio_path)


        for data in os.listdir("./data/A2I_Set/A2I_syn"):       
            img_path = os.path.join("./data/A2I_Set/A2I_syn", data)
            if self.only_real:
                continue
            if not os.path.exists(img_path):
                continue
            self.image_path.append(img_path)
            name = img_path.split("/")[-1].split(".")[0]
            audio_path = f"./data/A2I_Set/A2I_audio/{name}.flac" if os.path.exists(f"./data/A2I_Set/A2I_audio/{name}.flac") else f"./data/A2I_Set/A2I_audio/{name}.wav"
            self.audio_path.append(audio_path)

        with open("./data/A2I_Set/caption.json", "r") as f:
            self.data = json.load(f)



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

    def process_image(self, image, resolution=(512, 512)):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if h == w:
            transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((584,584)),
                v2.RandomCrop(resolution),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                v2.RandomHorizontalFlip(),
            ])
        else:
            scale = 512 / min(h, w)
            new_h = int(max(0.8 * h * scale, 512))
            new_w = int(max(0.8 * w * scale, 512))
            transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((new_h, new_w)),
                v2.RandomCrop(resolution),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                v2.RandomHorizontalFlip(),
            ])
        sd_image = transforms(image)
        return sd_image

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

    def __getitem__(self, i):
        result = {}
        audio_path = self.audio_path[i]
        image_path = self.image_path[i]
        name = "t2i_caption_" + audio_path.split("/")[-1].split(".")[0]
        objects = self.data[name]['[Main Subject]'].replace("<", "").replace(">", "").split(', ')
        caption = self.data[name]['[Visual Generation Instruction]']
        raw_text = caption if random.random() < 0.3 else " and ".join(objects)
        if not self.audio_only:
            try:
                result["pixel_values"] = self.process_image(image_path)
            except:
                print(image_path)
        process_audio = self.process_audio(audio_path)
        audio = self.audio_extractor(audios = process_audio, sampling_rate = 48000, return_tensors="pt", padding=True)
        
        result["audio_values"] = audio
        result['ytid'] = image_path.split("/")[-1].replace(".png","")
        result['raw_text'] = raw_text

        return result