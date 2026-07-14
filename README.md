<div align="center"> 
  
## Towards Expressive and Faithful Audio-to-Image Generation: <br/> A Unified Multimodal Dataset and Synthesis Framework

[Dongxu Ge], [Shansong Liu], [Cheng Gong], [Xiao-Lei Zhang], [Chi Zhang], [Xuelong Li]

University of Science and Technology of China, Tele AI

</div>
<p align="center">
<a href="http://arxiv.org/abs/2408.01343">
    <img src="https://img.shields.io/badge/arXiv-2408.01343-green" /></a>
<a href="https://huggingface.co/datasets/gdx123/A2I-Set">
    <img src="https://img.shields.io/badge/Dataset-A2ISet-red.svg" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" /></a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

Official implementation of paper "Towards Expressive and Faithful Audio-to-Image Generation: A Unified Multimodal Dataset and Synthesis Framework".

</div>

## 💬 Introduction

As an important subfield of cross-modal generation, synthesizing static visual content in the form of images from audio, namely audio-to-image (A2I) generation, has attracted increasing research attention in recent years. Nevertheless, despite the remarkable visual quality of modern text-to-image (T2I) models, the performance of A2I remains fundamentally limited by traditional datasets, which often lack both high-fidelity images and precise cross-modal alignment. As a result, existing methods still struggle to achieve high-quality audio-to-image generation through finetuning strong T2I models, thereby constraining practical applications in this area. Motivated by this gap, we introduce A2I-Set, a unified, high-quality tri-modal dataset consisting of 323K paired audio, images, and detailed text captions, specifically designed for audio-visual research, including audio-conditioned image generation. Besides, we developed a new mixed-source test set for the A2I task through human supervision. We further propose an A2I model, AudioCanvas, fine-tuned on our A2I-Set. Experiments show that AudioCanvas achieves more visually expressive as well as cross-modal alignment results that consistently surpass existing approaches. 

## 🌟 News
🎉 Our Paper is accepted by **ACM International Conference on Multimedia (ACMMM 2026)**, congratulations! 🎉


## 🚀 Updates​
- [x] 2026/4/29: Released our dataset A2I-Set.​
- [x] 2026/7/14: Updated environment setup for AudioCanvas.

## 🚩 TO DO
- [ ] Open source evaluation scripts.
- [ ] Open source training and inference codes of AudioCanvas.
- [ ] Upload AudioCanvas' training weights.
​
## 📦 Data preparation

Download our dataset A2I-Set from: https://huggingface.co/datasets/gdx123/A2I_Set

Merge the compressed packages and extract them, A2I_audio for example：
```
cat A2I_audio.tar.gz.part* > A2I_audio.tar.gz
tar -zxvf A2I_audio.tar.gz
```

Then, put all dataset under `data` directory as follows:
```
data/
├── A2I_audio
│   ├── [audio_tag1].wav
│   ├── [audio_tag2].wav
│   ├── [audio_tag3].flac
│   ├── ...

├── A2I_real
│   ├── [audio_tag1].png
│   ├── [audio_tag1].png
│   ├── ...

├── A2I_syn
│   ├── [audio_tag1].mp4_[frame_idx1].png
│   ├── [audio_tag1].mp4_[frame_idx2].png
│   ├── [audio_tag2].mp4_[frame_idx1].png
│   ├── ...

├── eval_audio_all
│   ├── [audio_tag1]|cls1|cls2.wav
│   ├── [audio_tag2]|cls1|cls2|cls3.wav
│   ├── ...

├── caption.json
```

To sum, A2I-Set has 260,060 audios with 241 hierarchical labels and captions, 193,653 synthetic images, 129,793 real images and 3280 human-labeled audios for evaluation.

A caption example shows below:
```
"t2i_caption_Skz--pje298": {
        "[Main Subject]": "<male voice>",
        "[Visual Generation Instruction]": "A well-lit presentation room with a man standing confidently at a podium, focused on delivering an instructional talk. He is dressed in business casual attire, gesturing slightly with his hands to emphasize points. The background is uncluttered, with a projected slide behind him featuring technical diagrams, creating a professional and engaging atmosphere.",
        "[Top Level]": "speech"
    }
```

## ⚙️ Environment Setup

We tested the code on:

- **OS**: Ubuntu 22.04
- **Hardware**: 8 × NVIDIA H100 80GB GPUs
- **Python**: 3.12.12

```
cd /path/to/A2I-Generation
# Create and activate the virtual environment if needed
conda env create -n A2I
conda activate A2I
pip install -r requirements.txt
```


## 🔍 AudioCanvas


## 🙏 Acknowledgements

This work builds upon and integrates components from the following open-source projects:

- MACS
- [AudioToken](https://github.com/guyyariv/AudioToken)
- [FusionAudio-1.2M](https://github.com/FreedomIntelligence/FusionAudio)
- [AudioSet](https://research.google.com/audioset/)
- [VGGSound](https://github.com/hche11/VGGSound)

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


## 📚 Citation




