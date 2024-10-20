# Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis

<div align="center">
<img width="1421" alt="Meissonic Banner" src="https://github.com/user-attachments/assets/703f6882-163a-42d0-8da8-3680231ca75e">

[![arXiv](https://img.shields.io/badge/arXiv-2410.08261-b31b1b.svg)](https://arxiv.org/abs/2410.08261)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/MeissonFlow/Meissonic)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/viiika/Meissonic)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorial_EN-FF0000?logo=youtube)](https://www.youtube.com/watch?v=PlmifElhr6M)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorial_JA-FF0000?logo=youtube)](https://www.youtube.com/watch?v=rJDrf49wF64)
[![Demo](https://img.shields.io/badge/Live-Demo-blue?logo=huggingface)](https://huggingface.co/spaces/MeissonFlow/meissonic)

</div>

![Meissonic Demos](./assets/demos.png)

## 🚀 Introduction

Meissonic is a non-autoregressive mask image modeling text-to-image synthesis model that can generate high-resolution images. It is designed to run on consumer graphics cards.

**Key Features:**
- 🖼️ High-resolution image generation (up to 1024x1024)
- 💻 Designed to run on consumer GPUs
- 🎨 Versatile applications: text-to-image, image-to-image

## 🛠️ Prerequisites

### Step 1: Clone the repository
```bash
git clone https://github.com/viiika/Meissonic/
cd Meissonic
```

### Step 2: Create virtual environment
```bash
conda create --name meissonic python
conda activate meissonic
pip install -r requirements.txt
```

### Step 3: Install diffusers
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

## 💡 Usage

### Gradio Web UI

```bash
python app.py
```

### Command-line Interface

#### Text-to-Image Generation

```bash
python inference.py --prompt "Your creative prompt here"
```

#### Inpainting and Outpainting

```bash
python inpaint.py --mode inpaint --input_image path/to/image.jpg
python inpaint.py --mode outpaint --input_image path/to/image.jpg
```

### Advanced: FP8 Quantization

Optimize performance with FP8 quantization:

Requirements:
- CUDA 12.4
- PyTorch 2.4.1
- TorchAO

```shell
python inference_fp8.py --quantization fp8
```

#### Performance Benchmarks

| Precision (Steps=64, Resolution=1024x1024) | Batch Size=1 (Avg. Time) | Memory Usage |
|-------------------------------------------|--------------------------|--------------|
| FP32                                      | 13.32s                   | 12GB         |
| FP16                                      | 12.35s                   | 9.5GB        |
| FP8                                       | 12.93s                   | 8.7GB        |

## 🎨 Showcase

<div align="center">
  <img src="https://github.com/user-attachments/assets/b30a7912-5453-48ba-aff4-bfb547bbe626" width="320" alt="A pillow with a picture of a Husky on it.">
  <p><i>"A pillow with a picture of a Husky on it."</i></p>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/b23a1603-399d-40d6-8e16-c077d3d12a08" width="320" alt="A white coffee mug, a solid black background">
  <p><i>"A white coffee mug, a solid black background"</i></p>
</div>

## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{bai2024meissonic,
  title={Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis},
  author={Bai, Jinbin and Ye, Tian and Chow, Wei and Song, Enxin and Chen, Qing-Guo and Li, Xiangtai and Dong, Zhen and Zhu, Lei and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2410.08261},
  year={2024}
}
```

## 🙏 Acknowledgements

We thank the community and contributors for their invaluable support in developing Meissonic. We thank apolinario@multimodal.art for making Meissonic DEMO. We thank @NewGenAI and @飛鷹しずか@自称文系プログラマの勉強 for making YouTube toturials. We thank @pprp for making fp8 and int4 quantization. 

---

<p align="center">
  <a href="https://star-history.com/#viiika/Meissonic&Date">
    <img src="https://api.star-history.com/svg?repos=viiika/Meissonic&type=Date" alt="Star History Chart">
  </a>
</p>

<p align="center">
  Made with ❤️ by the MeissonFlow Research
</p>
