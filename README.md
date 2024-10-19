# Meissonic: Revolutionary Masked Generative Transformers for High-Resolution Text-to-Image Synthesis

<div align="center">
<img width="1421" alt="Meissonic Banner" src="https://github.com/user-attachments/assets/703f6882-163a-42d0-8da8-3680231ca75e">

[![arXiv](https://img.shields.io/badge/arXiv-2410.08261-b31b1b.svg)](https://arxiv.org/abs/2410.08261)
[![Hugging Face](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/MeissonFlow/Meissonic)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/viiika/Meissonic)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorial-FF0000?logo=youtube)](https://www.youtube.com/watch?v=PlmifElhr6M)
[![Demo](https://img.shields.io/badge/Live-Demo-blue?logo=huggingface)](https://huggingface.co/spaces/MeissonFlow/meissonic)

</div>

![Meissonic Demos](./assets/demos.png)

## 🚀 Introduction

Meissonic is a groundbreaking non-autoregressive masked image modeling model for text-to-image synthesis. It pushes the boundaries of what's possible in content creation, producing stunning high-resolution images with unprecedented efficiency on consumer-grade hardware.

**Key Features:**
- 🖼️ High-resolution image generation (up to 1024x1024)
- 💻 Designed to run on consumer GPUs
- 🎨 Versatile applications: text-to-image, image-to-image

## 📰 News
- **[Oct 19, 2024]** FP8 inference code is available!
- **[Oct 18, 2024]** Gradio webui for local inference is available!
- **[Oct 14, 2024]** [Official Website-Meissonic](https://sites.google.com/view/meissonic/home) is launched!
- **[Oct 14, 2024]** Meissonic-1.0 is released!


## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (11.0+)

### Quick Start

```bash
git clone https://github.com/viiika/Meissonic
cd Meissonic
conda create --name meissonic python=3.8
conda activate meissonic
pip install -r requirements.txt
```

## 💡 Usage

### Gradio Web UI

Experience Meissonic through our intuitive web interface:

```bash
python app.py
```

### Command-line Interface

#### Text-to-Image Generation

```shell
python inference.py --prompt "Your creative prompt here"
```

#### Inpainting and Outpainting

```shell
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

If Meissonic contributes to your research, please cite our paper:

```bibtex
@article{bai2024meissonic,
  title={Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis},
  author={Bai, Jinbin and Ye, Tian and Chow, Wei and Song, Enxin and Chen, Qing-Guo and Li, Xiangtai and Dong, Zhen and Zhu, Lei and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2410.08261},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## 📄 License

Meissonic is released under the Apache-2.0 license. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgements

We'd like to thank the open-source community and our contributors for their invaluable support in making Meissonic a reality.

---

<p align="center">
  <a href="https://star-history.com/#viiika/Meissonic&Date">
    <img src="https://api.star-history.com/svg?repos=viiika/Meissonic&type=Date" alt="Star History Chart">
  </a>
</p>

<p align="center">
  Made with ❤️ by the MeissonFlow Research
</p>
