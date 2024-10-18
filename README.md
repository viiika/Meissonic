# Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis
<div align="center">
<img width="1421" alt="image" src="https://github.com/user-attachments/assets/703f6882-163a-42d0-8da8-3680231ca75e">

<a href='https://arxiv.org/abs/2410.08261'><img src='https://img.shields.io/badge/arXiv-2410.08261-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/MeissonFlow/Meissonic'><img src='https://img.shields.io/badge/Huggingface-Page-Green'></a> &nbsp;
<a href='https://github.com/viiika/Meissonic'><img src='https://img.shields.io/badge/Github-Page-e6cfe6'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=PlmifElhr6M'><img src='https://img.shields.io/badge/Youtube-Toturial-FF8000.svg'></a>&nbsp;
<a href='https://huggingface.co/spaces/MeissonFlow/meissonic'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Demo-blue'></a> &nbsp;

</div>

![demo](./assets/demos.png)

## Introduction
Meissonic is a non-autoregressive mask image modeling text-to-image synthesis model that can generate high-resolution images. It is designed to run on consumer graphics cards.

**Note: This is a project under development. If you encounter any specific performance issues or find significant discrepancies with the results reported in the paper, please submit an issue on the GitHub repository! Thank you for your support!**
## Prerequisites

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


## Usage

### Gradio UI (text2image)
```bash
python app.py
```

### Command-line inference

#### text2image
```shell
python inference.py
```
#### zero-shot inpaint or outpaint
```shell
python inpaint.py --mode inpaint
python inpaint.py --mode outpaint
```

## Some Interesting Examples
```bash
Prompt: "A pillow with a picture of a Husky on it."
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/b30a7912-5453-48ba-aff4-bfb547bbe626" width="320" alt="A pillow with a picture of a Husky on it.">
</div>

```bash
Prompt: "A white coffee mug, a solid black background"
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/b23a1603-399d-40d6-8e16-c077d3d12a08" width="320" alt="A white coffee mug, a solid black background">
</div>



## Citation
If you find this work helpful, please consider citing:
```bibtex
@article{bai2024meissonic,
  title={Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis},
  author={Bai, Jinbin and Ye, Tian and Chow, Wei and Song, Enxin and Chen, Qing-Guo and Li, Xiangtai and Dong, Zhen and Zhu, Lei and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2410.08261},
  year={2024}
}
```
