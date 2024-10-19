# Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis

[Paper](https://arxiv.org/abs/2410.08261) | [Model](https://huggingface.co/MeissonFlow/Meissonic) | [Code](https://github.com/viiika/Meissonic) | [Gallery](https://sites.google.com/view/meissonic/gallery?authuser=0)


![demo](./assets/demos.png)

## Introduction
Meissonic is a non-autoregressive mask image modeling text-to-image synthesis model that can generate high-resolution images. It is designed to run on consumer graphics cards.

**Note: This is a project under development. If you encounter any specific performance issues or find significant discrepancies with the results reported in the paper, please submit an issue on the GitHub repository! Thank you for your support!**
## Prerequisites

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

## Usage
#### text2image
```shell
python inference.py
```
#### zero-shot inpaint or outpaint
```shell
python inpaint.py --mode inpaint
python inpaint.py --mode outpaint
```

### fp8 quantization 

Requirements:
- CUDA 12.4 
- torch==2.4.1
- torchao 

```shell
python inference_fp8.py --quantization fp8 
```

| **Precision (Step=64, Res=1024)** | **BS=1 (Average Time)** | **Mem**  |
|-----------------------------------|--------------------------|---------|
| fp32                              | 13.32s                   | 12G     |
| fp16                              | 12.35s                   | 9.5G    |
| fp8                               | 12.93s                   | 8.7G    |


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
