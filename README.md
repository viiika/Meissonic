# Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis

[Paper](https://arxiv.org/abs/2410.08261) | [Model](https://huggingface.co/MeissonFlow/Meissonic) | [Code](https://github.com/viiika/Meissonic) | [Gallery](https://sites.google.com/view/meissonic/gallery?authuser=0)


![demo](./assets/demos.png)

## Introduction
Meissonic is a non-autoregressive mask image modeling text-to-image synthesis model that can generate high-resolution images. It is designed to run on consumer graphics cards.

## Prerequisites

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

## Usage

```bash
python inference.py
```


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
