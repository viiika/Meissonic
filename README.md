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

## Some Interesting Examples
```bash
Prompt: "a Highly-Beautiful LOGO with a 3D Pure-Color Letter 'M' painted on top with Art Background."
```
![a Highly-B_1024_64_9](https://github.com/user-attachments/assets/fddbb297-2598-4230-8add-45da385daf78)


```bash
Prompt: "A white coffee mug, a solid black background"
```
![A white co_1024_64_9](https://github.com/user-attachments/assets/b23a1603-399d-40d6-8e16-c077d3d12a08)




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
