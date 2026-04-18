# 【ICCV' 2025】Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model
###  Kai Tong, Kang Pan, Xiao Zhang, Erli Meng, Run He, Yawen Cui, Nuoyan Guo, Huiping Zhuang* 

## Introduction
This is the official implementation for Any-SSR [Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model](https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html).

## Overview

<div align="center">
<img src="imgs/overview.png" width="800px">
</div>

## Environment
We recommend using the [Anaconda](https://anaconda.org/) to install the development environment.

```bash
git clone --depth=1 https://github.com/ZHUANGHP/Any-SSR.git

cd Any-SSR
conda env create -n anyssr-olora
conda activate anyssr-olora
pip install -r requirements.txt

```
## Quick Start

### Lora Model Training

You can use
```bash
bash train_lora.sh
```
to train a lora model for each task in the Trace dataset. 


<div align="center">
<img src="imgs/infer_results.png" width="800px">
</div>



## From new branch called Analytic Continual Learning
This is the first LLM member from the continual learning branch: [Analytic Continual Learning](https://github.com/ZHUANGHP/Analytic-continual-learning). We have published over 20 papers in this branch (check [My Scholar](https://scholar.google.com.sg/citations?user=vCXxuLkAAAAJ&hl=en))!

## Cite our paper
If you find our paper or this repository useful, please kindly consider citing our paper.

```bib
@InProceedings{Tong_2025_ICCV,
    author    = {Tong, Kai and Pan, Kang and Zhang, Xiao and Meng, Erli and He, Run and Cui, Yawen and Guo, Nuoyan and Zhuang, Huiping},
    title     = {Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {3047-3057}
}
```