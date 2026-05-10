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
conda create -n anyssr-olora python=3.11
conda activate anyssr-olora
pip install -r requirements.txt

pip uninstall torch
pip3 install torch --index-url https://download.pytorch.org/whl/cu128
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn==2.7.2.post1 --no-build-isolation

```

Create `.env` file in the root directory. Fill in your Wandb API key:
```
bash
WANDB_API_KEY=your_api_key
```

## Quick Start

### Baselines
To use O-LoRA
```bash
mkdir logs/
bash scripts/train_OLoRA.sh 2>&1 | tee logs/train_OLoRA.log
```

To use LwF
```bash
mkdir logs/
bash scripts/train_LwF.sh 2>&1 | tee logs/train_Lwf.log
```

To use PP
```bash
mkdir logs/
bash scripts/train_PP.sh 2>&1 | tee logs/train_PP.log
```


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