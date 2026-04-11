


<h1>
    <img src="figures/CCD.png" alt="Image description" style="width: 1.5em; height: 1.5em;vertical-align: bottom; margin-right: 5px;">
    <span>CCD: Capturing Cross Correlations with Deformable Convolutional Networks for Multivariate Time Series Forecasting</span>
</h1>


This code is the official PyTorch implementation of paper: CCD: Capturing Cross Correlations with Deformable Convolutional Networks for Multivariate Time Series Forecasting

## Update Description

We have made the fixed version available at `ts_benchmark/baselines/ccd_fixed`. Specifically, the following fixes have been implemented:

1. In `ccd_fixed/layers/Shuffle.py`, restored the missing `* shuffled_scores` at line 53 compared to the previous version. This line implements one step in Eq. (5), attaching the ranking scores’ gradients to `x` to realize the straight-through proxy for adaptive shuffle learning.

2. In `ccd_fixed/layers/Shuffle.py` at line 13, changed the initialization method to `torch.empty(shuffle_vector_shape, device=device)` to avoid hardcoded CUDA-specific logic, with the device determined by the passed parameter.

3. Removed redundant ablation code from `ccd_fixed/layers/CrossDConv.py` to ensure the simplicity and stability of the open-source version.

4. Updated scripts in the `scripts/CCD` directory to ensure that script parameters correspond to the actual test targets.

## Introduction

In this study, we address these challenges by proposing a gen-
eral framework called **CCD**, which Capturing **C**ross **C**orrelations
with **D**eformable convolutional networks for multivariate time
series Forecasting. First, we design *Adaptive 2-Dimensions Shuffle*,
which adaptively reorders the rows and columns of the image-like
structures, grouping potentially dependent patches to establish lo
cal continuity. Second, we propose *Cross Deformable Convolution*,
which is enhanced by *Non-uniform Cross-Extension initialization*
and *Dynamic mask-based modulation*. Collectively, these designs
render the module more adapted to temporal image-like structures,
empowering it to capture complex and sparse cross correlations
effectively. Extensive experiments on real-world datasets demon-
strate the state-of-the-art performance of CCD.

<div align="center">
<img alt="Logo" src="figures/main.png" width="90%"/>
</div>

## Quickstart

> [!IMPORTANT]
> this project is fully tested under python 3.8, it is recommended that you set the Python version to 3.8.
1. Requirements

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

2. Data preparation

You can obtained the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link). Then place the downloaded data under the folder `./dataset`. 

3. Train and evaluate model

- We provide all the experiment scripts for CCD and other baselines under the folder `./scripts/multivariate_forecast/CCD`.  For example you can reproduce all the experiment results as the following script:

```shell
sh ./scripts/CCD/ETTh2.sh
```

# Reproducibility
We provide all the train logs in `./log` and all the result in `./result` 
