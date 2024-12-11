# Masked Autoencoders are Parameter-Efficient Federated Continual Learners

<p align="center">
  <img src="./images/overview.png" width="800px">
</p>

## Introduction

This repository contains the official source code for the paper [Masked Autoencoders are Parameter-Efficient Federated Continual Learners](https://arxiv.org/abs/2411.01916).
```
@article{he2024pmae,
  title   = {Masked Autoencoders are Parameter-Efficient Federated Continual Learners},
  author  = {Yuchen He and Xiangfeng Wang},
  year    = {2024},
  journal = {arXiv preprint arXiv:2411.01916}
}
```

## Requirements

To ensure smooth execution of the code, we recommend setting up a dedicated environment using `conda`.

### Steps:

1. First, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Create a new conda environment:

    ```bash
    conda create -n pMAE python==3.9.18
    ```

3. Activate the environment:

    ```bash
    conda activate pMAE
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Datasets

ImageNet-R and CUB-200 dataset can be downloaded from the link provided in [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT). Please specify the folder of your dataset in `src/utils/conf.py`.

## Checkpoints
Please download pre-trained MAE models from the [Releases](https://github.com/ycheoo/pMAE/releases/tag/checkpoints) and then put the pre-trained models to the folder specified in `src/utils/conf.py`.

The frozen pre-trained encoders for the Sup-based MAE and iBOT-based MAE are obtained from [vision_transformer](https://github.com/google-research/vision_transformer) and [ibot](https://github.com/bytedance/ibot), respectively.

## Experiments

Set the `[DATASET]` and `[MODEL]` options using the filenames of the .json files in the configs folder. If the selected model includes pMAE, set the `[METHOD]` to pmae; otherwise, set it to fedavg.

```bash
python src/main_fcl.py --dataset [DATASET] --model [MODEL] --method [METHOD] --device 0
```

### Examples:

```bash
python src/main_fcl.py --dataset cub_T20_beta5e-1 --model sup_pmae --method pmae --device 0
```

---

```bash
python src/main_fcl.py --dataset cub_T20_beta5e-1 --model sup_coda_prompt --method fedavg --device 0
```

---

```bash
python src/main_fcl.py --dataset cub_T20_beta5e-1 --model sup_coda_prompt_w_pmae --method pmae --device 0
```

## Metrics

Run the `results_processor.py` script after completing a specific experiment.

```bash
python results_processor.py --dataset [DATASET] --model [MODEL]
```

### Examples:

```bash
python results_processor.py --dataset cub_T20_beta5e-1 --model sup_pmae
```

---

```bash
python results_processor.py --dataset cub_T20_beta5e-1 --model sup_coda_prompt
```

---

```bash
python results_processor.py --dataset cub_T20_beta5e-1 --model sup_coda_prompt_w_pmae
```

## Acknowledgments

This repo is heavily based on [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT), [MarsFL](https://github.com/WenkeHuang/MarsFL), and [mae](https://github.com/facebookresearch/mae), many thanks.
