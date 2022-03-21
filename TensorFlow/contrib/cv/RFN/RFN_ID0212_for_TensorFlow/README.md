# RFN for Tensorflow

This repository provides a script and recipe to train the RFN model. The code is based on [RFN's PyTorch implementation](https://github.com/Zheng222/PPON), modifications are made to run on NPU.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)
  * [Default configuration](#default-configuration)
* [Quick start guide](#quick-start-guide)
  * [Train model](#train-model)
  * [Evaluate model](#evaluate-model)
* [Performance](#performance)
  * [Training accuracy](#training-accuracy)
  * [Training speed](#training-speed)

## Model overview
This is an PyTorch implementation of the RFN network proposed in "Progressive Perception-Oriented Network for Single Image Super-Resolution [ [pdf](https://arxiv.org/abs/1907.10399) ]". The single image super-resolution aims to estimate the SR image  from its LR counterpart.


### Model architecture
An overall structure of the proposed basic model (RFN)
is shown as Figure 1. This network mainly consists of two parts: content feature extraction module (CFEM) and reconstruction part, where the first part extracts content features for conventional image SR task (pursuing high PSNR value), and the second part naturally reconstructs SR through the front features related to the image content. 

<p align="center">
  <img src="images/RFN.png" width="500px"/> 
</p>
<p align="center">
  Figure 1: The network architecture of RFN
</p>

<p align="center">
  <img src="images/RRFB.png" width="500px"/> 
</p>
<p align="center">
  Figure 2: The basic blocks proposed in RFN
</p>


### Default configuration
The following sections introduce the default configurations and hyperparameters for RFN model. For detailed hpyerparameters, please refer to corresponding script `main.py`.

- batch_size 8
- patch_size 192 * 192
- learning_rate 2e-4
- lr_decay 0.7 per 100000 steps
- total steps 1000000

The following are the command line options about the training scrip:

    --lr_dir                      Path to the lr dataset.
    --hr_dir                      Path to the hr dataset.
    --model_dir                   Path to save model ckpt.
    --mini_steps                  Steps for a stage
    --stage                       Number of stages
    --lr_decay                    Decay rate between stages


## Quick Start Guide

### Clone the respository

```shell
git clone xxx
```

### Train model
1) Download the DF2K (DIV2K + Flickr2k) dataset and unzip it into a folder.

2) Train the model with the following commend:

```
python train.py --lr_dir ./path/to/lr_dataset/ --hr_dir ./path/to/hr_dataset/ --model_dir ./path/to/save/model/
```

You can download the pre_trained model throught the [obs link](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=SffjI83GBWcDQoSGC7c4BJvHjv2MeaQRWR5yKayURPzED5vrzXhTEgT5FNDrBfn+OJOczeraXlJbsTj54iVDnOUP6pr43VQmTrCIU9YebNbopyoPzM15bAmlcGrt55F3ckTwHF7sXcjUtCKDYSa8ue3RFjGIX9pxpLDgzyp2/RMfu6UPu8V3nU1IIMxqHksa1bDlNapKc9F8s7iilYSqYmpIQaNzsU9PoGwJSHr0c5ZLwikVchpuaEQKC8F2fMKUjxiol0K2qpbAeuw4RXt3STg8DXmZkxnhK0wInY4WJ0O1IWB5kEYCpETWRo8ZfySjvFNxa783dx5Q2xNhIMRFVi7rGQ8LCv5V3KTnK+z5AXgK+wj7YCbusCtZT4mvFM7TGtO9M947YNHTXvROgzn7v1Mt11CL2kBJjZA1bKEUEaicW18UGb4F+dwdctI1W4U5bXdVIZOfRqHaFEhOehhVaZekRaDfCoReor9Dnkt6e2v+HEvA21zjWhwJDYl97wHjlLWCsODEIsC4DZ75UePqrkPZl72fIJn1htDiglUTaCD9xu67939oN7fjixakL8h1) (password: 123456) or [BaiduNetDisk](https://pan.baidu.com/s/10im7mJNoQbjsUueQv3lLLw) (password: 3404)

### Evaluate model
Then you can evaluate the pre_trained model with the following commend:
```
python test.py --lr_dir_test ./path/to/lr_dataset/ --hr_dir_test ./path/to/hr_dataset/ --model_dir ./path/to/pre_trained/model/ --gpu gpu_id --path ./path/to/specific/model/
```

## Performance

### Training accuracy
We trained the model with DF2K dataset and evaluated the model with Set14 dataset with PSNR 28.75 which is 28.95 in the paper.


### Training speed
Here is the training speed comparison between NVIDIA TESLA V100 and Ascend 910.

| NVIDIA TESLA V100 | Ascend 910 |
| ------- | -------|
| 0.607s / step | 0.180s / step |
| 60.735s / 100steps | 18.020s / 100steps |