# DeblurGAN for Tensorflow

This repository provides a script and recipe to train the DeblurGAN model. The code is based on [DeblurGAN's tensorflow implementation](https://github.com/dongheehand/DeblurGAN-tf), modifications are made to run on NPU

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)
  * [Default configuration](#default-configuration)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
  * [Train using GOPRO dataset](#train)
  * [Test model](#test)
  * [Model format conversion](#frozen_model)
* [Advanced](#advanced)
  * [Command line arguments](#command-line-arguments)
  * [Training process](#training-process)
* [Performance](#performance)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)

## Model overview

Tensorflow implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.
`Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, Jiri Matas. "DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks". <https://arxiv.org/abs/1711.07064>.`
reference implementation:  <https://github.com/dongheehand/DeblurGAN-tf>

### Model architecture

The DeblurGAN network takes blurry image as an input and procude the corresponding sharp estimate, as in the example:
<img src="images/animation3.gif" width="400px"/> <img src="images/animation4.gif" width="400px"/>

The model we use is Conditional Wasserstein GAN with Gradient Penalty + Perceptual loss based on VGG-19 activations. Such architecture also gives good results on other image-to-image translation problems (super resolution, colorization, inpainting, dehazing etc.)

<p align="center">
  <img height="360" src="images/architecture.png">
</p>

### Default configuration

The following sections introduce the default configurations and hyperparameters for DeblurGAN model. We use the same training parameters as in the paper.
For detailed hpyerparameters, please refer to corresponding script `main.py`.

- batch_size 1
- patch_size 256 * 256
- learning_rate 1e-4
- max_epoch 300

The following are the command line options about the training scrip:
Parameters for training (Training/Evaluation):
    --train_Sharp_path             Path for sharp train images.
    --train_Blur_path              Path for blur train images.
    --model_path                   Checkpoint save path.
    --model_save_freq              Checkpoint save frequency.
    --learning_rate                Learning_rate.
    --mode                         Train or inference mode.

#### Optimizer

This model uses Momentum optimizer from Tensorflow with the following hyperparameters:

- Adam : 0.9, 0.99

#### Dataset

The first model to which we refer as DeblurGANWILD was trained on a random crops of size 256x256 from 1000

please prepare the GOPRO datasets by yourself 

Use follwoingg scripts to preprocess the datasets:

`python GOPRO_preprocess.py --GOPRO_path ./GOPRO/data/path --output_path ./data/output/path`

The processed images will be saved in the `./data` folder.


## Setup
The following section lists the requirements to start training the DeblurGAN model.
### Requirements

- Python 3.6.5
- Tensorflow 1.10.1
- Pillow 5.0.0
- numpy 1.14.5
- Pretrained VGG19 file : [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) (for training!)

## Quick Start Guide

### Clone the respository

```shell
git clone xxx
```

### Train using GOPRO dataset
1) Download the pre-processed GOPRO datasets
You can use the processed datasets in :

[obs path](s3://deblur-gan/data) `s3://deblur-gan/data`

[obs link](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=6HAr7zmWkpT6AB2oYv4B9uX/MwTPPkcH10N9OLs24+lbkbiPb4xY4ExZ/G4UAtxfDZiAINC/6+j/1Vf64tDp74iYTErNVlXPJZLtfRBdPAk09QILgV/aBDgFOvRTIWhuJoJIeavWatJXxkaJhEi7ChvxvdnJmzKhCwbnNSSNFZKlEhaLJJhpGpjqviFb7d4y2GAcI6KlDViDK0fa/6OdJjtB4bxKeucK536IOpgVQu6aox8UoQcF77bZeyFadx6Qu3OYGeYxMxbOcIduJI4YPIUcta9tWioo/cl6uZ4EA3GxoNtLpTNlhYZVPrr958IF5Pme0JjgtZeWNQbze+Afhz6vhZW/f3IAowLGDziQgdkT9w7pgDFiJ6vqot9N8HsvlFhCjigsi0/ZjOeK89XAzOojy+V5/pt0ciRRYlrq7A3IxNAdeUrJl0AHkLvG3z3tUz1X/ZvEzDwhyFJwvBXoUeZMeldZI3myldU87ZSQxFBXyNS/7WxNBK1JqbDl8hLxJXEhEfVWtkV7UMV94iIu0t2mr4e8wiNW0QiKU7Oq+sE=)

`password : 123456`

In order to calculate the perceptual loss, the weights of the pre-trained vgg19 network are provided???

[obs path](s3://deblurgan/pre_train_model) `obs://deblurgan/pre_train_model`

2) Training with GOPRO dataset.
```
python main.py --train_Sharp_path ./GOPRO/path/sharp --train_Blur_path ./GOPRO/path/blur
```

### Test model
1) Download pre-trained model:

[obs path](obs://deblurgan/pre_train_model) `obs://deblurgan/pre_train_model`

[obs link](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=6HAr7zmWkpT6AB2oYv4B9uX/MwTPPkcH10N9OLs24+lbkbiPb4xY4ExZ/G4UAtxfDZiAINC/6+j/1Vf64tDp74iYTErNVlXPJZLtfRBdPAk09QILgV/aBDgFOvRTIWhuJoJIeavWatJXxkaJhEi7ChvxvdnJmzKhCwbnNSSNFZKlEhaLJJhpGpjqviFb7d4y2GAcI6KlDViDK0fa/6OdJjtB4bxKeucK536IOpgVQu5u8kTXHTZTQfDuJQnf4elxsdd6lpzgKfvxm+Q6kGzw+WTlFqsPdaIm+6xbrYbPNuLYQjRmPdlWZ4IivLVC6+tg+FXOMUi/DSO/XM3s+fqz38coKqth6GgorMyFjzdAYYTq00xcdfa8Og6WafcHP0GyUgNoigo475AAReEqDDfmSVAhAUhvyyDhPwmLZWAOAK6HZ3yIZKb1H/BAW5y/EqQpQUVArr4KhGjZkGlKEUkTlzf8fXc5og4qAWoSztixra1eI94+Zi/dKRpC7WG+O7vf9lhCltRIRkKyipYDwN16X5NULiRa5UfopxiufNlmput5DsXxdZ5iG+ctGqqh2Gkv)

`passwod : 123456`

2) Verify inference results

To verify the model inference results, please use the evalauation.py script:
```
python evaluation.py --HR_data_dir inference_result
```

3) Deblur your own images
```
python main.py --mode test_only --pre_trained_model ./path/to/model --test_Blur_path ./path/to/own/images
```

4) If you have an out of memory(OOM) error, please use chop_forward option
```
python main.py --mode test_only --pre_trained_model ./path/to/model --test_Blur_path ./path/to/own/images --in_memory True --chop_forward True
```

### 4. Model format conversion
You can use ckpt2pb script to convert checkpoint to pb format froze model:
```
python ckpt2pb.py --model_path --ckpt_path --result_path
```
## Advanced
### Commmand-line options

For a complete list of options, please refer to `main.py`

### Training process

All the logs of the training will be stored in the directory `log`.
The pre-trained log is stored in [OBS link of log]:

## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Evaluation results
The model is trained and evaluated on GOPRO dataset. The pre_trained model got PSNR of ** 27.926 ** on 1111 paired test images.

