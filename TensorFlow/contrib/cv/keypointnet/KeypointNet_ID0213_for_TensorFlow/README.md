# Keypointnet for Tensorflow

This repository provides a script and recipe to train the Keypointnet model. The code is based on [Keypointnet's tensorflow implementation](https://github.com/tensorflow/models/tree/v1.13.0/research/keypointnet), modifications are made to run on NPU.

## Table Of Contents

* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)
  * [Default configuration](#default-configuration)
* [Quick start guide](#quick-start-guide)
  * [Dataset](#dataset)
  * [Train model](#train-model)
* [Performance](#performance)
  * [Training accuracy](#training-accuracy)
  * [Training speed](#training-speed)

## Model overview
This is an tensorflow implementation of the keypoint network proposed in "Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning [ [pdf](https://arxiv.org/pdf/1807.03146.pdf) ]". Given a single 2D image of a known class, this network can predict a set of 3D keypoints that are consistent across viewing angles of the same object and across object instances. These keypoints and their detectors are discovered and learned automatically without keypoint location supervision.


### Model architecture
During training, two views of the same object are given as input to the KeypointNet. The known rigid transformation (R, t) between the two views is provided as a supervisory signal. We optimize an ordered list of 3D keypoints that are consistent in both views and enable recovery of the transformation.

<p align="center">
  <img src="images/Keypointnet.png" width="600px"/> 
</p>

During inference, KeypointNet extracts 3D keypoints from an individual input image. The output image is as follows.

<p align="center">
  <img  src="images/result.png" width="600px"/>
</p>


### Default configuration
The following sections introduce the default configurations and hyperparameters for Keypointnet model. For detailed hpyerparameters, please refer to corresponding script `main.py`.

- batch_size 16
- patch_size 128 * 128
- learning_rate 2e-4
- steps 700000

The following are the command line options about the training scrip:

    --data_url                     Path to the dataset.
    --model_dir                    Path to save model ckpt.
    --tf_log_dir                   Path to save summary file.
    --test                         Evaluate pre_trained model. 


## Quick Start Guide

### Clone the respository

```shell
git clone xxx
```

### Dataset:
  ShapeNet's rendering for cars,plants,chairs;

dataset can be  preprocessed according to GPU reference.

  Each set contains:
1. tfrecords
2. train.txt, a list of tfrecords used for training.
2. dev.txt, a list of tfrecords used for validation.
3. test.txt, a list of tfrecords used for testing.
4. projection.txt, storing the global 4x4 camera projection matrix.
5. job.txt, storing ShapeNet's object IDs in each tfrecord.

### Train model
1) Download the dataset mentioned above and unzip it into a folder.


2) Train the model with the following commend:
```
python train.py --data_url ./path/to/dataset/ --model_dir ./path/to/save/model/ --tf_log_dir ./path/to/summary/file/
```

You can download the pre_trained model throught the [obs link](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=SffjI83GBWcDQoSGC7c4BJvHjv2MeaQRWR5yKayURPxhfmMQCdngxHqwltBEkNshagw/+WTtXTpDdUGSxD0Z2kruI8SDAESj/4Gj+bsVunp1Wpn0k706qYzEfuE6Vg99xplHUYF/XfCrSm5/EvSdYsHhCmvX2dEQDE+9UkS+x8rkNmV31Pvhj46+Pxia/skI5pB8nnNBkGNqaSPe1WUlqif6TvGPdAHeyeF2AWpY35EX8ADhmbb/lhNxk5FQPmmRGqxPlrPipuq62G8KBbZcfDSQJyYXRh+SB8STGTjGVINS8f4FUkBmGRxje69d0NEoMsr3gio4XmafyT5+cg1gWPQd+kRVwmDcRw1VupT/sHuGQ1NMLKWpodGIJk5pU6EkSoUU8M8fdLBhr8E8d1tXaiURHvwY7PgENNVm8UkhrzdtVpn2omFA22HIdFghXHrR921OdBnYC3PUf3DVi58osMdPMAHVwhlpzPa2OBn+RLxyEQegUBhw++hLY042dCEJZaby8BD1hLhGZzxZImZhmn/0q2LlnNKAv5sH41X2lLVhnxjwFYH3VhjSOpXRqvYF9MM/WRImbqhLDaZROJEzXYASVP9IlhdBOKu5eMr2Ln9zn6cyeiP5GcEZ368977L3rfZ0dv/+k8JA/GlJzb0JDyO1hLEV3t6pfMf5QKiGPT4=) (password: 123456) or [BaiduNetDisk](https://pan.baidu.com/s/1IWHL7ZLeHdoRuIN1lIAx-A) (password: wr43)

### Evaluate model
Then you can evaluate the pre_trained model with the following commend:
```
python test.py --data_url ./path/to/dataset/ --model_dir ./path/to/pre_trained/model/ --gpu gpu_id --path ./path/to/specific/model/
```

## Performance

### Training accuracy
We trained and evaluated the model in the ShapeNet's rendering for chairs. The mean angular distance error of the pretrained model is 13.61?? which is 14.57?? in the paper(smaller is better).

### Training speed
Here is the training speed comparison between NVIDIA TESLA V100 and Ascend 910.

| NVIDIA TESLA V100 | Ascend 910 |
| ------- | -------|
| 0.310s / step | 0.175s / step |
| 31.053s / 100steps | 17.468s / 100steps |
