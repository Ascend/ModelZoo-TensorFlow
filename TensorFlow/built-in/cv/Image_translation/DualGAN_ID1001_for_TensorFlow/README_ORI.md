
 # ICCV paper of DualGAN
<a href="https://arxiv.org/abs/1704.02510">DualGAN: unsupervised dual learning for image-to-image translation</a>

please cite the paper, if the codes has been used for your research.

# How to setup

## Prerequisites

* Linux

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

* download datasets (e.g., sketch-photo), run:

```
bash ./datasets/download_dataset.sh sketch-photo
```

* download pre-trained model (e.g., sketch-photo), run:

```
bash ./checkpoint/download_ckpt.sh sketch-photo
```

* train the model:

```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

* test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

## optional

Similarly, run experiments on facades dataset with the following commands:

```
bash ./datasets/download_dataset.sh facades

python main.py --phase train --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100

python main.py --phase test --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
