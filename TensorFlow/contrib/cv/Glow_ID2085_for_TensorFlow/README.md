**Status:** Archive (code is provided as-is, no updates expected)

# Glow

- Code for reproducing results in ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf)
- Official Source Code [openai/glow: Code for reproducing results in "Glow: Generative Flow with Invertible 1x1 Convolutions"](https://github.com/openai/glow)

## Requirements

 - python 3.7.5
 - Tensorflow (tested with v1.15.0)
 - Huawei Ascend

Run
```
pip install -r requirements.txt
```

## Download datasets
- MNIST
- cifar-10


## Simple Train with 1 GPU

Run wtih small depth to test
```
CUDA_VISIBLE_DEVICES=0 python train.py --depth 1
```


## CIFAR-10 Quantitative result

```
python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

### Training Log


- GPU：
````
参数：{"verbose": false, "debug": false, "debug_init": false, "restore_path": "", "problem": "cifar10", "data_dir": null, "dal": 1, "check_test_iterator": false, "fmap": 1, "pmap": 16, "n_train": 50000, "n_test": 10000, "n_batch_train": 64, "n_batch_test": 50, "n_batch_init": 256, "optimizer": "adamax", "lr": 0.001, "lr_scalemode": 0, "beta1": 0.9, "polyak_epochs": 1, "beta3": 1.0, "epochs": 1000000, "epochs_warmup": 10, "epochs_full_valid": 50, "gradient_checkpointing": 1, "shift": 1, "image_size": 32, "anchor_size": 32, "width": 512, "depth": 32, "weight_y": 0.0, "n_bits_x": 8, "n_levels": 3, "n_sample": 1, "epochs_full_sample": 50, "eps_beta": 0.95, "learntop": true, "ycond": false, "seed": 0, "flow_permutation": 2, "flow_coupling": 0, "n_y": 10, "local_batch_train": 64, "local_batch_test": 50, "local_batch_init": 256, "direct_iterator": false, "train_its": 196, "test_its": 40, "full_test_its": 50, "debug_logdir": "/root/results/zeus/2018-06-05_ckpt/ours-0000/", "n_bins": 256.0, "top_shape": [4, 4, 48]}

{"epoch": 1, "n_processed": 50176, "n_images": 50176, "train_time": 238, "local_loss": "5.0522", "bits_x": "5.0522", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 2, "n_processed": 100352, "n_images": 100352, "train_time": 439, "local_loss": "4.4926", "bits_x": "4.4926", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 3, "n_processed": 150528, "n_images": 150528, "train_time": 640, "local_loss": "4.3506", "bits_x": "4.3506", "bits_y": "0.0000", "pred_loss": "1.0000"}
……
{"epoch": 298, "n_processed": 14952448, "n_images": 14952448, "train_time": 58669, "local_loss": "3.2133", "bits_x": "3.2133", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 299, "n_processed": 15002624, "n_images": 15002624, "train_time": 58865, "local_loss": "3.2146", "bits_x": "3.2146", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 300, "n_processed": 15052800, "n_images": 15052800, "train_time": 59061, "local_loss": "3.2138", "bits_x": "3.2138", "bits_y": "0.0000", "pred_loss": "1.0000"}

````
- NPU:
````
参数：
Namespace(anchor_size=32, beta1=0.9, category='', dal=1, data_dir=None, data_url='/home/ma-user/modelarts/inputs/data_url_0/', depth=32, direct_iterator=False, epochs=300, epochs_full_sample=50, epochs_full_valid=5, epochs_warmup=10, flow_coupling=0, flow_permutation=2, fmap=1, full_test_its=200, gradient_checkpointing=1, image_size=32, inference=False, learntop=False, local_batch_init=1, local_batch_test=50, local_batch_train=50, logdir='./logs', lr=0.001, n_batch_init=1, n_batch_test=50, n_batch_train=50, n_bins=256.0, n_bits_x=8, n_levels=3, n_sample=1, n_test=10000, n_train=50000, n_y=10, optimizer='adamax', pmap=16, polyak_epochs=1, problem='cifar10', restore_path='', rnd_crop=False, seed=0, test_its=200, top_shape=[4, 4, 48], train_its=10, train_url='/home/ma-user/modelarts/outputs/train_url_0/', verbose=False, weight_decay=1.0, weight_y=0.0, width=512, ycond=False)
epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg
1 50000 50000 8.4 5975.1 0.0 439.5 6414.6 [6.2762194 6.2762194 0.        1.       ] [] 
2 100000 100000 54.9 910.7 0.0 0.0 910.7 [4.081265 4.081265 0.       1.      ] [] 
3 150000 150000 55.0 909.1 0.0 0.0 909.1 [3.8218997 3.8218997 0.        1.       ] [] 
4 200000 200000 54.9 911.6 0.0 0.0 911.6 [3.6507013 3.6507013 0.        1.       ] [] 
5 250000 250000 54.8 911.8 1216.4 0.0 2128.1 [3.5184724 3.5184724 0.        1.       ] [3.9819248 3.9819248 0.        1.       ]

...

70 3500000 3500000 165.3 302.4 14.6 0.0 1527.6 [2.9541063 2.9541063 0. 1. ] [3.5134225 3.5134225 0. 1. ] *
75 3750000 3750000 165.1 302.8 14.5 0.0 1527.7 [2.949904 2.949904 0. 1. ] [3.5093305 3.5093305 0. 1. ] *
80 4000000 4000000 165.4 302.4 14.2 0.0 1527.2 [2.9465308 2.9465308 0. 1. ] [3.5025241 3.5025241 0. 1. ] *
85 4250000 4250000 165.2 302.7 12.2 0.0 1526.1 [2.9442344 2.9442344 0. 1. ] [3.5053678 3.5053678 0. 1. ]
90 4500000 4500000 164.9 303.2 14.7 0.0 1529.7 [2.9408553 2.9408553 0. 1. ] [3.4954607 3.4954607 0. 1. ] *
95 4750000 4750000 165.0 303.0 12.2 0.0 1527.0 [2.9360971 2.9360971 0. 1. ] [3.496417 3.496417 0. 1. ]

````

#### 精度达标

bits_x NPU：2.9

bits_x GPU: 3.213

#### 性能达标

NPU Ascend910： 0.84 s / step

GPU v100： 0.85 s / step


