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
参数：{"data_url": "/home/ma-user/modelarts/inputs/data_url_0/", "train_url": "/home/ma-user/modelarts/outputs/train_url_0/", "verbose": false, "restore_path": "", "inference": false, "logdir": "./logs", "problem": "cifar10", "category": "", "data_dir": null, "dal": 1, "fmap": 1, "pmap": 16, "n_train": 500, "n_test": 10000, "n_batch_train": 50, "n_batch_test": 50, "n_batch_init": 1, "optimizer": "adamax", "lr": 0.001, "beta1": 0.9, "polyak_epochs": 1, "weight_decay": 1.0, "epochs": 300, "epochs_warmup": 10, "epochs_full_valid": 5, "gradient_checkpointing": 1, "image_size": 32, "anchor_size": 32, "width": 512, "depth": 32, "weight_y": 0.0, "n_bits_x": 8, "n_levels": 3, "n_sample": 1, "epochs_full_sample": 50, "learntop": false, "ycond": false, "seed": 0, "flow_permutation": 2, "flow_coupling": 0, "n_y": 10, "rnd_crop": false, "local_batch_train": 50, "local_batch_test": 50, "local_batch_init": 1, "direct_iterator": false, "train_its": 10, "test_its": 200, "full_test_its": 200, "n_bins": 256.0, "top_shape": [4, 4, 48]}

{"epoch": 1, "n_processed": 500, "n_images": 500, "train_time": 3786, "loss": "15.6115", "bits_x": "15.6115", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 2, "n_processed": 1000, "n_images": 1000, "train_time": 3796, "loss": "7.3706", "bits_x": "7.3706", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 3, "n_processed": 1500, "n_images": 1500, "train_time": 3806, "loss": "6.3911", "bits_x": "6.3911", "bits_y": "0.0000", "pred_loss": "1.0000"}
……
{"epoch": 297, "n_processed": 148500, "n_images": 148500, "train_time": 6726, "loss": "3.2871", "bits_x": "3.2871", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 298, "n_processed": 149000, "n_images": 149000, "train_time": 6736, "loss": "3.2421", "bits_x": "3.2421", "bits_y": "0.0000", "pred_loss": "1.0000"}
{"epoch": 299, "n_processed": 149500, "n_images": 149500, "train_time": 6746, "loss": "3.2897", "bits_x": "3.2897", "bits_y": "0.0000", "pred_loss": "1.0000"}

````

#### 性能分析


#### 精度分析


