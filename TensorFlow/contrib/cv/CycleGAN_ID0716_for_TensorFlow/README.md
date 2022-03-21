# 概述
CycleGAN TensorFlow实现，适配Ascend平台。


* 循环生成对抗网络，实现图像域到图像域之间的翻译。
* Original paper: https://arxiv.org/abs/1703.10593


## 训练环境

* TensorFlow 1.15.0
* Python 3.7.0

## 代码及路径解释


```
cyclegan
└─ 
  ├─README.md
  ├─LICENSE
  ├─offline_inference 310推理
    ├─freeze_graph.py  冻结图
    ├─jig2bin.py 图片转bin格式数据
    ├─bin2jpg.py 310推理结果转为.jpg图片
  ├─scripts
    ├─download_dataset.sh 原作者代码仓下载数据集脚本
    ├─inference_gpu.sh   cpu/gpu推理启动脚本，可推理一个目录下的所有图片
    ├─run_1p.sh 模型训练启动脚本，支持 [cpu/gpu/npu]
  ├─build_data.py 制作tfrecords数据集
  ├─config.py 
  ├─discriminator.py 判别器网络
  ├─generator.py 生成器网络
  ├─model.py 
  ├─ops.py
  ├─reader.py
  ├─utils.py
  ├─train.py
  ├─export_graph.py  导出模型
  ├─inference.py     生成器生成图片
```

## 数据集
```
选择使用 horse2zebra 数据集。
```

## 快速上手

* 下载数据集  
原文的官方代码仓中有下载数据集的链接，请自行下载。
或参考如下脚本：  
```bash
$ sh download_dataset.sh
```
  
* 制作tfrecords数据集  
网络训练数据集需制作为tfrecords格式。数据集结构：
 ```
      horse2zebra  
            ├─X.tfrecords  #trainA
            ├─Y.tfrecords  #trainB
```

```bash
$ python3 build_data.py --X_input_dir=${trainA_dir} --Y_input_dir=${trainB_dir} --X_output_file=X.tfrecords --Y_output_file=Y.tfrecords
```

Check `$ python3 build_data.py --help` for more details.

* NPU 训练

```bash
$ sh run_1p.sh
```

若想更改参数设置, 可通过命令行执行训练:

```bash
$ python3 train.py  \
    --chip=npu \
    --result=../result/ \
    --platform=apulis \
    --train_epochs=100 \
    --dataset=../data/tfrecords/horse2zebra/ \
    
```
* GPU 训练  
指定train.py中的参数 chip = gpu 即可：
```bash
$ python3 train.py  \
    --chip=gpu \
    --platform=linux \
    --train_epochs=100 \
    --dataset=../data/tfrecords/horse2zebra/ \ 
```

  
* 网络参数说明:
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan]
                [--norm NORM] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--pool_size POOL_SIZE] [--ngf NGF] 
                [--load_model LOAD_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_size IMAGE_SIZE
                        image size, default: 256
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True
  --nouse_lsgan
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default:
                        10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --pool_size POOL_SIZE
                        size of image buffer that stores previously generated
                        images, default: 50
  --ngf NGF             number of gen filters in first conv layer, default: 64
  
  --load_model LOAD_MODEL
                        folder of saved model that you wish to continue
                        training (e.g. 20170602-1936), default: None
```




若需要加载断点继续训练,可通过设置参数  `load_model`:

```bash
$ python3 train.py  \
    --load_model 20170602-1936
```

Notes: If high constrast background colors between input and generated images are observed (e.g. black becomes white), you should restart your training!
Train several times to get the best models.


* 导出模型

You can export from a checkpoint to a standalone GraphDef file as follow:

```bash
$ python3 export_graph.py --checkpoint_dir checkpoints/${datetime} \
                          --XtoY_model horse2zebra.pb \
                          --YtoX_model zebra2horse.pb \
                          --image_size 256
```
* GPU 推理

After exporting model, you can use it for inference. Input picture size: 256*256


For example:

```bash
$ python3 inference.py --model pretrained/horse2zebra.pb \
                     --input input_sample.jpg \
                     --output output_sample.jpg \
                     --image_size 256
```
## 超参设置
```
* 数据集：
 datasets size: 256*256  format: .jpg
      horse2zebra  
            ├─trainA(1067)
            ├─trainB(1334)
            ├─testA(120)
            ├─testB(140)
          
* epoch: 200k steps(~200 epochs)
* 学习率: 2e-4 for the first 100k steps (~100 epochs),a linearly decaying rate that goes to zero over the next 100k steps.
* 其他超参: 默认值.
* 训练时长：NPU单卡约 67 h.
```
## 精度对比
训练 200k steps的生成器生成图片样例：

* GPU训练结果  
  testA:  
  ![pics](./imgs/out_gpu/a/output_gpu_1.jpg)![pics](./imgs/out_gpu/a/output_gpu_2.jpg)![pics](./imgs/out_gpu/a/output_gpu_3.jpg)![pics](./imgs/out_gpu/a/output_gpu_4.jpg)  
  testB:  
  ![pics](./imgs/out_gpu/b/output_gpu_0.jpg)![pics](./imgs/out_gpu/b/output_gpu_1.jpg)![pics](./imgs/out_gpu/b/output_gpu_2.jpg)![pics](./imgs/out_gpu/b/output_gpu_3.jpg)

* NPU训练结果  
testA:  
![pics](./imgs/out_npu/a/out70.jpg)![pics](./imgs/out_npu/a/out27.jpg)![pics](./imgs/out_npu/a/out29.jpg)![pics](./imgs/out_npu/a/out31.jpg)  

testB:  
![pics](./imgs/out_npu/b/out38.jpg)![pics](./imgs/out_npu/b/out54.jpg)![pics](./imgs/out_npu/b/out85.jpg)![pics](./imgs/out_npu/b/out113.jpg)



## 训练性能

仅支持单卡训练。训练性能对比：

| 平台     | BatchSize | 训练性能(FPS) |
|----------|---|--------------|
| NPU      |  1 |   1.002  steps/sec     |
| GPU V100 |  1 |   4.098  steps/sec     |


