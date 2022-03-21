# Revisiting self-supervised
## 概述
自我监督技术最近成为了无监督的有效解法之一，然而目前对于自监督的研究大量集中在pretext任务，而如卷积神经网络(CNN)的选择，却没有得到同等的重视。因此，作者回顾了之前提出的许多自我监督模型，进行了一项彻底的大规模研究，结果发现了多个关键的见解。作者挑战了自我监督视觉表征学习中的一些常见做法，并大大超过了之前发表的最先进的结果。
### 参考论文
[Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)
## 默认配置

- 训练数据集预处理：
    - 图像的输入尺寸为224*224
    - 图像输入格式：TFRecord
    - 根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理
    - 图像的输入尺寸为224*224（将图像最小边缩放到256，同时保持宽高比，然后在中心裁剪图像）
    - 图像输入格式：TFRecord
    - 根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参
   - task=supervised
   - dataset=imagenet
   - train_split=trainval
   - val_split=val
   - batch_size=128
   - eval_batch_size=32
   - preprocessing=inception_preprocess
   - lr=0.1
   - lr_scale_batch_size=256
   - epochs=90
   - warmup_epochs=5
## 数据集
训练数据数据集为经过tf.slim转化为tfrecorad的IMAGENET数据集，[操作链接](https://github.com/tensorflow/models/tree/master/research/slim)

也可以使用OBS数据链接如下：

URL:

https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=GUjq8NJk03QKp6VZwTF5f3HZQEloJ4n35ln5YCj3rDSTr4WBRyHstX/iPdnIbFgvo16/YZIXJvFhqGxQETyFcbWNPgknqNLLHnOpu78UIvIKsUA1MlQsACqAWY3BoIqzzgOF+Unvw+GKoGQ2i4rfXAvyjEgdvdIxqlo8U3TWCIOoHzlVKZ3pHSOWRjCg43afOw6aR5GDoPGElFP0rRo5ddHuaolQIMEFvEUSbj/mOFjFjXMdq2+JEkKJI0bG15jaz89BzjZoWniDeV08j8vdmjV2R/loATfJJsv5VDwWjcFdVJbqdqVT2FcefTqvpPyc7okuDf8gIfXx1Gn5s1EaGsEOlBvtZto69AafODRpklRBTWdQKJBjz2Hc5g16hVUqjdlH8setvGC8ElmcqBUMt1pfeSqdDF0HYHSVD0mk+TAvapHrX+NUc+OgQRthOCZLvbxVvFS0KgkXwYZEFwsHdEiUvx2aEhee2r4IiVZb+wfW361gvYPz6PXzsn41W7urtCALKAi9ahLYSQf26SrzjEKMSrlBB0b8G/cbMGXWo8m1pq5kr3mubz8kclIPZrI+sfWaE4RsFLOs3UyLcFf/qA==

提取码:
123456

*有效期至: 2022/12/23 20:35:38 GMT+08:00

## 代码目录
```
├─config  //运行设置
│  ├─evaluation
│  │      jigsaw_or_relative_patch_location.sh
│  │      rotation_or_exemplar.sh      
│  ├─exemplar
│  │      imagenet.sh     
│  ├─jigsaw
│  │      imagenet.sh     
│  ├─relative_patch_location
│  │      imagenet.sh   
│  ├─rotation
│  │      imagenet.sh
│  │      
│  └─supervised
│          imagenet.sh
├─models //模型文件
│  │  resnet.py
│  │  utils.py
│  │  vggnet.py
│  │  __init__.py
├─self_supervision
│  │  exemplar.py
│  │  jigsaw.py
│  │  linear_eval.py
│  │  patch_model_preprocess.py
│  │  patch_utils.py
│  │  relative_patch_location.py
│  │  rotation.py
│  │  self_supervision_lib.py
│  │  supervised.py
│  │--__init__.py        
│  boot_modelarts.py  启动脚本
│  copy2obshook.py   输出loss信息
│  datasets.py   数据集处理
│  help_modelarts.py  modelarts辅助脚本
│  inception_preprocessing.py
│  permutations_100_max.bin
│  preprocess.py
│  setup.py
│  trainer.py  创建训练
│  train_and_eval.py  主函数
│  utils.py  工具库
│  __init__.py
```
## 快速复现
训练过程分为有监督训练和无监督训练
进行自监督训练:
```aidl
bash config/supervised/imagenet.sh --workdir=<work(result)dir> --dataset_dir=<datasetdir>
```
验证自监督训练结果：
```aidl
bash config/supervised/imagenet.sh --workdir=<work(result)dir> --dataset_dir=<datasetdir> --run_eval
```

## NPU训练结果!

NPU以跑通，训练step多于GPU的情况下loss低于GPU，npu上eval结果如下：

![npu_eval.png](npu_eval.png)
top5准确率：0.97952724

loss结果如下 (batch-size:128)
```aidl
INFO:tensorflow:loss = 0.8321978, step = 900029 (22.301 sec)
I1212 21:59:05.964472 281473521922416 basic_session_run_hooks.py:260] loss = 0.8321978, step = 900029 (22.301 sec)
INFO:tensorflow:Saving checkpoints for 900101 into /cache/result/model.ckpt.
I1212 21:59:22.676009 281473521922416 basic_session_run_hooks.py:606] Saving checkpoints for 900101 into /cache/result/model.ckpt.
INFO:tensorflow:global_step/sec: 3.09606
I1212 21:59:38.262868 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 3.09606
复制checkpiont到obs==================>
workdir== /home/ma-user/modelarts/workspace/device0
/cache/result
/home/ma-user/modelarts/outputs/train_url_0/result
===>>>Copy Event or Checkpoint from modelarts dir:/cache/result to obs:/home/ma-user/modelarts/outputs/train_url_0/result
INFO:tensorflow:loss = 1.0207675, step = 900129 (49.726 sec)
I1212 21:59:55.690677 281473521922416 basic_session_run_hooks.py:260] loss = 1.0207675, step = 900129 (49.726 sec)
INFO:tensorflow:global_step/sec: 2.23308
I1212 22:00:23.044453 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 2.23308
INFO:tensorflow:loss = 0.9054604, step = 900229 (27.355 sec)
I1212 22:00:23.045186 281473521922416 basic_session_run_hooks.py:260] loss = 0.9054604, step = 900229 (27.355 sec)
INFO:tensorflow:global_step/sec: 4.23395
I1212 22:00:46.662705 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 4.23395
INFO:tensorflow:loss = 0.733233, step = 900329 (23.618 sec)
I1212 22:00:46.663298 281473521922416 basic_session_run_hooks.py:260] loss = 0.733233, step = 900329 (23.618 sec)
INFO:tensorflow:global_step/sec: 4.27441
I1212 22:01:10.057892 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 4.27441
INFO:tensorflow:loss = 0.93613726, step = 900429 (23.395 sec)
I1212 22:01:10.058795 281473521922416 basic_session_run_hooks.py:260] loss = 0.93613726, step = 900429 (23.395 sec)
INFO:tensorflow:global_step/sec: 4.42444
I1212 22:01:32.659451 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 4.42444
INFO:tensorflow:loss = 0.7250228, step = 900529 (22.602 sec)
I1212 22:01:32.660283 281473521922416 basic_session_run_hooks.py:260] loss = 0.7250228, step = 900529 (22.602 sec)
INFO:tensorflow:global_step/sec: 4.48429
I1212 22:01:54.959711 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 4.48429
INFO:tensorflow:loss = 1.0412902, step = 900629 (22.300 sec)
I1212 22:01:54.960587 281473521922416 basic_session_run_hooks.py:260] loss = 1.0412902, step = 900629 (22.300 sec)
INFO:tensorflow:global_step/sec: 4.46345
I1212 22:02:17.363854 281473521922416 basic_session_run_hooks.py:692] global_step/sec: 4.46345
INFO:tensorflow:loss = 0.82551265, step = 900729 (22.404 sec)
I1212 22:02:17.364636 281473521922416 basic_session_run_hooks.py:260] loss = 0.82551265, step = 900729 (22.404 sec)
INFO:tensorflow:Saving checkpoints for 900810 into /cache/result/model.ckpt.
I1212 22:02:36.937217 281473521922416 basic_session_run_hooks.py:606] Saving checkpoints for 900810 into /cache/result/model.ckpt.
INFO:tensorflow:Loss for final step: 0.98728836.
I1212 22:02:51.659879 281473521922416 estimator.py:371] Loss for final step: 0.98728836.
```
## 精度对比（以GPU训练epoch为准）

精度对比| GPU | NPU
---- | ----- | ------  
top1 | 0.854 | 0.845
top5 | 0.982 | 0.979

## 性能对比
对比指标：globe_steps/sec(每秒进行的step数量,越大性能越好)：batch_size:32

性能对比| GPU | NPU
---- | ----- | ------  
globe_steps/sec  | 8.951 | 9.973

NPU训练性能已经达标