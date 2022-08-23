#  Pixel CNN++

#### 介绍
{Pixel CNN++是一个图像生成模型，这项工作建立在2016年6月van der Oord等人最初提出的PixelCNNs上。}

#### 参考论文
[pixel-cnn++](https://openreview.net/pdf?id=BJrFC6ceg)

#### 参考实现
[pixel-cnn++](https://github.com/openai/pixel-cnn)

#### 通过Git获取对应commit_id的代码方法如下：
```
git clone {repository_url}    # 克隆仓库的代码
cd {repository_name}    # 切换到模型的代码仓目录
git checkout  {branch}    # 切换到对应分支
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

#### 默认配置
##### 超参数
- batch_size: 16
- learning_rate 0.001
- lr_decay 0.999995
- max_epochs 200

#### 支持特性
| 特性列表 | 是否支持 |
|------|------|
| 混合精度 | 否    |

脚本默认关闭混合精度，开启混合精度会导致较大的精度损失

#### 训练环境准备
- 硬件环境： Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB
- 运行环境：ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1224

#### 快速上手

#### 数据集准备
模型训练使用Cifar-10数据集或mini-imagenet数据集，需要下载至./data/目录下
若使用ModelArts平台，请将数据上传至OBS中并在ModelArts中指定data path

#### 模型训练

在ModelArts平台中，直接将modelarts_entry_acc.py文件或modelarts_entry_perf.py文件作为Boot File即可

#### 训练结果
- 精度结果对比

| 精度指标项        | GPU实测  | NPU实测  |
|--------------|--------|--------|
| bits_per_dim | 2.9282 | 2.9595 |

####【执行结果打屏信息】
https://cann--id2101.obs.cn-north-4.myhuaweicloud.com:443/npu/Pixel%20CNN%2B%2B_ID2101_for_TensorFlow/modelarts-job-f0e4f360-9294-411c-9eec-86eb9615f85a-worker-0.log?AccessKeyId=FH7STWYX1HSCOLV0WQ2S&Expires=1692347125&Signature=AeqWBORUSd3Ki4i7WrH%2BuVfF1H0%3D
------------------ INFO NOTICE START------------------
INFO, your task have used Ascend NPU, please check your result.
------------------ INFO NOTICE END------------------
------------------ Final result ------------------
Final Performance images/sec : 3.99
Final Performance sec/step : 4.85
E2E Training Duration sec : 992970s
Final Train Accuracy : 2.9595
####【数据集OBS链接】
obs://cann--id2101/dataset/
####【执行任务OBS链接】




