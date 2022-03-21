# 概述
DSRC TensorFlow实现，适配Ascend平台。
* DSRC是一种一种基于深度学习的基于稀疏表示的分类方法。 
* Original paper: https://arxiv.org/abs/1904.11093


## 训练环境
* 硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB
* 软件环境：TensorFlow1.15

## 代码及路径解释


```
DSRC
└─ 
  ├─README.md
  ├─LICENSE
  ├─models 模型存放路径
  ├─logs 训练记录文件存放路径
  ├─data 存放数据集路径
  ├─dsrc_usps_gpu.py 适用于USPS数据集的gpu训练代码 
  ├─dsrc_usps_npu.py 适用于USPS数据集的npu训练代码
  ├─requirement.txt 环境配置要求
  ├─ops_info.json 混合精度训练算子黑白名单
  ├─modelzoo_level.txt 模型迁移进展
```

## 数据集
```
选择使用 USPS数据集
```

## 快速上手

* 下载数据集  
自行下载USPS.mat, 将USPS.mat放置于data路径下即可
  
* NPU 训练

```bash
$ python train_usps_npu.py
```

* GPU 训练
```bash
$ python train_usps_gpu.py
```

* 训练脚本参数
```
--mat           \\数据集
--model         \\保存模型名称
--rate          \\训练集占全部数据集比例
--max_step      \\训练的总次数
--pretrain_step \\预训练的次数
--display_step  \\打印训练过程损失和测试集预测结果的频率
--data_path     \\数据集路径
```

## 训练结果

论文指标：

| 分类任务| 论文精度 | GPU精度 |NPU精度|
| :-----| --------| -------|------|
| USPS  | 96.25   | 94~96|   94~96   |



| 性能指标| GPU耗时 |NPU耗时|
| :-----| -------|------|
| 200epoch耗时 | 3.9~4.1s|  2.8~3.1s   |

## 更新

2021/12/6通过修改数据加载方式将NPU速度从原来的200个epoch 4.3-4.6s 加快到了 2.8~3.1s，在原有代码上进行了改动。

2022/1/19添加了loss scale，完善了readme。

