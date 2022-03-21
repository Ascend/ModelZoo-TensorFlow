# 模型功能
- 图像超分：通过subpiexl实现上卷积，替代插值或解卷积的方法对缩小后的特征图实现相应倍数的upscale。
- 原论文：[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)
- 附加论文：[Is the deconvolution layer the same as a convolutional layer](https://arxiv.org/abs/1609.07009)
- 参考：[kweisamx/TensorFlow-ESPCN](https://github.com/kweisamx/TensorFlow-ESPCN.git)

## 环境
### pip安装
* Tensorflow
* Opencv
* h5py

## 数据集
可从SRCNN官方网页[Image Super-resolution Using Deep Convolutional Networks](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)中获得训练集set91和测试集set5。


## 代码及路径解释
```
ESPCN
└─ 
  |-README.md
  |-train 用于存放训练数据集
  |-test 用于存放测试数据集
  |-checkpoint 保存模型和中间h5数据集
  |-log 存放日志
  |-main.py 模型启动入口
  |-model.py 存放模型结构、训练和测试代码
  |-PSNR.py 模型评估
  |-utils.py 对数据进行预处理
```

## 训练和测试入口
```
python main.py
```

## 超参
```
epoch = 15000
batch_size = 128
learning_rate = 1e-5
上采样系数scale = 3
训练时长约1小时
```

## 验证精度
|   |原论文|本文|
|---|---|---|
|PSNR| 32.55 | 30.31|

## checkpoint百度云链接及提取码
链接：https://pan.baidu.com/s/1l3TB1wgYdl5lEbmQE-DoAQ 
提取码：j2gm
