![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)

# CircleLoss

**Circle Loss: A Unified Perspective of Pair Similarity Optimization**

Yifan Sun et al.

CVPR 2020

ArXiv: [https://arxiv.org/abs/2002.10857](https://arxiv.org/abs/2002.10857)

本文提出了一种新的pair-wise的相似度优化损失函数，能够在学习过程中自适应地调整对不同类型相似度的惩罚程度，从而达到更加高效学习效果。

![image-20210906114857004](https://picbed-1301760901.cos.ap-guangzhou.myqcloud.com/image-20210906114857004.png)

## 1. 依赖库安装

见Requirements.txt， 需要安装 numpy， scikit-image， tensorflow等库。

## 2.  数据集

模型训练采用MARKET1501数据集，将文件夹解压到 /Data/ 下

https://www.kaggle.com/pengcw1/market-1501/data

## 3. 训练

将模型预训练权重放置到 /checkpoint/ 下

训练脚本为`train_npu.py`，跳转至项目根目录下，直接执行以下命令即可：

```bash
bash script/train.sh
```



## 4. NPU性能



## 5.Loss曲线

![image-20211206125601106](https://picbed-1301760901.cos.ap-guangzhou.myqcloud.com/image-20211206125601106.png)

## 6. 精度对比

论文中的测试精度如下所示

![image-20211206124710842](https://picbed-1301760901.cos.ap-guangzhou.myqcloud.com/image-20211206124710842.png)

使用NPU训练后得到的测试集精度如下所示：

表格：Market-1501数据集上Circle loss 应用于ResNet50训练的测试结果，其中包括了R-1准确率（%）和mAP（%）

| Method      | R-1  | mAP  |
| ----------- | ---- | ---- |
| 原论文精度  | 94.2 | 84.9 |
| GPU复现精度 | 81.0 | 62.1 |
| NPU复现精度 |  80.9 |  61.5 |

## Contact

Please assign issues to [@louis-yx](https://gitee.com/louis-yx) if you have any question.

## Credits

This implementation is derived from [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) by [layumi](https://github.com/layumi).

