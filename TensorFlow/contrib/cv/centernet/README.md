# CenterNet

## 概述

CenterNet是一个Anchor free的one-stage目标检测模型。通过引入中心点信息，辅助降低不正确的预测框的数量，从而提高模型性能。

CenterNet模型经过主干网络提取特征，然后预测中心点，偏移量和目标框大小，即不需要Anchor就可以选取Bounding box。

### 参考论文

[Objects as Points](https://arxiv.org/abs/1904.07850)

### 参考实现

* [CenterNet(Official Repo, Pytorch) - xingyizhou](https://github.com/xingyizhou/CenterNet)
* [TF_CenterNet - MioChiu](https://github.com/MioChiu/TF_CenterNet)
* [CenterNet-tensorflow - Stick-To](https://github.com/Stick-To/CenterNet-tensorflow)
* [CornerNet - princeton-vl](https://github.com/princeton-vl/CornerNet)

## 准备

### 环境

```bash
PRJPATH=`pwd`
PYTHONPATH=`realpath ./src`:$PYTHONPATH
pip install -r src/dataset/requirements.txt
pip install -r requirements.txt
```

### COCOAPI

```bash
cd $PRJPATH
mkdir ext && cd ext && bash ../src/dataset/script/build_pycocotools.sh
```

### 数据集

Pascal VOC 2007(train, test), Pascal VOC 2012(train)

PASCAL VOC Annotations in COCO Format

解压放置为
```
.
├── checkpoint
├── dataset
│   └── VOC
│       ├── raw
│       │   ├── PASCAL_VOC
│       │   │   ├── pascal_test2007.json
│       │   │   ├── pascal_train2007.json
│       │   │   ├── pascal_train2012.json
│       │   │   ├── pascal_val2007.json
│       │   │   └── pascal_val2012.json
│       │   ├── test
│       │   │   └── VOCdevkit
│       │   │       └── VOC2007
│       │   │           ├── Annotations
│       │   │           ├── ImageSets
│       │   │           │   ├── Layout
│       │   │           │   ├── Main
│       │   │           │   └── Segmentation
│       │   │           ├── JPEGImages
│       │   │           ├── SegmentationClass
│       │   │           └── SegmentationObject
│       │   └── train
│       │       └── VOCdevkit
│       │           ├── VOC2007
│       │           │   └──...
│       │           └── VOC2012
│       │               └──...
│       └── tfrecord
├── ext
├── log
└── src
```

构造tfrecord
```bash
cd $PRJPATH/src
python dataset/script/build_tfrecords.py dataset.config.cfg_voc_object_detection
```

### 训练
```bash
cd $PRJPATH/src
python ./test/test_centernet.py --platform config.platform.cfg_ascend
```
