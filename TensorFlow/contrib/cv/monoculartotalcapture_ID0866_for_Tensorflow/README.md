# Readme_NPU

## 1、关于项目

本项目的目的是复现“Monocular Total Capture: Posing Face, Body, and Hands in the Wild ”论文算法。

论文链接为：[paper](https://arxiv.org/abs/1812.01598)

开源代码链接为：[code](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture/)

该文章提出了一种从单目照片输入中捕获目标人体的三维整体动作姿态的方法。效果如图所示。

![示例图](https://images.gitee.com/uploads/images/2021/1108/200620_2c36b961_5720652.png "3.png")

## 2、关于依赖库

见requirements.txt，需要安装numpy>=1.18.1 scipy py-opencv等库。

## 3、关于数据集

training_e2e_PAF训练脚本选用COCO2017数据集。

COCO2017数据集需要经过处理，处理方式参照github [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)。处理后的数据集目录为：

+ COCO_data
  + train2017
    + 000000581921.jpg
    + ....
  + mask2017
    + train2017_mask_miss_000000581921.png
    + ...
  + COCO.json

处理过的数据集的网盘分享链接在download.md中。

## 4、关于训练

下载上述提到的数据集到本地，数据集存储的路径可以在data/COCOReader.py中修改。

下载预训练模型（文件夹名：Final_qual_domeCOCO_chest_noPAF2D）。网盘分享链接在download.md中。将下载的文件夹放入snapshots文件夹。

训练脚本为training_e2e_PAF.py，训练时在终端输入python3 training_e2e_PAF.py或sh train.sh即可。

## 5、性能比较

GPU（使用华为弹性云服务器ECS）每个iteration耗时约 0.78s。

NPU（使用modelArts训练）每个iteration耗时2.70s。

导致性能较低的算子是resize正反向和stridedslicegrad，尚待开发后解决。

## 6、Loss收敛曲线

![loss曲线](https://gitee.com/wwxgitee/pictures/raw/master/loss_drop.png)

## 7、Loss比较

训练过程中，只有loss值没有精度值。GPU和NPU训练下的loss值均收敛至0.005附近。
gpu训练过程回显
![gpu训练过程回显](https://gitee.com/wwxgitee/pictures/raw/master/gpu_training.png)
npu训练过程回显
![npu训练过程回显](https://gitee.com/wwxgitee/pictures/raw/master/npu_training.png)