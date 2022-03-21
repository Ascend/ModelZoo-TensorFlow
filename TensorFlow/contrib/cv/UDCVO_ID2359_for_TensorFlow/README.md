# 基本信息

**应用领域（Application Domain）**： Computer Version

**版本（Version）**：1.0

**修改时间（Modified）**：2021.12.06

**框架（Framework）**：TensorFlow_1.15.0

**模型格式（Model Format）**：ckpt

**处理器（Processor）**：昇腾910

**描述（Description）**：通过摄像头运动和视觉惯性里程计估计出的稀疏深度，推断密集深度

# 概述

## 简述

此模型通过给定场景的RGB图像来增密稀疏点云，继而恢复3D场景。首先构造场景的分段平面脚手架，然后使用它与图像以及稀疏点一起推断密集深度。使用类似于“自我监督”的预测性交叉模态准则，跨时间测量光度一致性，前后姿势一致性以及与稀疏点云的几何兼容性。使用网格三角剖分和线性插值作为预处理步骤，使网络的参数减少了 80%，同时性能优于其他方法。提出了第一个视觉惯性+深度数据集。
* 参考论文：
[Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano, "Unsupervised Depth Completion from Visual Inertial Odometry", in ICRA, 2020.](https://arxiv.org/pdf/1905.08616.pdf)
```
@article{wong2020unsupervised,
 title={Unsupervised Depth Completion From Visual Inertial Odometry},
  author={Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={1899--1906},
  year={2020},
  publisher={IEEE}
}
```
* 参考实现：
https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry#unsupervised-depth-completion-from-visual-inertial-odometry
* 适配昇腾 AI 处理器的实现：
https://gitee.com/Goudaneer/modelzoo/tree/master/contrib/TensorFlow/Research/cv/UDCVO_ID2359_for_TensorFlow
* 通过Git获取对应commit_id的代码方法如下: 
```
git clone {repository_url}    # 克隆仓库的代码  
cd {repository_name}    # 切换到模型的代码仓目录  
git checkout  {branch}    # 切换到对应分支  
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id  
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```
## 默认配置

* 网络类型
   * VGGNet11
* 训练超参(单卡)：
   * batch-size: 8
   * height: 480
   * width: 640
   * channel: 3
   * epoch: 20
   * learning-rate: $0.50\times10^{-4}$（前 12 epoch），$0.25\times10^{-4}$（4 epoch），$0.12\times10^{-4}$（后 4 epoch）

# 训练环境准备

启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：

[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

# 训练

## 数据集准备

复现代码使用密度为 0.5%（1500个稀疏点）的室内场景数据集，用户需自行准备VOID_1500数据集到训练环境。训练脚本读取training下的.txt文件获得训练图片的路径，通过读取到的路径名，进而从data文件夹下读入训练图片。数据集目录放置结构如下（所有训练图片、测试图片及.txt文件均已放于obs中并共享）：

```
├── bash
├── src
├── testing
│    ├──void_test_ground_truth_1500.txt
│    ├──void_test_image_1500.txt
│    ├──void_test_interp_depth_1500.txt
│    ├──void_test_intrinsics_1500.txt
│    ├──void_test_sparse_depth_1500.txt
│    ├──void_test_validity_map_1500.txt
├── training
│    ├──testcase_image_1500.txt
│    ├──testcase_interp_depth_1500.txt
│    ├──testcase_sparse_depth_1500.txt
│    ├──testcase_validity_map_1500.txt
│    ├──void_train_ground_truth_1500.txt
│    ├──void_train_image_1500.txt
│    ├──void_train_interp_depth_1500.txt
│    ├──void_train_intrinsics_1500.txt
│    ├──void_train_sparse_depth_1500.txt
│    ├──void_train_validity_map_1500.txt
├── data
│    ├──void_release
│    ├──void_voiced
```



## 模型训练

在项目路径下执行如下 shell 命令进行训练：
```
sh bash/train_voiced_void.sh
```

可以使用 Tensorboard 监控训练情况：
```
tensorboard --logdir trained_models/<model_name> --host=127.0.0.1
```

## 模型评估

运行如下 shell 命令来评估预训练模型：
```
sh bash/evaluate_voiced_void.sh
```

可以替换 shell 脚本中的 restore_path 和 output_path 路径来评估自己的 checkpoints 。

# License and disclaimer

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](license). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).