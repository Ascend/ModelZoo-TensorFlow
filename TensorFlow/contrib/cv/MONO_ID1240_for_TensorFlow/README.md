#### MONO网络概述**

考虑到像素级深度数据的ground truth难以获取，监督学习条件十分受限，所以基于自监督学习或者无监督学习(使用内在几何，通常是多视图几何关系)的图像深度估计越来越受到重视和研究。自监督单目训练研究通常探索增加模型复杂度，损失函数和图像生成模型来帮助缩小与全监督模型的差距。已有研究表明仅使用双目图像对或者单目图像来训练单目深度估计模型是可行的，在使用单目图像训练深度估计时，同时还要训练时间序列图像对之间的位姿估计模型。使用双目数据训练会使相机位姿估计成为一次性的离线标定，但可能引起相关的遮挡和纹理复制伪影问题。

Monodepth2模型使用深度估计和姿态估计网络的组合来预测单帧图像中的深度。该模型的主要创新点有3项：

1. 一种先进的外观匹配损失，解决当使用单目监督时出现的像素遮挡问题。

2. 一种先进且简单的auto-masking方法来忽略在单目训练中没有相对运动的像素点 。

3. 在输入分辨率下执行所有图像采样的多尺度外观匹配损失，这可以导致深度伪影的减少。

   

#### **参考论文**

**Digging into Self-Supervised Monocular Depth Prediction**

[Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  

[arXiv 2018](https://arxiv.org/abs/1806.01260)



**参考实现**

https://github.com/FangGet/tf-monodepth2



#### **默认配置**

1. **数据集**：KITTI(http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)

   For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following commandkitti_raw_eigen

   ```bash
   python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
   ```

2. **训练及测试数据集**

   训练数据集：obs://cann-share/kitti_dataset/kitti_final_dataset/

   测试数据集：obs://cann-share/kitti_dataset/kitti_raw_eigen/

   ```
   数据集组织
   ├── kitti_dataset						----数据集文件
       ├── kitti_final_dataset				---训练数据集
       │   ├── train.txt
       │   ├── val.txt
       │   2011_09_26_drive_0001_sync_01
       │   ....
       │
       ├── kitti_raw_eigen		----测试及验证数据集
       │   ├── 2011_09_26
       │   ├── 2011_09_28
       │   ...
   ```

3. **训练超参数**

   见config文件夹中的monodepth2_kitti.yml及monodepth2_kitti_eval.yml文件

#### **代码及路径解释**

```
MONO
└─ 
  ├─README.md
  ├─requirements.txt 配置要求文件 
  ├─model 模型实现
  ├─config 模型训练及验证超参数
  ├─monodepth2.py 模型训练代码
  ├─npu_train.sh 启动脚本
  ├─kitti_eval 模型性能验证
```

#### **训练过程及结果**

待更新





