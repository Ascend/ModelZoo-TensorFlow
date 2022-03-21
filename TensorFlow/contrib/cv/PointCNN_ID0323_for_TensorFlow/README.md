# PointCNN: Convolution On X-Transformed Points

## 概述
PointCNN提出一种从点云中学习特征的通用框架。CNN网络成功的关键是卷积操作，它可以利用网格中密集数据在局部空间上的相关性。然而，点云是不规律且无序的，因此，直接对点的特征进行卷积，会导致形状信息和点排列多样性的损失。论文提出通过在输入点云中学习X-转换，以此改善两个问题：（1）输入点特征的权重（2）将点排列为潜在、规范的顺序。论文将典型卷积操作符的乘与和运算，应用在X-转换后的特征上。
- 参考论文：

  [Li Y, Bu R, Sun M, et al. Pointcnn: Convolution on x-transformed points[J]. Advances in neural information processing systems, 2018, 31: 820-830.](https://arxiv.org/pdf/1801.07791.pdf)
- 参考实现：

  [PointCNN](https://github.com/yangyanli/PointCNN)
- 适配昇腾 AI 处理器的实现：

  https://gitee.com/nuaayqm/modelzoo/edit/master/contrib/TensorFlow/Research/cv/PointCNN
- 通过Git获取对应commit_id的代码方法如下：

       git clone {repository_url}    # 克隆仓库的代码
       cd {repository_name}    # 切换到模型的代码仓目录
       git checkout  {branch}    # 切换到对应分支
       git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
       cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换


## 环境配置
- python3.7
- Tensorflow 1.15.0
- Ascend 910
- matplotlib
- plyfile
- python-mnist
- requests
- scipy
- svgpathtools
- tqdm
- transforms3d

requirements.txt 内记录了所需要的环境与对应版本，可以通过命令配置所需环境。
- pip install -t requirements.txt

## 结果
迁移PointCNN到Ascend 910平台，使用的环境是 ModelArts

使用mnist数据集在ModelArts Ascend 910 TensorFlow平台上上训练，在测试集上测试结果如下：
### 训练精度
|   |accuracy(%)|Enviroment|device|batch size|iterations|
|---|---|---|---|---|---|
|Report in paper|99.54|TensorFlow, GPU|Unknown|Unknown|Unknown|
|Report on GPU|99.15|TensorFlow, GPU|1|256|15000|
|Report on Ascend 910|98.26|ModelArts, Ascend 910|1|256|15000|

### 训练性能
|Platform|Batch size|Throughout|
|---|---|---|
|1xAscend 910|256|0.38s/iter|
|1xTesla V100-16G|256|0.26s/iter|


## 训练及测试
### 参数说明

```
--path 训练集txt文件路径
--path_val 验证集txt文件路径
--load_ckpt 加载checkpoint文件的路径
--save_folder 保存checkpoint和summary的文件夹路径
--model 使用模型类型
--setting 加载设置
--platform 运行设备
--epochs 训练次数
--batch_size 批次大小
--log 日志文件保存，默认log.txt
```
### GPU版本
```
cd data_conversions
python3.7 ./prepare_mnist_data.py -f ../../data/mnist
cd ../pointcnn_cls
sh train_val_mnist_gpu.sh -g 0 -x mnist_x2_l4
```
### NPU版本
```
cd data_conversions
python3.7 ./prepare_mnist_data.py -f ../../data/mnist
cd ../pointcnn_cls
sh train_val_mnist_npu.sh -g 0 -x mnist_x2_l4
```

测试结果保存在 save_folder 路径下log.txt文件内