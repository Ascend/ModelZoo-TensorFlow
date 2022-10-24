# SVD_ID2019_for ACL

#### 概述
给定两个三维点云图，利用SVD正交化过程SVDO+(M)将其投射到SO(3)上，要求网络预测最佳对齐它们的3D旋转。

- 开源代码：训练获取

    https://github.com/google-research/google-research/tree/master/special_orthogonalization。

- 参考论文：

    [An Analysis of SVD for Deep Rotation Estimation](https://arxiv.org/abs/2006.14616)

#### 数据集

训练数据集 points 
	
测试数据集 points_test
        
旋转后数据集 points_test_modified

#### 模型固化
-直接获取

直接下载获取，百度网盘
链接：https://pan.baidu.com/s/17zKWq2aY06cF9IQW6htn_A 
提取码：2019
-训练获取

训练获取
训练完成saved_model模型网盘链接：https://pan.baidu.com/s/1Y4ato6Ob-6-rcXr31AvgoA 
提取码：2019

1.按照SVD_ID2019_for_Tensorflow中的流程训练，模型保存为saved_model格式

2.将saved_model格式文件冻结为pb文件（需要在freeze.py文件中修改路径）

python freeze.py

得到svd.pd

#### 使用ATC工具将pb文件转换为om模型
命令行代码示例

atc --model=/home/test_user04/svd.pb --framework=3 --output=/home/test_user04/svd --soc_version=Ascend310 --input_shape="data_1:1,1410,3;rot_1:1,3,3"

注意所使用机器的Ascend的型号

模型直接下载百度网盘链接：https://pan.baidu.com/s/14-m0ZhPQyIr8enpUgVytpg 
提取码：2019

得到svd.om
#### 制作数据集
-直接下载,数据集在svd_inference/data_1



-自己制作
原数据链接：链接：https://pan.baidu.com/s/1aGAO3os8ifDnYm1yXrxndQ 
提取码：2019

使用pts2txt制作数据集(注意修改数据路径，数据路径为/xxx/points_test_modified/*.pts,注意修改产生数据集后的路径)

python pts2txt.py

#### 获取离线推理输出bin文件

推理文件在压缩包svd_inference中，下载百度网盘链接：https://pan.baidu.com/s/1OfCxHMUJcnyqp2IcvV3eWg 
提取码：2019

脚本在src文件夹中，直接运行

python svdom_inference.py

推理结果直接下载网盘链接：https://pan.baidu.com/s/1NFNfJkTUW4u7YJcaHK9mLw 
提取码：2019

#### 使用输出的bin文件验证推理精度

运行脚本

python calc_acc.py

得到推理精度:3.150504164928697

与在线推理精度近似

关于以上所有文件的百度网盘链接：https://pan.baidu.com/s/1sR8gYK8jM6xCZwbq7eK50A 
提取码：2019