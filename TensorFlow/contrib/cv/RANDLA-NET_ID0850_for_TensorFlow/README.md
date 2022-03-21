# RandLA-Net

#### 介绍
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds (CVPR 2020)，一种高效且轻量级的网络结构，可以直接处理大规模点云，得到每一个点的分割结果。论文中的主要关键点是使用随机采样算法替代了当前其他算法中复杂的点云采样技术。论文中提出了一个新颖的本地特征聚合模块，通过逐渐的增加每一个3D点的感受野，从而可以有效的保留因为随机采样丢失的几何细节特征。
- 参考论文：
https://arxiv.org/abs/1911.11236
- 参考实现：
- 适配昇腾 AI 处理器的实现：
https://gitee.com/xuanzhang_1/modelzoo/tree/master/contrib/TensorFlow/Research/cv/RANDLA-NET_ID0850_for_TensorFlow


#### 默认配置

- 训练超参

    - train_steps 1000
    - batch_size 3
    - val_steps 100
    - val_batch_size 3
    - max_epoch 3

    修改超参数请在npu_train.sh文件中修改


#### 使用说明

1.数据集预处理

 S3DIS数据集[下载地址](http://https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)，下载名为"Stanford3dDataset_v1.2_Aligned_Version.zip"的文件并解压，并及时更新数据预处理程序utils/data_prepare_s3dis.py内文件地址，之后执行下面预处理的命令。预处理之后的数据下载链接：https://pan.baidu.com/s/1vy7VPMzqXrijYp9lz96vFw 提取码：wo5m。下载好数据之后，上传至OBS桶。预处理数据的obs地址：obs://randla-net/data_retry/

```
python utils/data_prepare_s3dis.py
``` 

2.  训练命令、验证命令、测试命令执行下面.py文件

```
python modelarts_entry.py
```

3.  标杆性能比对结果
Six-fold cross validation前三次训练精度对比：

|              | GPU    | NPU    |
|--------------|--------|--------|
| Area1_epoch0 | 31.98% | 35.68% |
| Area1_epoch1 | 44.4%  | 43.64% |
| Area1_epoch2 | 49.9%  | 54.3%  |
| Area2_epoch0 | 26.5%  | 25.7%  |
| Area2_epoch1 | 27.81% | 31.33% |
| Area2_epoch2 | 36.5%  | 38.45% |
| Area3_epoch0 | 30.6%  | 31.51% |
| Area3_epoch1 | 43.43% | 40.43% |
| Area3_epoch2| 49.14% | 51.31% |

S3DIS数据集中有6个area,分别是area1-6.表格中的Area1_epoch1表示的是本次训练验证数据集是area1，训练数据集是area2-area6。由于NPU训练性能很慢，所以只训练了前3个epoch的精度数据进行对比，训练是用的默认精度+关闭融合规则（请自行配置precision_tool工具）。对比可知精度是达标的

4.文件

```
├── utils                           //数据预处理文件和自定义算子相关编译文件
├── LICENSE
├── README.md                       //说明文件
├── RandLANet.py                    //完整的网络
├── helper_ply.py                   //读写ply文件
├── helper_requirements.txt         //gpu训练环境配置信息
├── helper_tf_util.py               //网络结构文件
├── helper_tool.py                  //配置文件
├── main_S3DIS.py                   //训练文件
├── modelarts_entry.py              //训练启动文件
├── npu_train.sh                    //训练启动相关文件
├── tester_S3DIS.py                 //测试文件

```
