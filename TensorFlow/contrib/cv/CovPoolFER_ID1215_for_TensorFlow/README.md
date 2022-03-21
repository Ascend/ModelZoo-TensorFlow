###   **基本信息** 

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.12.23**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的CovPoolFER网络训练代码** 


###   **概述** 
CovPoolFer将面部表情分类为不同类别需要捕获面部关键点的区域扭曲。论文提出协方差等二阶统计量能够更好地捕捉区域面部特征中的这种扭曲。论文探索了使用流形网络结构进行协方差合并以改善面部表情识别的好处。特别地，CovPoolFer首次将这种类型的流形网络与传统的卷积网络结合使用，以便以端到端的深度学习方式在各个图像特征映射内进行空间汇集。此外，CovPoolFer利用协方差池来捕捉基于视频的面部表情的每帧特征的时间演变。论文显示了通过在卷积网络层之上堆叠设计的协方差池的流形网络来暂时汇集图像集特征的优点。CovPoolFer在SFEW2.0和RAF数据集上获得了state-of-the-art的结果。

- 网络架构：
  
总览图：

![img](./Img4Doc/Total.png)

SPDNet：  

![img](./Img4Doc/SPD.png)

- 相关参考：  
    - 参考论文：[Covariance Pooling for Facial Expression Recognition](https://arxiv.org/abs/1805.04855)
    - 参考实现：[https://github.com/iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)

### 默认配置<a name="section91661242121611"></a>
-   训练超参（单卡）：
    - Batch size: 128 (set as default)
    - num_epochs: 95 (set as default)

###   **Requirements** 

1. NPU环境
```
硬件环境：NPU: 1*Ascend 910 CPU: 24*vCPUs 96GB

运行环境：ascend-toolkit 5.0.4.alpha001_x86_64-linux
```
2. 第三方依赖
```
numpy @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/numpy-1.17.5-cp37-cp37m-linux_aarch64.whl
importlib-metadata==4.8.1
h5py @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/h5py-2.10.0-cp37-cp37m-linux_aarch64.whl
Pillow @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/Pillow-7.0.0-cp37-cp37m-linux_aarch64.whl
scikit-learn @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/scikit_learn-0.20.0-cp37-cp37m-linux_aarch64.whl
scipy @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/scipy-1.3.3-cp37-cp37m-linux_aarch64.whl
npu-bridge @ file:///tmp/selfgz1881668/tfplugin/bin/npu_bridge-1.15.0-py3-none-any.whl
tensorboard==1.15.0
tensorflow @ http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/compiled-wheel/tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
tensorflow-estimator==1.15.1
tensorflow-probability==0.10.1
```
3. Python

python 3.7.5

4. Tensorflow

Tensorflow 1.15.0 [需要修改]

修改方式：
将python环境下的site-packages/tensorflow_core/python/ops/linalg_grad.py 替换为 src/linalg_grad.py    [替换原因](https://gitee.com/ascend/modelzoo/issues/I494PG#note_7384472)）


###   **代码及路径解释** 

```
CovPoolFER
└─ 
  ├─README.md
  ├─train.sh                   训练脚本
  ├─Img4Doc
    │─SPD.png                  SPDNet网络结构图
    │─Total.png                总览网络结构图
  ├─data                       用于存放训练数据集 obs://cov/data/SFEW_100/
  	├─SFEW_100               
  	  └─
           ├─Train             训练集
           ├─val               验证集
        ├─learning_rate.txt    动态学习率
        
  ├─src 
    │─framework.py             工具函数代码
    │─train.py                 训练代码
    │─linalg_grad.py           规避溢出代码
    └─models                   模型代码
       └─
        ├─covpoolnet.py        论文模型一
        |─covpoolnet2.py       论文模型二
    └─precision_tool
       └─
        ├─fusion_switch.cfg    融合规则
```

### **数据集准备** 
训练数据集为SFEW数据集，位于obs路径：obs://cov/data/SFEW_100/

也可使用百度网盘链接：[百度网盘获取链接](https://pan.baidu.com/s/14zMX4-izJTSL4L6uWySLnw)

提取码：4rdy 


###  **模型迁移对比** 

| 迁移模型    | 训练次数 | NPU loss |GPU loss |
| :---------- | ----- | ------ | ------ |
| CovPoolFER | 95 | 0.6406|0.7004|

评价方法：最后一个epoch（第95个）的loss平均值。


| 迁移模型    | 训练次数 | NPU Seconds per Step |GPU Seconds per Step |
| :---------- | ----- | ------ | ------ |
| CovPoolFER | 95 | 2.902 |3.220|

评价方法：最后一个epoch（第95个）的执行单个step时间平均值。




### **脚本参数**
```
--logs_base_dir               //日志生成路径
--models_base_dir             //模型生成路径
--data_dir                    //数据集路径
--image_size                  //图片大小
--model_def                   //模型定义，指向包含模型定义的模块。
--optimizer                   //学习器，可选：['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
--learning_rate               //学习率，为-1则表示使用动态学习率
--max_nrof_epochs             //epoch数目
--keep_probability            //保持全连接层的 dropout概率
--learning_rate_schedule_file //若learning_rate为-1，则此处为学习率动态变化表 
--weight_decay                //L2正则化权重
--center_loss_factor          //center_loss参数
--center_loss_alfa            //center_loss参数
--epoch_size 95               //单个epoch中的iter迭代数
--batch_size                  //batch_size大小

```

###  **启动训练和测试过程**

执行shell脚本：
```
sh train.sh
```