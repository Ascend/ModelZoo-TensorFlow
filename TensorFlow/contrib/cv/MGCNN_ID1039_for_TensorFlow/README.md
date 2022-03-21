## Introduction

MGCNN提出了一个卷积神经网络，根据它们生成的统计上相似的弱透镜图对不同的宇宙学情景进行分类,证明了在模拟收敛映射上训练的机器学习网络可以比传统的高阶统计数据更好地区分这些模型.为了加速训练，实现了一种新的数据压缩策略，该策略融合了对典型收敛地图特征形态的先验知识。在无噪声数据下，该方法将ΛCDM与最相似的MG模型完全区分开来，在使用完整的红移信息时，能够正确识别MG模型，准确率至少达到80%。加入噪声降低了所有模型的正确分类率，但神经网络仍然显著优于之前分析中使用的峰值统计量


- 参考论文：

    [Distinguishing standard and modified gravity cosmologies with machine learning](https://arxiv.org/abs/1810.11030) (Peel et al. 2018).
    
   

## Requirements
运行`MGCNN`模型需安装以下依赖：
- tensorflow 1.15.0
- scikit-learn 0.24.2
- numpy 1.21.2
- keras 2.3.1
- python 3.7.10



## Dataset
In the paper we use weak-lensing data obtained by ray-tracing through the DUSTGRAIN-pathfinder simulation set presented in [Weak lensing light-cones in modified gravity simulations with and without massive neutrinos](https://academic.oup.com/mnras/article-abstract/481/2/2813/5094586) (Giocoli et al. 2018).  These simulations are not yet public, so we are not able to include the original convergence maps derived from the various cosmological runs. We do provide, however, the wavelet PDF datacubes derived for the four models as described in the paper: one standard `LCDM` and three modified gravity `f(R)` models.





## Training MGCNN

先把data文件夹下载下来，并在train_mgcnn.py代码里面配置好路径
运行 python train_mgcnn.py -n0即可运行文件
n 0，1，2表示数据中含有噪声的等级，0表示没有噪声，数字越大噪声越大，对结果影响越大 

## Transfer learning

- 使用与GPU训练相同的数据集

- 模型修改

  通过使用npu自动迁移工具进行模型的迁移，详细过程请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgtool_13_0006.html)

- 配置启动文件`boot_modelarts.py`,启动训练时，需设置好`train_url` 和 `data_url` 两个路径，详情请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgma_13_0004.html) 。通过修改以下命令中的`python_command`来选择训练`train_mgcnn.py` ：

  ```
    ## start to train on Modelarts platform
    python_command = "train_mgcnn.py"
    print('python command:', python_command)
    os.system(python_command)
  ```

## Reference

脚本和示例代码

```
├──train_mgcnn.py         //MGCNN GPU上运行的代码
├──mgcnn_npu.py                //MGCNN NPU上运行的代码
├──figures                //无噪声最后的精度
├── README.md             //代码说明文件
├── data                  //数据集文件
│    ├──clean             //无噪声数据集
│    ├──sigma035          //噪声系数为0.35
│    ├──sigma070          //噪声系数为0.7

npu上代码使用方法：

1配置ModelArts相关参数:
AI Engine：Ascend-Power-Engine tensorflow_1.15-cann_5.0.2-py_37

Boot File Path: （代码放置在本地的路径）D:\programming\pycharm2020\code\test\MGCNN_ID1039_for_TensorFlow\mgcnn_cpu.py

Code Directory: （代码放置在本地的路径）D:\programming\pycharm2020\code\test\MGCNN_ID1039_for_TensorFlow

OBS Path: (代码输出到obs上的路径)/mgcnn/

Data Path in OBS:（obs上数据存放文件夹，也就是data文件夹放的路径） /mgcnn/mgcnn1/
2运行modelarts






## Result
- 训练性能

参数设置 -n 设置1  -f 设置为pdf_j37.npy
GPU  2h 40min  前五个iteration训练时间： 98.112s  97.721s  96.383s  95.473s  96.369s   2-5个iteration平均值：96.487
NPU  2h 25min  前五个iteration训练时间： 261.438s 94.439s  94.559s  95.051s  91.394s   2-5个iteration平均值：93.861

说明：NPU上总时间比GPU上总时间更短，在前五个iter中除了第一个iterNPU训练时间明显长于GPU，其他NPU训练时间都略短于GPU，可以看出，NPU上的性能比GPU上性能略好。

- 训练精度

参数设置 -n 设置1  -f 设置为pdf_j37.npy
以下以混淆矩阵表示精度：
GPU 0.96875  0        0.015625  0.015625
    0        0.9375   0.625     0
    0        0.046875 0.890625  0.625
    0.0625   0        0.03125   0.90625

NPU 1        0        0         0
    0        0.96875  0.03125   0
    0.015625 0        0.984375  0
    0        0        0.150625  0.984375


说明：表中的行和列分别表示四类，表中的数值表示预测结果，比如第一行第一列就表示第一类被预测为第一类的概率，因此对角线上数值越大表示精度越高，可以看出NPU上运行结果精度比gpu上要高很多，gpu除了第一类预测精度为0.96，其他都为0.9上下，npu的预测精度平均在0.98以上，分辨能力强了很多

