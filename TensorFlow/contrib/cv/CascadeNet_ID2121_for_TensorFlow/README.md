# 简介
Cascade模型是论文“A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction”的Tensorflow实现（基于晟腾NPU）。该框架使用卷积神经网络（CNN）的深层级联从欠采样数据中重建2D心脏磁共振（MR）图像的动态序列，以加速数据采集过程，论文提出了Data sharing层，有效引入了核磁影像已知的先验信息，降低了网络的学习难度，有效保证了网络学习的效果。
# 文件说明
* saveModels：为训练输出保存的文件夹，checkpoints以及记录的tensorboard文件
* data：数据集,包括图片以及打包好的Hdf5文件
* models：网络构建相关代码
* utils：一些依赖函数
* make_h5data:打包数据脚本，为保证采样模板一致，训练集打包时保存模板，打包测试集时直接加载模板，不重新生成
* train：训练脚本
* test：测试脚本
* image_to_bin: 图像文件转bin文件
* ckpt_to_pb: 保存的ckpt网络模型转为pb文件
* view_pb: 查看pb文件节点
* om_precision: 读取离线推理的结果文件（bin），精度统计
# 依赖安装
* pip install tensorflow-gpu==1.15.0
* pip install numpy
* pip install openc-python
# 数据集
* 胸部核磁影像，训练集共1200张核磁图片，测试集36张
* 数据集地址：obs://imagenet2012-lp/cascade_re/data/
* 数据集说明：chest_train、chest_test文件夹分别存放训练集测试集图片；chest_train_acc3.hdf5、chest_test_acc3.hdf5为相应的打包好的数据集
# 训练命令
* python train.py
# 测试
* python test.py
# 指标计算
* 测试脚本中会计算MSE与PSNR，提供了相关函数
# 训练损失对比
|                | 论文  | GPU       | Ascend   |
|----------------|------|-----------|----------|
| MSE | 0.89*e-3 | ~1.11*e-3 | ~1.2*e-3 |
### GPU
* 论文未提供数据集，训练数据集与论文中不一致，无法达到论文给出的指标
### NPU
* 训练损失达到1.2*e-3，与GPU差距5%，基本符合要求
# 测试集精度
|                |  GPU | NPU  | 离线推理 |
|----------------|------|--------|--------|
| MSE | 3.57*e-3 | 3.07*e-3 | 3.08*e-3  |
| PSNR | 38.94 | 39.61 | 39.60  |
* 两个精度指标NPU均略优于GPU，约1.7%；离线推理与NPU在线推理精度相当，差距小于0.1%
# 训练性能
|                |  GPU | NPU   |
|----------------|------|--------|
| Epoch | 2 min | 6 min |
* 训练性能单个epoch耗时NPU约为GPU3倍，提交ISSUE，分析后主要原因为： 网络涉及大量FFT与IFFT操作，该算子NPU训练时不支持，训练时速度无法提高
* 但离线推理时，推理性能不受影响，离线推理速度： 重建一组数据耗时0.52s
# 离线推理命令参考
* ./out/msame --model="cascade_om.om" --input="./feature/,./mask/" --output="./" --outfmt BIN
# pb转om命令参考
* ./Ascend/ascend-toolkit/latest/atc/bin/atc --input_shape="feature:1,256,256,2;mask:1,256,256" --input_format=NHWC --output="./cascade_om" --soc_version=Ascend310 --framework=3 --model="cascade.pb"
# obs文件地址
* GPU复现及保存的模型：obs://imagenet2012-lp/cascade_re/
* NPU复现（包括一些保存的BIN文件）：obs://imagenet2012-lp/cascade_modelarts/
* NPU训练log及保存的模型：obs://imagenet2012-lp/cascade_log/MA-new-cascade_modelarts-11-24-11-26/
* NPU测试集结果：obs://imagenet2012-lp/cascade_log/MA-new-cascade_modelarts-12-02-10-21/
* NPU ckpt转pb：obs://imagenet2012-lp/cascade_log/MA-new-cascade_modelarts-12-02-12-28/