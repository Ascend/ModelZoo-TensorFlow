# 简介
GMan_Net模型是论文“Single Image Dehazing with a Generic
Model-Agnostic Convolutional Neural Network”的Tensorflow实现（基于晟腾NPU）。该
框架使用卷积神经网络（CNN）的深层级联从受云雾遮盖的图像数据中恢复无污染的高清图像。
# 文件说明
* gman_flags：运行参数配置
* gman_config：config配置
* gman_constant：超参定义
* gman_model：网络构建相关代码
* gman_net：网络封装类
* gman_learningrate：学习率
* gman_log：log信息定义
* gman_tools：网络定义工具函数
* gman_tower：损失函数定义
* datasets: 数据集生成及加载
* gman_train：训练脚本
* gman_eval：测试脚本
* test：测试脚本(固定分辨率)
* image_to_bin: 图像文件转bin文件
* ckpt_to_pb: 保存的ckpt网络模型转为pb文件
* view_pb: 查看pb文件节点
* om_precision: 读取离线推理的结果文件（bin），精度统计
# 依赖安装
* pip install tensorflow-gpu==1.15.0
* pip install numpy
* pip install opencv-python
# 数据集
* 户外图像数据集，训练集共140000张，测试集共14000张。
* obs地址：**obs://imagenet2012-lp/Gman_re/**
* 文件夹**ClearResultImages**、**HazeImages**分别存放label和加雾的图片；
* **Train_record**文件夹存放生成的TFRecord文件用于训练；
* 文件夹**Test_images**存放生成的固定分辨率的测试集图片
* 离线推理的bin文件及结果存放在：**obs://imagenet2012-lp/GMan_modelarts/Test_bin/**
# 训练命令
* python gamn_train.py
# 测试
* python test.py
# 指标计算
* 测试脚本中会计算PSNR，提供了相关函数实现
# 训练损失对比
|                 | GPU      | Ascend 910 |
|----------------------|----------|------------|
| MSE  | ~2.0*e-3 | ~2.1*e-3   |
* NPU与GPU损失差距大约1%以内，符合要求。

# 训练性能对比
|              | GPU  | Ascend 910 |
|--------------|--------|--------|
| 完成训练总耗时      | 121 min | 101 min |
| 平均每s训练图片张数   | ~190  | 160-710 |
* 整体训练性能NPU优于GPU，平均领先约20%；NPU训练性能有所波动，每秒处理图片160-710张，GPU稳定在190张左右；
# 测试集精度
* 说明：动态尺寸图片GPU测试集PSNR:  28.714060，略好于论文中给出的指标（28.474217），但是后续离线推理不支持动态shape的输入，故按照训练集的处理流程将测试集图像随机crop为224*224分辨率的图像，以便测试，以下精度指标均为固定shape测试集下的结果。

|                 | GPU  | Ascend 910 |离线推理 310 |
|----------------------|--------|--------|------ |
| PSNR  | 26.869251 |  27.065545 |27.065233 |
* 测试集测试精度NPU略优于GPU，约0.7%；离线推理精度与910NPU在线推理相当，差距几乎不可计（小于0.1%）
# 测试性能
|                 | GPU  | Ascend 910 |离线推理 310 |
|----------------------|--------|--------|------ |
| 单张图片处理耗时 | 15 ms |  2.2 ms |2.6 ms |
* 测试性能NPU优于GPU，约领先85%；离线推理与910NPU在线推理性能相当
# 离线推理命令参考
* ./out/msame --model="gman_om.om" --input="./Test_bin/HazedImages/TestImages/" --output="./" --outfmt BIN
# pb转om命令参考
* ./Ascend/ascend-toolkit/latest/atc/bin/atc --input_shape="feature:1,224,224,3" --input_format=NHWC --output="./gman_om" --soc_version=Ascend310 --framework=3 --model="gman.pb"
# obs文件地址
* GPU复现及保存的模型：**obs://imagenet2012-lp/Gman_re/**
* NPU复现（包括一些保存的BIN文件）：**obs://imagenet2012-lp/GMan_modelarts/**
* NPU训练log及保存的模型：**obs://imagenet2012-lp/GMan_log/MA-new-GMan_modelarts-12-09-14-21/**
* NPU测试集结果：**obs://imagenet2012-lp/GMan_log/MA-new-GMan_modelarts-12-09-16-20/**
* NPU ckpt转pb：**obs://imagenet2012-lp/GMan_log/MA-new-GMan_modelarts-12-09-17-18/**
* 离线推理精度计算：**obs://imagenet2012-lp/GMan_log/MA-new-GMan_modelarts-12-09-18-22/**
