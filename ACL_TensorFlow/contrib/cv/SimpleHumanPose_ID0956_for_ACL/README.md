## SimpleHumanPose
### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Human Pose Estimation**

**版本（Version）：1.0**

**修改时间（Modified） ：2022.05.23**

**框架（Framework）：TensorFlow 1.15.0**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的SimpleHumanPose网络离线推理代码**

### 概述

SimpleHumanPose模型主要针对人体姿态估计和姿态追踪，提出一种简化的baseline。当前流行的人体姿态估计方法都过于复杂，各种模型在结构上看起来差异性很大，但是性能上又很接近，很难判断究竟是什么在起作用。相比于其他人体姿态估计模型，该模型并没有使用过多的技巧和理论依据。它以简单的技术为基础，通过全面的消融实验进行了验证，提出一个较为简化直观的模型，取得了良好的性能。

- 参考论文：

  https://arxiv.org/abs/1804.06208

- 参考实现：

  https://github.com/mks0601/TF-SimpleHumanPose

### 文件结构
```bash
├── README.md                                 //代码说明文档
├── LICENSE                                   //许可证
├── src
│    ├── data/                                 //包含数据加载代码
│    ├── lib/                                  //包含2D多人姿态估计系统的核心代码                   
│    ├── main/                                 //包含train/test网络、ckpt转pb、数据集转bin等代码
|    ├── boot_pb_frozen.py                     //ModelArts启动pb固化代码
|    ├── boot_inference.py                     //ModelArts启动计算精度代码
|    ├── boot_test.py                          //ModelArts启动测试代码
|    ├── pip-requirements.txt                  //配置文件，用于指定依赖包的包名及版本号    
```

