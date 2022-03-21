# Unsupervised Learning of Object Structure and Dynamics from Videos

对应论文：https://arxiv.org/abs/1906.07889.

本目录将作者开源的代码移植到了华为Ascend 910机器上进行训练和离线推理，作者的代码连接： [project website](https://mjlm.github.io/video_structure/).

## 1.交付件说明
```bash
├── checkpoints                     # ckpt文件
│   ├── keras                       # keras model.save_weights()得到的ckpt
│   │   ├── checkpoint
│   │   ├── struc_vrnn.data-00000-of-00002
│   │   ├── struc_vrnn.data-00001-of-00002
│   │   └── struc_vrnn.index
│   ├── pb                          # 基于session 的ckpt 得到的pb
│   │   └── model.pb
│   └── session                     # 通过keras.backend.get_session()方法得到的ckpt
│       ├── checkpoint
│       ├── session_struc_vrnn.ckpt.data-00000-of-00001
│       ├── session_struc_vrnn.ckpt.index
│       └── session_struc_vrnn.ckpt.meta
├── testdata                        # 作者提供的测试数据集
│   ├── acrobot_swingup_random_repeat40_00006887be28ecb8.npz
│   └── acrobot_swingup_random_repeat40_00006887be28ecb8_short_sequences.npz
├── om                              # 离线推理
│   ├── data                            # 转换成bin格式的数据
│   │   ├── batch_0.bin
│   │   ......
│   │   └── batch_99.bin
│   ├── models                          # 模型的pb和om文件
│   │   ├── model.om
│   │   └── model.pb
│   ├── msame                           # 在Ascend910上编译的msame推理工具
│   ├── om_output                       # 通过msame工具推理om得到的输出，模型有四个输出节点对应编号0-3
│   │   └── 20210804_173815
│   │       ├── batch_0_output_0.bin
│   │       ├── batch_0_output_1.bin
│   │       ├── batch_0_output_2.bin
│   │       ├── batch_0_output_3.bin
│   │       ......
│   │       ├── batch_99_output_0.bin
│   │       ├── batch_99_output_1.bin
│   │       ├── batch_99_output_2.bin
│   │       └── batch_99_output_3.bin
│   └── scripts                         # 使用的相关脚本
│       ├── ckpt2pb.py                      # 将session的ckpt转成pb模型
│       ├── data2bin.py                     # 将数据导出成bin文件的形式
│       ├── eval.py                         # 验证离线推理的输出精度
│       ├── export.sh                       # 将pb模型转换成om模型的shell脚本
│       └── inference.sh                    # 使用msame工具推理pb的shell脚本
├── datasets.py                     # 读取testdata，生成tf.Dataset                      
├── dynamics.py                     # 论文中的dynamics模型
├── eval.py                         # 模型验证脚本
├── hyperparameters.py              # 超参设置
├── __init__.py                     # __init__.py文件
├── losses.py                       # 自定义loss
├── ops.py                          # 自定义op
├── README.md                       # 本文件
├── train.py                        # 训练脚本
└── vision.py                       # 论文中的vision模型


```
## 2.如何使用

### 2.1 前期准备


### 2.1训练

```bash
python train.py
```

### 2.2验证

```bash
python eval.py
```

### 2.3 运行pb转om脚本

```bash
cd om/scripts
./export.sh
```

### 2.4 运行msame执行om的脚本

```bash
cd om/scripts
./inference.sh
```

### 2.5 验证msame得到的离线推理结果

```bash
cd om/scripts
python3.7 eval.py
```

## 精度对比

跑了作者提供的小数据集，小数据集论文中没有给出精度指标，这里将CPU和NPU的精度做一下对比。

|  环境   | loss 精度  |
|  ----  | ----  |
| CPU  | 0.85 |
| NPU  | 0.82 |