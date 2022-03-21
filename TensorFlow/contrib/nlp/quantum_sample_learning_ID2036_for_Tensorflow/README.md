# Quantum Sample Learning

## 概述

迁移Quantum Sample Learning到Ascend910平台，使用NPU运行，并将结果与原论文进行对比

+ 参考论文 Learnability and Complexity of Quantum Samples
+ 参考项目 https://github.com/google-research/google-research/tree/master/quantum_sample_learning



## Requirements

python==3.6  absl-py  cirq==0.8.0  numpy>=1.16.4  scipy>=1.2.1  TensorFlow==1.15

Ascend: 1*Ascend 910 CPU: 24vCPUs 96GiB




## 代码及路径解释

```
Quantum Sample Learning
└─
  ├─README.md
  ├─LICENSE  
  ├─data        存放数据集文件夹
  ├─training_checkpoints_lm    存放checkpoint文件夹
  ├─data_loader.py     训练数据工具类 
  ├─data_loader_test.py     
  ├─circuit.py     生成数据集
  ├─run_lm.py     程序入口
```



## 数据集

数据集：q12c0



## 训练执行

```
python run_lm.py 
```

参数解释：

```
num_qubits : Number of qubits to be learnt. In the original paper, it was 12

epochs : Number of epochs to train, default is 20
```




## 训练结果

CPU

```
Min sampled probability 0.000000
Max sampled probability 0.002621
Mean sampled probability 0.000489
Space size 4096
Linear Fidelity: 1.002173
Logistic Fidelity: 1.001621
Number of bitstrings used in eval: 499968.000000
theoretical_linear_xeb: 1.016568
theoretical_logistic_xeb: 1.003946
linear_xeb: 1.002173
logistic_xeb: 1.001621
```

NPU

```
Min sampled probability 0.000001
Max sampled probability 0.002621
Mean sampled probability 0.000496
Space size 4096
Linear Fidelity: 1.032573
Logistic Fidelity: 1.009519
theoretical_linear_xeb: 1.015251
theoretical_logistic_xeb: 1.004565
linear_xeb: 1.032573
logistic_xeb: 1.009519
```

## 下载链接

### 数据集下载

链接：https://pan.baidu.com/s/1hFvneK9EJ6b36eSgDLSRNA  提取码：f88i

### checkpoint文件
链接：https://pan.baidu.com/s/1hc43pD9ntuyFXLsBvRrz2g  提取码：ytr0