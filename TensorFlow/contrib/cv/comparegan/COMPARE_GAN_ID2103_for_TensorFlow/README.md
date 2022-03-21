## COMPARE_GAN
## 概述

COMPARE_GAN模型是关于GANs的综合实验，GAN通过框架中的生成模型和判别模型的互相博弈学习，从而产生相当好的输出，该模型为许多与生成性对抗网络相关的组件提供TensorFlow实现，COMPARE_GAN是一个用于训练和评估GAN的库，该库包括现有 GAN 中常用的损失函数，正则化和规范化模式，神经架构和量化指标等等。 

- 参考论文：

   A Large-Scale Study on Regularization and Normalization in GANs(https://arxiv.org/abs/1807.04720v3) 

- 参考源码：
  https://github.com/google/compare_gan
## 默认配置

- 训练数据集预处理：

  - 图像的输入尺寸为32*32
  - 图像输入格式：TFRecord

- 测试数据集预处理

  - 图像的输入尺寸为32*32
  - 图像输入格式：TFRecord

- 训练超参

  - Batch size: 64
  - lamba: 1
  - z_dim: 128
  - standardize_batch.decay = 0.9
  - standardize_batch.epsilon = 1e-5
  - Learning rate(LR): 0.0002
  - Optimizer: AdamOptimizer
  - AdamOptimizer.beta1: 0.5
  - AdamOptimizer.beta2: 0.999
  - Training_steps: 60000

## 训练环境
  - tensorflow-datasets==1.2.0
  - tensorflow-hub==0.8.0
  - tensorflow-gan==0.0.0.dev0
  - gin-config==0.4.0
  - tensorflow-probability==0.8.0
  - pstar==0.1.9
  - mock

## 数据集
选择cifar10数据集
## 模型训练及验证
- 执行NPU训练及验证入口
  - boot_modelarts.py
```
#!/bin/sh
### Modelarts Platform train command
export TF_CPP_MIN_LOG_LEVEL=2               ## Tensorflow api print Log Config
export ASCEND_SLOG_PRINT_TO_STDOUT=0        ## Print log on terminal on(1), off(0)
export ASCEND_GLOBAL_LOG_LEVEL=3
code_dir=${1}
data_dir=${2}
result_dir=${3}
pip install mock &&
pip install -r ${code_dir}/pip-requirements.txt &&
#start exec
#eval_after_train
python ${code_dir}/compare_gan/main.py  \
  --model_dir=${result_dir} \
	--tfds_data_dir=${data_dir} \
	--gin_config=${code_dir}/compare_gan/resnet_cifar10_false.gin \
	--myevalinception=${data_dir}/frozen_inception_v1_2015_12_05.tar.gz \
	--schedule=eval_after_train \
	--eval_every_steps=60000 \
```

- 脚本参数
  - model_dir 保存模型路径
  - tfds_data_dir 数据集路径
  - gin_config resnet_cifar网络参数配置gin
  - myevalinception 验证FID工具压缩包
  - schedule 模式：train(训练)、eval_after_train(训练后评估)
  - eval_every_steps 每训练多少步进行一次评估

- 数据集和验证FID工具压缩包obs桶

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=6IzahpC6+KkieR3BBmvBTSIQMWIfno8eyYY2tpwnWoXxTvmie6iYv2WCo/VPQ6LuvVXcLFHuVbcG6rE8sjOjPItb2jaY7OmqukWvs3Us6ZFAlmr5Fb1grPCjx9r54UXSY8NNRLUnyDIJ8Y8a7Wh7bNSibKQXQHKaEqxrVjflLliMnZauCZ4C58ai1s0d3eBu7iOR8S6yG+MleXkIJLpAh2lc++V0WEj0a2kRVhjIfZTJWCa5nLlUItXAmE0xpLnCgm0/P4z8OSXI0l4FyB0G/4f2fSC8g+WwlcIbDCmcfu9XVHqjdHnAFyrCpCv2S4POIrYErJQNVbKDDVbavw/BUflxzPIhaJdPTzWcUNFBscBA8ChzbqTbqM+NPTnJsFWR50cXx+uCw5Q3hkWoMpd6G8OR2LilK/lvHVRbsutppGtyIcegWb6OcJ77PzBdqAbqddHGa/GFBclciW8z5KO3OcqV+aMtIyajxexWEYTs9ZRICyozR1NiNX5UPXASQPWAYwg9h3c+3H3a/sEMYU5Xh3mU9AXrEdRvhAlB7ZfYkIImawq25aAon4sK0qNBgAyC

提取码:
123456

*有效期至: 2022/12/20 23:31:33 GMT+08:00
## 脚本和示例代码

```
├── LICENSE                                 
├── README.md                         //instructions       
├── boot_modelarts.py                 //the boot file for ModelArts platform            
├── help_modelarts.md                 //the help file for ModelArts platform           
├── compare_gan
│    ├──asess_util.py                //Migration interface            
│    ├──datasets.py                  //Interface for Image datasets based on TFDS (TensorFlow Datasets)
│    ├──eval_gan_lib.py              //Evaluation library 
│    ├──eval_utils.py                //Data-related utility functions            
│    ├──hooks.py                     //Contains SessionRunHooks for training  
│    ├──main.py                      //Binary to train and evaluate one GAN configuration       
│    ├──runner_lib.py                //Binary to train and evaluate a single GAN configuration.      
│    ├──test_utils.py                //Utility classes and methods for testing   
│    ├──utils.py                     //Utilities library     
│    ├──pip-requirements.py          //Environmental dependence       
│    ├──resnet_cifar10_false.py      //resnet_cifar configuration        
│    ├──run_modelarts_1p.py          //Program running entry file            
│    ├──architectures
│          ├── abstract_arch.py      //Defines interfaces for generator and discriminator networks 
│          ├── arch_ops.py           //Provides a library of custom architecture-related operations
│          ├── resnet_cifar.py       //Resnet generator and discriminator for CIFAR
│          ├── resnet_ops.py         //ResNet specific operations                  
│    ├──gans   
│          ├── abstract_gan.py       //Interface for GAN models that can be trained using the Estimator API
│          ├── consts.py             //Defines constants used across the code base.  
│          ├── loss_lib.py           //Implementation of popular GAN losses
│          ├── modular_gan.py        //Provides ModularGAN for GAN models with penalty loss
│          ├── ops.py                //Customized TensorFlow operations
│          ├── penalty_lib.py        //Implementation of popular GAN penalties
│          ├── utils.py              //Utilities library       
│    ├──metrics  
│          ├── eval_task.py          //Abstract class that describes a single evaluation task
│          ├── fid_score.py          //Implementation of the Frechet Inception Distance          
│    ├──tpu
│          ├── tpu_ops.py            //Tensorflow operations specific to TPUs
│          ├── tpu_random.py         //Provide methods for generating deterministic pseudorandom values
│          ├── tpu_summaries.py      //Provide a helper class for using summaries on TPU via a host call                      
```


## 精度
FID越低代表生成的假图片越好

* 论文精度：

| FID | 
| :--------: |
|   31.40  | 

* GPU精度：

| FID | 
| :--------: |
|   30.00  | 

* Ascend精度：
评估代码涉及平台不支持的float64，所以float64改为float32才能成功评估，因此出现了精度误差，下列是npu训练+gpu评估和npu训练+npu评估的结果，gpu评估代码用的float64，npu评估代码用的float32。

| FID(gpu评估)|FID(npu评估)| 
| :--------: | :--------:|
|   31.13  | 45.09|

GPU和NPU复现结果obs桶
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=6IzahpC6+KkieR3BBmvBTSIQMWIfno8eyYY2tpwnWoXxTvmie6iYv2WCo/VPQ6LuvVXcLFHuVbcG6rE8sjOjPItb2jaY7OmqukWvs3Us6ZFAlmr5Fb1grPCjx9r54UXSY8NNRLUnyDIJ8Y8a7Wh7bNSibKQXQHKaEqxrVjflLliMnZauCZ4C58ai1s0d3eBu7iOR8S6yG+MleXkIJLpAh34SZL6FGS8ZzEqdVzAjsCoKY1pnQeu/MsF5QKtzlIZhTpuMRiF23NeY4o5iy4H0a3zoA7b4tB7p+R/4ZDBm1OA0lvsTI5uHcTQ1uaFfaQCMXmsLK2bWmNVpy4MqtEOqyM2iWSBRNCigw/wK7eUMq3nl3JtCKeIuuCK2ojR91Bu79uWoMvYSx0PBviWYi6NpajTlseq8hCZHgMcsUdpLFErGPMcN7tXfLRXYJ7ueLv2cR2kHiZzGiCHXe7zCTCSa6QLqnFK4Mi4GUWtYSsJ1KZnLTGkUo+2ytV1SEAHr7udsk2usghUduHXwRWHdyjJA5iJkG76upxWnJLgN54VLV7g66eY1boGVekt8+o4sLae8

提取码:
123456

*有效期至: 2022/06/23 23:28:57 GMT+08:00

## 性能对比：

| GPU V100  | Ascend 910 | 
|:--------:| :--------:| 
|   5.2steps/s  | 11.4steps/s   | 


