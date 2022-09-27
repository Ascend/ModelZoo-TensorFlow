-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)

<h2 id="概述.md">概述</h2>

现有的图像之间转换的方法，大部分都是需要图像对的方法，但是实际上有的场景下，很难得到这样的图像对。DualGAN使用对偶学习模式的GAN网络结构来进行image to image translation，将 domain A 到 domain B 之间的转换，构成一个闭环（loop）。通过 minimize 该图和重构图像之间的 loss 来优化学习的目标：
给定一个 domain image A，用一个产生器 P 来生成对应的 domain image B，由于没有和A匹配的图像对，这里是没有ground truth的。如果该图伪造的很好，那么反过来，用另一个产生器 Q，应该可以很好的恢复出该图，即Q(P(A, z), z') 应该和 A 是类似的。对于 domain image B 也是如此。

- 参考论文：

    https://arxiv.org/abs/1704.02510

- 参考实现：

    https://github.com/duxingren14/DualGAN 

- 适配昇腾 AI 处理器的实现：  
        
  https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/DualGAN_ID1001_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - batch_size: 1
  - lambda_A: 1000.0
  - lambda_B: 1000.0
  - train_epochs: 100

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 混合精度  | 否    |

脚本已默认关闭混合精度，因为开启之后loss正常收敛但是生成的图像会糊掉。


<h2 id="训练环境准备.md">训练环境准备</h2>

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB

运行环境：ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0107


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备

   模型训练使用[sketch-photo](http://www.cs.mun.ca/~yz7241/dualgan/dataset/)数据集，需要下载到./datasets/目录下。也可直接执行该目录下的download_dataset.sh脚本进行下载。

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：
     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置batch_size、dataset_name、train_epochs等参数。

     ```
		batch_size=1
		phase=train
		dataset_name=sketch-photo
		lambda_A=1000.0
		lambda_B=1000.0
		train_epochs=100
     ```

  2. 启动训练。

     启动单卡训练 （脚本为DualGAN_ID1001_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|GPU实测|NPU实测|
|---|---|---|
|min loss|0.026|0.030|

- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|sec/step|1.146|0.996|

- 生成图像
    - GPU训练结果

      A域原图
      ![pics](./result/gpu_realA.jpg)
      A域到B域
      ![pics](./result/gpu_A2B.jpg)

      B域原图
      ![pics](./result/gpu_realB.jpg)
      B域到A域
      ![pics](./result/gpu_B2A.jpg)
	  
    - NPU训练结果

      A域原图
      ![pics](./result/npu_realA.jpg)
      A域到B域
      ![pics](./result/npu_A2B.jpg)

      B域原图
      ![pics](./result/npu_realB.jpg)
      B域到A域
      ![pics](./result/npu_B2A.jpg)

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── main.py                                   //执行主函数代码
├── README.md                                 //代码说明文档
├── model.py                                  //定义模型结构
├── ops.py                                    //自定义算子
├── utils.py                                  //工具函数
├── requirements.txt                          //训练python依赖列表
├── test
│    ├──train_performance_1p.sh               //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                      //单卡全量训练启动脚本

```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  参考脚本的模型存储路径为./output/checkpoint。


