-   [简介](#简介md)
-   [环境变量](#环境变量.md)
-   [数据集说明](#数据集说明.md)
-   [模型训练](#模型训练.md)
-   [验证过程](#验证过程.md)
-   [训练精度](#训练精度.md)
-   [脚本和示例代码](#脚本和示例代码.md)
-   [训练日志](#训练日志.md)
-   [模型性能](#模型性能.md)

<h2 id="简介.md">简介</h2>

**发布者（Publisher）：** Huawei

**应用领域（Application Domain）：** visible and infrared image fusion

**修改时间（Modified） ：** 2021.11.20

**框架（Framework）：** TensorFlow 1.15.0

**模型格式（Model Format）：** ckpt

**描述（Description）：** 基于TensorFlow框架的FusionGAN红外和可见光图像融合网络训练代码。FusionGAN提出了一种生成式对抗网络融合这可见光和红外信息，通过建立一个生成器和鉴别器之间的对抗博弈，其中生成器的目标是生成具有主要红外强度和附加可见光梯度的融合图像，而鉴别器的目标是迫使融合图像具有更多可见图像中的细节。这使得最终的融合图像能够同时保持红外图像中的热辐射强度和可见光图像中的纹理。

**参考论文 ： **    Ma J ,  Wei Y ,  Liang P , et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J]. Information Fusion, 2019, 48:11-26.（https://www.researchgate.net/publication/327393843_FusionGAN_A_generative_adversarial_network_for_infrared_and_visible_image_fusion）

**参考实现 ： ** https://github.com/yugezi0128/FusionGAN

**启动训练 ： ** bash train_testcase.sh

<h2 id="环境变量.md">环境变量</h2>

- numpy==1.16.4
- Pillow==8.1.0
- scipy==1.2.1
- opencv-python==4.5.3.56
- tensorflow-gpu==1.15.0
- numpy==1.20.0
- imageio==2.9.0

 modelart中的配置 ：

 - NPU：1*Ascend 910 CPU：24*vCPUs 96GB
 - AI Engine：Ascend-Powered-Engine tensorflow_1.15


<h2 id="数据集说明.md">数据集说明</h2>

- 数据集1 ： TNO database

- 数据集2 ： INO database


以上数据集均开源，且为可见光和红外图像融合常用数据集，数据集请用户自行获取。


在obs中已经上传了本文中训练使用的数据集TNO，路径为：obs://deserant-fusiongan/dataset/，内有训练和测试的可见光和近红外图像。其中，测试图像有两组，一组为背景较为复杂的图像，一组是一个图像序列，背景较为单一。


<h2 id="模型训练.md">模型训练</h2>


1. 配置训练参数 首先在脚本train_testcase.sh中，可直接修改main.py中的训练超参数；也可在train_testcase.py中传入超参数的值。配置训练数据集路径、模型输出路径、图片输出路径，请用户根据实际路径配置，数据集参数。

2. 启动训练 第一步：模型预训练，首先修改main.py 中IS_TRAINING参数为True，输入数据为数据集、输出数据为训练模型。

3. modelart配置如下： 

```
  - obs path ： /deserant-fusiongan/
  - data path in obs ： /deserant-fusiongan/dataset/
```

4. 启动训练。

```angular2
bash train_testcase.sh
```

<h2 id="验证过程.md">验证过程</h2>

1. 将train_testcase.sh中改变执行文件为test_one_image.py，训练：main.py;测试：test_one_image.py

2. 在code目录下创建checkpoint文件夹，将训之前训练好的模型从obs下载到本地checkpoint目录下，并和本地的code一起上传到服务器中。

3. modelart配置如下： 

```angular2
obs path : /deserant-fusiongan/
data path in obs : /deserant-fusiongan/dataset/
```

4. 启动测试。

```angular2
bash train_testcase.sh
```


<h2 id="训练精度.md">训练精度</h2>

参考EN\SD\SSIM\CC\VIF五个指标作为精度验证。以下各个指标为一组序列图像的精度。

|     |  论文精度	| GPU精度  |  NPU精度  |
|  ----     |  ----     |  ----    |   ----   |
| EN   | 6.8-7.1 |6.7-7.0|6.8-7.1 |
| SSIM | 0.6-0.68 |0.5-0.65 |0.4-0.61 |
| CC |0.64-0.65 |0.7-0.74 |0.7-0.76 |
| SF |6-7.9 |7.5-13 |8.0-12.1 |
| VIF |0.34-0.38|0.425-0.45 |0.415-0.45 |


**训练后的checkpoint文件归档obs地址:obs://deserant-fusiongan/workspace/**



<h2 id="脚本和示例代码.md">脚本和示例代码</h2>


```
├── main.py                                  //网络训练
├── README.md                                //代码说明文档
├── metrics
│    ├──CC.py                                //计算指标CC
│    ├──EN.py                                //计算指标EN
│    ├──SF.py                                //计算指标SF
│    ├──SD.py                                //计算指标SD
│    ├──SSIM.py                              //计算指标SSIM
│    ├──VIF.py                               //计算指标VIF
├── train_testcase.sh                        //单卡运行启动脚本
├── cfg.sh                                   //make cfg
├── utils.py                                 //数据预处理
├── model.py                                 //定义模型
├── modelart_entry.py                        //modelart准备
├── test_one_image.py                        //得到测试图像融合结果，并给出精度指标曲线
```

<h2 id="训练日志.md">训练日志</h2>

```
Epoch: [ 1], step: [10], time: [366.3574], loss_d: [1.39310193],loss_g:[25.23447227]
Epoch: [ 1], step: [20], time: [369.0135], loss_d: [0.62126660],loss_g:[12.92138290]
Epoch: [ 1], step: [30], time: [371.6750], loss_d: [1.03893065],loss_g:[9.52834511]
Epoch: [ 1], step: [40], time: [374.3323], loss_d: [0.42926469],loss_g:[6.30722237]
Epoch: [ 1], step: [50], time: [377.0307], loss_d: [0.41075188],loss_g:[8.82999134]
Epoch: [ 1], step: [60], time: [379.6944], loss_d: [0.92173839],loss_g:[8.75182629]
Epoch: [ 1], step: [70], time: [382.3554], loss_d: [1.03094649],loss_g:[15.73189926]
Epoch: [ 1], step: [80], time: [384.9997], loss_d: [1.03490973],loss_g:[16.21470451]
Epoch: [ 1], step: [90], time: [387.6526], loss_d: [2.75148726],loss_g:[10.69058228]
Epoch: [ 1], step: [100], time: [390.3027], loss_d: [3.56997538],loss_g:[17.26455116]
Epoch: [ 1], step: [110], time: [392.9647], loss_d: [1.56379306],loss_g:[29.88542747]
Epoch: [ 1], step: [120], time: [395.6154], loss_d: [2.12289810],loss_g:[18.99460030]
Epoch: [ 1], step: [130], time: [398.2671], loss_d: [0.50320888],loss_g:[15.47661018]
Epoch: [ 1], step: [140], time: [400.9041], loss_d: [0.83181250],loss_g:[14.21201706]
Epoch: [ 1], step: [150], time: [403.5500], loss_d: [0.29238495],loss_g:[16.66788864]
Epoch: [ 1], step: [160], time: [406.2123], loss_d: [1.42455351],loss_g:[36.14822769]
Epoch: [ 1], step: [170], time: [408.8590], loss_d: [0.47839081],loss_g:[8.02831078]
Epoch: [ 1], step: [180], time: [411.5836], loss_d: [0.41331083],loss_g:[10.92359352]
Epoch: [ 1], step: [190], time: [414.2585], loss_d: [1.01939440],loss_g:[15.61238575]
Epoch: [ 1], step: [200], time: [416.9215], loss_d: [0.23530087],loss_g:[23.65291214]
```

<h2 id="模型性能.md">模型性能</h2>
NPU的性能大概在GPU的2倍左右。