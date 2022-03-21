### 概述

STGAN是一个建立在AttGAN基础上的人脸属性编辑模型，通过差分属性标签下选择性传输单元的跳跃连接实现了人脸高精度属性的编辑。

STGAN主要由两部分组成，一个生成器G和一个判别器D。在生成器G中，主要由编码器Genc和解码器Gdec，编码器用于抽象潜在属性表示，包含5个卷积层；解码器用于生成目标图像，包含5个转置卷积层。并在编码器和解码器过程中，加入STU选择单元，从而获得人脸属性编辑后的输出。在判别器D中有两个分支Dadv和Datt，Dadv用于区分图像是假图像还是真图像，由五个卷积层和两个全连接层组成；Datt预测属性向量，Datt和Dadv共享前5个卷积层，但通过另外两个全连接层预测属性向量，具体实现细节如下表。

- 参考论文

  [**STGAN: A Unified Selective Transfer Network for Arbitrary Image Attribute Editing**](https://arxiv.org/abs/1904.09709)

- 参考实现

  https://github.com/csmliu/STGAN

### 准备

- 环境
  - pip install numpy==1.20.1
  - pip install matplotlib==3.4.1
  - pip install imageio==2.9.0
  - pip install scipy==1.6.2
  - conda install tensorflow-gpu=1.15
- 数据集
  - CelebA数据集（请用户自行下载好）
    - 图片放在***DATAROOT/img_align_celeba/\*.jpg***
    - 属性标签放在***DATAROOT/list_attr_celeba.txt***
- 预训练模型
  - 从[百度网盘(vjf6)](https://pan.baidu.com/s/1ValVSxP0M-rMsBkF3jaHHA)下载，并解压缩至***./output/***

### 代码结构

```
STGAN
└─
  ├─att_classification
  ├─image
  ├─imlib
  ├─precision_tool
  ├─pylib
  ├─tflib
  ├─data.py       数据预处理代码
  ├─LICENSE  
  ├─models.py     模型代码
  ├─psnr_ssim.py  指标psnr/ssim
  ├─README.md     
  ├─test.py       测试代码
  ├─train.py      训练代码(高精度模式下)
  ├─train_.py     训练代码(混合精度模式下)
```

### 训练

- ```
  python train.py --experiment_name xxx
  ```

### 测试

- ```
  python test.py --experiment_name xxx --test_int 2
  ```

### psnr/ssim指标

- ```
  python psnr_ssim.py --experiment_name xxx
  ```

### 属性分类精度指标

- ```
  cd att_classification
  python test.py --img_dir ../output/xxx/sample_testing
  ```

### 结果对比

- 高精度模式下

  ![image-20210926212057237](image/image-20210926212057237.png)

  修改./tflib/utils.py，关闭TbeMultiOutputFusionPass这个UBFusion，在高精度模式下，性能不达标，psnr/ssim和分类精度指标都已达标。

  - 性能

    - 论文

      训练：27min/epoch

      推理：11.78ms/image

    - npu

      训练：52min/epoch

      推理：3.33ms/image

    - gpu

      训练：13.2/epoch

      推理：7.11ms/image

  - 精度

    - 论文

      psnr/ssim：31.67/0.948

      平均分类精度：86.05%

    - npu

      psnr/ssim：34.15/0.971

      平均分类精度：88.48%

    - gpu

      psnr/ssim：32.30/0.950

      平均分类精度：87.09%

  - 图片效果

    - 论文

      ![image-20210926211251680](image/image-20210926211251680.png)

    - npu

      ![image-20210926211645297](image/image-20210926211645297.png)

    - gpu

      ![image-20210926211429698](image/image-20210926211429698.png)

  

- 混合精度模式下

  ![image-20210926212311479](image/image-20210926212311479.png)

  修改./tflib/utils.py，开启混合精度，并且使用动态loss scale策略，性能达标，psnr/ssim指标也达标，但是分类精度指标差了3%。

  - 性能

    - 论文

      训练：27min/epoch

      推理：11.78ms/image

    - npu

      训练：5.2min/epoch

      推理：3.33ms/image

    - gpu

      训练：13.2/epoch

      推理：7.11ms/image

  - 精度

    - 论文

      psnr/ssim：31.67/0.948

      平均分类精度：86.05%

    - npu

      psnr/ssim：33.78/0.972

      平均分类精度：83.32%

    - gpu

      psnr/ssim：32.30/0.950

      平均分类精度：87.09%

  - 图片效果

    - 论文

      ![image-20210926211308958](image/image-20210926211308958.png)

    - npu

      ![image-20210926211536690](image/image-20210926211536690.png)

    - gpu
  
      ![image-20210926211424269](image/image-20210926211424269.png)

