## STGAN离线推理

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

  - CelebA数据集（请用户自行准备好数据集）
    - 图片放在***DATAROOT/img_align_celeba/\*.jpg***
    - 属性标签放在***DATAROOT/list_attr_celeba.txt***

- [ckpt文件(uurq)](https://pan.baidu.com/s/1tUd0-eLNuUSgA_9yskT4Gw)

  


### 代码结构

```
STGAN_ID1473_for_ACL
└─
  ├─att_classification
  ├─image
  ├─imlib
  ├─output
  	  ├─1024
  	  	  ├─checkpoints   ckpt模型路径
  	  	  ├─msame_out     msame推理结果（.bin）
  	  	  ├─om_model      om模型路径
  	  	  ├─om_testing    msame推理结果（.png）
  	  	  ├─pb_model      pb模型路径
  	  ├─xxx
  	  ├─...
  ├─pylib
  ├─tflib
  ├─bin2img.py         bin格式转img格式
  ├─ckpt2pb.py         ckpt转pb
  ├─data.py            数据预处理代码
  ├─img2bin.py         img格式转bin格式
  ├─LICENSE  
  ├─models.py          模型代码
  ├─modelzoo_level.txt
  ├─psnr_ssim.py       指标psnr/ssim
  ├─README.md     
  ├─requirements.txt
```

### ckpt文件转pb文件

- 运行ckpt2pb.py文件，将ckpt文件转为pb文件，ckpt_path为ckpt模型路径，pb_path为转化后的pb模型路径

  ```
  python3 ckpt2pb.py --ckpt_path xxx --pb_path xxx
  
  example: 
  python3 ckpt2pb.py --ckpt_path
  './output/1024/checkpoints/Epoch_(69)_(473of473).ckpt' --pb_path 
  './output/1024/pb_model'
  ```

- 下载已转好的[pb文件(uurq)](https://pan.baidu.com/s/1tUd0-eLNuUSgA_9yskT4Gw)

### pb文件转om文件

- 使用ATC模型转换工具进行模型转换

  ```
  /usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=./output/1024/pb_model/STGAN.pb --framework=3 --output=./output/1024/om_model/STGAN --soc_version=Ascend910 --input_shape="input1:1,128,128,3;input2:1,13;input3:1,13"
  ```

- 下载已转好的[om文件(uurq)](https://pan.baidu.com/s/1tUd0-eLNuUSgA_9yskT4Gw)

### 数据转bin文件

- 运行img2bin.py文件，将数据转化为bin格式，data_dir为原数据集路径，save_dir为转化后的bin数据集路径

  ```
  python3 img2bin.py --data_dir xxx --save_dir xxx
  
  example: 
  python3 img2bin.py --data_dir '/home/test_user02/yuxh/celeba/' --save_dir '/home/test_user02/yuxh/celeba_bin/'
  ```

### 使用msame工具离线推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法

获取到msame可执行文件之后，进行推理测试

```
/home/test_user02/yuxh/STGAN_NPU/tools/msame/out/msame --model "./output/1024/om_model/STGAN.om"  --output "./output/1024/msame_out/" --outfmt BIN --input "/home/test_user02/yuxh/celeba_bin/input1,/home/test_user02/yuxh/celeba_bin/input2,/home/test_user02/yuxh/celeba_bin/input3"
```

### bin文件转图片

msame工具离线推理的输出结果为.bin格式，需要将其转化为.png图片格式，data_dir为原数据集路径，bin_dir为离线推理的输出结果（.bin格式），save_dir为转化后的离线推理结果（.png格式）

```
python3 bin2img.py --data_dir xxx --bin_dir xxx --save_dir xxx

example:
python3 bin2img.py --data_dir '/home/test_user02/yuxh/celeba/' --bin_dir './output/1024/msame_out/20211104_165416' --save_dir './output/1024/om_testing'
```

### 精度验证

- psnr/ssim指标：

  ```
  python psnr_ssim.py --experiment_name xxx
  ```

- 属性分类精度指标：

  ```
  cd att_classification
  python test.py --img_dir ../output/xxx/om_testing
  ```

### 推理结果

- 性能

  ![image-20211104171459906](image/image-20211104171459906.png)

- 精度

  - 论文

    psnr/ssim：31.67/0.948

    分类精度：86.05%

  - npu

    psnr/ssim：33.78/0.972

    分类精度：83.32%

  - gpu

    psnr/ssim：32.30/0.950

    分类精度：87.09%

- 图片效果

  - 论文

    ![img](image/image-20210926211308958.png)

  - npu

    ![img](image/image-20210926211536690.png)

  - gpu

    ![img](image/image-20210926211424269.png)

