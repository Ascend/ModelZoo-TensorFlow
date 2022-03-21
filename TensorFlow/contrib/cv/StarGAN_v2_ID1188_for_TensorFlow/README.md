# StarGAN_v2_for_TensorFlow
## 目录
[TOC]


## 基本信息
**发布者（Publisher）：Huawei**
**应用领域（Application Domain）：Image Synthesis
**版本（Version）：1.1
**修改时间（Modified） ：2021.09.28
**大小（Size）：1.6 MB
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：FP32
**处理器（Processor）：昇腾910
**应用级别（Categories）：Research
**描述（Description）：基于TensorFlow框架的StarGAN_v2生成式对抗网络训练代码

## 概述

![1636701707490](./pics/overview.png)

-     完善的image-to-image translation模型应当能够在多个视觉域之间进行映射，同时还应兼顾所生成图像的多样性和多域扩展性。StarGAN v2提出只使用单一的框架即可同时解决二者，同时在baseline模型上得到了卓越的性能提升。同时，作者们也发布了新的动物人脸数据集AFHQ。

![1636633669151](./pics/sample.png)

-   参考论文：

        https://arxiv.org/abs/1912.01865

-   参考实现：
    
        https://github.com/taki0112/StarGAN_v2-Tensorflow
    

### 默认配置<a name="section91661242121611"></a>
-   训练数据集预处理：
    -   resize图像的输入尺寸为256*256
    -   随机水平翻转图像
    -   resize286*286后随机裁剪到原始尺寸


-   训练超参：
    -   iteration: 200000
    -   batch_size: 4
    -   decay_iter: 50000
    -   lr: 0.0001
    -   ema_decay: 0.999
    -   r1_weight: 1
    -   gp_weight: 10
    -   ch: 32
    -   n_layer: 4
    -   style_dim: 16
    -   img_height: 256
    -   img_width: 256
    


### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度<a name="section20779114113713"></a>
相关代码示例

```
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = 1
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
```

## 训练环境准备

1.  硬件环境准备请参见
    [各硬件产品文档](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)
    需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/index)
    获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手
### 数据集准备

1. AFHQ数据集：即Animal Face HQ（AFHQ），由16130张高质量图像组成，分辨率为512×512。通过具有多个（三个）域和每个域中各种品种的不同图像，AFHQ提出了一个具有挑战性的图像间转换问题。
2. CelebA-HQ是一个大型人脸属性数据集，拥有超过200K张人脸图像，每个图像都有40个属性注释。
3. 请用户自行准备好数据集，数据集下载后，放入dataset目录下即可。或通过以下OBS共享链接获取

> URL:
> https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=Uehhty+Pvk1/WI/UeBN3Vn2lSgb4hZW/xR7anU2x7q11wOHP4+hS4bE13hhGMbXmmokwR+EAoaqAB2Fn4tHphMtZgY9m8EAdWj46HK9QwZ+iLXgW4HA+9h1wEKSnIg/z/N9E1HHQ5JoW26ZCp9QZvKz/cuMuh04s4Fm4P7+UecGEWODf/QbqSPbBkSldjwBzbvhF/ggG8dgpYOYxnCer2zOIJbuO/zw0yz6z6CKH76SpOA6M7dKVtJSxHhQ9XmL65PGWmr2iZ2mnywnHCGg1T0PAJqYVi7X6SVwz8GcSRjjGqXrgw5SJvqMVOkAWBvGrzdmQdcD2rTS3sKCOiY+JMltuE7JaKDQiMDKC7eCY/o8bO0aAj8CD+1/ItKu/w6YAfEwV1D5dMlw9bNXPFi4WpN3+ukcdBZ3qg5qn6oQzV1gqs4nLAQ9GU/nnuPd/clDMBvcLUBQeTnKesAwOnYE6UD0eDGO92n1Cbwy22GdGJj8TivbhgpjB5XoBLTS/E4hVpJ73rUwF0Wjw7CzIwgRvLfSKT22BAjj2mu7k9bFMrbbkm8m1fkUvad7MT9rmuWcw
>
> 提取码:
> 666666
>
> *有效期至: 2022/10/25 15:03:40 GMT+08:00

### 模型训练

- 下载工程代码，并选择合适的下载方式下载源码包
  
- 启动训练之前，首先要配置程序运行相关环境变量，环境变量配置信息参见：

   [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练

  - 配置训练参数

     首先在scrips/train_1p.sh中配置训练数据集等参数，具体参数可参考下文

     ```
     python3 main.py --dataset afhq-raw --phase train
     ```
  - 启动单卡训练  
  
     ```
     bash scripts/train_1p.sh
     ```
  
- 在线推理

  - 配置推理参数

     首先在scrips/test_1p.sh中配置推理数据集等参数；其次以afhq数据集为例，在afhq-raw文件夹下建立test文件夹，并将验证集中的图片放置于test文件夹下

     ```
     python3 main.py --dataset afhq-raw --phase test
     ```
  - 启动在线推理
  
     ```
     bash scripts/test_1p.sh
     ```
  
- 生成pb模型

  - 执行ckpt转pb的冻结脚本，请用户根据实际路径配置参数：

     ```
     python3.7 frozen_graph.py --model_phase frozen_graph --input_checkpoint ./checkpoint/StarGAN_v2_afhq-raw_gan_1adv_0.3sty_1ds_0.1cyc/StarGAN_v2.model-20001 --out_pb_path ./stargan_v2_model.pb
     ```
  

- 转为om模型

  - 以refer_test模式为例，使用如下atc命令，即可得到对应的om模型：

    ```
    atc --model=./output_model/pb_model/frozen_model.pb --framework=3 --output=./model --soc_version=Ascend310 --input_shape='input_node1:1,256,256,3;input_node2:1,256,256,3'
    ```

- 在310服务器上进行测试

  - 根据转换得到的om模型，使用msame工具进行测试：

    ```
    msame --model model.om --output output/ --loop 10
    ```

    ```
    [INFO] acl init success
    [INFO] open device 0 success
    [INFO] create context success
    [INFO] create stream success
    [INFO] get run mode success
    [INFO] load model model.om success
    [INFO] create model description success
    [INFO] get input dynamic gear count success
    [INFO] create model output success
    output//20211111_224034
    [INFO] model execute success
    Inference time: 105.365ms
    [INFO] model execute success
    Inference time: 105.487ms
    [INFO] model execute success
    Inference time: 105.386ms
    [INFO] model execute success
    Inference time: 105.419ms
    [INFO] model execute success
    Inference time: 105.323ms
    [INFO] model execute success
    Inference time: 105.42ms
    [INFO] model execute success
    Inference time: 105.423ms
    [INFO] model execute success
    Inference time: 105.461ms
    [INFO] model execute success
    Inference time: 105.383ms
    [INFO] model execute success
    Inference time: 105.504ms
    [INFO] get max dynamic batch size success
    [INFO] output data success
    Inference average time: 105.417100 ms
    Inference average time without first time: 105.422889 ms
    [INFO] destroy model input success.
    [INFO] unload model success, model Id is 1
    [INFO] Execute sample success
    [INFO] end to destroy stream
    [INFO] end to destroy context
    [INFO] end to reset device is 0
    [INFO] end to finalize acl
    ```

- NPU与GPU训练速度对比

  NPU使用ASCEND910，32GB显存，GPU使用TITAN V，12GB显存。TensorFlow版本为1.15。Batch_size=4情况下，训练速度对比：

  |    GPU    |    NPU    |
  | :-------: | :-------: |
  | 1.6s/iter | 9.6s/iter |

- NPU与GPU精度对比

  - afhq数据集上，通过FID量化评价指标评测模型精度，使用如下实现[FID](https://github.com/bioinf-jku/TTUR)进行测试，结果如下：

    |  GPU   |   NPU    |
    | :----: | :------: |
    | 38.87  | 41.23    |

## 高级参考

### 脚本和示例代码<a name="section08421615141513"></a>

    ├──LICENES                          //参考模型license文件
    ├──README.md                        //说明文档
    ├──requirements.txt                 //依赖
    ├──frozen_graph.py                  //ckpt转pb的冻结脚本
    ├──main.py                          //模型训练、测试入口代码
    ├──dataset							//数据集目录
    │    ├──afhq-raw                    //AFHQ数据集					 
    ├──utils.py                         //预处理等操作
    ├──ops.py                           //构建模型的网络层和算子								 
    ├──StarGAN_v2.py                    //模型运行的相关实现
    ├──scripts									 								 
    │    ├──train_1p.sh                 //单卡训练脚本									 
    │    ├──test_1p.sh                  //单卡测试脚本									 
    │    ├──download.sh                 //数据集下载脚本


### 脚本参数<a name="section6669162441511"></a>

```
  --phase PHASE         train or test or refer_test 
  --dataset DATASET     dataset_name
  --refer_img_path REFER_IMG_PATH
                        reference image path
  --iteration ITERATION
                        The number of training iterations
  --batch_size BATCH_SIZE
                        The size of batch size
  --print_freq PRINT_FREQ
                        The number of image_print_freq
  --save_freq SAVE_FREQ
                        The number of ckpt_save_freq
  --decay_flag DECAY_FLAG
                        The decay_flag
  --decay_iter DECAY_ITER
                        decay start iteration
  --lr LR               The learning rate
  --ema_decay EMA_DECAY
                        ema decay value
  --adv_weight ADV_WEIGHT
                        The weight of Adversarial loss
  --sty_weight STY_WEIGHT
                        The weight of Style reconstruction loss
  --ds_weight DS_WEIGHT
                        The weight of style diversification loss
  --cyc_weight CYC_WEIGHT
                        The weight of Cycle-consistency loss
  --r1_weight R1_WEIGHT
                        The weight of R1 regularization
  --gp_weight GP_WEIGHT
                        The gradient penalty lambda
  --gan_type GAN_TYPE   gan / lsgan / hinge / wgan-gp / wgan-lp / dragan
  --sn SN               using spectral norm
  --ch CH               base channel number per layer
  --n_layer N_LAYER     The number of resblock
  --n_critic N_CRITIC   number of D updates per each G update
  --style_dim STYLE_DIM
                        length of style code
  --num_style NUM_STYLE
                        number of styles to sample
  --img_height IMG_HEIGHT
                        The height size of image
  --img_width IMG_WIDTH
                        The width size of image
  --img_ch IMG_CH       The size of image channel
  --augment_flag AUGMENT_FLAG
                        Image augmentation use or not
  --checkpoint_dir CHECKPOINT_DIR
                        Directory name to save the checkpoints
  --result_dir RESULT_DIR
                        Directory name to save the generated images
  --log_dir LOG_DIR     Directory name to save training logs
  --sample_dir SAMPLE_DIR
                        Directory name to save the samples on training

```

### 训练过程<a name="section1589455252218"></a>

```
bash scripts/train_1p.sh
```


### 推理/验证过程<a name="section1465595372416"></a>

```
bash scripts/test_1p.sh
```


