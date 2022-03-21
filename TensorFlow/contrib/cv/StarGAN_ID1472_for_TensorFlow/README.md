# StarGAN_for_TensorFlow
## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息
**发布者（Publisher）：Huawei**
**应用领域（Application Domain）：Image Synthesis
**版本（Version）：1.1
**修改时间（Modified） ：2021.08.27
**大小（Size）：13.6 MB
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：FP32
**处理器（Processor）：昇腾910
**应用级别（Categories）：Research
**描述（Description）：基于TensorFlow框架的StarGAN生成式对抗网络训练代码

## 概述
-     StarGAN作为一种生成式对抗网络，它能从多域的图像中有效地训练模型，只使用一个生成器和一个识别器就能学习多个域之间的映射。
      StarGAN具有将输入图像灵活地翻译到任意目标域的新能力，在人脸属性转换和表情合成任务上都有较好的效果。

-   参考论文：

        https://arxiv.org/pdf/1711.09020.pdf

-   参考实现（官方开源pytorch版）：
    
        https://github.com/yunjey/StarGAN

-   模型的Tensorflow复现（GPU版）：
    
        https://gitee.com/runner42195/StarGAN_for_TensorFlow
   
-   适配昇腾 AI 处理器的实现
    
        https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/StarGAN_ID1472_for_TensorFlow

-   通过Git获取对应commit\_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

### 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   model phase: train/test
    -   image_size: 128
    -   g_conv_dim: 64
    -   d_conv_dim: 64
    -   g_repeat_num: 6
    -   d_repeat_num: 6
    -   lambda_cls: 1.
    -   lambda_rec: 10.
    -   lambda_gp: 10.
    
-   训练数据集预处理（当前代码以CelebA数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为128*128
    -   按照尺寸为128×128进行中心裁剪
    -   随机水平翻转图像
    -   根据平均值和标准偏差对输入图像进行归一化


-   训练超参（单卡）：
    -   batch_size: 16
    -   c_dim: 5
    -   selected_attrs: Black_Hair Blond_Hair Brown_Hair Male Young
    -   d_train_repeat: 5
    -   init_learning_rate: 1e-4
    -   lr_update_step: 1000
    -   num_step_decay: 100000
    -   beta1: 0.5
    -   beta2: 0.999
    -   training_iterations: 200000
    


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

1.模型训练使用**CelebA**数据集，CelebA是香港中文大学的开放数据，包含10177个名人身份的202599张图片，并且都做好了特征标记，这对人脸相关的训练是非常好用的数据集。

2.数据集请用户在官网自行获取图像img_align_celeba.zip和标签list_attr_celeba.txt文件，或通过如下链接获取：

  - 图像数据 [CelebA images]
    
  - 标签数据 [CelebA attribute labels]

3.数据集下载后，放入相应目录下，在训练脚本中指定数据集路径，可正常使用

### 模型训练

- 下载工程代码，并选择合适的下载方式下载源码包
  
- 启动训练之前，首先要配置程序运行相关环境变量，环境变量配置信息参见：

   [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练

  - 配置训练参数

     首先在scrips/train_1p.sh中配置训练数据集路径等参数，请用户根据实际路径配置

     ```
     image_root="../datasets/celeba/images"    # 数据集路径
     metadata_path="../datasets/celeba/list_attr_celeba.txt"   # 数据集标注文件路径
     batch_size=16   # batch size
     c_dim=5   # 训练特征维度
     selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"   # 训练的属性
     output_dir="../logs/model_output"   # 权重、日志保存路径
     ...
     --image_root="$image_root" \
     --metadata_path="$metadata_path" \
     --batch_size="$batch_size" \
     --c_dim="$c_dim" \
     --selected_attrs="$selected_attrs" \
     --output_dir="$output_dir"
     ...
     ```
  - 启动单卡训练 （脚本为StarGAN_ID0725_for_TensorFlow/scripts/train_1p.sh） 

     ```
     bash train_1p.sh
     ```
  
- 在线推理

  - 配置推理参数

     首先在scrips/test_1p.sh中配置推理数据集路径等参数，请用户根据实际路径配置

     ```
     image_root="../datasets/celeba/images"    # 数据集路径
     metadata_path="../datasets/celeba/list_attr_celeba.txt"   # 数据集标注文件路径
     batch_size=1   # batch size
     c_dim=5   # 训练特征维度
     selected_attrs="Black_Hair Blond_Hair Brown_Hair Male Young"   # 训练的属性
     output_dir="../logs/model_output"   # 权重、日志与推理输出的保存路径
     checkpoint="model-200000"   # 推理调用的ckpt文件
     ...
     --image_root="$image_root" \
     --metadata_path="$metadata_path" \
     --batch_size="$batch_size" \
     --c_dim="$c_dim" \
     --selected_attrs="$selected_attrs" \
     --output_dir="$output_dir" \
     --checkpoint="$checkpoint"
     ...
     ```
  - 启动在线推理 （脚本为StarGAN_ID0725_for_TensorFlow/scripts/test_1p.sh） 

     ```
     bash test_1p.sh
     ```

- 生成pb模型

  - 执行ckpt转pb的冻结脚本，请用户根据实际路径配置参数

     ```
     python3.7 frozen_graph.py --phase frozen_graph --input_checkpoint ./logs/model_output/checkpoint/model-200000 --out_pb_path ./stargan_model.pb
     ```
  
- NPU与GPU训练性能对比
  
  - 在相同超参下，Ascend 910和GPU V100的训练性能对比如下：
    ```
    NPU上每1000次迭代平均耗时: 112s
    GPU上每1000次迭代平均耗时: 250s    
    ```
    
  
- NPU与GPU训练精度对比
  
  - 在CelebA数据集上抽取了2000张图像作为测试数据，通过FID量化评价指标评测模型精度。FID 为从原始图像的计算机视觉特征的统计方面的相似度来衡量两组图像的相似度，分数越低代表两组图像越相似，或者说二者的统计量越相似。FID相关实现可参见官方github工程: [FID](https://github.com/bioinf-jku/TTUR)
。
    
  - 在相同超参下，模型训练训练相同的迭代次数（200000次），训练精度对比如下：
     ```
     NPU上模型推理的FID精度为: 15.339
     GPU上模型推理的FID精度为: 15.285
     ```

     精度对比图片文件:[StarGAN精度对比图片](https://pan.baidu.com/s/1iZVRHqtc8eXjxgcruaP_7g)
     （提取码：0000）
  

## 高级参考

### 脚本和示例代码<a name="section08421615141513"></a>

    ├──LICENES                          //参考模型license文件
    ├──README.md                        //说明文档
    ├──requirements.txt                 //依赖
    ├──frozen_graph.py                  //ckpt转pb的冻结脚本
    ├──main.py                          //模型训练、测试入口代码
    ├──data									 
    │    ├──data_loader.py              //数据集加载代码
    ├──model									 
    │    ├──models.py                   //网络模型代码
    │    ├──ops.py                      //构建模型的网络层和算子								 
    │    ├──stargan.py                  //模型运行的相关实现
    ├──scripts									 
    │    ├──npu_set_env_1p.sh           //单卡环境变量配置文件									 
    │    ├──train_1p.sh                 //单卡训练脚本									 
    │    ├──test_1p.sh                  //单卡测试脚本									 
    │    ├──download.sh                 //数据集下载脚本


### 脚本参数<a name="section6669162441511"></a>

```
    --phase                     model phase: train/test
    --image_root                the root path of images
    --metadata_path             the path of metadata(labels)
    --batch_size                batch size for training
    --c_dim                     the dimension of condition
    --selected_attrs            selected attributes for the CelebA dataset
    --training_iterations       number of total iterations for training D
    --lr_update_step            learning_rate update step       
    --num_step_decay            step for starting decay learning_rate
    --output_dir                checkpoint and summary directory
    --summary_steps             summary period
    --save_steps                save sample period
    --checkpoint_steps          checkpoint period
```

### 训练过程<a name="section1589455252218"></a>

NA


### 推理/验证过程<a name="section1465595372416"></a>

NA
