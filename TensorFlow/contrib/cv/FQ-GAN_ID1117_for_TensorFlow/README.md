- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.12**

**大小（Size）：416020KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的特征量化提升GAN训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

特征量化（FQ），将真数据样本和假数据样本嵌入到共享离散空间中。FQ的量化值被构造为一个进化词典，与最近分布历史的特征统计一致。因此，FQ隐式地在紧凑空间中实现了鲁棒的特征匹配。我们的方法可以很容易地插入现有的GAN模型中，在训练中几乎没有计算开销。

- 参考论文：
  
  [https://arxiv.org/abs/2004.02088](Feature Quantization Improves GAN Training)

- 参考实现：

  https://github.com/YangNaruto/FQ-GAN

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/FQ-GAN_ID1117_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 1
    - dataset=${data_path}/dataset/selfie2anime
    - phase='train'
    - test_train=True
    - quant=True
    - checkpoint_dir=${output_path}/checkpoint
    - result_dir=${output_path}/results
    - log_dir=${output_path}/logs
    - sample_dir=${output_path}/samples
    - epoch=1
    --iteration=10000


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否      |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         #precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```

混合精度相关代码示例:

 ```
    precision_mode="allow_mix_precision"

 ```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练使用selfie2anime数据集，数据集请用户自行获取https://github.com/taki0112/UGATIT

2、FQ训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练


        1. 配置训练参数。
        
        训练参数已经默认在脚本中设置，需要在启动训练时指定数据集路径和输出路径
        
        ```
        parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')
    
        parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
        parser.add_argument('--iteration', type=int, default=10000, help='The number of training '
                                                                      'iterations')
        parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
        parser.add_argument('--print_freq', type=int, default=10, help='The number of '
                                                                        'image_print_freq')
        parser.add_argument('--save_freq', type=int, default=10, help='The number of ckpt_save_freq')
        parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
        parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')
    
        parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
        parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
        parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
        parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
        parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
        parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
        parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect'）
        parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
        parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
        parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
        parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
        parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')
        parser.add_argument('--img_size', type=int, default=256, help='The size of image')
        parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
        parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint'）
        parser.add_argument('--result_dir', type=str, default='/cache/results'）
        parser.add_argument('--log_dir', type=str, default='logs'）
        parser.add_argument('--sample_dir', type=str, default='/cache/samples'）        
        ```
        
        2. 启动训练。
        
        ```
        python3.7 UGATIT.py
        ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── UGATIT.py                              //网络训练与测试代码
├── README.md                                 //代码说明文档
├── logger.py                                 
├── main.py                           
├── modelarts_entry_acc.py                          
├── ops.py                          
├── test                          
│    ├──train_full_1p.sh                //训练验证full脚本
│    ├──train_performance_1p.sh              //训练验证perf性能脚本
├──requirements.txt
```

## 脚本参数<a name="section6669162441511"></a>

```
-- Batch size: 1
-- dataset=${data_path}/dataset/selfie2anime
-- phase='train'
-- test_train=True
-- quant=True
-- checkpoint_dir=${output_path}/checkpoint
-- result_dir=${output_path}/results
-- log_dir=${output_path}/logs
-- sample_dir=${output_path}/samples
-- epoch=1
--iteration=10000
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。