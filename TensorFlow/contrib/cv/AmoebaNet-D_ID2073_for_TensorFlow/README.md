- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.25**

**大小（Size）：75MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的AmoebaNet-D图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

 AmoebaNet-D是由AmoebaNet演化神经架构搜索算法搜索出的一个图像分类神经网络。

- 参考论文：

  [https://arxiv.org/pdf/1802.01548.pdf](Regularized Evolution for Image Classifier Architecture Search)

- 参考实现：

  https://github.com/tensorflow/tpu/tree/master/models/official/amoeba_net

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/AmoebaNet-D_ID2073_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

      - data_dir
      - model_dir
      - num_cells
      - image_size
      - num_epochs
      - train_batch_size
      - eval_batch_size
      - lr=2.56 
      - lr_decay_value
      - lr_warmup_epochs
      - mode=train_and_eval 
      - iterations_per_loop=1251
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
    npu_config = NPURunConfig(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        save_summary_steps=0,
        # dump_config=dump_config,
        # fusion_switch_file="/home/test_user03/tpu-master/models/official/amoeba_net/fusion_switch.cfg",
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
        #precision_mode="allow_mix_precision")
        precision_mode="allow_fp32_to_fp16")
        #precision_mode="force_fp32")

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

1、模型训练使用ImageNet2012数据集，数据集请用户自行获取

2、数据集训练前需要做预处理操作，将数据集封装为tfrecord格式(方法见https://github.com/tensorflow/models/tree/master/research/slim)

3、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

4、AmoebaNet训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数。
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：
        
             ```
             # 路径参数初始化
              --data_dir=${data_path} 
              --model_dir=${output_path} 
              --num_cells=6 
              --image_size=224 
              --num_epochs=35 
              --train_batch_size=64 
              --eval_batch_size=64 
              --lr=2.56 
              --lr_decay_value=0.88 
              --lr_warmup_epochs=0.35 
              --mode=train_and_eval 
              --iterations_per_loop=1251  
             ```
        
          2. 启动训练。（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── amoeba_net.py                          //训练与测试入口
├── README.md                               //代码说明文档
├── amoeba_net_model.py                    //模型功能
├── model_builder.py                       //根据用户传入的参数构建模型
├── model_specs.py                         //Amoeba_net架构配置
├── network_utils.py                       //Amoeba-net使用的常见操作的自定义模块
├── network_utils_test.py                  //对network_utils自定义模块的测试
├── tf_hub.py                               //模型导出和评估
├── inception_preprocessing.py            //图像预处理
├── train_testcase.sh                      //训练测试用例
├── online_inference_testcase.sh           //在线推理测试用例
├── modelzoo_level.txt                     //网络状态描述文件
├── requirements.txt        
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
├── common
│    ├──imagenet.py                         //为ImageNet ILSVRC 2012数据集提供数据帮助程序
│    ├──inference_warmup.py                //inference warmup实现```
```

## 脚本参数<a name="section6669162441511"></a>

```
--use_tpu              是否使用tpu，默认：False（由于该代码从tpu版本迁移过来，在昇腾910上只能是False）
--mode                 运行模式，可选：train_and_eval，train，eval
--data_dir             数据集目录
--model_dir           保存checkpoint的目录
--num_cells             网络结构中cell的数量，默认：6
--image_size            图像尺寸，默认：224
--num_epochs           训练迭代次数，默认：35
--train_batch_size     训练的batch size，默认：64
--eval_batch_size      验证的batch size， 默认：64    
--lr                     初始学习率，默认：2.56
--lr_decay_value        学习率指数衰减 默认：0.88
--lr_warmup_epochs      初始学习率从0增长到指定学习率的迭代数，默认：0.35
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。