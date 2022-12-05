- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.25**

**大小（Size）：1.3G**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的albert_xlarge_zh代码**

## 概述

ALBert基于Bert，但有一些改进。它在主要基准测试中实现了最先进的性能，参数减少了30%。对于albert_base_zh，它只与原始bert模型相比只有10个百分比的参数，并且保留了主要精度。

- 参考论文：
  
    https://arxiv.org/pdf/1909.11942.pdf

- 参考实现：

    https://github.com/brightmart/albert_zh

- 适配昇腾 AI 处理器的实现：    
  
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/albert_xlarge_zh_ID2348_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

#### 默认配置

- 网络结构

    - albert_xlarge_zh

-   训练超参（单卡）：

    pretrain:
    - Learning rate(LR): 0.00176
    - Batch size: 4
    - max_seq_length: 512

    finetune:
    - Learning rate(LR): 1e-6
    - Batch size: 32
    - max_seq_length: 128

#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |

#### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度


```
  session_config=tf.ConfigProto(allow_soft_placement=True)
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```


## 训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录


## 快速上手

#### 数据集准备<a name="section361114841316"></a>

- 用户需自行准备训练数据集，例如LCQMC数据集，可参考github源获取。
- 用户需提前下载预训练模型，参考github源里提供的albert_xlarge_zh_183k等模型下载方式

#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 单卡训练 
    
        2.1 pretrain训练：配置train_full_1p_pretrain.sh脚本中`data_path`（脚本路径albert_xlarge_zh_ID2348_for_TensorFlow/test/train_full_1p_pretrain.sh）,请用户根据实际路径配置，数据集参数如下所示：

             --input_file=${data_path}/tf*.tfrecord
        
        2.2 单p指令如下:

            bash train_full_1p_pretrain.sh --data_path=./lcqmc

        2.3 finetune训练：配置train_full_1p_finetune.sh脚本中`data_path`和`ckpt_path`（脚本路径albert_xlarge_zh_ID2348_for_TensorFlow/test/train_full_1p_finetune.sh）,请用户根据实际路径配置，数据集和预训练模型参数如下所示：

            --data_dir=${data_path} \
            --init_checkpoint=${ckpt_path}/albert_model.ckpt \
        
        2.4 单p指令如下:

            bash train_full_1p_finetune.sh --data_path=./lcqmc--ckpt_path=albert_xlarge_zh_183k

- 验证。

    1. 执行训练时设置以下参数：
    
       ```
       --do_eval=true
       ```

## 迁移学习指导

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    
- 模型训练。

    参考“模型训练”中训练步骤。

- 模型评估。

    参考“模型训练”中验证步骤。

## 高级参考

#### 脚本和示例代码

```
albert_xlarge_zh_ID2348_for_TensorFlow/
├── albert_config
│   ├── albert_config_base_google_fast.json
│   ├── albert_config_base.json
│   ├── albert_config_large.json
│   ├── albert_config_small_google.json
│   ├── albert_config_tiny_google_fast.json
│   ├── albert_config_tiny_google.json
│   ├── albert_config_tiny.json
│   ├── albert_config_xlarge.json
│   ├── albert_config_xxlarge.json
│   ├── bert_config.json
│   └── vocab.txt
├── args.py
├── bert_utils.py
├── classifier_utils.py
├── create_pretrain_data.sh
├── create_pretraining_data_google.py
├── create_pretraining_data.py
├── data
│   └── news_zh_1.txt
├── lamb_optimizer_google.py
├── LICENSE
├── modeling_google_fast.py
├── modeling_google.py
├── modeling.py
├── modelzoo_level.txt
├── optimization_finetuning.py
├── optimization_google.py
├── optimization.py
├── README.md
├── requirements.txt
├── resources
│   ├── create_pretraining_data_roberta.py
│   └── shell_scripts
│       └── create_pretrain_data_batch_webtext.sh
├── run_classifier_clue.py
├── run_classifier_clue.sh
├── run_classifier_lcqmc.sh
├── run_classifier.py
├── run_classifier_sp_google.py
├── run_pretraining_google_fast.py
├── run_pretraining_google.py
├── run_pretraining.py
├── similarity.py
├── test
│   ├── train_full_1p_finetune.sh
│   ├── train_full_1p_pretrain.sh
│   ├── train_performance_1p_finetune.sh
│   └── train_performance_1p_pretrain.sh
├── test_changes.py
├── tokenization_google.py
└── tokenization.py
```




#### 脚本参数<a name="section6669162441511"></a>

```
--data_path  训练数据集路径
--ckpt_path  预训练模型路径   
```

​                    

#### 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p*.sh）中的data_path、ckpt_path设置为训练数据集和预训练模的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/ASCEND_DEVICE_ID”，包括训练的log文件。
4. 以单卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
