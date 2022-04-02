- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

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

**描述（Description）：基于TensorFlow框架的Roberta_ZH预训练代码**

<h2 id="概述.md">概述</h2>

    在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。
    为了进一步促进中文信息处理的研究发展，基于全词遮罩（Whole Word Masking）技术的中文预训练模型BERT-wwm应运而生，
    以及与此技术密切相关的模型：BERT-wwm-ext，RoBERTa-wwm-ext，RoBERTa-wwm-ext-large, RBT3, RBTL3。
  
- 参考论文：
    
    https://arxiv.org/abs/1906.08101

- 参考实现：

    https://github.com/brightmart/roberta_zh

- 适配昇腾 AI 处理器的实现：    
    
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/Roberta_ID2349_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
      git clone {repository_url}    # 克隆仓库的代码
      cd {repository_name}    # 切换到模型的代码仓目录
      git checkout  {branch}    # 切换到对应分支
      git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
      cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

## 默认配置

- 网络结构

    - roberta-xlarge

-   训练超参（多卡）：
    - Learning rate(LR): 2e-5
    - Batch size: 16
    - max_seq_length: 256

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度


```
  session_config=tf.ConfigProto(allow_soft_placement=True)
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
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

1. 用户需自行准备训练数据集，例如wiki2019zh中文词条，可参考github源获取并转换成训练所需要的tfrecord数据。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 多卡训练 

        2.1 配置train_full_8p.sh脚本中`data_path`（脚本路径Roberta_ID2349_for_TensorFlow/test/train_full_8p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --input_file=${data_path}/wiki_zh.tfrecord \
            
        2.2 8p指令如下:

            bash train_full_8p.sh --data_path=./wiki_zh

- 验证。

    1. 执行训练时设置以下参数：
    
       ```
       --do_eval=true

        ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    
- 模型训练。

    参考“模型训练”中训练步骤。

- 模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
Roberta_ID2349_for_TensorFlow/
├── bert_config_large.json
├── create_pretrain_data.sh
├── create_pretraining_data.py
├── LICENSE
├── modeling.py
├── modelzoo_level.txt
├── optimization_finetuning.py
├── optimization.py
├── README.md
├── requirements.txt
├── resources
│   └── vocab.txt
├── run_classifier.py
├── run_pretraining.py
├── test
│   ├── 8p.json
│   ├── train_full_8p.sh
│   └── train_performance_8p.sh
└── tokenization.py

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path  训练数据集路径
                      
```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动多卡训练。
2. 将训练脚本（train_full_8p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/ASCEND_DEVICE_ID”，包括训练的log文件。
4. 以多卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

## 推理/验证过程<a name="section1465595372416"></a>

```
 NA

```
