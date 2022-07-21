<h2 id="基本信息.md">基本信息</h2>

发布者（Publisher）：Huawei

版本（Version）：1.1

修改时间（Modified） ：2022.7.19

大小（Size）：74M

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：ckpt

精度（Precision）：Mixed

处理器（Processor）：昇腾910

应用级别（Categories）：Official



<h2 id="概述.md">概述</h2>

Here we use byte as the unit of our model for quantum sample learning. In our case, the inputs to the language model are samples of bitstrings
and the model is trained to predict the next bit given the bits observed so far,
starting with a start of sequence token. We use a standard LSTM language model
with a logistic output layer. To sample from the model, we input the start of
sequence token, sample from the output distribution then input the result as
the next timestep. This is repeated until the required number of samples is
obtained. 

- 参考论文：[[2010.11983\] Learnability and Complexity of Quantum Samples (arxiv.org)](https://arxiv.org/abs/2010.11983)

- 参考实现：https://github.com/google-research/google-research/tree/master/quantum_sample_learning

- 适配昇腾 AI 处理器的实现：
  
  [TensorFlow/contrib/nlp/quantum_sample_learning_ID2036_for_Tensorflow · Ascend/ModelZoo-TensorFlow - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/nlp/quantum_sample_learning_ID2036_for_Tensorflow)


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```



## 默认配置

- 训练超参

  - epoch：20
  - batch_size：64
  - learning_rate：0.001
  - num_qubits：12
  - rnn_units：256

  


<h2 id="训练环境准备.md">训练环境准备</h2>

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. requirements

   ```
   python==3.6  
   absl-py
   cirq==0.8.0
   numpy==1.16.4
   scipy==1.2.1
   tensorflow==1.15
   
   Ascend: 1*Ascend 910 
   CPU: 24vCPUs 96GiB
   ```

   

## 快速上手

- 数据集准备
  - 模型训练使用q12c0数据集



## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

- 单卡训练 

  1. 配置训练参数。

     在`run_lm.py`中，配置checkpoint保存路径，请用户根据实际路径配置，参数如下所示：

     ```
     flags.DEFINE_string('checkpoint_dir', './checkpoint',
                         'Where to save checkpoints')
     ```

  2. 启动训练。

     ```
     python run_lm.py 
     ```

  3. 在`evaluate.py`中配置进行验证的checkpoint路径

    ```
    flags.DEFINE_string('checkpoint_dir', './checkpoint',
                        'Where to save checkpoints')
    ```

  4. 进行验证

    ```
    python evaluate.py
    ```



## 训练结果

论文

```
Linear Fidelity: 0.982864
Logistic Fidelity: 0.979632
theoretical_linear_xeb: 1.018109
theoretical_logistic_xeb: 1.006301
linear_xeb: 0.982864
logistic_xeb: 0.979632
kl_div: 0.021964
```

GPU

```
Linear Fidelity: 1.006005
Logistic Fidelity: 1.001702
theoretical_linear_xeb: 1.015967
theoretical_logistic_xeb: 1.005840
linear_xeb: 1.006005
logistic_xeb: 1.001702
kl_div: 0.004158
```

NPU

```
Linear Fidelity: 1.058425
Logistic Fidelity: 1.029696
theoretical_linear_xeb: 1.020069
theoretical_logistic_xeb: 1.006884
linear_xeb: 1.058425
logistic_xeb: 1.029696
kl_div: 0.004556
```

+ 精度对比

  |              | 论文     | GPU      | NPU      |
  | ------------ | -------- | -------- | -------- |
  | linear_xeb   | 0.982864 | 1.006005 | 1.058425 |
  | logistic_xeb | 0.979632 | 1.001702 | 1.029696 |

  

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
Quantum Sample Learning
└─
  ├─README.md
  ├─LICENSE  
  ├─data        存放数据集文件夹
  ├─training_checkpoints_lm    存放checkpoint文件夹
  ├─data_loader.py     训练数据工具类 
  ├─data_loader_test.py     
  ├─circuit.py     生成数据集
  ├─run_lm.py     模型训练程序入口
  ├─evaluate.py     模型评估程序入口
```



## 脚本参数

```
flags.DEFINE_string('data_url', './dataset',
                    'Where to save datasets')
flags.DEFINE_string('train_url', './output',
                    'Where to save Output')
flags.DEFINE_string('checkpoint_dir', './checkpoint',
                    'Where to save checkpoints')
flags.DEFINE_string('save_data', '', 'Where to generate data (optional).')
flags.DEFINE_string('eval_sample_file', '',
                    'A file of samples to evaluate (optional).')
flags.DEFINE_boolean(
    'eval_has_separator', False,
    'Set if the numbers in the samples are separated by spaces.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('eval_samples', 500000,
                     'Number of samples for evaluation.')
flags.DEFINE_integer('training_eval_samples', 4000,
                     'Number of samples for evaluation during training.')
flags.DEFINE_integer('num_qubits', 12, 'Number of qubits to be learnt')
flags.DEFINE_integer('rnn_units', 256, 'Number of RNN hidden units.')
flags.DEFINE_integer(
    'num_moments', -2,
    'If > 12, then use training data generated with this number of moments.')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_boolean('use_adamax', False,
                     'Use the Adamax optimizer.')
flags.DEFINE_boolean('eval_during_training', False,
                     'Perform eval while training.')
flags.DEFINE_float('kl_smoothing', 1, 'The KL smoothing factor.')
flags.DEFINE_boolean(
    'save_test_counts', False, 'Whether to save test counts distribution.')
flags.DEFINE_string(
    'probabilities_path', './data/q12c0.txt',
    'The path of the theoretical distribution')
flags.DEFINE_string(
    'experimental_bitstrings_path',
    'quantum_sample_learning/data/experimental_samples_q12c0d14.txt',
    'The path of the experiment measurements')
flags.DEFINE_integer('train_size', 500000, 'Training set size to generate')
flags.DEFINE_boolean('use_theoretical_distribution', True,
                     'Use the theoretical bitstring distribution.')
flags.DEFINE_integer(
    'subset_parity_size', 0,
    'size of the subset for reordering the bit strings according to the '
    'parity defined by the bit string of length specified here')
flags.DEFINE_boolean('random_subset', False,
                     'Randomly choose which subset of bits to '
                     'evaluate the subset parity on.')
flags.DEFINE_boolean('porter_thomas', False,
                     'Sample from Poter-Thomas distribution')
```



## 下载链接

### 数据集下载

链接：https://pan.baidu.com/s/1WAl4C_EnQi4wp6l684yI9w  提取码：unp5

### checkpoint文件

链接：https://pan.baidu.com/s/1wckJSk7sNv0HvFuzJKWSdA  提取码：s4yz
