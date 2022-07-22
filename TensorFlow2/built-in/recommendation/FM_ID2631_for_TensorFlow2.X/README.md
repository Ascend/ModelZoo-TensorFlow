- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Recommendation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.21**

**大小（Size）：16M**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow2.X框架的推荐算法CTR预估模型的训练代码**


<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

    因子分解机(Factorization Machine, FM)是由Steffen Rendle提出的一种基于矩阵分解的机器学习算法。目前，被广泛的应用于广告预估模型中，相比LR而言，效果强了不少。是一种不错的CTR预估模型，也是我们现在在使用的广告点击率预估模型，比起著名的Logistic Regression, FM能够把握一些组合的高阶特征，因此拥有更强的表现力。


  - 参考论文：
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074)

  - 参考实现：
    https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/FM(https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/FM)


  - 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/recommendation/FM_ID2631_for_TensorFlow2.X

  - 通过Git获取对应commit\_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```




## 默认配置<a name="section91661242121611"></a>


-   网络结构
    -  SVM模型与factorization模型的结合，可以在非常稀疏的数据中进行合理的参数轨迹。
    -  考虑到多维特征之间的交叉关系，其中参数的训练使用的是矩阵分解的方法。
    -  在FM中，每个评分记录被放在一个矩阵的一行中，从列数看特征矩阵x的前面u列即为User矩阵，每个User对应一列，接下来的i列即为item特征矩阵，之后数列是多余的非显式的特征关系。后面一列表示时间关系，最后i列则表示同一个user在上一条记录中的结果，用于表示用户的历史行为。

-   训练超参（单卡）：
    -   file：Criteo文件；
    -   read_part：是否读取部分数据，True(full脚本为False)；
    -   sample_num：读取部分时，样本数量，1000000；
    -   test_size：测试集比例，0.2；
    -   k：隐因子，8；
    -   dnn_dropout：Dropout， 0.5；
    -   hidden_unit：DNN的隐藏单元，[256, 128, 64]；
    -   learning_rate：学习率，0.001；
    -   batch_size：4096；
    -   epoch：10；


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 数据并行  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
  config_proto = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config_proto)
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

    采用Criteo数据集进行测试。数据集的处理见../data_process文件，主要分为：
1. 考虑到Criteo文件过大，因此可以通过read_part和sample_sum读取部分数据进行测试；
2. 对缺失数据进行填充；
3. 对密集数据I1-I13进行离散化分桶（bins=100），对稀疏数据C1-C26进行重新编码LabelEncoder；
4. 整理得到feature_columns；
5. 切分数据集，最后返回feature_columns, (train_X, train_y), (test_X, test_y)；


## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于FM_ID2631_for_TensorFlow2.X/test/train_full_1p.sh），示例如下。
            
        
        ```
        batch_size=4096
        #训练step
        train_epochs=10
        #训练epoch
        ```

        2.2 单卡训练指令（FM_ID2631_for_TensorFlow2.X/test） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=xx
        数据集应为txt类型，配置data_path时需指定为Criteo这一层，例：--data_path=/home/data/Criteo
        ├─data
          ├──Criteo
                │	├──demo.txt
                │	├──.DS_Store
                │	├──train.txt
   
        ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备
    
- 模型训练

    请参考“快速上手”章节

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>


    |--modelzoo_level.txt                           #状态文件
    |--LICENSE
    |--README.md									#说明文档
    |--criteo.py
    |--model.py                                     #模型结构代码
    |--modules.py
    |--train.py    									#训练代码
    |--requirements.txt		   						#所需依赖
    |--run_1p.sh
    |--utils.py
    |--test			           						#训练脚本目录
    |	|--train_full_1p.sh							#全量训练脚本
    |	|--train_performance_1p.sh					#performance训练脚本

 
## 脚本参数<a name="section6669162441511"></a>

```
batch_size                                       训练batch_size
epochs                                           训练epoch数
precision_mode                                   default="allow_mix_precision", type=str,help='the path to save over dump data'
over_dump                                        type=ast.literal_eval,help='if or not over detection, default is False'
data_dump_flag                                   type=ast.literal_eval,help='data dump flag, default is False'
data_dump_step                                   data dump step, default is 10
profiling                                        type=ast.literal_eval help='if or not profiling for performance debug, default is False'
profiling_dump_path                              type=str, help='the path to save profiling data'
over_dump_path                                   type=str, help='the path to save over dump data'
data_dump_path                                   type=str, help='the path to save dump data'
use_mixlist                                      type=ast.literal_eval,help='use_mixlist flag, default is False'
fusion_off_flag                                  type=ast.literal_eval,help='fusion_off flag, default is False'
mixlist_file                                     type=str,help='mixlist file name, default is ops_info.json'
fusion_off_file                                  type=str,help='fusion_off file name, default is fusion_switch.cfg'
auto_tune                                        help='auto_tune flag, default is False'
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
