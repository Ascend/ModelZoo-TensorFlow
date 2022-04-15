- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection

**版本（Version）：1.1

**修改时间（Modified） ：2022.04.08

**大小（Size）：140KB

**框架（Framework）：TensorFlow 1.15.0

**模型格式（Model Format）：ckpt

**精度（Precision）：Mixed

**处理器（Processor）：昇腾910

**应用级别（Categories）：Official

**描述（Description）：Wide&Deep是一个同时具有Memorization和Generalization功能的CTR预估模型，该模型主要由广义线性模型（Wide网络）和深度神经网络（Deep网络）组成，对于推荐系统来说，Wide线性模型可以通过交叉特征转换来记忆稀疏特征之间的交互，Deep神经网络可以通过低维嵌入来泛化未出现的特征交互。与单一的线性模型（Wide-only）和深度模型（Deep-only）相比，Wide&Deep可以显著提高CTR预估的效果，从而提高APP的下载量。

- 参考论文：
   https://arxiv.org/abs/1606.07792

- 参考实现：
   https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Recommendation/WideAndDeep

- 适配昇腾 AI 处理器的实现：

    https://gitee.com/chen-yucheng113/research_TF/tree/master/built-in/TensorFlow/Research/debugging_model/WideDeep_ID2712_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```
### 默认配置<a name="section91661242121611"></a>

1、 训练超参（单卡）

  batch_size：131072
  pos_weight: 1.0
  train_per_epoch: 59761827
  test_per_epoch: 1048576


### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |

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

### 数据集准备<a name="section361114841316"></a>

- 请用户参考"参考实现"从源码里下载outbrain数据集

```
### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 
 以数据集放在/data为例
    ```
	cd test
	bash train_performance_1p.sh --data_path=/data  (功能和性能)
	bash train_full_1p.sh --data_path=/data        （全量）
    ```
<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

├── README.md                            //代码说明文档 
├── requirements.txt                     //安装依赖 
├── make_docker.sh                    
├── configs                              
│    ├──config.py                        //参数配置 
├── widedeep                                //模型结构 
│    ├──WideDeep_fp16_huifeng.py
│    ├──data_utils.py
│    ├──features.py     
│    ├──tf_util.py                      
├── test
│    ├──train_full_1p.sh                 //单卡运行启动脚本 
│    ├──train_full_8p.sh                 //8卡执行脚本 
│    ├──train_performance_1p.sh          //单卡性能运行启动脚本 
│    ├──train_performance_8p.sh          //8卡性能执行脚本 
│    ├──8p.json                         //8卡IP的json配置文件 

### 脚本参数

在`configs/config.py`中进行设置。

--record_path                     train data dir, default : path/to/data
--num_inputs                      number of features of dataset. default : 39
--batch_size                      mini-batch size ,default: 128 
--n_epoches                       initial learning rate,default: 0.06

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。


