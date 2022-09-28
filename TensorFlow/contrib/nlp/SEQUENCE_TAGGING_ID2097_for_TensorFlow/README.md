-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
# 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Natural Language Processing**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.01.07**

**大小（Size）：25M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Single**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Sequence_Tagging序列标注网络训练代码** 

#概述

Sequence_Tagging是一种自然语言处理网络架构，该架构通过使用双向LSTM、CNN和CRF的组合，自动受益于词级和字符级表示。 整个网络是真正的端到端，不需要特征工程或数据预处理，因此适用于广泛的序列标记任务。

- 参考论文：

    [Xuezhe Ma, Eduard Hovy. “End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.” arXiv:1603.01354](https://arxiv.org/pdf/1603.01354.pdf) 

- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
  
  []()      


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 数据集预处理（CoNLL2003语料库）：

  - 去除数据集中与命名实体识别无关的属性（第2、3列）和DOCSTART行  
    处理前：
    ```
    -DOCSTART- -X- -X- O
    
    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    call NN I-NP O
    to TO B-VP O
    boycott VB I-VP O
    British JJ B-NP B-MISC
    lamb NN I-NP O
    . . O O
    ```
    处理后：
    ```
    EU B-ORG
    rejects O
    German B-MISC
    call O
    to O
    boycott O
    British B-MISC
    lamb O
    . O
    ```
  - 数据集文件路径 
    
    训练集：./data/coNLL/eng/eng.train.iob  
    测试集：./data/coNLL/eng/eng.testb.iob  
    验证集：./data/coNLL/eng/eng.testa.iob  
- 词向量库预处理：
  
  - glove.6B下载
  - 词向量库文件路径  
    ./data/glove.6B
- 训练超参

  - Batch size: 10
  - Dropout: 0.5
  - Learning rate(LR): 0.002
  - Learning rate decay: 0.05
  - Optimizer: adam
  - Gradient clip: 10
  - Train epoch: 50
  - Max sequence length: 128
  - Max word length: 64


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 并行数据  | 否   |

#训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


#快速上手

- 数据集准备
1. 模型训练使用CoNLL2003数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请参考默认配置中的数据集预处理小结。

3. 数据集处理后，放入sequence_tagging/data目录下，可正常使用。
   
4. 训练前需初始化数据环境，在数据集和词向量库处理完毕后，执行以下脚本即可完成初始化。
    ```
    python build_data.py
    ```

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 训练

  ```
  python train.py --device_target npu
  ```

- 验证。

  ```
  python evaluate.py --device_target npu
  ```

#高级参考

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── build_data.py                             //创建单词表
├── README.md                                 //代码说明文档
├── train.py                                  //训练模型
├── evaluate.py                               //评估模型
├── modelarts_entry_acc.py                    //modelarts全量训练
├── modelarts_entry_perf.py                   //modelarts性能验证
├── requirements.txt                          //环境依赖
├── LICENSE.txt                               //证书
├── data                                      
│    ├──words.txt                            //单词
│    ├──tags.txt                             //标签
│    ├──chars.txt                            //字母
│    ├──glove.6B.100d.trimmed.npz            //词向量
│    ├──coNLL                                //CoNLL2003数据集
│    │   ├──eng
│    │   │   ├──eng.testa.iob                //验证集
│    │   │   ├──eng.testb.iob                //测试集
│    │   │   ├──eng.train.iob                //训练集
│    ├──glove.6B                             //词向量库
│    │   ├──glove.6B.50d.txt                 //50维词向量
│    │   ├──glove.6B.100d.txt                //100维词向量
│    │   ├──glove.6B.200d.txt                //200维词向量
│    │   ├──glove.6B.300d.txt                //300维词向量
├── test 
│    ├──train_performance_1p.sh              //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                     //单卡全量训练启动脚本                                   
├── model                                     
│    ├──__init__.py
│    ├──base_model.py                        //基础模型
│    ├──ner_model.py                         //网络结构
│    ├──config.py                            //参数设置
│    ├──data_utils.py                        //数据集处理
│    ├──general_utils.py                     //通用工具
├── results                                   //训练结果
│    ├──test                                 
│    │   ├──events.out.tfevents.XXX          //summary文件
│    │   ├──log.txt                          //日志文件
│    │   ├──model.weights                    //checkpoint文件
```

## 脚本参数<a name="section6669162441511"></a>

```
--dir_dataset            数据集路径，默认：./data
--dir_result             训练结果保存路径，默认：./results
--resume                 恢复训练的起始epoch，默认：0
--dir_ckpt               评估模型checkpoint路径，默认：./results/model.weights
--dim_word               单词向量纬度，默认：100
--dim_char               字符向量纬度，默认：30
--max_sequence_length    句子中最大单词数量，默认：128
--max_word_length        单词中最大字符数量，默认：64
--dir_glove              词向量库路径，默认：./data
--train_embeddings       是否训练词嵌入向量，默认：False
--epochs                 训练epoch数，默认：100
--dropout                dropout率，默认：0.5
--batchsize              batch size，默认：10
--optimizer              优化算法，默认adam；可选：sgd，momentum，adagrad，rmsprop
--lr                     初始学习率，默认：0.001
--lr_decay               学习率衰减率，默认：0.005
--grad_clip              梯度截断，默认：-1(无截断)
--early_stop             early stop，默认：10
--mix_precision          是否启用混合精度，默认：False
--loss_scale             loss scale，默认：0
--hidden_size_lstm       lstm隐含层大小，默认：200
--use_crf                是否启用crf，默认：True
--use_chars              是否使用字符向量，默认：True
--conv_kernel_size       cnn卷积核大小，默认：3
--conv_filter_num        cnn卷积核数量，默认：30
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动训练。

2.  参考脚本的模型存储路径为./results/test，训练脚本log.txt中包括如下信息。

```
2021-12-23 09:46:02,633:INFO: Initializing tf session
2021-12-23 09:46:03,162:INFO: Epoch 1 out of 100
2021-12-23 09:48:05,065:INFO: precision 81.25 - recall 81.51 - f1 81.38
2021-12-23 09:48:05,288:INFO: - new best score!
2021-12-23 09:48:05,288:INFO: Epoch 2 out of 100
2021-12-23 09:50:10,927:INFO: precision 84.37 - recall 84.56 - f1 84.47
2021-12-23 09:50:11,110:INFO: - new best score!
2021-12-23 09:50:11,110:INFO: Epoch 3 out of 100
2021-12-23 09:52:18,773:INFO: precision 86.09 - recall 85.65 - f1 85.87
2021-12-23 09:52:18,955:INFO: - new best score!
2021-12-23 09:52:18,955:INFO: Epoch 4 out of 100
2021-12-23 09:54:22,882:INFO: precision 87.02 - recall 86.00 - f1 86.51
2021-12-23 09:54:23,095:INFO: - new best score!
2021-12-23 09:54:23,096:INFO: Epoch 5 out of 100
2021-12-23 09:56:22,388:INFO: precision 87.50 - recall 87.30 - f1 87.40
2021-12-23 09:56:22,572:INFO: - new best score!
```

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印测试集的precision，recall和f1-score，如下所示。

```
Reloading the latest trained model...
Testing model over test set
precision 90.95 - recall 91.09 - f1 91.02
```
