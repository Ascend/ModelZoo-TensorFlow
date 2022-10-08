- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
<h2 id="基本信息.md">基本信息</h2>
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.09.02**

**大小（Size）：2M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的smith代码**

<h2 id="概述.md">概述</h2>
  许多自然语言处理和信息检索问题可以形式化为语义匹配任务。以往的工作主要集中在短文本之间的匹配或短文本和长文本之间的匹配。长篇文档之间的语义匹配在新闻推荐、相关文章推荐和文档聚类等许多重要应用中的应用相对较少，需要更多的研究工作。这项工作通过提出用于长格式文档匹配的Siamese Multi-depth Transformer-based Hierarchical (SMITH) 编码器来解决这个问题。

- 参考论文：
    https://dl.acm.org/doi/abs/10.1145/3340531.3411908

- 参考实现：
    https://github.com/google-research/google-research/tree/master/smith

- 适配昇腾 AI 处理器的实现：   
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/nlp/smith_ID2025_for_TensorFlow
    

## 默认配置

-   训练超参（单卡）：
    -   Learning rate(LR): 5e-05
    -   Batch size: 32
    -   num_train_steps: 10000
    -   num_warmup_steps: 1000

## 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度
```
session_config = tf.ConfigProto(allow_soft_placement=True)
run_config = NPURunConfig(
  session_config=session_config,
  model_dir=FLAGS.output_dir,
  save_checkpoints_steps=exp_config.train_eval_config.save_checkpoints_steps,
  iterations_per_loop=exp_config.train_eval_config.iterations_per_loop,
  precision_mode='allow_mix_precision', 
  hcom_parallel=True
)
```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
    ```
    pip3 install requirements.txt
    ```
说明：1.依赖配置文件requirements.txt文件位于模型的根目录
     2.根据原始的[git地址](https://github.com/google-research/google-research/tree/master/smith)下载nltk的数据
    ```
        import nltk
        nltk.download('punkt')
    ```
     3.根据原始git地址进行各模块的流程测试
     ```
        python -m smith.loss_fns_test
        python -m smith.metric_fns_test
        python -m smith.modeling_test
        python -m smith.preprocessing_smith_test
     ```
备；硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

<h2 id="快速上手.md">快速上手</h2>
## 数据集准备<a name="section361114841316"></a>
1. 模型使用数据集gwikimatch[介绍](https://github.com/google-research/google-research/tree/master/gwikimatch)，数据路径[下载](http://storage.googleapis.com/gresearch/smith_gwikimatch/README.md)。请用户自行准备好数据集，数据集实际可用的为样例tfrecord（本次迁移所使用）
2. 模型使用分为2种方式，即：SMITH-Short方式下使用bert官方预训练模型uncased_L-12_H-768_A-12以及SMITH-WP+SP方式下使用作者预训练smith_pretrain_model_ckpts。下方"训练部分"提供路径下载。
3. 使用protobuf工具将原作者提供的wiki_doc_pair.proto及experiment_config.proto转成wiki_doc_pair_pb2.py和experiment_config_pb2.py(已完成，可直接使用。具体过程见原GitHub的README.md)
4. 数据集转换说明，正式训练前需使用smith/preprocessing_smith.py将原始的训练集(small_demo_data.external_wdp.filtered_contro_wiki_cc_team.tfrecord)做预处理。然后作为模型训练的输入。
   执行脚本 preprocessing_smith.sh（需配置`DATA_PATH`执行路径参数）
   ```
    source ~/env.sh
    DATA_PATH="../data"
    DATA_PATH_OUT="../data/output_file"
    if [ ! -d "${DATA_PATH_OUT}" ]; then
      mkdir ${DATA_PATH_OUT}
    fi
    python3 preprocessing_smith.py --input_file=${DATA_PATH}/input_file/small_demo_data.external_wdp.filtered_contro_wiki_cc_team.tfrecord --output_file=${DATA_PATH}/output_file/smith_train_sample_input.tfrecord --vocab_file=${DATA_PATH}/uncased_L-12_H-768_A-12/vocab.txt
   ```

## 模型训练<a name="section715881518135"></a>

- 训练部分
    1. 启动训练之前，首先要配置程序运行相关环境变量。
       环境变量配置信息如下：
       ```
       source ~/env.sh
       export JOB_ID=10001
       export ASCEND_DEVICE_ID= 1 # 根据实际npu-smi info查看到的npu使用情况定
       ```
       
    2. 配置smith/config/dual_encoder_config.smith_wsp.32.48.pbtxt中的网络超参数，主要指定预训练模型和数据输入的路径(上述3的输出)。
       ``` 
       init_checkpoint # 预训练模型的ckpt路径
       bert_config_file # 配置config/sent_bert_4l_config.json
       doc_bert_config_file # 配置config/doc_bert_3l_256h_config.json
       vocab_file # 配置bert官方uncased_L-12_H-768_A-12/vocab.txt
       input_file_for_train # 配置数据预处理的output_file
       input_file_for_eval # 配置数据预处理的output_file （该代码由于原始数据无法获取，训练和验证使用同一套作者所提供样例tfrecord数据集）
       ```
       下载依赖模型文件：
       bert预训练模型[下载](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
       作者提供wsp训练依赖预训练模型[smith_wsp_pretrain_ckpt_opensource](http://storage.googleapis.com/gresearch/smith_gwikimatch/smith_wsp_pretrain_ckpt_opensource.zip)
       
    3.  单卡训练
    单卡训练指令如下（smith/bash/train_eval_predict_export/train_wsp.sh） 
    注意配置eval_wsp.sh中的参数，'DATA_PATH', 'CODE_PATH' 设置为自己的路径（下同）；
        ```
        cd smith/bash
        bash train_wsp.sh
        ```
    训练完成后ckpt文件[路径](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OVWrEUKDR1ADCmJjkBGQ3htGaizfvxlSf9IlnYwV2/viY4ioin+PR2KRtwadLKxE1gt4UUhppUs5f/woysQbGg9EGEcwk+17FznTTlVaDwMq/+IOPf44FDjQSSDcfB80gaA9iw1wn0I54iM5Ay3J2PDUDpr9bDe3faMdnnFv05ROdjODdyWRuoJPBK4YRNwwGAEJ7qhjWbF0IbVkmzXKPMkfdE9/KdHhSrOkGHtKcBFr7Ng1T/37noZ6kNMj8dGYofVLwMcdR51fm6hmlCbnC9jA0y9xyKD6TJPno0O+WFMpYos4IbhHPec1EWau0MY0+iMU2HTQTURNnMIfp28oR+TH2uM3RTV7kXNZMFijcKtX7Nxn6yVMOx4Fo0ycWImWRbBQUIKNLGCeD2XcMB++5tYV6y8LdzBOyQEGC/i1iYuQ/K+r1/IYDkdy59FZcZ/C/LV8tYe1u+I5F4eWe2tuguhK1qGRmBusr/StF7hwnl0xSbOY5hv3mkUczWH8bRA9zSlQ5C4ZzUExK2lok0qldw==)
    密码：111111
  
-  验证和推理部分
    1. 执行eval_wsp.sh，验证最后的打屏显示出最终的accuracy/precision/recall指标
        ```
        cd smith/bash/train_eval_predict_export
        bash eval_wsp.sh
        ```
        精度对比说明：通过比对GPU复现输出精度与NPU对比一致
        accuracy = 1.0   precision = 1.0  recall = 1.0  

    2. 执行predict_wsp.sh，验证输出的prediction_results.json
        ```
        cd smith/bash/train_eval_predict_export
        bash predict_wsp.sh
        ```
        备：在预测的结果prediction_results.json 在--output_dir的路径下
       
    3. 执行export_wsp.sh
        ```
        cd smith/bash/train_eval_predict_export
        bash export_wsp.sh 
        ```
    
<h2 id="高级参考.md">高级参考</h2>
## 脚本和示例代码<a name="section08421615141513"></a>

```

└─smith 
    ├─README.md 
    ├─config
        ├─bert_config.json
        └─doc_bert_3l_256h_config.json...
        └─doc_bert_3l_768h_config.json
        └─dual_encoder_config.smith_short.32.8.pbtxt
        └─dual_encoder_config.smith_wsp.32.48.pbtxt
        ...
    ├─bert 
        |─modeling.py
        └─optimization.py
    ├─input_fns.py
    ├─layers.py
    ...
    ├─run_smith.py
    └─modeling.py
```

## 脚本参数<a name="section6669162441511"></a>

```
#Training 
--dual_encoder_config_file   config for train and eval.
--output_dir                 the output for ckpt. 
--train_mode                 finetune.  
--num_train_steps            the step of train. 
--num_warmup_steps           the Patience factor of lr. 
--schedule                    train/eval/predict/export. 
```