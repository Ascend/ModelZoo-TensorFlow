# 基本信息
## 发布者（Publisher）：Huawei
## 应用领域（Application Domain）：Online Shopping
## 修改时间（Modified） ：2021.12.13
## 框架（Framework）：TensorFlow_1.15.0
## 模型格式（Model Format）：ckpt
## 精度（Precision）：Mixed
## 处理器（Processor）：昇腾910
## 描述（Description）：基于TensorFlow框架的NGNN网络训练代码

# 概述
--NGNN对服装的兼容性进行评估，研究时尚推荐的实际问题。
* 参考论文
https://arxiv.org/abs/1902.08009
* 参考实现
https://github.com/CRIPAC-DIG/NGNN

# 默认配置
数据集等一些无法上传的大文件可通过如下网盘得到。 
链接：https://pan.baidu.com/s/1lxktOwl7lOstFrIrXjASKQ 
提取码：21i7 
* 网络结构
* 数据集
    * category_summarize_100.json -- id, name, frequency and items of category
    * cid2rcid_100.json -- category id list
    * train_no_dup_new_100.json -- 训练集
    * test_no_dup_new_100.json -- 测试集
    * polyvore_image_vectors.zip -- the vector of all items by inception-v3 [outfitid]_[itemid].json
    * polyvore_text_onehot_vectors.zip -- the vector of all items by Muti-hot
* 训练超参
    * Batch size：16
    * Train epoch：15

# 支持特性
| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 数据并行  | 否    |

# 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

# 开启混合精度
```
config_proto = tf.ConfigProto()
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = 'NpuOptimizer'
config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
# open mix_precision
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config = npu_config_proto(config_proto=config_proto)
```

# 训练环境准备
1. pycharm和PyCharm Toolkit
2. modelarts3.0
3. 镜像：ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_1116


# 快速上手
# 数据集准备
在训练脚本中指定数据集路径，即可正常使用。

# 模型训练
* 下载训练脚本
* 开始训练--单卡训练
1.设置单卡训练参数（在`main_multi_modal.py`中）

```
#max_epoch =15
for epoch in range(15):
...
<batch_size = 16
```

2.单卡训练指令
`bash train_1p_ci.sh`

# 迁移学习指导
* 数据集准备。
请参见“快速上手”中的数据集准备。
* 模型训练。
请参考“快速上手”章节。 

# 脚本和示例代码
```
├── data
│   ├── category_summarize_100.json
│   ├── cid2rcid_100.json
│   ├── test_no_dup_new_100.json
│   └── train_no_dup_new_100.json
├── scipts_gpu                               // gpu 代码文件夹
│   ├── load_data_multimodal.py
│   ├── main_multi_modal.py                  // gpu训练入口脚本
│   └── model_multimodal.py			                
├── scipts_npu                               // npu 代码文件夹
│   ├── load_data_multimodal.py
│   ├── main_multi_modal.py                   // npu训练入口脚本
│   └── model_multimodal.py
├── LICENSE
├── README.md
├── loss perf_gpu.txt
├── loss perf_npu.txt
├── modelzoo_level.txt
├── requirements.txt
├── run_1p.sh                                 // 执行训练脚本
├── train_1p_ci.sh                            // 执行训练脚本						                    
```

# 精度和性能对比
|     | Accuracy | AUC      | 运行总时长     | 单步性能                                           |
|-----|----------|----------|-----------|------------------------------------------------|
| GPU | 0.7701   | 0.9600   | 21h23m15s | 第一步需要80.79041242599487s，其余每步需要≈0.35s |
| NPU | 0.771598 | 0.970337 | 30h39m22s | 第一步和第二步需要2066.6089763641357，其余每步需要≈0.27s |
