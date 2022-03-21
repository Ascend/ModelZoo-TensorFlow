# 概述
    milking_cowmask是一种用于图像分类的半监督学习方法，能够在拥有少量有标签数据的情况下训练出分类准确率很高的网络模型。
    本项目在应用了Shake-Shake正则化的26层的Wide-Resnet的基础网络上应用cowmask,得到了很好的分类效果
    脚本精度与论文相差约2%，推测可能是改变了训练策略和avgpooling使用fp16浮点溢出导致,已反馈澄清。

- 论文链接: [Milking CowMask for Semi-Supervised Image Classification](https://arxiv.org/abs/2003.12022)

- 官方代码(JAX): [链接](https://github.com/google-research/google-research/tree/master/milking_cowmask)

- tensorflow在GPU复现OBS路径: obs://milking-cowmask/milking_cowmask_gpu/

- 精度比较

|  | 论文 | GPU复现 | Ascend |
| ------ | ------ | ------ | ------ |
| Top-1 accuracy | 95.27% | 94.31% | 93.21% |

# 环境
    - python 3.7.5
    - Tensorflow 1.15
    - Ascend 910

# 训练
## 数据集
    1.模型默认使用cifar10数据集训练
    2.配置数据集路径,默认为:./dataset/ 支持自动下载,已上传于obs://milking-cowmask/dataset/
    2.训练脚本自动调用dada_sources/small_image_data_source.py预处理数据集
##训练超参见train.py参数列表
## 单卡训练命令
```commandline
sh ./test/train_full_1p.sh
```

# 功能测试
少量step(单epoch)运行
```commandline
sh ./test/train_performance_1p.sh
```
# 模型固化
准备checkpoint,默认为 ./checkpoint/milking_cowmask.ckpt-300
```commandline
python3 freeze_graph.py
```

# 部分脚本和示例代码
```text
├── README.md                                //说明文档
├── requirements.txt			//依赖
├──test		                         //训练脚本目录								 
│    ├──train_performance_1p.sh			 
│    ├──train_full_1p.sh
├──train.py                 	     //训练脚本
├──freeze_graph.py              //固化脚本
```

# 输出
模型存储路径为test/output/ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。loss信息在文件test/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。 
模型固化输出为pb_model/milking_cowmask.pb
## 预训练结果obs路径:
   - checkpoint: obs://milking-cowmask/Milking_cowmask_ID0712_for_TensorFlow/checkpoint/
   - pb模型: obs://milking-cowmask/Milking_cowmask_ID0712_for_TensorFlow/pb_model/