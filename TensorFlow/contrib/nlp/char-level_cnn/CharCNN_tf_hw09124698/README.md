# Text Classification with CNN and RNN

使用卷积神经网络以及循环神经网络进行中文文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

本文是基于TensorFlow在中文数据集上的简化实现，使用了字符级CNN和RNN对中文文本进行分类，达到了较好的效果。

文中所使用的Conv1D与论文中有些不同，详细参考官方文档：[tf.nn.conv1d](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)

## 环境

- Python 2/3 (感谢[howie.hu](https://github.com/howie6879)调试Python2环境)
- TensorFlow 1.3以上
- numpy
- scikit-learn
- scipy

## 目录
```
│  cnn_model.py									#模型定义
│  LICENSE
│  modelzoo_level.txt
│  predict.py
│  README.md
│  requirements.txt
│  rnn_model.py
│  run_cnn.py									#训练脚本
│  run_cnn_2.py								#循环训练脚本
│  run_rnn.py
│  train_testcase.sh							#启动脚本
│
├─checkpoints									# 95.33%的ckpt
│      best_validation.data-00000-of-00001
│      best_validation.index
│      best_validation.meta
│
├─data
│      cnews_loader.py							#数据加载
│      __init__.py
│
├─helper
│      cnews_group.py							#训练数据集生成
│      __init__.py
│
├─images
│      cnn_architecture.png
│      GPU.png
│      GPU2.png
│      NPU.png
│      NPU2.png
│      tb.png
│
└───────────────────────────────────────────────────────────────

```



## 数据集

使用THUCNews进行训练与测试，[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)，请遵循数据提供方的开源协议。


- thucnews.train.txt: 训练集
- thucnews.val.txt: 验证集
- thucnews.test.txt: 测试集
- train.x: 处理后的训练集
- train.y: 处理后的训练集标签
- val.x: 处理后的验证集
- val.y: 处理后的验证集标签
- test.x: 处理后的测试集
- test.y: 处理后的测试集标签



## 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

由于数据集预处理大概需要20-30分钟，所以已附上处理好的文件！你也可以通过注释掉run_cnn.py 150-153行并去除154、155行注释自行进行数据预处理。


## CNN卷积神经网络

### 配置项

CNN可配置的参数如下所示，在`cnn_model.py`中。

```python
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64			# 词向量维度
    seq_length = 600			# 序列长度
    num_classes = 14  			# 类别数
    num_filters = 1024 			# 卷积核数目
    kernel_size = 5  			# 卷积核尺寸
    vocab_size = 5000  			# 词汇表达小

    hidden_dim = 1024 			# 全连接层神经元

    dropout_keep_prob = 0.5  	# dropout保留比例
    learning_rate = 1e-3  		# 学习率

    batch_size = 512  			# 每批训练大小
    num_epochs = 100000 		# 总迭代轮次

    print_per_batch = 100  		# 每多少轮输出一次结果
    save_per_batch = 10  		# 每多少轮存入tensorboard
```

### CNN模型

具体参看`cnn_model.py`的实现。

大致结构如下：

![images/cnn_architecture](images/cnn_architecture.png)

### 训练与验证
依瞳环境(CANN 5.0.1)下需要先手动将`aic-ascend910-ops-info.json` 覆盖到`/home/HwHiAiUser/Ascend/ascend-toolkit/5.0.1/arm64-linux/opp/op_impl/built-in/ai_core/tbe/config/ascend910` 目录下  

运行 `python run_cnn.py train`，可以开始训练。

> 若之前进行过训练，请把tensorboard/textcnn删除，避免TensorBoard多次训练结果重叠。

![images/npu](images/NPU.png)

直到精度长时间不在提升后会自动退出


### 测试

运行 `python run_cnn.py test` 在测试集上进行测试。

![images/npu](images/NPU2.png)


在测试集上的准确率达到了95.33%

### 精度对比
V100:  

![images/GPU](images/GPU2.png)  

Ascend-910:  

![images/npu](images/NPU2.png)  

GPU和NPU均在95%左右

### 性能对比


![images/npu](images/tb.png)  


训练性能：  
V100平均13秒/100step   
![images/npu](images/GPU.png)  

Ascend-910平均4.3秒/100step  
![images/npu](images/NPU.png)

推理性能 V100: 2.418秒 Ascend-910: 1.79秒  

| bs512     | 精度    | 训练速度step/s | 推理时间   |
|------|-------|------------|--------|
| 910  | 95.33 | 21.23    | 1.79s  |
| V100 | 95.07 | 7.6244     | 2.418s |

### run_cnn_2.py魔改脚本
由于随机性的问题需要训练多次才可能得到最好的精度所以稍微改了下源码，使其可以自动循环训练和测试

