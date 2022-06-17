## FM

### 1. 论文
Factorization Machines

**创新**：**经典因子分解机模型**  



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_15.png" width="50%;" style="float:center"/></div>

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_16.png" width="50%;" style="float:center"/></div>



### 3. 实验数据集

采用Criteo数据集进行测试。数据集的处理见`../data_process`文件，主要分为：

1. 考虑到Criteo文件过大，因此可以通过`read_part`和`sample_sum`读取部分数据进行测试；
2. 对缺失数据进行填充；
3. 对密集数据`I1-I13`进行离散化分桶（bins=100），对稀疏数据`C1-C26`进行重新编码`LabelEncoder`；
4. 整理得到`feature_columns`；
5. 切分数据集，最后返回`feature_columns, (train_X, train_y), (test_X, test_y)`；



### 4. 模型API

```python
class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
```



### 5. 实验超参数

- file：Criteo文件；
- read_part：是否读取部分数据，`True`；
- sample_num：读取部分时，样本数量，`1000000`；
- test_size：测试集比例，`0.2`；
- 
- k：隐因子，`8`；
- dnn_dropout：Dropout， `0.5`；
- hidden_unit：DNN的隐藏单元，`[256, 128, 64]`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`4096`；
- epoch：`10`；



### 6. 实验结果

1. 采用Criteo数据集中前`500w`条数据，最终测试集的结果为：`AUC: 0.778358, loss: 0.4765`；
2. 采用Criteo数据集全部内容：
   - 学习参数：264,501,784；
   - 单个Epoch运行时间【GPU：Tesla V100S-PCI】：323s；
   - 测试集结果：`AUC: 0.787504, loss: 0.4762`；

