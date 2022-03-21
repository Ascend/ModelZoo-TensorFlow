# HAN：Heterogeneous Graph Attention Network

模型出处：

By Xiao Wang, Houye Ji, Chuan Shi*, Bai Wang. Heterogeneous Graph Attention Network



## 提交文件夹说明：

```shell
├── acc_compare.py                                       # 精度对比文件
├── ckpt_prediction.py									# ckpt验证文件
├── data																
├── ex_acm3025.py										 #训练文件
├── jhyexp.py													 #引用的一些函数
├── load_data													#数据集
├── MA_LOG														#Pycharm Toolkit在昇腾上的训练日志
├── models														  #训练网络依赖的GAT网络模型
├── om																  #om模型和其生成的bin文件数据
├── om_input_code.py								#om输入数据预处理
├── Pb_converter.py										#将ckpt文件转化为pb文件
├── pb_model												 #生成的pb模型文件
├── Pb_prediction.py								   # pb预测文件 
├── predication_bin										#ckpt文件预测的bin文件数据
├── prediction_data										#om_input_code.py生成的4个bin文件
├── pre_trained												#预训练模型
├── Readme.md
├── __tb
├── trained														#在npu上训练的ckpt文件
├── Untitled.ipynb
└── utils															#相关的助手函数
```



## 代码使用说明

通过终端执行训练脚本，其中 ex_acm3025.py 相关路径需要修改（文件中已标出） 

```python
python ex_acm3025.py
```

训练完成后训练脚本集成了在test集上计算准确度。

或者可以生成ckpt文件后，执行ckpt验证脚本，注意修改path

```python
python ckpt_prediction.py
```



在使用Pycharm Toolkit在ModelArts平台上训练

![](img/pycharm-toolkit.png)



## 相关环境

* Ascend Tensorflow 1.15



## 论文来源

* [arxiv] https://arxiv.org/abs/1903.07293



## 数据集

* https://github.com/Jhy1993/HAN



## 论文引用

> *Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and Philip S Yu. 2019. Heterogeneous Graph Attention Network. In* *The World Wide Web Conference* *(**WWW '19**).*



## 模型精度

目标数据集：`ACM`

我们在npu上通过训练最终在test集上进行验证，结果准确度在86.54%

![](img/test_acc.png)

我们在本机cpu上在test集上的验证结果为87.7%

![](img/cpu_acc.png)

而在论文中提到的准确度为89%

![](img/paper_acc.png)


## 数据集和checkpoint文件
* 链接: https://pan.baidu.com/s/12qjfY_byID01DcSevxxJdw  密码: 6es7


