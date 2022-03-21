## PIN

模型出处：By Yanru Qu, Bohui Fang, Weinan Zhang, Ruiming Tang, Minzhe Niu, Huifeng Guo, Yong Yu, Xiuqiang He.



### 提交文件夹说明：

```
├──datasets									# 数据集处理脚本
├──image									# README中使用的图片
├──iPinYou-all								# 数据集
├──__init__.py								# 包初始化
├──eval.py									# 验证启动脚本
├──model.py									# pin模型定义
├──print_hook.py							# print_hook类定义
└──train.py									# 训练启动脚本
```



### 代码使用说明

通过终端执行训练脚本

```shell
python train.py
```

通过终端执行验证脚本

```shell
python eval.py
```



### 相关环境

- Ascend TensorFlow 1.15



### 数据集

[iPinYou](https://gitee.com/Apochen/i-pin-you)

说明：拉取数据集后将hdf整个文件夹放在`pin_on_-ascend/pin/iPinYou-all/`目录下（在拷贝之前iPinYou-all为空文件夹。同时将该仓库中的ckpts目录下checkpoints文件夹复制到`../log`目录下。

由于ckpt-data文件太大，上传至[网盘](https://pan.baidu.com/s/1eQxpRps-LxrirT4BuJ33XQ)，提取码`svcm`。



### 论文引用

>  Product-Based Neural Networks for User Response Prediction over Multi-Field Categorical Data (2018)



### 模型精度

目标数据集：iPinYou

| Method | AUC   | Log Loss |
| ------ | ----- | -------- |
| paper  | 78.22 | 0.005547 |
| CPU    | 78.00 | 0.005613 |
| NPU    | 77.30 | 0.005844 |

论文中的精度：

![performance](./images/performance.png)

本地cpu运行精度：

![local_cpu_performance](./images/local_cpu_performance.jpg)
