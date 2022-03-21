# kernelNet MovieLens-1M

State of the art model for MovieLens-1M.

This is a minimal implementation of a kernelNet sparsified autoencoder for MovieLens-1M. 
See http://proceedings.mlr.press/v80/muller18a.html

原始模型参考[github链接](https://github.com/lorenzMuller/kernelNet_MovieLens)，迁移训练代码到NPU Ascend 910

## 结果展示

|                   | 精度（Val RMSE） | 性能（s/epoch） |
| :---------------: | :--------------: | --------------- |
|       基线        |    82.30000%     | 0.3245          |
| NPU（Ascend 910） |    82.09000%     | 0.3245          |

## 快速启动
在本级目录下，本地模式可以在终端直接运行如下代码即可：

```bash train_testcase.sh```

也可以执行归一化脚本：

```bash train_full_1p.sh --data_path=./ml-1m```  或

```bash train_performance_1p.sh --data_path=./ml-1m```

```
|-- ml-1m                 ----数据集目录
    |--movies.dat     
    |--ratings.dat 
    |--README
    |--users.dat 
|-- dataLoader.py         ----数据集加载文件
|-- kernelNet_ml1m.py     ----训练启动文件（NPU Ascend 910训练）
|-- LICENSE-2.0.txt       ----LICENSE
|-- README.md             ----使用前必读
|-- requirements.txt      ----第三方组件清单
|-- train_testcase.sh     ----NPU训练入口shell
|-- test                  ----NPU训练归一化shell
    |--env.sh
    |--launch.sh
    |--train_full_1p.sh
    |--train_performance_1p.sh
```

## Setup

Download this repository

## Requirements
* TensorFlow 1.15.0.
* Ascend 910

## Dataset（可选，如使用本项目，建议直接跳过）

注意：本项目代码中已经包含数据集，无需再次下载，如果是第一次执行，请参照下方介绍执行。

Expects MovieLens-1M dataset in a subdirectory named ml-1m.
Get it here https://grouplens.org/datasets/movielens/1m/

or on linux run in the project directory

```wget --output-document=ml-1m.zip http://www.grouplens.org/system/files/ml-1m.zip ```

```unzip ml-1m.zip```

## Run
```bash train_testcase.sh``` 或

```bash train_full_1p.sh --data_path=./ml-1m```  或

```bash train_performance_1p.sh --data_path=./ml-1m```

