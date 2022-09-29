# 分布式插件使用指南

## 1 分布式插件简介

工具主要用于基于昇腾AI处理器快速拉起分布式训练，简化了分布式参数配置，用户只需要输入单卡训练指令即可快速进行分布式训练，同时加入了AOE分布式梯度调优功能，用户可以在完成分布式训练后直接进行调优以提升分布式训练的性能

### 1.1 工具获取

1. 下载压缩包的方式获取 将https://gitee.com/ascend/ModelZoo-TensorFlow 以压缩包形式下载
2. 使用git命令方式获取
3. 移动 Tools/ascend_distribute 目录至常用公共路径

### 1.2 使用约束

- 本插件仅适用于TensorFlow1.X/2.X框架的训练网络

- 使用前请确保网络单卡训练指令可以正常在昇腾AI处理器上进行训练，且分布式训练代码已修改
- 执行脚本和训练代码中不要设置任何分布式环境变量，包括但不限于:ASCEND_DEVICE_ID,RANK_TABLE_FILE,RANK_ID,RANK_SIZE...
- 多机训练时，请保证每个服务器的训练代码路径、数据集路径、脚本参数一致

### 1.3 已完成功能

- 自动根据传入的分布式参数生成对应的RANK_TABLE_FILE
- 自动设置执行分布式需要设置的环境变量
- 对拉起的进程每10分钟检测一次
- 多服务器训练时只需在一个服务器下发任务



### 1.2 环境准备

#### 1.2.1 训练环境准备

硬件环境和运行环境请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》



#### 1.2.2 插件依赖安装

运行以下命令安装分布式工具所需依赖

```
pip3 install requirements.txt
```



#### 1.2.3 AOE工具安装

AOE工具的下载和安装请参见《AOE工具使用指南》，如需使用分布式插件中的AOE梯度调优功能请按照该指南中的说明进行AOE工具的下载和安装，如仅需调用知识库或不进行梯度调优则可以不安装使用AOE工具



## 2 分布式插件的使用<TODO>

### 2.1 使用分布式插件运行单机或多机分布式训练

#### 2.1.1 使用流程

**环境初始化**

为了实现多机可以跨服务器拉起，执行训练之前，请完成环境配置，相关文件保存在```~/ascend_tools/ascend_distribute``` 目录下，已配置无需重复配置，华为不会记录你的任何环境信息

```
# config传入参数格式 {ip}:{username}:{password} 多个环境之间用','隔开

python3 distrbute_npu.py --config 10.10.10.10:root:huawei,10.10.10.11:root:huawei
```



**单机多卡分布式训练**



说明：```--np  n```  设置在n卡上训练，```--env {ip}:{device数量}:{device id}`` 设置在指定IP服务器上的8张卡上进行训练，当设置8卡时可以不写:{device id}，详细见《分布式插件参数说明章节》

```
# 单机8卡
python3 $path/distrbute_npu.py --np 8 --env 10.10.10.10:8 --train_command "bash train_1p.sh --data_path=/npu/traindata"

# 单机4卡
python3 $path/distrbute_npu.py --np 4 --env 10.10.10.10:4:0123 --train_command "bash train_1p.sh --data_path=/npu/traindata"
```



**多机多卡分布式训练**

使用以下多机16卡示例命令拉起多机多卡分布式训练

说明：类似于单机多卡训练，--env 参数中不同机器用“,”分隔

```
# 两机16卡
python3 distrbute_npu.py --np 16 --env 10.10.10.10:8,10.10.10.11:8 --train_command "bash train_1p.sh --data_path=/npu/traindata"

# 两机8卡,每个服务器分别在device 0123上执行训练
python3 distrbute_npu.py --np 8 --env 10.10.10.10:4:0123,10.10.10.11:4:0123 --train_command "bash train_1p.sh --data_path=/npu/traindata"
```



### 2.2 使用AOE工具进行分布式梯度切分调优

#### 2.2.1 调优流程

**使用AOE生成自定义知识库**

当用户可以拉起单机多卡分布式训练后，可以开启AOE梯度调优，仅需在拉起单机分布式的命令后加一个--aoe=True的参数即可。执行该命令后，会默认在device0上拉起单个进程进行梯度调优，梯度调优结束后会生成一个{芯片名}_gradient_fusion.json的自定义知识库，例如Ascend910A_gradient_fusion.json

说明：对于一个网络的某一个场景，AOE只用调优一次；对于已经进行过AOE梯度调优的网络，无需再次进行AOE



**使用生成的知识库**

AOE调优完毕后，会生成一个自定义知识库文件，通过环境变量调用知识库进行分布式训练



#### 2.2.2 快速上手

**使用AOE生成自定义知识库**

使用以下示例中的命令进行AOE调优

```
python3 distribute_npu.py --np 8 train_command "bash train_1p.sh --data_path=/npu/traindata" --aoe=True
```

说明：AOE调优前需确保该命令可以进行分布式训练，在可执行分布式训练的命令后添加 --aoe=True即可



**使用/禁用调优后生成的知识库<TODO>**

调优完毕后再次拉起分布式训练即可调用自定义知识库，当进行多机训练时会自动将自定义知识库传输到其他机器上

如果用户不想调用自定义知识库时可以按照以下示例在训练命令后添加 --use_library=False禁用知识库

```
python3 distribute_npu.py --np 8 train_command "bash train_1p.sh --data_path=/npu/traindata" --use_library=False
```



## 3 常见问题处理<TODO>

残留进程处理

报错



## 4 分布式插件参数说明

| 参数名          | 默认值 | 类型 | 参数说明                                                     |
| --------------- | ------ | ---- | ------------------------------------------------------------ |
| -h 或 --help    | 无     | 无   | 打印帮助信息，使用python3 distribute_npu.py --help打开帮助信息 |
| --env           | None   | 必须 | 环境信息，按照ip:device数量:device_id的格式进行输入，多机时请用','进行分隔。示例：--env 10.10.10.10:4:0123,10.10.10.11:4:1234 |
| --np            | 8      | 必须 | 总共使用的device数量，默认为8卡。示例：--np 16               |
| --train_command | None   | 必须 | 启动单卡训练的指令。示例：--train_command "bash train_1p.sh --data_path=/home/data" |
| --aoe           | False  | 可选 | 是否使用AOE工具进行分布式梯度调优，默认为False。使用 --aoe=True启动 |
| --use_library   | <TODO> | 可选 | 是否使用AOE调优生成的知识库，默认为True，当用户              |

 

