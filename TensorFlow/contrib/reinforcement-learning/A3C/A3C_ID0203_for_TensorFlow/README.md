

# A3C模型

## 概述

迁移强化学习模型A3C到ascend910平台，将Ascend平台训练结果的精度和论文在cpu上的训练结果进行对比。

| 测试游戏  | 论文得分收敛值 | 论文训练时间 | Ascend得分收敛值 | Ascend训练时间 |
| --------- | -------------- | ------------ | ---------------- | -------------- |
| Pong      | 11.4           | 1 day        | 17.4             | 1.5 hours      |
| Tutankham | 144            | 4 days       | 200              | 6.5 hours      |
| Alien     | 945            | 4 days       | 950              | 6 hours        |

推理性能对比：

| 测试游戏 | GPU推理pb模型速度(NVIDIA T4) | npu推理om模型速度(Ascend310) |
| -------- | ---------------------------- | ---------------------------- |
| Pong     | 2.63 s                       | 2.21 s                       |
| Alien(lstm)    | 1.29 s                       | 8.14 s                       |



## Requirement

* Tensorflow 1.15.0
* Ascend910
* gym[atari]
* gym=0.10.5

## 代码路径解释

```shell
A3C
└─ 
  ├─README.md
  ├─pb_model 转换得到的pb模型
  	├─a3c_pong_model.pb
  	├─a3c_Alien_model.pb
  ├─om_model 转换得到的om模型
  	├─a3c_pong_model.om
  	├─a3c_alien_model.om
  ├─Ascend_Infer_no_lstm Ascend310平台上推理脚本（模型中无lstm模块）
  	├─A3C_Inferance.py	推理脚本
  	├─acl_model.py		定义模型类，完成模型推理过程中资源管理
  	├─constants.py		常量定义
  	├─utils.py			常用操作定义
  	├─envs.py			gym环境类的重载
  
  ├─Params.py 	全局变量定义
  ├─AcNet.py 	核心网络结构
  ├─envs.py		gym环境类的重载
  ├─model.py 	模型结构
  ├─workerm.py 	每一个线程下工作单元worker结构
  ├─train_cpu.py 启动训练脚本(cpu)
  ├─train_npu.py 启动训练脚本(npu)
  ├─model_Converter.py ckpt模型转换pb模型脚本
  ├─om_converter.sh	atc转换命令
  ├─train_testcase.sh 训练命令
```

## 流程描述

### 1：使用A3C目录下的train脚本完成模型训练

### 2：使用A3C目录下model_Converter脚本完成ckpt模型转换pb模型

### 3：使用CANN(3.3.0.alpha005)中的atc工具，pb模型转换为om模型

### 4：使用A3C/Ascend_Infer/A3C_Inferance.py脚本完成om模型部署推理



## 主要脚本解释

### 1: A3C/train.py

#### 参数解释

|       参数名       | 参数作用     |
| :----------------: | ------------ |
|     output_url     | 输出模型目录 |
|      log_url       | 输出日志目录 |
|      env_name      | 测试游戏名称 |
|    threads_num     | 线程数       |
| UPDATE_GLOBAL_ITER | 更新步长     |
|   MAX_GLOBAL_EP    | 总训练步长   |
|         lr         | 学习率       |
|       GAMMA        | 贡献递减率   |
|    ENTROPY_BETA    | 熵损失参数   |
|        LSTM        | 是否使用LSTM |
|     model_name     | 保存模型名称 |

#### 示例

```bash
python train_npu.py --env_name PongDeterministic-v4 --threads_num 16 --model_name a3c_Pong_model

python train_npu.py --env_name TutankhamDeterministic-v4 --threads_num 8 --MAX_GLOBAL_EP 5000 --lr 0.0005 --LSTM --model_name a3c_Tutankham_model_lstm

python train_cpu.py --env_name AlienDeterministic-v4 --threads_num 8 --MAX_GLOBAL_EP 20000 --model_name a3c_Alien_model_lstm --lr 0.0005 --LSTM --UPDATE_GLOBAL_ITER 512
```

### 2：A3C/model_Converter.py

#### 参数解释

|   参数名    |           参数作用            |
| :---------: | :---------------------------: |
| output_path |        pb模型输出目录         |
| input_path  |       ckpt模型输入目录        |
| model_name  |         ckpt模型名称          |
|    type     | 卷积网络(conv)/LSTM网络(lstm) |
|  env_name   |         游戏环境名称          |

#### 示例

``` bash
python model_Converter.py --output_path ./pb_model --input_path ./mdoel --model_name model_Pong_Conv --type conv --env_name PongDeterministic-v4

python model_Converter.py --output_path ./pb_model --input_path ./mdoel --model_name model_Alien_Conv --type lstm --env_name AlienDeterministic-v4
```

### 3: atc转换模型

无lstm模块：

```bash
./om_converter.sh（无lstm）
```

有lstm模块：

参考命令：

```shell
(1)获取V1网络模型的输出
summarize_graph --in_graph=model_Alien_new.pb

(2)构造config.pbtxt（例子如下）
fetch{
  id{ node_name:"W_0/actor/rnn/while/Exit_3" }
}
fetch{
  id{ node_name:"W_0/actor/rnn/while/Exit_4" }
}
fetch{
  id{ node_name:"out_a"}
}

(3)生成V2的配置文件
xlacompile --graph=model_Alien_new.pb --config=config.pbtxt --output=./out/Switch_v1_v2

(4)V2网络模型生成graph子图
python3.7.5 ${install_path}/atc/python/func2graph/func2graph.py -m ./Switch_v1_v2.pb

(5)V2网络模型转换成适配昇腾AI处理器的离线模型
atc --model=./Switch_v1_v2.pb --framework=3 --output=./Switch_v1_v2_to_om --soc_version=Ascend310
```



### 4：Ascend310推理

具体推理流程见[无lstm推理部分](./Ascend_Infer_no_lstm/README.md) 或 [lstm推理部分](./Ascend_Infer_lstm/README.md)

模型文件可以从仓库获取：https://gitee.com/HengYueTM/a3-c_-final