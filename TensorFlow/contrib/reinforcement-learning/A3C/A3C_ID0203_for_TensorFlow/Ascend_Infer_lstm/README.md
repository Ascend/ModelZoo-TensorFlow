# A3C推理---带lstm

## 推理效果

| 测试游戏 | GPU推理pb模型速度(NVIDIA T4) | npu推理om模型速度(Ascend310) | 推理得分 |
| -------- | ---------------------------- | ---------------------------- | -------- |
| Alien    | 1.29 s                       | 8.14 s                       | 1000     |

## Requirement

* python3.7
* Tensorflow 1.15.0
* gym=0.10.5
* atari-py
* Ascend310 + CANN(3.3.0.alpha005)

## 代码路径解释

```bash
├─Ascend_Infer Ascend310平台上推理脚本
  	├─A3C_Inferance.py	推理脚本
  	├─acl_model.py		定义模型类，完成模型推理过程中资源管理
  	├─constants.py		常量定义
  	├─utils.py			常用操作定义
  	├─envs.py			gym环境类的重载
```

## 推理流程描述

A3C_Inferance.py 脚本对指定om模型调用Ascend推理资源进行推理

## 推理脚本：A3C_Inferance.py 

#### 参数

|   参数名    |     参数作用     |
| :---------: | :--------------: |
| input_model | 输入om模型的路径 |
|  game_name  | 待测游戏环境名称 |
|  test_num   |     测试轮数     |

#### 示例

```bash
python A3C_Inferance.py --input_model ../om_model/a3c_alien_model.om --game_name AlienDeterministic-v4 --test_num 5
```

