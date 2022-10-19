中文|[English](README_EN.md)
# <font face="微软雅黑">
# KGAT TensorFlow离线推理

***
此链接提供KGAT TensorFlow模型在NPU上离线推理的脚本和方法

* [x] KGAT 推理, 基于 [knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)

***

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/recommendation/KGAT_for_ACL
```

### 2. 下载数据集和预处理

请自行下载测试数据集，详情见: [amazon-book](./Data/README.md)

### 3. 获取pb模型

获取pb模型, 详情见: [pb](./Model/pb_model/README.md)

### 4. 编译程序
编译推理应用程序, 详情见: [xacl_fmk](./xacl_fmk/README.md)

### 5. 离线推理

**KGAT**
***
* KGAT 在 KGAT_for_ACL 中 使用静态batch, 设置 predict_batch_size=2048 作为输入参数, 所以我们舍弃了最后一批测试数据(batch size=959)
* 在./Model目录中执行以下命令
***
**环境变量设置**

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
```Bash
python3 offline_inference/data_preprocess.py
```

在Model/input_bin目录中生成bin文件


**Pb模型转换为om模型**

[pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/KGAT_for_ACL.zip)

```Bash
atc --model=KGAT_tf.pb --framework=3 --output=ID1376_KGAT_tf_gpu --soc_version=Ascend310 --input_shape="Placeholder:2048;Placeholder_1:24915;Placeholder_4:3" --log=info
```

将转换后的om文件放入Model目录

**运行推理与后处理**
```Bash
python3 offline_inference/xacl_inference.py
```

```log
2021-08-19 19:38:24.124 - I - [XACL]: Om model file is: ID1376_KGAT_tf_gpu.om
2021-08-19 19:38:24.124 - I - [XACL]: Input files are: input_bin/input1,input_bin/input2,input_bin/input3
2021-08-19 19:38:24.124 - I - [XACL]: Output file prefix is: output_bin/kgat_output_bin
2021-08-19 19:38:24.124 - I - [XACL]: Input type is director
2021-08-19 19:38:24.272 - I - [XACL]: Init acl interface success
2021-08-19 19:38:24.866 - I - [XACL]: Load acl model interface success
2021-08-19 19:38:24.866 - I - [XACL]: Create description interface success
2021-08-19 19:38:24.866 - I - [XACL]: The input file: input_bin/input1/users_00000.bin is checked
2021-08-19 19:38:24.866 - I - [XACL]: The input file: input_bin/input2/pos_items_00000.bin is checked
2021-08-19 19:38:24.866 - I - [XACL]: The input file: input_bin/input3/node_dropout_00000.bin is checked
...
2021-08-19 19:41:22.743 - I - [XACL]: The input file: input_bin/input1/users_00033.bin is checked
2021-08-19 19:41:22.743 - I - [XACL]: The input file: input_bin/input2/pos_items_00033.bin is checked
2021-08-19 19:41:22.743 - I - [XACL]: The input file: input_bin/input3/node_dropout_00033.bin is checked
2021-08-19 19:41:22.743 - I - [XACL]: Create input data interface success
2021-08-19 19:41:22.782 - I - [XACL]: Create output data interface success
2021-08-19 19:41:27.705 - I - [XACL]: Run acl model success
2021-08-19 19:41:27.705 - I - [XACL]: Loop 0, start timestamp 1629373282783, end timestamp 1629373287705, cost time 4922.59ms
2021-08-19 19:41:27.914 - I - [XACL]: Dump output 0 to file success
2021-08-19 19:41:27.914 - I - [XACL]: Single batch average NPU inference time of 1 loops: 4922.59 ms 0.20 fps
2021-08-19 19:41:27.914 - I - [XACL]: Destroy input data success
2021-08-19 19:41:28.134 - I - [XACL]: Destroy output data success
2021-08-19 19:41:28.460 - I - [XACL]: Start to finalize acl, aclFinalize interface adds 2s delay to upload device logs
2021-08-19 19:41:30.197 - I - [XACL]: Finalize acl success
2021-08-19 19:41:30.197 - I - [XACL]: 34 samples average NPU inference time of 34 batches: 4931.11 ms 0.20 fps
output_bin
[INFO]    推理结果生成结束
{'precision': array([0.01522007, 0.01111792, 0.00916784, 0.00797109, 0.00710147]), 'recall': array([0.14694857, 0.20585731, 0.2472628 , 0.28113915, 0.30843164]), 'ndcg': array([0.09972443, 0.12063131, 0.13402407, 0.14439348, 0.15252858]), 'hit_ratio': array([0.25184514, 0.34177161, 0.39911603, 0.44301682, 0.47649134]), 'auc': 0.0}
```


### 6. 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|                 | ascend310 |
|----------------|--------|
| precision |  [0.01522007, 0.01111792, 0.00916784, 0.00797109, 0.00710147]  |
| recall |  [0.14694857, 0.20585731, 0.2472628 , 0.28113915, 0.30843164]  |
| ndcg |  [0.09972443, 0.12063131, 0.13402407, 0.14439348, 0.15252858]  |
| hit_ratio |  [0.25184514, 0.34177161, 0.39911603, 0.44301682, 0.47649134]  |
