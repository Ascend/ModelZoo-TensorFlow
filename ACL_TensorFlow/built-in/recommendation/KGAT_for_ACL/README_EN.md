English|[中文](README.md)
# <font face="微软雅黑">
# KGAT Inference for TensorFlow

***
This repository provides a script and recipe to Inference the KGAT Inference

* [x] KGAT Inference, based on [knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)

***

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/recommendation/KGAT_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset by yourself, more details see: [amazon-book](./Data/README.md)

### 3. Obtain the pb model

Obtain the pb model, more details see: [pb](./Model/pb_model/README.md)

### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)

### 5. Offline Inference

**KGAT**
***
* KGAT in KGAT_for_ACL use static batch size, set predict_batch_size=2048 as input parameter, so we throw away the last batch of test data(batch size=959)
* The following commands are executed in the ./Model directory
***
**configure the env**

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

**PreProcess**
```Bash
python3 offline_inference/data_preprocess.py
```

The generated bin file is in the Model/input_bin directory


**Convert pb to om**

[pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/KGAT_for_ACL.zip)

```Bash
atc --model=KGAT_tf.pb --framework=3 --output=ID1376_KGAT_tf_gpu --soc_version=Ascend310 --input_shape="Placeholder:2048;Placeholder_1:24915;Placeholder_4:3" --log=info
```

Put the converted om file in the Model directory.

**Run the inference and PostProcess**
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


### 6. Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|                 | ascend310 |
|----------------|--------|
| precision |  [0.01522007, 0.01111792, 0.00916784, 0.00797109, 0.00710147]  |
| recall |  [0.14694857, 0.20585731, 0.2472628 , 0.28113915, 0.30843164]  |
| ndcg |  [0.09972443, 0.12063131, 0.13402407, 0.14439348, 0.15252858]  |
| hit_ratio |  [0.25184514, 0.34177161, 0.39911603, 0.44301682, 0.47649134]  |
