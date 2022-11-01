中文|[English](README_EN.md)

# MTCNN TensorFlow离线推理

此链接提供MTCNN TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL/Official/cv/MTCNN_for_ACL
```


### 2. 离线推理


**Pb模型转换为om模型与推理**

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/MTCNN_for_ACL.zip)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- 编译程序

  ```
  cd xacl_fmk-master
  bash xacl_fmk.sh
  ```

- 开始运行:


  ```
  cd ..
  export DUMP_GE_GRAPH=2
  python3 acltest.py ompath data_in_om data_out_om Ascend310P3 ./mtcnn_pnet_tf.pb ./mtcnn_rnet_tf.pb ./mtcnn_onet_tf.pb
  ```
  注意: 
  默认情况下，“picture”目录中的图像已被推理。如果要替换图像，请替换“picture”目录中的图像。

  ompath、data_in_om和data_out_om目录将自动创建。

