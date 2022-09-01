# Transformer-CNN离线推理

Transformer离线推理：regression (water solubility) models

模型论文：https://arxiv.org/pdf/1911.06603.pdf

训练代码参考：https://github.com/bigchem/transformer-cnn

## 环境

 Python版本：3.6.13

 CANN 版本：5.0.3 

 推理芯片： 昇腾310 

 第三方库：参考requirements.txt

## 快速入门

### 1. 数据集

使用solubility数据，训练集(train.csv)和测试集(test.csv)数据位于data文件夹下

### 2. 数据预处理

将csv文件预处理成bin文件.

```
cd scripts
mkdir input_bins
python3 preprocessing.py
```

将生成的bin文件放在scripts文件夹下

### 3. 配置环境

```
export install_path=/usr/local/Ascend
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

### 4. 将pb文件转为om模型

首先touch一个自定义算子dtypes文件

```
touch customize_dtypes
vim customize_dtypes
输入并保存：
OpType::MatMulV2:InputDtype:float16,float16,float32,OutputDtype:float32
OpType::BatchMatMul:InputDtype:float16,float16,OutputDtype:float32
OpType::BatchMatMulV2:InputDtype:float16,float16,OutputDtype:float32
```

将pb文件（encoder和model）转为om模型

```
atc --model=/root/Transformer_CNN/pb_model/encoder.pb --framework=3 --output=/root/Transformer_CNN/encoder --output_type=FP32 --soc_version=Ascend310 --input_shape="input_1:10,86;input_2:10,86" --customize_dtypes=/root/Transformer_CNN/code/Encoder_test/Siamese_for_ACL/scripts/customize_dtypes --precision_mode=force_fp32
atc --model=/root/Transformer_CNN/pb_model/model.pb --framework=3 --output=/root/Transformer_CNN/model --soc_version=Ascend310 --input_shape="input_4:10,86,64"
```

PB和OM模型的链接：

https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Transformer_CNN/model.pb
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Transformer_CNN/model.om
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Transformer_CNN/encoder.om
https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/Transformer_CNN/encoder.pb

### 5. 构建推理程序

```
bash build.sh
```

### 6. 进行离线推理

进行离线推理得到输出bin文件，需要修改benchmark_tf.sh中的路径。

```
cd scripts
bash benchmark_tf.sh
```

## 推理结果

COEFFICIENT OF DETERMINATION $r^{2}$: 0.9175694121226837

