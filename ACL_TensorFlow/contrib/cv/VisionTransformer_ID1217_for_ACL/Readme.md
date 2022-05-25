推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame | NPU精度  |GPU精度  |
|--|--|--|---| -- | --| -- |-- |
| VisionTransformer | Cifar100  | `4*328*328*3` | `4*100`  |0.25s~ | 0.826 | 0.871 |0.871| 

## 0. 目录分析 
```log
../VisionTransformer_ID1217_for_ACL
├── Convert2pb.py  # 将模型转换为pb文件
├── Readme.md # 文档
├── bindataset # bin数据集
├── ckpt
    ├── model
       ├── model.ckpt-0
├── data_generation.py # 数据生成
├── dataset # 数据集
    ├── train
    ├── test
├── inference.py # 推理
├── inference.sh # msame 推理脚本 
├── pb2om.sh # pb2om 转换脚本
└── requirements.txt
```

## 1、原始模型
训练后保存[ckpt文件](obs://cann-id1217/dataset/model/),下载ckpt到到指定目录`./ckpt`，使用`Convert2pb.py`脚本将`ckpt`转成`pb`模型。

## 2、转om模型

atc转换命令参考：

```sh
atc --model=./vit.pb --framework=3 --output=./vit --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 4、数据集预处理：

下载[数据集](obs://cann-id1217/dataset/)，拷贝到指定目录`./dataset`。

测试集总共10000张图片，每4张图片一个bin。

使用`data_preprocess.py`脚本,设定好路径后,
执行`python3 data_preprocess.py`会在`./bindataset`目录下生成`bin`文件。


## 5、执行推理和精度计算
根据msame编译输出的位置以及数据，模型所在的位置，修改`./inference.sh`中相应位置，并执行`./inference.sh`
该命令主要功能就是加载om执行推理同时计算精度

修改`inference.py`的路径得到推理结果 

## 6. msame 日志 
```log
Inference time: 1004.22ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/96.bin
[INFO] model execute success
Inference time: 1003.84ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/960.bin
[INFO] model execute success
Inference time: 1004.24ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/961.bin
[INFO] model execute success
Inference time: 1003.94ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/962.bin
[INFO] model execute success
Inference time: 1004.16ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/963.bin
[INFO] model execute success
Inference time: 1003.96ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/964.bin
[INFO] model execute success
Inference time: 1004.74ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/965.bin
[INFO] model execute success
Inference time: 1003.82ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/966.bin
[INFO] model execute success
Inference time: 1003.83ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/967.bin
[INFO] model execute success
Inference time: 1004.15ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/968.bin
[INFO] model execute success
Inference time: 1004.06ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/969.bin
[INFO] model execute success
Inference time: 1003.69ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/97.bin
[INFO] model execute success
Inference time: 1003.48ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/970.bin
[INFO] model execute success
Inference time: 1004.5ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/971.bin
[INFO] model execute success
Inference time: 1004.38ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/972.bin
[INFO] model execute success
Inference time: 1004.07ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/973.bin
[INFO] model execute success
Inference time: 1003.87ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/974.bin
[INFO] model execute success
Inference time: 1004.05ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/975.bin
[INFO] model execute success
Inference time: 1004.02ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/976.bin
[INFO] model execute success
Inference time: 1004.24ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/977.bin
[INFO] model execute success
Inference time: 1003.66ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/978.bin
[INFO] model execute success
Inference time: 1003.81ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/979.bin
[INFO] model execute success
Inference time: 1003.22ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/98.bin
[INFO] model execute success
Inference time: 1003.87ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/980.bin
[INFO] model execute success
Inference time: 1004.2ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/981.bin
[INFO] model execute success
Inference time: 1004.12ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/982.bin
[INFO] model execute success
Inference time: 1003.72ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/983.bin
[INFO] model execute success
Inference time: 1003.89ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/984.bin
[INFO] model execute success
Inference time: 1003.66ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/985.bin
[INFO] model execute success
Inference time: 1004.06ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/986.bin
[INFO] model execute success
Inference time: 1003.66ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/987.bin
[INFO] model execute success
Inference time: 1003.65ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/988.bin
[INFO] model execute success
Inference time: 1004.05ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/989.bin
[INFO] model execute success
Inference time: 1003.93ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/99.bin
[INFO] model execute success
Inference time: 1003.73ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/990.bin
[INFO] model execute success
Inference time: 1003.75ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/991.bin
[INFO] model execute success
Inference time: 1003.72ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/992.bin
[INFO] model execute success
Inference time: 1004.09ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/993.bin
[INFO] model execute success
Inference time: 1003.98ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/994.bin
[INFO] model execute success
Inference time: 1003.98ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/995.bin
[INFO] model execute success
Inference time: 1003.99ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/996.bin
[INFO] model execute success
Inference time: 1004.22ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/997.bin
[INFO] model execute success
Inference time: 1003.49ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/998.bin
[INFO] model execute success
Inference time: 1003.72ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/root/zwy/bindataset/999.bin
[INFO] model execute success
Inference time: 1004.15ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 1003.89 ms
Inference average time without first time: 1003.89 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
````

