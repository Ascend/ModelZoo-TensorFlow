推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame | NPU精度(Inception Score 50000 mean std)  |GPU精度  |
|--|--|--|---| -- | --| -- |-- |
| Transferring-Gan | cifar10  | `label:100,noise:100*128` | `100*3072`  |0.0006s~ | 0.06s | 7.05 0.1 |7.9 0.09| 

## 0. 目录分析 
```log
../Transferring-Gan_ID1252_for_ACL
├── Convert2pb.py  # 将模型转换为pb文件
├── Readme.md # 文档
├── bindataset32 # bin数据集
    ├── data0 # label 数据
    ├── data1 # noise 数据  
├── ganCkpt # GAN模型的checkpoint
    |-- final # 最终的checkpoint
├── datamake.py # 数据生成
├── inference.py # 推理
├── inference.sh # msame 推理脚本 
├── pb2om.sh # pb2om 转换脚本
└── requirements.txt
```

## 1、原始模型

参考 obs://cann-id1252/inference/pbmake/  由于本模型时GAN模型，所以在推理时不需要判别器，只需要固定生成器。 使用 obs://cann-id1252/inference/pbmake/pbmaker.py 实现只保存生成器。 使用 obs://cann-id1252/inference/pbmake/convert2pb.py 或者 Convert2pb.py 实现将生成器转换为pb文件。   


## 2、转om模型

atc转换命令参考：

```sh
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/home/TestUser08/zhegongda/transfergan/inference/transfergan2.pb --framework=3  --input_shape="label:100;noise:100,128" --output=/home/TestUser08/zhegongda/transfergan/inference/transfergan5  --soc_version=Ascend910  --precision_mode=allow_fp32_to_fp16 --log=error 
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 4、数据集预处理：

使用`datamake.py`脚本生成模型的输入文件。  

## 5、执行推理和精度计算
根据msame编译输出的位置以及数据，模型所在的位置，修改`./inference.sh`中相应位置，并执行`./inference.sh`
该命令主要功能就是加载om执行推理同时计算精度

修改`inference.py`的路径得到推理结果 

> 注意： 为了更快完成推理建议使用gpu完成inception score的计算。 

## 6. msame 日志 
```log
[INFO] model execute success
Inference time: 6.487ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/65.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/65.bin
[INFO] model execute success
Inference time: 6.499ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/66.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/66.bin
[INFO] model execute success
Inference time: 6.5ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/67.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/67.bin
[INFO] model execute success
Inference time: 6.501ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/68.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/68.bin
[INFO] model execute success
Inference time: 6.489ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/69.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/69.bin
[INFO] model execute success
Inference time: 6.488ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/7.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/7.bin
[INFO] model execute success
Inference time: 6.485ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/70.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/70.bin
[INFO] model execute success
Inference time: 6.497ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/71.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/71.bin
[INFO] model execute success
Inference time: 6.513ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/72.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/72.bin
[INFO] model execute success
Inference time: 6.487ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/73.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/73.bin
[INFO] model execute success
Inference time: 6.505ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/74.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/74.bin
[INFO] model execute success
Inference time: 6.486ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/75.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/75.bin
[INFO] model execute success
Inference time: 6.509ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/76.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/76.bin
[INFO] model execute success
Inference time: 6.498ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/77.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/77.bin
[INFO] model execute success
Inference time: 6.505ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/78.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/78.bin
[INFO] model execute success
Inference time: 6.487ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/79.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/79.bin
[INFO] model execute success
Inference time: 6.502ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/8.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/8.bin
[INFO] model execute success
Inference time: 6.511ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/80.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/80.bin
[INFO] model execute success
Inference time: 6.507ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/81.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/81.bin
[INFO] model execute success
Inference time: 6.504ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/82.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/82.bin
[INFO] model execute success
Inference time: 6.501ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/83.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/83.bin
[INFO] model execute success
Inference time: 6.507ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/84.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/84.bin
[INFO] model execute success
Inference time: 6.521ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/85.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/85.bin
[INFO] model execute success
Inference time: 6.514ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/86.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/86.bin
[INFO] model execute success
Inference time: 6.504ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/87.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/87.bin
[INFO] model execute success
Inference time: 6.501ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/88.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/88.bin
[INFO] model execute success
Inference time: 6.515ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/89.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/89.bin
[INFO] model execute success
Inference time: 6.502ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/9.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/9.bin
[INFO] model execute success
Inference time: 6.497ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/90.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/90.bin
[INFO] model execute success
Inference time: 6.506ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/91.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/91.bin
[INFO] model execute success
Inference time: 6.498ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/92.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/92.bin
[INFO] model execute success
Inference time: 6.51ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/93.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/93.bin
[INFO] model execute success
Inference time: 6.503ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/94.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/94.bin
[INFO] model execute success
Inference time: 6.528ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/95.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/95.bin
[INFO] model execute success
Inference time: 6.5ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/96.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/96.bin
[INFO] model execute success
Inference time: 6.504ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/97.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/97.bin
[INFO] model execute success
Inference time: 6.492ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/98.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/98.bin
[INFO] model execute success
Inference time: 6.501ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data0/99.bin
[INFO] start to process file:/home/TestUser08/zhegongda/transfergan/inference/bindata_32/data1/99.bin
[INFO] model execute success
Inference time: 6.505ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 6.49 ms
Inference average time without first time: 6.49 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
````
