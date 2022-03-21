# FlowNet2
## 离线推理
### 1、原始模型转PB模型
```
python3 ./offline_infer/freeze_graph.py \
        --ckpt ./log/flownet2/model-loss-2.019588.ckpt
```
--ckpt参数指定需要转换的模型路径，
转成的PB模型会保存在```offline_infer/flownet2.pb```  
我们转换好的PB模型在
```
https://pan.baidu.com/s/1np0efAvxqru9moXVcpLtLA 提取码：6m3q
```
OBS地址：
```
obs://flownet2/flownet2/offline_infer/flownet2.pb
```

### 2、PB模型转OM模型
```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc  \
     --model=./offline_infer/flownet2.pb \
     --framework=3 --output=./offline_infer/flownet2_base \
     --soc_version=Ascend910 \
     --input_shape="input_a:1,2,448,1024,3" \
     --log=info --out_nodes="output:0"
```
使用ATC模型转换工具转换PB模型到OM模型  
我们转换好的OM模型在
```
https://pan.baidu.com/s/1PtaairxlAAdGdZ7vaqthhA 提取码：x2o2
```
OBS地址：
```
obs://flownet2/flownet2/offline_infer/flownet2_base.om
```

### 3、数据预处理
读取数据集中所有图片，对其进行预处理，保存在--output路径下gt和image两个文件夹
```
python3 ./offline_infer/dataprep.py \
       --dataset ./data/ \
       --data_file ./data/sintel/val.txt \
       --output ./offline_infer/Bin/
```
生成的gt和image文件夹会保存在```./offline_infer/Bin/```下,分别存储光流信息和相邻两帧图像
或者从 ```OBS://flownet2/data/Bin.zip``` 下载并解压到```./offline_infer/```目录下,
```
https://pan.baidu.com/s/1qL8MomNczz1zHc0XgmtY8A 提取码：xsbb
```

OBS地址:
```
obs://flownet2/flownet2/offline_infer/Bin/gt
obs://flownet2/flownet2/offline_infer/Bin/image
```

### 4、pb模型推理精度与性能
使用pb模型推理得到预测的光流信息
#### 推理性能测试
```
python3 ./offline_infer/verify_pbmodel.py
```
生成的预测光流信息会保存在```./offline_infer/Bin/outputs_pb/```路径下,
OBS地址:
```
obs://flownet2/flownet2/offline_infer/Bin/outputs_pb
```
Batchsize=1, input shape = [1, 2, 448, 1024, 3], 平均推理时间1100.45ms

#### 推理精度测试
```
python3 ./offline_infer/evaluate.py  \
      --gt_path ./offline_infer/Bin/gt/ \
      --output_path ./offline_infer/Bin/outputs_pb/
```
pb模型推理精度EPE=2.0184，与在线模型精度一致

### 5、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/master/msame)

### 6、om模型推理性能精度测试
#### 推理性能测试
使用如下命令进行性能测试：
```
./msame --model ./offline_infer/flownet2_base.om --output ./offline_infer/Bin/output/ --loop 100
```
测试结果如下：
```
[INFO] output data success
Inference average time: 81.343710 ms
Inference average time without first time: 81.342939 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
```
Batchsize=1, input shape = [1, 2, 448, 1024, 3], 平均推理时间81.34ms

#### 推理精度测试
使用OM模型推理结果，运行：
```
./msame \
      --model ./offline_infer/flownet2_base.om \
      --input ./offline_infer/Bin/image/ \
      --output ./offline_infer/Bin/outputs_om/
```
所有的输出会保存在```./offline_infer/Bin/outputs_om/```目录下，
OBS地址:
```
obs://flownet2/flownet2/offline_infer/Bin/outputs_om/yyyymmdd_hhmmss/
(yyyymmdd_hhmmss是以生成的时间命名,格式为: 年月日_时分秒)
```

运行以下命令进行离线推理精度测试
```
python3 ./offline_infer/evaluate.py \
      --gt_path ./offline_infer/Bin/gt/ \
      --output_path ./offline_infer/Bin/outputs_om/yyyymmdd_hhmmss/
(yyyymmdd_hhmmss是以生成的时间命名,格式为: 年月日_时分秒)
```
离线推理精度EPE=2.022，与在线模型精度一致
