# UDCVO

推理部分实现，模型概述详情请看UDCVO_ID2359_for_TensorFlow README.md

## 代码及路径解释

```
UDCVO_ID2359_for_ACL
├── bash
│    ├──eval_om.sh				OM模型精度测试启动shell
│    ├──png2bin.sh				推理数据预处理启动shell
├── binfile
│    ├──input					OM模型输入（.bin格式）
│    ├──output					OM模型输出（.bin格式）
├── data						推理所用图像数据集（.png格式）
│   ├──void_release
│   ├──void_voiced	
├── src
│    ├──ckpt2pb.py				PB固化脚本
│    ├──data_utils.py			数据路径读取子函数
│    ├──dataloader.py			数据载入脚本
│    ├──eval_om.py				OM模型精度测试脚本
│    ├──eval_utils.py			精度评估子函数
│    ├──png2bin.py				推理数据预处理脚本
├── testing						推理所用图像的路径
│    ├──void_test_ground_truth_1500.txt
│    ├──void_test_image_1500.txt
│    ├──void_test_interp_depth_1500.txt
│    ├──void_test_validity_map_1500.txt          
```

## 1. ckpt转pb

模型固化：
```
python3.7.5 src/ckpt2pb.py
```
转换完成的pb模型obs路径：
obs://udcvio/data/binfile/udcvo.pb

## 2. pb转om模型

使用ATC模型转换工具进行模型转换时可参考如下指令：
```
atc --model=/home/HwHiAiUser/AscendProjects/UDCVO/pb_model/udcvo.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/UDCVO --soc_version=Ascend310 --input_shape="dataloader/IteratorGetNext:8,480,640,3;concat:8,480,640,2" --log=info --out_nodes="truediv:0" --debug_dir=/home/HwHiAiUser/AscendProjects/UDCVO/debug_info
```
转换完成的om模型obs路径：
obs://udcvio/data/binfile/UDCVO.om

## 3. 将测试集图片转为bin文件

om模型的输入由两种tensor构成，shape分别为：
im0: [8, 480, 640, 3]
sz0: [8, 480, 640, 2]

执行入口：

```
sh bash/png2bin.sh
```
生成的bin文件obs路径：
obs://udcvio/data/binfile/output/

## 4. 编译msame推理工具

参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。

## 5. 性能测试

使用msame工具进行性能测试时可参考如下指令：
```
./msame --model /home/HwHiAiUser/AscendProjects/UDCVO.om --input /home/HwHiAiUser/AscendProjects/UDCVO/input/im0/,/home/HwHiAiUser/AscendProjects/UDCVO/input/sz0/ --output /home/HwHiAiUser/AscendProjects/UDCVO/output/ --outfmt BIN
```
测试结果如下：

```
Inference average time : 122.92 ms
Inference average time without first time: 122.92 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

1Batch, shape: [8, 480, 640, 1], 平均推理性能122.92ms

om推理得到的bin文件obs路径：
obs://udcvio/data/binfile/input/

## 6. om模型精度测试

执行入口：

```
sh bash/eval_om.sh
```

测试结果如下：

| Model            |   MAE   |   RMSE   |  iMAE   |  iRMSE  |
| :--------------- | :-----: | :------: | :-----: | :-----: |
| 原项目           |  82.27  |  141.99  |  49.23  |  99.67  |
| GPU V100 ECS推理 | 94.9901 | 146.9962 | 53.7138 | 91.8339 |
| OM 模型离线推理  | 95.1431 | 147.0940 | 53.7585 | 91.8702 |