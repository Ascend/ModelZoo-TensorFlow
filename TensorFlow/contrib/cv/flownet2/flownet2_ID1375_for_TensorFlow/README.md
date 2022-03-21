# FlowNet2
## 模型简介
根据“FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks”，Flownet2通过设计更加复杂的网络结构来改进FlowNet。光流估计背后的基本思想是根据从一帧到下一帧时像素的位置会发生变化,估计出相邻两帧对应像素点之间的运行信息。Flownet2通过堆叠FlownetC与FlownetS的结构加深网络结构,通过逐步refine的手法来得到更好的结果。并且将warp操作加入至不同模块信息传输中,使用中间光流warp图片，并以此作为一个监督信号辅助训练。使用第二帧图像和预测的光流信息进行warp，这样，下一个模块就可以集中学习warp后的图像和第一帧图像的差别，这是一种增量学习的思想。同时,网络也将warp后的图像和第一帧图像的误差图也显式地输入给了下一个网络模块。需要说明的是，warp的过程是用双线性插值，是可微的，故整个网络可以端到端训练。最终通过更加复杂的网络设计与逐步refine的训练方法,Flownet2在Flownet网络的基础上提升了光流信息预测的精度,最终的预测精度能够达到SOTA 。

![](./imgs/flownet2.jpg)

![](./imgs/sample.png)
## 环境配置
- python3.7
- Tensorflow 1.15.0
- Ascend 910
- opencv-python
- scipy1.2.0
- enum34
- pypng
- image
- numpy

requirements.txt 内记录了所需要的环境与对应版本，可以通过命令配置所需环境。
- pip install -t requirements.txt 

## 结果
迁移 [FlowNet2](https://github.com/sampepose/flownet2-tf) 到Ascend 910平台，使用的环境是 [ModelArts](https://www.huaweicloud.com/product/modelarts.html)

使用 FlyingChairs 数据集和 FlyThings3d两个数据集进行预训练的模型，之后在ModelArts Ascend 910 TensorFlow平台上在 MPI Sintel数据集上训练，在MPI Sintel training clean set上测试结果如下：

### 训练精度
<table>
    <tr>
        <td></td>
        <td >training clean set</td>
        <td colspan="5">training details</td>
    </tr>   
    <tr>
        <td></td>
        <td>Avg. EPE &#8595;</td>
        <td>Enviroment</td>
        <td>device </td>
        <td>batch size </td>
        <td>iterations </td>
        <td>lr schedule</td>
    </tr>
    <tr>
        <td>pretrained model</td>
        <td>10.56</td>
        <td>TensorFlow, GPU</td>
        <td>2</td>
        <td>16</td>
        <td>1200000</td>
        <td>multi-steps</td>
    </tr>
    <tr>
        <td>Report in paper</td>
        <td>2.02</td>
        <td>Caffe, GPU</td>
        <td>Unknown</td>
        <td>Unknown</td>
        <td>Unknown</td>
        <td>multi-steps</td>
    </tr>
    <tr>
        <td>Reproduce on GPU</td>
        <td>2.02</td>
        <td>TensorFlow, GPU</td>
        <td>1</td>
        <td>4</td>
        <td>40000</td>
        <td>multi-steps</td>
    </tr>
    <tr>
        <td>Reproduce on Ascend 910</td>
        <td>2.02</td>
        <td>ModelArts, Ascend 910</td>
        <td>1</td>
        <td>4</td>
        <td>40000</td>
        <td>multi-steps</td>
    </tr>
</table>

### 训练性能
<table>
    <tr>
        <td >Platform</td>
        <td >Batch size</td>
        <td >Throughout</td>
    </tr>  
    <tr>
        <td >1xAscend 910</td>
        <td >4</td>
        <td >9.75 imgs/s</td>
    </tr> 
    <tr>
        <td >1xTesla V100-16G</td>
        <td >4</td>
        <td >8.69 imgs/s</td>
    </tr>    
    <tr>
        <td >1xAscend 910</td>
        <td >8</td>
        <td >12.3 imgs/s</td>
    </tr> 
    <tr>
        <td >1xTesla V100-16G</td>
        <td >8</td>
        <td >memory out</td>
    </tr> 
</table>


---
## 数据准备
### 预训练模型
1、用户自行准备好数据集，包括训练数据集和验证数据集。使用的数据集包括 FlyingChairs 数据集， FlyThings3d和MPI Sintel

2、数据集的处理可以参考"简述->开源代码路径处理"

3、预训练模型的下载可以参考"简述->开源代码路径处理"

### Finetuned 模型
在MPI Sintel clean training set上，在ModelArts TensorFlow Ascend 910环境下训练好的模型在   
```
obs://flownet2/flownet2/log/flownet2/npu/model-loss-2.019588.ckpt.index
obs://flownet2/flownet2/log/flownet2/npu/model-loss-2.019588.ckpt.meta
obs://flownet2/flownet2/log/flownet2/npu/model-loss-2.019588.ckpt.data-00000-of-00001
obs://flownet2/flownet2/log/flownet2/npu/checkpoint
```

在GPU上复现的模型在
```
obs://flownet2/flownet2/log/flownet2/gpu/model-loss-2.020386.ckpt.index
obs://flownet2/flownet2/log/flownet2/gpu/model-loss-2.020386.ckpt.meta
obs://flownet2/flownet2/log/flownet2/gpu/model-loss-2.020386.ckpt.data-00000-of-00001
obs://flownet2/flownet2/log/flownet2/gpu/checkpoint
```
下载Finetuned模型至```./log/flownet2/```路径下

## 训练
### 参数说明
```
--train_step 训练的iteration个数
--result 保存checkpoint的文件夹路径
--save_step 每save_step轮训练进行一次验证
--batch_size 训练的batch_size大小
--dataset 数据集文件夹路径
--train_file 训练集文件路径
--val_file 验证集文件路径
--image_size 输入图像尺寸
--pretrained 预训练模型路径
--chip 运行设备

```

### 运行命令
#### GPU版本
```
sh run_gpu_1p.sh
```
或者
```
python3 flownet2_train.py \
	--dataset='./data/sintel' \
	--result='./log/flownet2' \
	--train_file='./data/sintel/train.txt' \
	--val_file='./data/sintel/val.txt' \
	--batch_size=4 \
	--save_step=500 \
	--image_size=436,1024 \
	--pretrained='./checkpoints/FlowNet2/flownet-2.ckpt-0' \
	--chip='gpu' \
	--gpu_device='0' \
	--train_step=40000
```

#### NPU版本
```
sh run_npu_1p.sh
```
或者
```
python3 flownet2_train.py \
	--dataset='./data/sintel' \
	--result='./log/flownet2' \
	--train_file='./data/sintel/train.txt' \
	--val_file='./data/sintel/val.txt' \
	--batch_size=4 \
	--save_step=500 \
	--image_size=436,1024 \
	--pretrained='./checkpoints/FlowNet2/flownet-2.ckpt-0' \
	--chip='npu' \
	--train_step=40000
```

## 测试 
### 参数说明
```
--dataset 数据集文件夹路径
--test_file 测试集文件路径
--checkpoint 需要测试的checkpoint的路径
```
### 运行命令
```
sh validation.sh {dataset_path} {file_path} {checkpoint_path} {chip}
eg: sh validation.sh ./data/sintel ./data/sintel/test.txt ./checkpoints/FlowNet2/flownet-2.ckpt-0 npu
```
或者
```
python3 flownet2_val.py --dataset {dataset_path} --test_file {file_path} --checkpoint {checkpoint_path}
eg: python3 flownet2_val.py --dataset ./data/sintel --test_file ./data/sintel/test.txt --checkpoint ./checkpoints/FlowNet2/flownet-2.ckpt-0
```
## 推理
### 参数说明
```
--input_a 输入图像A
--input_b 输入图像B
--out_path 保存路径
--checkpoint 需要导入的checkpoint的路径
--save_flo 是否生成 .flo 文件
```
### 运行命令
```
sh test.sh {input_a_path} {input_b_path} {out_path}
eg: sh test.sh data/samples/0img0.ppm data/samples/0img1.ppm ./
```
或者
```
python3 flownet2_test.py --input_a {input_a_path} --input_b {input_b_path} --out {out_path} [--checkpoint {checkpoint_path} --save_flo {True/Flase}]
eg: python -m flownet2_test.py --input_a data/samples/0img0.ppm --input_b data/samples/0img1.ppm --out ./
```

## 离线推理
### 1、原始模型转PB模型
```
python3 ./offline_infer/freeze_graph.py \
        --ckpt ./log/flownet2/model-loss-2.019588.ckpt
```
--ckpt参数指定需要转换的模型路径，
转成的PB模型会保存在```offline_infer/flownet2.pb```  

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

读取数据集中所有图片，对其进行预处理，保存在--output路径下gt和image两个文件夹

```
python3 ./offline_infer/dataprep.py \
       --dataset ./data/ \
       --data_file ./data/sintel/val.txt \
       --output ./offline_infer/Bin/
```
生成的gt和image文件夹会保存在```./offline_infer/Bin/```下,分别存储光流信息和相邻两帧图像
或者从 ```OBS://flownet2/data/Bin.zip``` 下载并解压到```./offline_infer/```目录下,

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
