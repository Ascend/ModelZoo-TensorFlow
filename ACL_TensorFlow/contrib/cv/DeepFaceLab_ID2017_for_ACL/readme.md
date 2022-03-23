# DeepFaceLab

## 1. 模型概述
详情请看ModelZoo-TensorFlow/contrib/Tensorflow/Research/cv/DeepFaceLab_ID2017_for_Tensorflow/
## 2. 准备工作  
### 2.1 模型转换--获取om  
新建model文件夹，在model文件夹下新建ckpt, pb, om文件夹，用于存放相应的模型文件。  
[2.1.1，2.1.2]中所涉及的模型文件均可点击[链接](https://pan.baidu.com/s/1afh9Z0DSrhWZmNnfhiXx3Q ) (提取码：d5hc) 进行下载。  
obs模型路径：obs://dflobs/DFL4Infer/model。
#### 2.1.1 ckpt转pb文件
获取ckpt模型，进入工程/code目录下运行ckpt2pb.py, 得到pb模型文件保存并命名为-o指定的文件。  
```
  python3.7 ckpt2pb.py -i "./model/ckpt/DFL" -o "./model/pb/frozen_model_sigmoid4.pb" -n "Sigmoid_4"

其中参数路径均为相对ckpt2pb的路径，请根据实际测试情况进行相应更换。
```

参数说明：   

|参数|说明|
| :-----:|:----|
|-i    |  输入的ckpt模型文件|
|-o    |  输出的pb模型文件(与 -n 参数进行对应修改用于区分模型)|
|-n    |  选择ckpt的相应输出节点用于冻结为pb模型|
> -n 参数中可供选择的节点为"Sigmoid_3", "Sigmoid_4", "Sigmoid_5"，其中用于完成模型功能用于推理的节点为Sigmoid4，其余两个节点获得的pb模型用于获取中间推理结果以完成推理精度评估。

分别选取3个节点运行转换脚本后将获得相应的3个pb文件:(frozen_model_sigmoid3.pb, frozen_model_sigmoid4, frozen_model_sigmoid5.pb)。
#### 2.1.2 pb转om文件 
分别将上述获得三个pb模型用atc进行模型转换得到相应的om模型。ATC工具使用环境搭建方法请参考[教程](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html )。
atc环境设置完成之后，进入model/pb目录下，执行以下命令获得相应的om模型：
```
* frozen_model_sigmoid3:  
atc --model=frozen_model_sigmoid3.pb --framework=3 --input_format=NHWC --input_shape="Placeholder_1:1,96,96,3" --output="../om/dfl_sigmoid3" --soc_version=Ascend310
* frozen_model_sigmoid4:
atc --model=frozen_model_sigmoid4.pb --framework=3 --input_format=NHWC --input_shape="Placeholder_1:1,96,96,3" --output="../om/dfl_sigmoid4" --soc_version=Ascend310
* frozen_model_sigmoid5:
atc --model=frozen_model_sigmoid5.pb --framework=3 --input_format=NHWC --input_shape="Placeholder_1:1,96,96,3" --output="../om/dfl_sigmoid5" --soc_version=Ascend310  

其中参数路径均为相对model的路径，请根据实际测试情况进行相应更换。

```
atc转换完成后，会在om模型文件下生成相应的离线模型, (dfl_sigmoid3.om, dfl_sigmoid_4.om, dfl_sigmoid5.om)。

### 2.2 数据集准备
生成om文件后，需要将用于推理的jpg图片转换为bin文件格式。[参考工具](https://gitee.com/ascend/tools/blob/master/img2bin/README.md)。
原始数据集及相应转换后的结果已上传至obs。
```
原始jpg数据集：obs://dflobs/DFL4Infer/data_dst_aligned_96
结果bin文件: obs://dflobs/DFL4Infer/data_dst_aligned_bin
```
在工程目录下新建dataset文件夹，下载原始jpg数据集及其对应的文件夹，进入code目录执行如下命令
```
python3.7 img2bin.py -i ../dataset/data_dst_aligned_96 -w 96 -h 96 -f BGR -a NHWC -t float32 -o ../dataset/data_dst_aligned_bin

参数中的路径均为相对路径，是相对img2bin.py的路径，请根据实际测试情况进行相应更换。
```

执行完成后会在-o指定的路径下生成对应的bin文件夹。
## 3. 使用msame工具推理  
### 3.1 Ascend310推理AiS环境  
创建昇腾 310推理 Ai1S环详情请见[快速创建离线推理Ai1S环境](https://gitee.com/ascend/modelzoo/wikis/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/%E5%BF%AB%E9%80%9F%E5%88%9B%E5%BB%BA%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86Ai1S%E7%8E%AF%E5%A2%83)
#### 3.1.1 第三方依赖
```
见requirements.txt
```
#### 3.1.2 获取msame推理工具
参考[链接](https://gitee.com/ascend/tools/tree/master/msame) ，选择msame.zip方式获取推理工具及使用方法。

### 3.2 进行推理
获得推理工具之后进行推理工具所在文件夹可参考如下命令获得推理结果：
```
* dfl_sigmoid3.om:  
    ./msame --model "../model/dfl_sigmoid3.om" --input "../data_dst_aligned_bin" --output "../dataset/msame_infered_bin_sigmoig3"

* dfl_sigmoid4.om:   
    ./msame --model "../model/dfl_sigmoid4.om" --input "../data_dst_aligned_bin" --output "../dataset/msame_infered_bin_sigmoig4"

* dfl_sigmoid5.om:  
    ./msame --model "../model/dfl_sigmoid5.om" --input "../data_dst_aligned_bin" --output "../dataset/msame_infered_bin_sigmoig5"  

其中参数路径均为相对路径，是相对msame推理工具(.exe)可执行文件的路径，请根据实际测试情况进行相应更换。
```
执行上述命令后会在工程/dataset/目录下生成相应的推理结果文件夹：（msame_infered_bin_sigmoig3， msame_infered_bin_sigmoig4， msame_infered_bin_sigmoig5）。
> dfl_sigmoid3.om 推理截图：  
![](img4md/Sigmoid3.jpg)

> dfl_sigmoid4.om 推理截图：  
![](img4md/Sigmoid4.jpg)
 
> dfl_sigmoid5.om 推理截图：  
![](img4md/Sigmoid5.jpg)

### 3.3 推理结果可视化(可选)  
sigmoid4对应的模型完成本网络的功能，故只需将msame_infered_bin_sigmoig4文件夹下对应的bin文件进行可视化。  
进入工程/code目录下运行bin2img.py，会在工程/dataset目录下生成相应的jpg推理文件。  
```
python3.7 bin2img.py -i "../dataset/msame_infered_bin_sigmoid4" -o "../dataset/msame_infered_img_sigmoid4"

其中参数均为相对bin2img.py的路径，请根据实际测试情况进行相应更换。
```

参数说明：   

|参数|说明|
| :-----:|:----|
|-i    |  输入的bin文件|
|-o    |  输出的img文件|

### 3.4 精度测试  
进入工程/code目录下执行SSIM.py，得到推理精度。
```
python3.7 SSIM.py -ddm "../dataset/msame_infered_bin_sigmoig3" -sd "../dataset/msame_infered_bin_sigmoig4" -sdm "../dataset/msame_infered_bin_sigmoid5" -dst "../dataset/data_dst_aligned_bin"
```
> 推理精度与训练精度如下图所示：  
![](img4md/precision.jpg)

结果显示推理精度与训练精度保持一致。

## 4. 工程结构

```
|-------- code
|           |---- img2bin.py                   // 将jpg格式文件转化为jpg图片
|           |---- bin2img.py                   // 将bin格式文件转化为jpg图片
|           |---- ckpt2pb.py                   // 将ckpt模型文件转化为pb模型
|           |---- SSIM.py                      // 计算推理精度指标
|-------- dataset
|           |---- data_dst_aligned_96           // 推理数据集(jpg格式)(2.2已提供下载链接)
|           |---- data_dst_aligned_bin          // 推理数据集(bin格式)(由2.2生成,2.2已提供下载链接)
|           |---- data_dst_aligned_bin          // 推理数据集(bin格式)(由2.2生成,2.2已提供下载链接)
|           |---- msame_infered_bin_sigmoig3    // sigmoid3节点的推理输出(用于验证推理精度)
|           |---- msame_infered_bin_sigmoig4    // sigmoid4节点的推理输出(最终需要的推理结果模型)
|           |---- msame_infered_bin_sigmoig5    // sigmoid5节点的推理输出(用于验证推理精度)
|-------- img4md
|-------- model
|           |---- ckpt                          // 训练保存的ckpt模型(2.1已提供下载链接)
|           |---- pb                            // 冻结的pb模型(结果文件由2.1生成，2.1已提供下载链接)      
|           |---- om                            // 离线推理om模型(结果文件由2.1生成，2.1已提供下载链接)           
|-------- modelzoo_level.txt
|-------- readme.md                              
```  