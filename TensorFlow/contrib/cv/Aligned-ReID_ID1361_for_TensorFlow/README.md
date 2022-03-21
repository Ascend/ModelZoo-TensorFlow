# AlignedReID
## 模型简介
根据“AlignedReID: Surpassing Human-Level Performance in Person Re-Identification”，这篇论文中提出的AlignedReID通过学习全局特征，并在学习时执行自动的部件对齐，不需要额外监督和特定的姿态估计。在学习阶段，通过两个分支来同时学习全局特征和局部特征。局部分支里引入最短路径loss来对齐局部部件，并在推理阶段丢弃局部分支只提取全局特征。论文在CUHK03数据集上达到了88.1%的top1精度。

![Structure](https://images.gitee.com/uploads/images/2021/0908/184744_9a9146bd_7793736.png "微信图片_20210908184737.png")
## 环境配置
- python3.7
- Tensorflow 1.15.0
- Ascend 910
- opencv-python
- scipy1.2.0
- numpy
- time
- os
- opencv


## 结果
迁移 [AlignedReID](https://github.com/Phoebe-star/AlignedReID) 到Ascend 910平台，使用的环境是 [ModelArts](https://www.huaweicloud.com/product/modelarts.html)

使用 Cuhk03 dataset 数据集

训练前数据集需要做预处理操作，请用户参考GPU开源链接进行处理

### 训练精度
<table>
    <tr>
        <td></td>
        <td >training clean set</td>
        <td colspan="5">training details</td>
    </tr>   
    <tr>
        <td></td>
        <td>top1 &#8595;</td>
        <td>Enviroment</td>
        <td>device </td>
        <td>batch size </td>
        <td>iterations </td>
        <td>lr schedule</td>
    </tr>
    <tr>
        <td>Report in paper</td>
        <td>88.1</td>
        <td>TensorFlow, GPU</td>
        <td>Unknown</td>
        <td>Unknown</td>
        <td>Unknown</td>
        <td>multi-steps</td>
    </tr>
    <tr>
        <td>Reproduce on GPU</td>
        <td>89</td>
        <td>TensorFlow, GPU</td>
        <td>1</td>
        <td>30</td>
        <td>64000</td>
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
        <td >1xV100</td>
        <td >30</td>
        <td >0.55 s/step</td>
    </tr>  
    <tr>
        <td >8xAscend910</td>
        <td >30</td>
        <td >0.42 s/step</td>
    </tr>      
</table>


---
## 数据准备
### 预训练模型
预训练数据集以及resnet预训练模型在
```
链接：https://pan.baidu.com/s/1-vGELSPSHkN0I1tqJrzuGQ 
提取码：cdjj 

下载后将压缩包中的四个文件放在根目录dataset中（两个数据集以及两个pretrained model）
```

## 训练
### 参数说明
```
--train_data 训练数存放位置
--result 保存checkpoint的文件夹路径
--max_step 训练的iteration个数
--batch_size 训练的batch_size大小
--chip 运行设备

```

### 运行命令
在modelarts上进行训练
使用镜像
```
swr.cn-north-4.myhuaweicloud.com/ascend-share/5.0.2.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_0709
```

boot command
```
/bin/bash run_train.sh 's3://alignedreid/alignedreid/code/' '/code/boot_modelarts.py' '/tmp/log/train.log'
#'s3://alignedreid/alignedreid/code/' '/code/boot_modelarts.py'为代码在桶中的存放地址
```

obs path
```
/alignedreid/alignedreid/log/
```

data path in obs
```
/alignedreid/alignedreid/dataset/
```


