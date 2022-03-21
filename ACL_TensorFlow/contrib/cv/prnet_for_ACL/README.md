# Readme

### 1. 关于项目
本项目为复现 “Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network” 论文算法  

论文链接为： [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.pdf)  

原作者开源代码链接为： [code](https://github.com/YadiraF/PRNet)  

该文章提出了一种能够同时完成3D人脸重建和稠密人脸对齐的端对端的方法–位置映射回归网络（PRN），效果示意图如下所示：  

![prnet](https://images.gitee.com/uploads/images/2021/0818/200336_92843d93_9227151.gif "prnet.gif")


### 2. 关于依赖库

需要安装 numpy>=1.14.3， scikit-image， scipy， opencv-python，dilb等库。


### 3. 关于数据集
已经上传百度云：[DataSet](https://pan.baidu.com/s/1A8sX_aK5vazWczr8Gcwd3g)（password：ypzm ）  
其中各数据集均经过处理，直接解压即可。图片尺寸统一处理为256*256   
将测试集 LFPA,  AFLW2000  解压后放入  /Dataset/TestData/  中  



### 4. pb模型：
原始ckpt文件以及上传[百度云](https://pan.baidu.com/s/1UJ98jmwBgAPEj3803E5cpA)（password：51ol），在/prnet/ckpt文件文件夹中，将三个文件下载后放于checkpoint/文件夹中，将 uv_kpt_ind.txt放入 Data/uv-data 文件夹中。  
（除此以外，百度云链接中也提供了我自己生成的prnet.pb以及prnet.om模型。经过测试，精度均达标。）
  
运行  

```
python3 pb.py
```
得到pb模型（pb.py中固定了pb模型的输出目录为 model/ ）。项目中提供了eval_from_pb.py脚本，是从pb模型进行推理的脚本。


### 5. 生成om模型：
使用ATC命令进行转换，参考命令如下

```
atc --model=/home/HwHiAiUser/prnet/model/prnet.pb --framework=3 --output=/home/HwHiAiUser/prnet/model/prnet --soc_version=Ascend310 --input_shape="input:1,256,256,3"  
```  
  
需要注意的是，我在一开始转换om模型时，om模型输出的结果与pb文件输出的结果完全不同，华为的工程师帮忙定位了问题，需要将在switch文件中添加上面两个图融合和UB融合。
推理：/usr/local/Ascend/atc/lib64/plugin/opskernel/fe_config/fusion_config.json
修改关闭如下规则“ConvBatchnormFusionPass”后可行：  

```
 "Switch":{
        "GraphFusion":{
                "ConvBatchnormFusionPass":"off"

        },
        "UBFusion":{

        }
    },
```
按照如上指示修改后，生成的om模型的推理结果才是正确的。  


### 6. 数据集生成bin  
运行`python3 pre_process.py`，生成数据集的bin文件。


### 7. om模型推理：
使用msame工具，有两个测试集，分别进行推理，两个测试集的参考命令为：  

```
./msame --model /home/HwHiAiUser/prnet/model/prnet.om --input /home/HwHiAiUser/prnet/Dataset/TestData_bin/AFLW2000-3D/ --output /home/HwHiAiUser/prnet/output_bin/
```

```
./msame --model /home/HwHiAiUser/prnet/model/prnet.om --input /home/HwHiAiUser/prnet/Dataset/TestData_bin/LFPA/ --output /home/HwHiAiUser/prnet/output_bin/
```

### 8. om模型离线推理性能：
数据集AFLW2000-3D的平均运行时间为 4.46ms
![AFLW2000-3D](https://images.gitee.com/uploads/images/2021/0908/145254_32e4d956_9227151.png "AFLW2000-3D性能.png")
数据集LFPA平均运行时间为 4.43ms
![LFPW](https://images.gitee.com/uploads/images/2021/0908/145326_ba5f58be_9227151.png "LFPA推理性能.png")


### 9. om模型离线推理精度对比：
AFLW2000-3D平均误差为：3.19%
![AFLW2000-3D推理精度](https://images.gitee.com/uploads/images/2021/0908/153805_889a5b88_9227151.png "AFLW2000-3D推理精度.png")  
  

LFPW平均误差为：2.65%
![LFPW推理精度](https://images.gitee.com/uploads/images/2021/0908/153840_2b3c8f3f_9227151.png "LFPA推理精度.png")  
  
  

两个数据集平均误差均优于原论文误差。
![精度对比](https://images.gitee.com/uploads/images/2021/0908/145431_5bcc828a_9227151.png "精度对比.png")

