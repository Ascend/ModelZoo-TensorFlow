## 概述

Unsupervised_Person_Re-identification，实现了一种渐进无监督学习（Progressive Unsupervised Learning, 简称PUL）方法，将预先训练的深度表示转移到未知领域。具体来说，PUL在行人聚类和卷积神经网络（CNN）微调之间迭代，以改进在无关标记数据集上训练的初始化模型。由于聚类结果可能非常噪杂，PUL在聚类和微调之间增加了一个选择操作。开始时，当模型较弱时，CNN在特征空间中靠近聚类中心的少量可靠示例上进行微调。随着模型的增强，在后续迭代中，更多的图像被自适应地选择为CNN训练样本。逐步地，行人聚类和CNN模型同时改进，直到算法收敛。这个过程被自然地表述为自步学习（self-paced learning）。

参考论文: Fan, H., Zheng, L., Yan, C., & Yang, Y. (2018). Unsupervised person re-identification: Clustering and fine-tuning. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 14(4), 1-18.

## 模型架构
![模型架构](picture/frame.jpg)

由上图，步骤（0）在一个无关的标记数据集中初始化CNN；然后进入迭代，在每次迭代中步骤（1）为未标记的数据集提取CNN特征并进行聚类和样本选择，步骤（2）使用选择的样本微调CNN模型。

## 数据集
请用户自行准备好数据集，可选用的数据集包括DukeMTMC-reID、Market-1501、CUHK03等。

这里以Market-1501数据集为例，[OBS下载链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=whjZKOD5ZjJHYwAZF/rtEhzzzyI6P3QnTqQnYpIJMsXVioipE2jSNYuVfZmP9g/CElla/zm4aa/Ewi+ggVjSfWt3Sx3MvCxbJObXlUvWyRpHOHkXYCxA8WwOJXJY0sqodvct6FdfHwHj1ryZWVgyOCgyFXuLQw0ABH9oXYMdJ1zedxtYNjbic1swo9npNjG3y7sHEtPtf1hxDze5Ix6MYMqy3YXSrizc2a6bGEt3vS5Yb8QQsBpLjHA0nzyXCBrsHk8SlqRCV4OOj1aQPl28OuQddRR4LltqmgngOgpKzz9EpAnSVXezhbOHiCi80gHsj/4SuMEZrlNqB3zGSEcRyvBiYYsh2IIX8t87+y9wUNHWFdMvQcbwevOg9M5lrzK5Yc5wdHeApEurhStkF96yGUA4RfFFxOCCKHFpNK6pBaQ50EtjwXWs1GaTXH3ZjIkACtC395J3FlaBhH1gynCVzjMqBj8pXyWAsBY3zRJtwcNKBV1wW0Aw3j44RDnZaWenj8IiECgfo8+uHysK/xB55tT0KXymn3uxStBIlv33EgeQAkM/m0I4ye6hRYdgF7gO)

提取码:
111111
*有效期至: 2022/12/08 15:04:30 GMT+08:00


目录结构为：

```
Market-1501
    - bounding_box_test
        - 0000_c1s1_000151_01.jpg
        - 0000_c1s1_000376_03.jpg
        - 0000_c1s1_001051_02.jpg
    - bounding_box_train
        - 0002_c1s1_000451_03.jpg
        - 0002_c1s1_000551_01.jpg
        - 0002_c1s1_000801_01.jpg
    - query
        - 0001_c1s1_001051_00.jpg
        - 0001_c2s1_000301_00.jpg
        - 0001_c3s1_000551_00.jpg
    - checkpoint
    - train.list
    - test.list
    - query.list
    - readme.txt
```

需要注意的是，后续实现对应于论文中的PUL(Duke)->Market.

## 环境要求
- python==3.7.5
- tensorflow==1.15.0
- keras==2.3.1
- sklearn==0.24.2
- h5py==2.10.0

## 快速上手
## 脚本以及简单代码
```
    - model_zoo
        - README.md
        - dataset
            - Market
                - query
                - checkpoint
                - bounding_box_train
                - bounding_box_test
                - train.list
                - test.list
                - query.list
        - train_on_gpu.py
        - train_on_npu.py
        - train_on_npu.sh
        - evaluate_on_gpu.py
        - evaluate_on_npu.py
        - evaluate_on_npu.sh
        - modelzoo_level.txt
        - evaluate_offline.py
        - export_test.py
```
## 脚本参数
```
    - 训练参数
        ```
        - STATR = 1
        - END = 25
        - LAMBDA = 0.85
        - NUM_EPOCH = 20
        - BATCH_SIZE = 16
        - ……
```
## 训练
- ### 入口
    `运行代码之前请将预训练checkpoint: 0.ckpt复制到./dataset/Market/checkpoint/下`

    - gpu(V100)

        `train_on_gpu.py`
    
        `python3.7.5 train_on_gpu.py --dataset=Market --data_path=./dataset/Market/ >train_on_gpu.log 2>&1 &`
    
    - NPU(ASCEND 910)
        
         `train_on_npu.py`
        
        `python3.7.5 train_on_npu.py --dataset=Market --data_path=./dataset/Market/ >train_on_npu.log 2>&1 &`
- ### 预训练checkpoint
    预训练使用的0.ckpt为论文作者开源提供。

    [论文作者Github](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning)
    
    [0.ckpt OBS地址](https://unsupervised.obs.cn-north-4.myhuaweicloud.com/Unsupervised_Person_Re-identification-PR/V100-ckpt/checkpoint_V100/0.ckpt?AccessKeyId=5VVGPWZODP1TNYXHXVI7&Expires=1670431679&Signature=CBG%2BPPvOPOBkUST9KgiahX4Ayo0%3D)
- ### 结果checkpoint
    `1.ckpt~25.ckpt`
    
    - gpu(V100)

    [checkpoint-V100 OBS地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=whjZKOD5ZjJHYwAZF/rtEhzzzyI6P3QnTqQnYpIJMsXVioipE2jSNYuVfZmP9g/CElla/zm4aa/Ewi+ggVjSfWt3Sx3MvCxbJObXlUvWyRpHOHkXYCxA8WwOJXJY0sqodvct6FdfHwHj1ryZWVgyOCgyFXuLQw0ABH9oXYMdJ1zedxtYNjbic1swo9npNjG3y7sHEtPtf1hxDze5Ix6MYDRJ17ZUQUEk9unsszqKZfAiyYo4UK7+IHkfE2fKBfRg4GrB8KhTodaqmxML1uwtygCllKBOAlMkflHl+3iP/N7NrWUl/fRFryG2z3AVmgL5373fKdD0uJbxmFjaH5k3rPZ7KEOOZS4y2DCanlX3Mny/yoQXly617sdnvGpYe2UCcKPWaQiFz74ERGwXIwkd+GMtytJBiZNvi877AIe+8ub1sYExtjmtOpvQ6igQPGEAEIz5/orX3MkzsA8MopGTcOJAllL1VBiylEnSB4HY0KuPtJ9ySeT6/EV/KyjAD3TVNOctLFPf5BFgvEu70VaO4H2VgLyR5B/NR13LXKmD88r0D+4r3spo6WfisRindRkvYSosh9BsIIoj4qHrJv726RcfwY9rhJC23VHsX1ty+apPlTD1M6UDnJbmnyYqnhIaTHly6oE3QofM5wnNloBazMuTkTb7/SxSyohSe4IPFw+5qsjh/0cLSgirmHCxPy7D+DMFaNOIo3mnQXiMADMBwUvQv5xcoFY7MDK6pSm0g3S6gtWExwzRCVxLGjdc3u28QJL1z8kauJcto1G3KXafEk9eKHIk5OZFTE7LrK2SN92cK9dJJXfRwdS7NGaHKwAr)
    提取码:111111

    *有效期至: 2022/12/08 00:34:42 GMT+08:00

    - NPU(ASCEND 910)

    [checkpoint-NPU OBS地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=whjZKOD5ZjJHYwAZF/rtEhzzzyI6P3QnTqQnYpIJMsXVioipE2jSNYuVfZmP9g/CElla/zm4aa/Ewi+ggVjSfWt3Sx3MvCxbJObXlUvWyRpHOHkXYCxA8WwOJXJY0sqodvct6FdfHwHj1ryZWVgyOCgyFXuLQw0ABH9oXYMdJ1zedxtYNjbic1swo9npNjG3y7sHEtPtf1hxDze5Ix6MYDRJ17ZUQUEk9unsszqKZfAjC96zqpCmZA3rlgHC+81YoKnnt0Wni+tr6OECIKR2/kNMlvZn5mTpvagiqPcmm4sB+TXa7/3h1m+fvzqCWgooXMRf5nCImCW3ps9+T0QYMYnKgVGetanC/Fy6xntw9aZdRRQtthtm/X1w5evmiIFUu7fodcFw0Zl84Us4i6vlsktTXWxHlcaQGe6O2qnBhLxqvJuC+oXTtjAZTy0HF2whXyd4WWlWGSPcs7vBjY3JreTmN67DybLGgQM+OXlfhxyY/TJNm2VfsHHnV3WiRlNKMcIX8Owbt5udutcQXfKlBzwhmSIt8Lr1T8VVZJmfliAzxt0X+8Y+X8jxrmECbwrZVlg0mLWgKvj2wimxfWAvt5TH5sHmYp2rNCzttCEunv/KuqZ7tNbbqFaBQleiaNxcdqgJfIYqkXu/AqoWX+ZexQmqR9JffFEuIZV6QrN7zw/WINHV/S8KBx9U3TwwI2WiqdaKlhnUFbr/OlT4WQ33HfRLzQ6omxEZYu/scj0Wxw5BZ4Rgt04qJzAC/MGJsEDCmoIjF3Kwu58yysVY+2JONew6Ex/bB1pF/KjuvCNedYk=)
    提取码:111111     
            
    *有效期至: 2022/12/08 00:49:09 GMT+08:00
## 测试
- ### 入口
    - gpu(V100)
        
        `evaluate_on_gpu.py`
        
        `python3.7.5 evaluate.py --dataset=Market >evaluate_on_gpu.log 2>&1 &`
    
    - NPU(ASCEND 910)
        
        `evaluate_on_npu.py`
        
        `python3.7.5 evaluate.py --dataset=Market >evaluate_on_npu.log 2>&1 &`
- ### 精度达标
- 原论文

    | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP  |
    |--------|--------|---------|---------|------|
    | 44.7   | 59.1   | 65.6    | 71.7    | 20.1 |

- gpu(V100)
    
    | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP  |
    |--------|--------|---------|---------|------|
    | 44.5   | 59.3   | 65.9    | 71.9    | 19.6 |

- NPU(ASCEND 910)
            
    | Rank-1 | Rank-5 | Rank-10 | Rank-20 | mAP  |
    |--------|--------|---------|---------|------|
    | 44.3   | 59.8   | 66.1    | 72.7    | 19.4 |

    说明，以上数据的单位是%，值越大越好。

- 总结

     基于以上数据，NPU精度达标。具体来讲，在Rank-1、Rank-5、Rank-10、Rank-20、mAP五个指标数据中，NPU的Rank-5、Rank-10、Rank-20三个指标均超过标准，Rank-1与mAP指标的误差在1%以内。

## 性能达标
以生成最后一个checkpoint（25.ckpt）为例，共20个Epoch，对比V100与ASCEND 910的性能，结果如下：

|            |                |
|------------|----------------|
| V100       | 51.95s / Epoch |
| ASCEND 910 | 51.35s / Epoch |

- 总结

    基于以上数据，NPU性能优于GPU性能，性能达标。

## 离线推理
### 1、原始模型转PB模型
```
python3 export_test.py
```
生成的PB模型为：
```
链接：https://pan.baidu.com/s/1fL_89xnTf7nj7BcSgA7rUQ 
提取码：p05x
```
### 2、PB模型转OM模型
```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc  \
     --model=/home/data/dataset/wxm/tensorflow_model/model.pb \
     --framework=3 --output=/home/data/dataset/wxm/tensorflow_model/net_test \
     --soc_version=Ascend310 \
     --input_shape="input_1:1,224,224,3" \
     --log=info --out_nodes="avg_pool/AvgPool"
```
使用ATC模型转换工具转换PB模型到OM模型
### 3、数据预处理
特别注意，数据预处理需要分别对bounding_box_test与query两个文件夹均执行以下操作：
```
使用git命令方式获取img2bin脚本：
在命令行中：$HOME/AscendProjects目录下执行以下命令下载代码。
git clone https://gitee.com/ascend/tools.git
```
```
进入img2bin脚本所在目录
cd $HOME/AscendProjects/img2bin
```
```
执行
python3 img2bin.py -i /home/data/dataset/unsupervised/dataset/Market/query/ -w 224 -h 224 -f BGR -a NHWC -t float32 -m [104,117,123] -c [1,1,1] -o /home/data/dataset/wxm/out/market_query2
```
过程如下图
![数据预处理](picture/p1.png)

### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/master/msame)
### 5、om模型推理性能精度测试
#### 推理性能测试
##### 无输入版本
使用如下命令进行性能测试：
```
msame ./msame --model "/home/HwHiAiUser/wxm/net_base.om" --output "/home/HwHiAiUser/wxm/out" --outfmt TXT --loop 100
```
测试结果如下：
![无输入版本1](picture/p2.png)
![无输入版本2](picture/p3.png)

##### 输入待处理的文件夹
使用如下命令进行性能测试：
```
msame ./msame --model "/home/HwHiAiUser/wxm/net_test.om" --input "/home/HwHiAiUser/wxm/input/market_test2/" --output "/home/HwHiAiUser/wxm/out" --outfmt TXT --loop 1
```
测试结果：
```
Inference average time : 2.41 ms
Inference average time without first time: 2.41 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
```
![输入待处理的文件夹1](picture/p4.png)
![输入待处理的文件夹2](picture/p5.png)

#### 推理精度测试
使用如下命令进行精度测试：
```
python3 evaluate_offline.py --query_path=/home/HwHiAiUser/wxm/out/query/ --test_path=/home/HwHiAiUser/wxm/out/test/
```


