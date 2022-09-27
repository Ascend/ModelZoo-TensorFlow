# README

## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision** 

**修改时间（Modified） ：2022.2.12**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架进行人脸图像嵌入学习的任务** 

本项目将PFE迁移到ascend 910进行训练和推理。
相比传统的将人脸图像确定性嵌入到隐空间的一个点，Probabilistic Face Embeddings(PFEs)提出在隐空间建立人脸图像的分布，这样可以在无约束的人脸识别环境下有更好的表现。而且可以基于传统的确定性嵌入模型作为预训练模型。在此基础上训练PFE模型。

![PFE](PFE.png)

- 参考论文： [paper](http://arxiv.org/abs/1904.09658)

- 参考实现： [code](http://github.com/seasonSH/Probabilistic-Face-Embeddings)

- 适配昇腾 AI 处理器的实现：
    
        
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/pfe/pfe_ID0982_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置
- 使用CASIA-WebFace数据集，并进行预处理成112*96大小，并进行标准化（均值127.5，标准差128）进行训练

    - 百度网盘下载地址： [BaiduDrive](http://pan.baidu.com/s/1Sv84mdM3Y3hy_PSQwCSBrQ) ，提取码：：3yju

    - obs训练数据集： [obs_casia](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIaNhRd6sMlvWfOd8dAmFgEjGPacPPn75WjtvHmk7Rn9UWMFLvTPCFjYsA1bLZhNJFjTSDEio+2PWSN7aBpMQFRrRAoBTjzQTDyEEy3P1HwSChco4zQVHRuyT7m/s6BkX34C76LFUbCcBaa2/0WLzxiyswxCQgC5MIVAmjsjuAUKZH23o3BKLjNO89jXCovZq0myTBxB22+RS6zfalaqlU6x7n5C3eMdgXQo9Xo/n5AwQ6xNCzrCsKfFkiZi6s9dAYjRXP6RBYy3wfYGf3+R8FAEkr4Zkwq7UuXhlURU8XzDUcLD9pyaxZKFjIzhk18QZTGFWjQQlO8Y0lMHogA4gIE/ASZwhr9MLJvWkObdTjhOf7jIyMvAAoPJUULnkwszjTkU1bN5D1rEusLe3aEg2b6KcJ4g5wVhERuD2AJ1VGww4Fr+Gf6xze3+5berZ27V76iiW1W7XlSWKw5QjiAc7dJtNOAAN6sQF+UJJ/u93DjIGUaAwL7sQd4grPU8z8OG/cVthSFzxJUtBFf2toWyZiHnF8c8eL2KstvjWRA2oOgxz8MI8ZJelRaSA0oL98uJ/ceXzz0rifggKQnCI4xStbE9kI8PhdT2rHtFQRVqBZIW+EmzRNq8cqLR7tVo4cY0Jf2PB8jBMekLSNHFo7t3QrPo=
) ,提取码： yzy123

- 训练超参
    见config/sphere64_casia.py

- 使用LFW数据集进行测试
- obs测试数据集： [obs_lfw](http://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIaNhRd6sMlvWfOd8dAmFgEjGPacPPn75WjtvHmk7Rn9UWMFLvTPCFjYsA1bLZhNJFjTSDEio+2PWSN7aBpMQFRrRAoBTjzQTDyEEy3P1HwSChco4zQVHRuyT7m/s6BkX34C76LFUbCcBaa2/0WLzxiyswxCQgC5MIVAmjsjuAUKZH23o3BKLjNO89jXCovZq0kVIh9Z+V8P9npbmKq7GTcLFg5GTxspBjAF7FncxogVXRKpuwCadGj41Vpo/mVMgM/zikysjbddlp8zHi4BXNloGmJuZ+mkPSiCpOS74AxRskpuQqVSGGEe2OZOZQ50WuT+kCSgOSum4PtfSaEdM2GMKWYUlYuNK5CM+zlDmF+SKc6vlboEOca4bXUluPMW85NChqLCoi8Fh6jAeA183NrPIuo9o7gSr8pEXP8PirJA8RyYkW4aLD1T9qDCY9Qs7u8pRTkhlVSze0hplh6Mne/rtZ6vJOL0ih0WYWs0iFCBpvvC8HDerT+wHGNTfApytT//G5eNwdH7vqhYOjMtJaHAb8nVFaS6bHvo3yKar4AdHZwRiGuy1RM5I7YNwU2TCjpdZadjUJPtR2KNTD/u1eoM=
) ，提取码： yzy123

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度

手动开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
    custom_op = npu_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_fp32_to_fp16')
    custom_op.name = "NpuOptimizer"
  ```

## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 关于训练
在预训练模型的基础上训练PFE
预训练模型下载地址： [pretrained](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=qwwgvJK2FYxY566/N/OLIaNhRd6sMlvWfOd8dAmFgEjGPacPPn75WjtvHmk7Rn9UWMFLvTPCFjYsA1bLZhNJFjTSDEio+2PWSN7aBpMQFRrRAoBTjzQTDyEEy3P1HwSChco4zQVHRuyT7m/s6BkX34C76LFUbCcBaa2/0WLzxiyswxCQgC5MIVAmjsjuAUKZH23o3BKLjNO89jXCovZq0ujjEdNOBDJjK/kwCXX+WeUAdIAPLD9OFirMvMAfY9pUB0VWYf/UvlNrcT83a8YtH/IJeC7ci/FD19fbb8pscKfZAxDkzJRME3ScmNyBKfPKETiaqB4HCvFmrp3mm7EGNVPZOLG/qd0ibQh5lEOSofxUxGdmiR7plKxzmbbbfvCrNTvaAmpP3zIM1KJYZ5ZJTq2trb1Px0KLHZSOJN94p4MRerAvn53iI7YKZJHEPI++TYg71dAcmNTlU1LI4bJOCGX+YO0mUPHRMItvTyUMwFIGfjwbsigddhoPdw6VJIzKrRXAgJWE8Zgai46FbNaTBTwMBcdI+KKXxQJRIznOX/0tFhztuGDd/5qZ2+iwxdTtuK6GPzQ8BJ2KTIbCMLDhxzM5APEbyz3Kx7YiWOBtr6LKdl3jRIsxgbgI8FAlrvhX) ，提取码: yzy123

放在 pretrained/sphere64_caisa_am/ 中

### GPU训练
共训练3000epoch，在GPU（使用华为弹性云服务器ECS）上训练结果如下

![输入图片说明](gpu_loss.png)
  
平均每100epoch训练时长约在26s。

loss是论文中定义的MLS_loss，没有下界，训练的loss在-2800至-2700之间，且用该范围loss训练结果的模型用于测试（测试集LFW）得到和论文近似的准确率99.38%（使用论文代码训练好的PFE模型准确率99.40%，论文准确率98.63%），因此认为训练完成。

### NPU训练

执行scripts/run_1p.sh

共训练3000epoch，在GPU上训练结果如下

![输入图片说明](npu_loss.png)

平均每100epoch训练时长不到26s。

训练的loss在-2800至-2700之间，且使用NPU训练的模型进行测试准确率在99.13%，认为训练完成。

## 快速训练流程
全部图片（约50w张）的上传时间约在2小时，适当修改可以快速过训练流程。

快速数据集只使用casia_mtcnncaffe_aligned/CASIA-Webface/0000045下的图片。

放在data/0000045/ 中

执行 sh run_triancase.sh

## 精度对比
|Model|Method|LFW|
|--|--|--|
|64-CNN CASIA-WebFace|Baseline	|99.20|
|64-CNN CASIA-WebFace|PFE|99.47|
|64-CNN CASIA-WebFace|PFE + GPU|99.40|
|64-CNN CASIA-WebFace|PFE + NPU|99.13|

## 性能对比

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|100 epochs|26.5s|25.4s|
