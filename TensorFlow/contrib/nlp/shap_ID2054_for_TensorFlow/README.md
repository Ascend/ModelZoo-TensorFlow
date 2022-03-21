- [SHAP](#shap)
    - [基本信息](#基本信息)
    - [概述](#概述)
    - [训练环境准备](#训练环境准备)
    - [快速上手](#快速上手)
    - [脚本和示例代码](#脚本和示例代码)
    - [训练过程](#训练过程)
        - [GPU结果（V100）](#GPU结果（V100）)
        - [NPU结果](#NPU结果)
        - [精度分析](#精度分析)
        - [性能分析](#性能分析)
# SHAP

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.28**

**大小（Size）：610M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的SHAP网络训练代码** 


## 概述

本项目实现了在NPU上的训练，迁移自SHAP。

- GitHub项目地址：https://github.com/slundberg/shap

- 参考论文：Learning Important Features Through Propagating Activation Differences

- 适配昇腾 AI 处理器的实现：
  [modelzoo: Ascend Model Zoo - Gitee.com](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/nlp//shap_ID2054_for_TensorFlow)

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```


## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
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




## 快速上手

- 数据集准备

模型训练使用ImageNet50数据集。

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=ZooFr93nESyM+SLxMFyJV9FzVnc8iEcIdAadkWaJmJKff/6hQoZtfsCzUSAdbnW1TPNhG31hUTB1QDRVDvLJaFQKI4SeLCM4zHCdPqKSEncxBAh/Wu41B3ynGIhGw6Vz9T4eXV6tnq8KCji47wT3aodqNWYhNEUAYMr6NDMvDJ9zJ2FAoNusJqmE3Iu9aCb1ytBdpJRq7IdqM/TlkKt5WGbTQJ73TDQYraemZyG4UXoDrhCy6MOLvsHqF0LC334RRVKZ4Lfxajcoq77azJ1rjptJ2CSzfNu2BcdVelJpIOijT4DEK68Dm4MMxVmHNUqyjlYi3Sj8COtpnqTvDcjCUuT9a+jkK28LUxT5bPqh36rhMFWyxBwqerlmmIOscghymqAi0Dn8Ys9GZNvG6EUxUIF4BJ4B9wmnKXVKbFzhgTuckWdRNOk54bcPxFKcG8RE0D0WP+3gbu+H7YkBYeall/mjIj0dqDyc+EzoR8EjeYGZKRaPA3CsXWdHk4ZEdk1scUVVljNGGL/eR/s1Yn191pbZMxRaH44STbivjXfL1LIOcTA7VezNpx0xKWaSXNfKPBEXSs4M0YGylqIbcRjTbFo142luS2bAmm3gL5fvb4SrRcAaXZL+AdJqihvt73lUj4x1R2rPEfQrtHpG82lShf0t+JK0YeAfLNWZmNiMwc0aiVf0IcUJi92bNdFw9oPOCMblWU0N3ea488NA5SYQJg==

提取码:
123456

*有效期至: 2022/12/24 09:54:02 GMT+08:00


- 预处理模型

预处理模型采用keras中的VGG16

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=ZooFr93nESyM+SLxMFyJV9FzVnc8iEcIdAadkWaJmJKff/6hQoZtfsCzUSAdbnW1TPNhG31hUTB1QDRVDvLJaFQKI4SeLCM4zHCdPqKSEncxBAh/Wu41B3ynGIhGw6Vz9T4eXV6tnq8KCji47wT3aodqNWYhNEUAYMr6NDMvDJ9zJ2FAoNusJqmE3Iu9aCb1ytBdpJRq7IdqM/TlkKt5WGbTQJ73TDQYraemZyG4UXpxuZSjtIKAgw1Yz/+5tXwaFT3kACIbololz3ZnpxbElXmit/Zt34Srb+lb/fmfZxJ5DWPd2mGoBcbiVuBAwPHAshQH59mR33aK/Q/qVes3KC2GKEvZRtwPnjRGGj2X/GueMyzGfzHzY86N+AL3nSqS3VmzwR4BNLnpattQEy5a0UnFqiua0xq/3K32GcFb541jjmo4fNc498RHUzjjs53Si8bCuw5MjG1ufGgPcKv0C3u8+5M3GLtQlql/ltfARcl3LNtKrNnSylhgfwIPeBFgFF6LwEputFxO7lcf/DueTKLQiKLniAUK1WtWz2UB9PyD20SE4gW+2ffRN9Z7MQk+NlvHU7KewzTDMP31TMDfvDpfMtwAPbEilgWjyoK5XIjtT65Ax8fWLEBDs77LsOzMFb/v6tQlEjyWuk8WiH/F0S4xwD21xJ3SLFdoFvF5Nrk=

提取码:
123456

*有效期至: 2022/12/24 09:57:42 GMT+08:00



## 脚本和示例代码

```
shap-master
└─
  ├─shap              原项目的迁移后的文件夹
  ├─LICENSE
  ├─modelzoo_level.txt
  ├─README.md
  ├─requirements.txt
  ├─train.py          代码运行脚本
```




## 训练过程

### GPU结果（V100）
运行时间：30s

![输入图片说明](https://images.gitee.com/uploads/images/2021/1122/151612_64304d2b_9994355.png "屏幕截图.png")



### NPU结果
运行时间：160s

![输入图片说明](https://images.gitee.com/uploads/images/2021/1122/151622_5c074b74_9994355.png "屏幕截图.png")


### 精度分析
没有具体指标衡量，人眼观察对比基本一致，精度达标。

### 性能分析
|         | NPU   | GPU(单卡V100) |
| --------| ----- | ------------- |
| s/epoch |  160  |      30       |
