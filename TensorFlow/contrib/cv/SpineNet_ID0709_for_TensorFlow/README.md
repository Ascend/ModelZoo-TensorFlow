# **SpineNet_ID0709_for_TensorFlow**

## 目录

-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)

## 基本信息

-   发布者（Publisher）：Huawei
-   应用领域（Application Domain）： CV
-   版本（Version）：1.0
-   修改时间（Modified） ：2021.10.22
-   大小（Size）：6.76M
-   框架（Framework）：TensorFlow 1.15.0
-   模型格式（Model Format）：ckpt
-   精度（Precision）：Mixed
-   处理器（Processor）：昇腾910
-   应用级别（Categories）：Research
-   描述（Description）：执行目标检测，实现实时计算

<h2 id="概述">概述</h2>

SpineNet是一种具有尺度排列中间特征的骨干网络与采用NAS专为目标检测任务而学习的跨尺度连接特性的网络。


- 参考论文：

    https://arxiv.org/abs/1912.05027v3

- 参考实现：

   https://github.com/tensorflow/tpu/tree/master/models/official/detection.


- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/SpineNet_ID0709_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

- 数据集预处理（以COCO2017训练集为例，仅作为用户参考示例）：

  请参考“概述”->“参考实现”

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否       |
| 并行数据   | 否       |

<h2 id="训练环境准备">训练环境准备</h2>

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

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
   <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
   </td>
   </tr>
   </tbody>
   </table>


<h2 id="快速上手">快速上手</h2>

- 数据集准备

  OBS下载地址：（下载的数据集为处理完的tf数据集）
  https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=/CZ/LNkKUldlXXGMVCN8HTxJVObUoG4oc6RqspCnhIP+qM2mwpMirVowqL6Gxe03oSsawHuOfaWP+TTQZbD0++htS5NpaWDrFiw1hN6KDVd+bBQTDa5RrmJGiC5GOcIJrd639iFKUiqh3iRpqtKVhquuDgLI5w8BC47Bbzp1cnQialGZwM5KKVEJ1kW0xR3R7UzoiFEkv/9IhCOknBO74FTw9KDGG37Z/uKTAsXACBQfYzBxJ2LQznEKTwli8jj+ENP+padxklKmClQExE03PeSLJzBTPaY4tlJP3OfhLwyA+Lc/pSvFclSH2zse/FE6sTIy483mo3EYkKnaHTdfKEroGiotGgBq1NJQunuc9vBJVHIkBO0D0ztS5ZK5CG0xEWSvA2ssO+i7xW7AlmSIKUMifVp1U4SB7P9PLf919SPjdUHDNJThTo9AAMv/SYlh5fzS7J4GZu9iZoaV72bLsCUEp+X3bj5D+R8RqRhb4s5FVRIFVArkx+qb93v7Ltc0vbPp5cC8JTOHzWnSTuQaIhnw8/wwFTr/YqImV8J1sAUIX9Vkhw8Azeb/oF/7qMOtN7oZvPNXtXzEsRa6N7HMtw==

  提取码:
  123456

  *有效期至: 2023/04/24 15:31:41 GMT+08:00
  
  
  
## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 单卡性能训练。

    用户可以执行test/train_performance_1p.sh脚本执行少量step获取性能信息：

    ```
	cd test
	bash train_performance_1p.sh --data_path=数据集路径
	
    train_performance_1p.sh中调用的训练命令示例如下：
  	python3 official/detection/main.py \
		--model_dir=./result --use_tpu=False \
		--mode=train --eval_after_training=True \
		--config_file="official/detection/configs/spinenet/spinenet49S_retinanet.yaml" \
		--params_override="{ train: { total_steps : 10, train_batch_size : 4, train_file_pattern: ${data_path}/train-* }, eval: { val_json_file: ${data_path}/annotations/instances_val2017.json, eval_file_pattern: ${data_path}/val-* } }" 

    ```

  2. 单卡精度训练。

	用户可以执行test/train_full_1p.sh脚本执行少量step获取性能信息：

    ```
    cd test
    bash train_full_1p.sh --data_path=数据集路径   

    train_full_1p.sh中调用的训练命令示例如下：
	python3 official/detection/main.py \
		--model_dir=./result --use_tpu=False \
		--mode=train --eval_after_training=True \
		--config_file="official/detection/configs/spinenet/spinenet49S_retinanet.yaml" \
		--params_override="{ train: { total_steps : 231500, train_batch_size : 4, train_file_pattern: ${data_path}/train-* }, eval: { val_json_file: ${data_path}/annotations/instances_val2017.json, eval_file_pattern: ${data_path}/val-* } }" 

  3. 执行结果。    

  |精度指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |mAP|xxx|0.198|0.164|

  |性能指标项|论文发布|GPU实测|NPU实测|
  |---|---|---|---|
  |FPS|XXX|0.24 sec/batch|0.114 sec/batch|

<h2 id="高级参考">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                                 //代码说明文档
├── run_train_1p.sh                           //训练脚本
├── models
│    ├──official                    		  //官方
│    │    ├──detection                   	  //目标检测
```
