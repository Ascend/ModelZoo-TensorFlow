# SYNET
## 1.模型概述
Symnet结合属性-对象转换的对称性原理和群论公理，由耦合网络和解耦网络两个模块组成，提出了基于Relative Moving Distance(RMD)的识别方法，利用属性的变化而非属性本身去分类属性。在Attribute-Object Composition零样本学习任务上取得了重大改进。

- 参考论文：

    [Li, Yong-Lu, et al. "Symmetry and group in attribute-object compositions." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.](https://arxiv.org/abs/2004.00587) 

- 官方实现：

    [SymNet](https://github.com/DirtyHarryLYL/SymNet)
## 2.环境
基于昇腾310推理Ai1S环境，参考[快速创建离线推理Ai1S环境](https://gitee.com/ascend/modelzoo/wikis/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/%E5%BF%AB%E9%80%9F%E5%88%9B%E5%BB%BA%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86Ai1S%E7%8E%AF%E5%A2%83)
配置环境。
## 3.数据准备
用户可运行download_data.sh下载数据集并进行预处理。预处理后的数据集已上传至obs中，包括原始数据，ckpt，pb，om，bin文件。
obs路径：obs://cann-id1292-symnet/data/data.tar.gz。可以下载解压到项目根目录(SYMNET_ID1292_for_ACL)，目录结构：
```
SYMNET_ID1292_for_ACL
├── data                # 数据集  
├── data_bin.py         # 制作bin文件
├── evaluate_acc.py     # 精度评估
├── freeze_graph.py     # ckpt转pb
├── inference.sh        # msame推理
├── modelarts_entry.py  # modelarts训练拉起
├── train_full_1p.sh    # 执行freeze_graph.py
├── LICENSE
├── modelzoo_level.txt  
├── pb_om.sh            # pb转om
├── README.md
├── requirement.txt     # 环境依赖
└── utils               
```
## 4.CKPT转PB
在ModelArts平台，通过`modelarts_entry.py`拉起训练，执行`train_full_1p.sh`，运行`freeze_graph.py`将ckpt转pb。
pb模型已上传至obs，路径 obs://cann-id1292-symnet/data/pb/
## 5.PB转OM
执行`pb_om.sh`，使用atc转换pb模型为om模型，atc模型转换参考[ATC模型转换](https://support.huaweicloud.com/atctool-cann51RC1alpha2/atlasatc_16_0005.html) 
om模型已上传obs，路径 obs://cann-id1292-symnet/data/om/
```
atc --model=./data/pb/symnet_new.pb --framework=3 --output=./data/om/symnet --soc_version=Ascend310 --input_shape="Placeholder_2:1,512;test_attr_id:116;test_obj_id:116;Placeholder_6:1,12" --out_nodes="Mul_18:0;Softmax_3:0;Placeholder_6:0"
```
## 6.bin文件制作
运行`data_bin.py`，将测试集数据制作为bin文件。
bin文件已上传至obs，路径 obs://cann-id1292-symnet/data/bin_file/
```
python3 data_bin.py --data_url=./data --obj_pred=UT_obj_lr1e-3_test_ep260.pkl --bin_file=./data/bin_file/
```
## 7.msame离线推理
参考 [msame](https://gitee.com/ascend/tools/tree/master/msame) 配置msame工具。
可将msame生成工具（tools/msame/out下)复制至本项目文件夹下，或修改`inference.sh`中msame路径为工具路径。
自定义bin文件输入路径，推理结果输出路径，执行`inference.sh`，进行推理。
## 8.精度计算
执行`evaluate_acc.py`，评估推理结果精度。
```
python3 evaluate_acc.py --input=/path/to/msame/output/ --data_url=./data --obj_pred=UT_obj_lr1e-3_test_ep260.pkl
```
|   | 数据集 | EPOCH| 精度 |
|-------|------|------|------|
| 原文 | UT | <700 | T1:52.1 &nbsp; T2:67.8 &nbsp; T3:76.0 |
| GPU  | UT | 574 | T1:0.5116 &nbsp; T2:0.6719 &nbsp; T3:0.7616 |
| NPU | UT | 636 | T1:0.5007 &nbsp; T2:0.6684 &nbsp; T3:0.7571 |
| NPU离线推理 | UT | 636 | T1:0.4991 &nbsp; T2:0.6696  &nbsp; T3:0.7561|
