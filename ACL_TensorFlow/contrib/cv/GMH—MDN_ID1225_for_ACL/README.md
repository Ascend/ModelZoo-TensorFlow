# Memnet

### 1.关于项目

GMH-MDN：基于TensorFlow框架的基于多峰混合密度网络生成多个可行的 3D 姿态假设的网络。

论文链接为：[paper](https://arxiv.org/pdf/1904.05547.pdf)

原作者开源代码链接为：[code](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network)

tensorflow版本代码链接：[tensorflow-code](https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network)

### 2.关于依赖库
见Requirements.txt，需要安装tensorflow==1.15.0, opencv-python。

### 3.关于数据集
1. 模型预训练使用 [Human3.6M]数据集  ，需用户自行申请。因申请较慢，故可在[此处](https://github.com/MendyD/human36m) 下载

2. 数据集下载后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

### 4.关于训练

**GPU**训练：bash train.sh

**NPU**训练：由于是在modelart中进行训练的，直接将将boot file设置为modelarts_entry_acc.py；

若是在NPU的其他环境下运行，需要将modelarts.entry中的data_url和train_url，分别设置为数据集所在目录以及输出文件目录，在运行python modelarts_entry_acc.py。

#### ckpt转pb

在华为云的ModelArt，执行toolkit中的脚本freeze_graph.py文件，可以将ckpt转换为pb文件。

#### pb转om

使用ATC模型转换工具进行模型转换，可参考如下命令：

    atc --model=$HOME/module/mdm_5_prior.pb 
        --framework=3 --output=$HOME/module/out/tf_mdm5_v1 
        --soc_version=Ascend310 
        --input_shape="inputs/enc_in:1,32" 
        --output_type=FP32 
        --precision_mode=allow_mix_precision 
        --fusion_switch_file=$HOME/module/fusion_switch.cfg

fusion_switch.cfg 见 [一键式关闭融合规则](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha003/infacldevg/atctool/atlasatc_16_0077.html)

#### 离线推理
制作数据集 见 toolkit/data_generalize.py

在Ascend310上进行离线推理 命令如下：

    ./msame --model "tf_mdm5_v1.om" 
        --input "./AscendProjects/bin/" 
        --output "./AscendProjects/out"

可视化 见 toolkit/sample_visualization.py

### 5.性能&精度

#### 性能

| GPU V100   | Ascend 910  |
| :--------- | :---------- |
| 5.3 ms/batch | 5.5 ms/batch |

#### 训练精度

|       | GPU V100 | Ascend 910 |
| :-----| :------- | :--------- |
| mm    | 58.23   | 58.18     |

#### 推理精度
同训练精度
