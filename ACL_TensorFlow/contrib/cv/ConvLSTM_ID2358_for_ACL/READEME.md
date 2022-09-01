**描述（Description）：基于TensorFlow框架的卷积长短期记忆实现视觉里程计训练代码** 

<h2 id="概述.md">概述</h2>

ConvLSTM最早由香港科技大学的团队提出，解决序列图片的时空预测问题。本网络的ConvLSTM结构用于处理车载摄像头序列图片，实现一个视觉里程计。

- 参考论文：

    [https://arxiv.org/abs/1709.08429](Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting)


- 参考实现：
  https://github.com/giserh/ConvLSTM-2
  https://github.com/Kallaf/Visual-Odometry/blob/master/VisualOdometry.ipynb


- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/ConvLSTM_ID2358_for_TensorFlow



<h2 id="概述.md">原始模型</h2>

obs地址：obs://convlstm/GPU/mini_dataset/GPU_output7/



步骤一:将model300转化成Convlstm_frozen_model.pb文件
通过代码ckpt_to_pb.py将ckpt转成pb



<h2 id="概述.md">pb模型</h2>

```
Convlstm_frozen_model.pb
```
obs地址：obs://convlstm/GPU/mini_dataset/



<h2 id="概述.md">om模型</h2>

转Convlstm_frozen_model.pb到Convlstm_OM86.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/home/HwHiAiUser/AscendProjects/Convlstm/pb_model/Convlstm_frozen_model.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/Convlstm/Convlstm_OM86
--soc_version=Ascend910 --input_shape="input/Placeholder:32,1,384,1280,6" --log=info --out_nodes="Wx_plus_b/xw_plus_b:0"
```

成功转化成Convlstm_OM86.om

Convlstm_OM86.om的obs地址：obs://convlstm/GPU/mini_dataset/



<h2 id="概述.md">数据集转换bin以及利用acllite工具推理</h2>

首先需要下载acl相关工具：
https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/6_other/colorization_picture
利用代码colorize.py进行前处理，om模型调用，后处理，生成txtcsv结果。
预测的txtcsv文件夹路径：obs://convlstm/GPU/mini_dataset/推理txtcsv/




<h2 id="概述.md">结果计算</h2>

利用evo工具进行推理txtcsv文件夹中的数据文件计算

计算参考，先cd到所在文件夹，再执行：

```
evo_ape kitti  output_12D_file.txt  estimated_12D_file.txt  -r full -va --plot --plot_mode xz --save_results results/ConvLSTM.zip

```








