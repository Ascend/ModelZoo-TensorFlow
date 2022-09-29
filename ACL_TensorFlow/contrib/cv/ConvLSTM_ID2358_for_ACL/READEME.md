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
pb文件共享地址：https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/ConvLSTM-ID2358/Convlstm_frozen_model.pb



<h2 id="概述.md">om模型</h2>

转Convlstm_frozen_model.pb到Convlstm_OM.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:
```
atc --model=/home/test_user07/Convlstm/READEME/Convlstm_frozen_model.pb --framework=3 --output=/home/test_user07/Convlstm/READEME/Convlstm_OM --soc_version=Ascend910  --input_shape="input/Placeholder:32,1,384,1280,6" --log=info --out_nodes="Wx_plus_b/xw_plus_b:0"  --precision_mode=force_fp32 --op_select_implmode=high_precision
```
成功转化成Convlstm_OM.om

om文件共享地址：https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/ConvLSTM-ID2358/Convlstm_OM86.om



<h2 id="概述.md">数据前处理+利用acllite工具推理+数据后处理</h2>

acllite是对当前开源社区样例中相关重复代码进行封装，为用户提供的一组简易公共接口。
首先需要下载acllite相关工具，用户需要解压acllite文件夹文件，或者去以下路径下载：

https://gitee.com/ascend/samples/tree/master/python/common

再利用代码Inference.py进行数据集前处理，om模型调用，数据后处理，生成结果存放在txtcsv文件夹:

```
python3 Inference.py
```

数据集文件共享地址，需要在Inference.py中替换datapath：https://modelzoo-atc-pb-om.obs.cn-north-4.myhuaweicloud.com/ConvLSTM-ID2358/dataset.tar

实际27机器上的路径：/home/test_user07/Convlstm/samples/python/level2_simple_inference/6_other/colorization_picture/src/

预测的txtcsv文件夹路径：obs://convlstm/GPU/mini_dataset/推理txtcsv/




<h2 id="概述.md">结果计算</h2>

安装evo工具：
```
pip install evo --upgrade --no-binary evo
```
利用evo工具对txtcsv文件夹中的数据进行计算，先cd到所在文件夹（其中output_12D_file.txt是真值文件，estimated_12D_file.txt是预测文件），执行：

```
evo_ape kitti  output_12D_file.txt  estimated_12D_file.txt  -r full -va --plot --plot_mode xz --save_results results/ConvLSTM.zip

```
推理计算后的均方根误差RMSE（Root Mean Square Error）为16.048021，对比GPU的结果16.047025，精度达标。







