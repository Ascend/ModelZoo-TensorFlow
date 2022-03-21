MIT License

Copyright (c) 2019 Tianwei Shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.  

 **基本信息** 

发布者（Publisher）：Huawei

应用领域（Application Domain）： CV

版本（Version）：1.1

修改时间（Modified） ：2021.12.27

大小（Size）：1913KB

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：ckpt

精度（Precision）：FP32

处理器（Processor）：昇腾910

应用级别（Categories）：Research

描述（Description）：基于TensorFlow框架的DeepMatchVO网络训练代码

 **概述** 
DeepMatchVO是一个视觉里程计系统，采用无监督的学习方法，获得位姿估计。
本文的主要贡献为加入传统算法作为loss信息
1. 加入对极几何误差
2. 弱位姿进行监督
3. 弥补光度误差

 **网络架构** 

![输入图片说明](architecture.png)

架构特点

1.采用两个网络 深度网络和为自网络

2.提供消融实验

3.传统信息融合进学习架构

 **相关参考** 
 **module source** : https://github.com/hlzz/DeepMatchVO  

 **默认配置** 
训练超参：-img_width=416 --img_height=128 --batch_size=4 --seq_length 3
--max_steps 300000 --save_freq 3000 --learning_rate 0.001 --num_scales 1 --match_num=100

 **支持特性** 
分布式训练：否

混合精度：是

数据并行：是

 **混合精度训练** ：影响较小

 **图融合** ：影响较大，需要关闭图融合

 **训练环境准备** 

硬件环境：NPU: 1*Ascend 910 CPU: 24*vCPUs 96GB

运行环境：ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1125

 **模型训练** 

1、数据集下载，按照论文提示方法下载，但是部分数据的生成作者没有公开，需要自行下载

2、执行命令 python3 train.py --dataset_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/genertate_wty'  --checkpoint_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/ckpt_pro' --img_width=416 --img_height=128 --batch_size=4 --seq_length 3 --max_steps 300000 --save_freq 3000 --learning_rate 0.001 --num_scales 1  --continue_train=False --match_num=100

3、测试命令--output_dir --ckpt_file需要自己更换

python test_kitti_pose.py --test_seq=10 --dataset_dir='/root/kitti/dataset/' --output_dir='/mnt/data/outputt/10/' --ckpt_file='/mnt/newckpt/model-90000' --seq_length=3 --concat_img_dir='/mnt/data/generatetestimage'
python kitti_eval/eval_pose.py --gtruth_dir='/root/DeepMatchVO-master/kitti_eval/pose_data/ground_truth/seq3/10/' --pred_dir='/mnt/data/outputt/10/'

 **训练打屏日志** 

NPU 910

Epoch: [13] [ 4421/ 4590] time: 0.5149
total/pixel/smooth loss: [0.263/0.073/0.000]

Epoch: [13] [ 4521/ 4590] time: 0.5151
total/pixel/smooth loss: [0.298/0.095/0.000]

Epoch: [14] [   31/ 4590] time: 0.5144
total/pixel/smooth loss: [0.451/0.166/0.000]

Epoch: [14] [  131/ 4590] time: 0.5148
total/pixel/smooth loss: [0.407/0.170/0.000]

Epoch: [14] [  231/ 4590] time: 0.5152
total/pixel/smooth loss: [0.378/0.133/0.000]

Epoch: [14] [  331/ 4590] time: 0.5149
total/pixel/smooth loss: [0.427/0.189/0.000]

 [*] Saving checkpoint step 60000 to /home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw...
INFO:tensorflow:/home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw/model-60000 is not in all_model_checkpoint_paths. Manually adding it.
I1216 01:41:30.187625 281473561675552 checkpoint_management.py:95] /home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw/model-60000 is not in all_model_checkpoint_paths. Manually adding it.

GPU Tesla V100

Epoch: [13] [ 1121/ 4590] time: 0.0871
total/pixel/smooth loss: [0.365/0.163/0.000]

Epoch: [13] [ 1221/ 4590] time: 0.0874
total/pixel/smooth loss: [0.388/0.185/0.000]

Epoch: [13] [ 1321/ 4590] time: 0.0873
total/pixel/smooth loss: [0.372/0.170/0.000]

Epoch: [13] [ 1421/ 4590] time: 0.0874
total/pixel/smooth loss: [0.323/0.126/0.000]

Epoch: [13] [ 1521/ 4590] time: 0.0871
total/pixel/smooth loss: [0.376/0.187/0.000]

Epoch: [13] [ 1621/ 4590] time: 0.0873
total/pixel/smooth loss: [0.355/0.182/0.000]

Epoch: [13] [ 1721/ 4590] time: 0.0873INFO:tensorflow:/mnt/newckpt/model-60000 is not in all_model_checkpoint_paths. Manually adding it.
I1116 17:18:26.870828 140652646106880 checkpoint_management.py:95] /mnt/newckpt/model-60000 is not in all_model_checkpoint_paths. Manually adding it.

 **训练结果** 

![输入图片说明](gpu_vs_npu.png)


