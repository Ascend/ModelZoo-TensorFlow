### 《Non-local Neural Network》

- [Non-local Neural Network](https://arxiv.org/pdf/1711.07971v1.pdf)

- [作者公开的源代码-caffe](https://github.com/facebookresearch/video-nonlocal-net)

- 传统的卷积操作一次只是处理的一个局部邻域（`local neighborhood`），没有考虑到其他位置的像素对其输出的影响。这篇文章呈现的non-local操作能够捕获远距离的像素之间的依赖关系（`long-range dependencies`）

- `non-local`公式如下：
  $$
  y_i = \frac{1}{C(x)}\sum_jf(x_i,x_j)g(x_j)
  $$
  以图像为例，$i, j$为像素位置，$f(·)$表示$i$和$j$之间的关联系数，也可以说是权重，$g(·)$表示像素j的信息，$C(x)$为归一化系数，整个公式表示遍历所有的点$j$，以$f$为权重，将信息$g$进行加权求和

- 输入和输出尺寸相同

- `non-local`对f和g的形式不敏感

- `Non-local  block`结构如下：
![输入图片说明](https://images.gitee.com/uploads/images/2021/0906/171846_d6f43fec_8550619.png "屏幕截图.png")


- 这个结构设置通道的数量为输入`x`通道数的一半，这样可以减少计算量，然后通过Wz的来让输出Z跟输出X通道数保持一致，这点参考了`ResNet`的`bottleneck`设计
- 这个结构实现起来不复杂，文章这个`block`添加到`ResNet50`中，分别添加1（`to res4`）、5（`3 to res4 and 2 to res3`）、10个（`to every residual block in res3 and res4`）`non-local block`

- 论文做的是视频分类，该代码做的实现的是在`MNIST`数据集上的分类，添加10个`non-local block`到`ResNet50`中

- 由于采用`ResNet50`，网络的输入尺寸为224x224，所以为了匹配尺寸，强行将28x28`pad`为224x224，纯粹是为了尺寸的匹配

## Requirements
运行NonLocalNet模型需安装以下依赖：
- tensorflow-gpu 1.15
- python3.7
    
## Dataset

 NonLocalNet 模型使用MNIST手写体数据集进行训练。

##Transfer learning

- 使用与GPU训练相同的数据集

- 从打印出来的结果来看训练是正常的，
- 模型修改
  通过使用npu自动迁移工具进行模型的迁移，详细过程请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgtool_13_0006.html)
- 配置启动文件`boot_modelarts.py`,启动训练时，需设置好`train_url` 和 `data_url` 两个路径，详情请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgma_13_0004.html) 。

## Reference
```
├── Network.py                               //用于non-local-net的网络模型
├── Non_local_Net_ModelArts.py               //在modelarts平台上训练的执行文件
├── no-local-net-tf_v100.py                  //在华为云平台上使用V100进行训练的执行文件
├── README.md                                  //代码说明文件
├── requirements.txt                           //模型依赖
├── PB_generate.py                           //模型固化脚本
├── LICENSE
```

本模型的超参数只有Epoch和batchsize，可以酌情设置，本实验设置Epoch=20,batchsize=128

## GPU(V100) Training 

- 执行文件中的 no-local-net-tf_v100.py, `python no-local-net-tf_v100.py --Epoch 20 --batchsize 128`
- 训练结束后的模型存储路径(OBS中)output/, log路径log/，训练脚本log中包括如下信息。


```
do nothing
[Modelarts Service Log]user: uid=1101(work) gid=1101(work) groups=1101(work)
[Modelarts Service Log]pwd: /home/work
[Modelarts Service Log]app_url: s3://no-local-net/no-local-net-tf/code/Non_Local_Net_ID0274_for_TensorFlow/
[Modelarts Service Log]boot_file: Non_Local_Net_ID0274_for_TensorFlow/no-local-net-tf.py
[Modelarts Service Log]log_url: /tmp/log/no-local-net-tf.log
[Modelarts Service Log]command: Non_Local_Net_ID0274_for_TensorFlow/no-local-net-tf.py --data_url=s3://no-local-net/dataset/ --train_url=s3://no-local-net/no-local-net-tf/output/V0025/ --num_gpus=1
[Modelarts Service Log]MODELARTS_IPOIB_DEVICE: 
[Modelarts Service Log]dependencies_file_dir: /home/work/user-job-dir/Non_Local_Net_ID0274_for_TensorFlow
[Modelarts Service Log][modelarts_create_log] modelarts-pipe found
[Modelarts Service Log]handle inputs of training job
INFO:root:Using MoXing-v2.0.0.rc2.b4eeb646-b4eeb646
INFO:root:Using OBS-Python-SDK-3.20.9.1
[ModelArts Service Log][INFO][2021/10/22 15:11:50]: env MA_INPUTS is not found, skip the inputs handler
INFO:root:Using MoXing-v2.0.0.rc2.b4eeb646-b4eeb646
INFO:root:Using OBS-Python-SDK-3.20.9.1
[ModelArts Service Log]2021-10-22 15:11:51,107 - modelarts-downloader.py[line:623] - INFO: Main: modelarts-downloader starting with Namespace(dst='./', recursive=True, skip_creating_dir=False, src='s3://no-local-net/no-local-net-tf/code/Non_Local_Net_ID0274_for_TensorFlow/', trace=False, type='common', verbose=False)
/home/work/user-job-dir
[Modelarts Service Log][modelarts_logger] modelarts-pipe found
INFO:root:Using MoXing-v2.0.0.rc2.b4eeb646-b4eeb646
INFO:root:Using OBS-Python-SDK-3.20.9.1
WARNING:tensorflow:From Non_Local_Net_ID0274_for_TensorFlow/no-local-net-tf.py:25: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From /home/work/anaconda/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
..........   .......................
Learning start...
2021-10-22 15:12:18.463080: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
Epoch: 1	Loss:0.136091576	Accuarcy: 95.99%	 sec:564.1334021091461
Epoch: 2	Loss:0.015844521	Accuarcy: 99.69%	 sec:554.599419593811
Epoch: 3	Loss:0.026863999	Accuarcy: 99.44%	 sec:555.3349287509918
Epoch: 4	Loss:0.015781297	Accuarcy: 99.66%	 sec:555.1155166625977
Epoch: 5	Loss:0.031282460	Accuarcy: 99.34%	 sec:555.0727603435516
Epoch: 6	Loss:0.015345256	Accuarcy: 99.71%	 sec:554.8869705200195
Epoch: 7	Loss:0.016597942	Accuarcy: 99.71%	 sec:553.523374080658
Epoch: 8	Loss:0.017079884	Accuarcy: 99.68%	 sec:554.426707983017
Epoch: 9	Loss:0.012021461	Accuarcy: 99.78%	 sec:555.364844083786
Epoch: 10	Loss:0.013407425	Accuarcy: 99.77%	 sec:555.3845937252045
Epoch: 11	Loss:0.018451432	Accuarcy: 99.73%	 sec:555.5793664455414
Epoch: 12	Loss:0.023705571	Accuarcy: 99.71%	 sec:555.5929570198059
Epoch: 13	Loss:0.011928293	Accuarcy: 99.78%	 sec:554.9410705566406
Epoch: 14	Loss:0.010916438	Accuarcy: 99.81%	 sec:555.1003584861755
Epoch: 15	Loss:0.011399707	Accuarcy: 99.81%	 sec:555.2023286819458
Epoch: 16	Loss:0.018137812	Accuarcy: 99.72%	 sec:553.4322748184204
Epoch: 17	Loss:0.009838819	Accuarcy: 99.82%	 sec:554.387039899826
Epoch: 18	Loss:0.008480139	Accuarcy: 99.84%	 sec:555.4488346576691
Epoch: 19	Loss:0.010529141	Accuarcy: 99.86%	 sec:555.8053321838379
Epoch: 20	Loss:0.019199413	Accuarcy: 99.80%	 sec:555.1669955253601
Learning finished!
```


## NPU Training 

- 执行迁移后文件Non_local_Net_ModelArts.py, `python Non_local_Net_ModelArts.py --Epoch 20 --batchsize 128`
- 训练结束后模型存储路径(OBS中)output/**/model, log路径log/，训练脚本log中包括如下信息。

```
do nothing
[Modelarts Service Log]user: uid=1101(work) gid=1101(work) groups=1101(work),1000(HwHiAiUser)
[Modelarts Service Log]pwd: /home/work
[Modelarts Service Log]app_url: s3://no-local-net/no-local-nets/code/Non_Local_Net_ID0274_for_TensorFlow/
[Modelarts Service Log]boot_file: Non_Local_Net_ID0274_for_TensorFlow/Non_local_Net.py
[Modelarts Service Log]log_url: /tmp/log/no-local-nets.log
[Modelarts Service Log]command: Non_Local_Net_ID0274_for_TensorFlow/Non_local_Net.py --data_url=s3://no-local-net/dataset/ --train_url=s3://no-local-net/no-local-nets/output/V0025/
[Modelarts Service Log]local_code_dir: 
[Modelarts Service Log]Training start at 2021-09-06-21:53:35
[Modelarts Service Log][modelarts_create_log] modelarts-pipe found
[ModelArts Service Log]modelarts-pipe: will create log file /tmp/log/no-local-nets.log
[Modelarts Service Log]handle inputs of training job
INFO:root:Using MoXing-v1.17.3-8aa951bc
INFO:root:Using OBS-Python-SDK-3.20.7
[ModelArts Service Log][INFO][2021/09/06 21:53:36]: env MA_INPUTS is not found, skip the inputs handler
ln: failed to create symbolic link '/home/work/modelarts/outputs': No such file or directory
INFO:root:Using MoXing-v1.17.3-8aa951bc
INFO:root:Using OBS-Python-SDK-3.20.7
[ModelArts Service Log]2021-09-06 21:53:37,019 - modelarts-downloader.py[line:623] - INFO: Main: modelarts-downloader starting with Namespace(dst='./', recursive=True, skip_creating_dir=False, src='s3://no-local-net/no-local-nets/code/Non_Local_Net_ID0274_for_TensorFlow/', trace=False, type='common', verbose=False)
[Modelarts Service Log][modelarts_logger] modelarts-pipe found
[ModelArts Service Log]modelarts-pipe: will create log file /tmp/log/no-local-nets.log
[ModelArts Service Log]modelarts-pipe: will write log file /tmp/log/no-local-nets.log
[ModelArts Service Log]modelarts-pipe: param for max log length: 1073741824
[ModelArts Service Log]modelarts-pipe: param for whether exit on overflow: 0
[ModelArts Service Log]modelarts-pipe: total length: 24
/home/work/user-job-dir
[Modelarts Service Log][modelarts_logger] modelarts-pipe found
[ModelArts Service Log]modelarts-pipe: will create log file /tmp/log/no-local-nets.log
[ModelArts Service Log]modelarts-pipe: will write log file /tmp/log/no-local-nets.log
[ModelArts Service Log]modelarts-pipe: param for max log length: 1073741824
[ModelArts Service Log]modelarts-pipe: param for whether exit on overflow: 0
INFO:root:Using MoXing-v1.17.3-8aa951bc
INFO:root:Using OBS-Python-SDK-3.20.7
[Modelarts Service Log]2021-09-06 21:53:38,069 - WARNING - stdout log /var/log/batch-task/job299293a7/job-no-local-nets/stdout.log is not found
[Modelarts Service Log]2021-09-06 21:53:38,078 - INFO - Ascend Driver: Version=20.2.0
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - you are advised to use ASCEND_DEVICE_ID env instead of DEVICE_ID, as the DEVICE_ID env will be discarded in later versions
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - particularly, ${ASCEND_DEVICE_ID} == ${DEVICE_ID}, it's the logical device id
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - Davinci training command
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - ['/usr/bin/python', '/home/work/user-job-dir/Non_Local_Net_ID0274_for_TensorFlow/Non_local_Net.py', '--data_url=s3://no-local-net/dataset/', '--train_url=s3://no-local-net/no-local-nets/output/V0025/']
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - Wait for Rank table file ready
[Modelarts Service Log]2021-09-06 21:53:38,079 - INFO - Rank table file (K8S generated) is ready for read
[Modelarts Service Log]2021-09-06 21:53:38,080 - INFO - 
..............................
Epoch: 1	Loss:0.101984661	Accuarcy: 0.9679	global_step/sec:202.57287979125977
Epoch: 2	Loss:0.026374630	Accuarcy: 0.9939	global_step/sec:108.59884691238403
Epoch: 3	Loss:0.020736671	Accuarcy: 0.9956	global_step/sec:108.67354011535645
Epoch: 4	Loss:0.012909167	Accuarcy: 0.9976	global_step/sec:108.67826795578003
Epoch: 5	Loss:0.013714831	Accuarcy: 0.9975	global_step/sec:108.59573554992676
Epoch: 6	Loss:0.012684588	Accuarcy: 0.9975	global_step/sec:108.56736826896667
Epoch: 7	Loss:0.011056191	Accuarcy: 0.998	global_step/sec:108.52921724319458
Epoch: 8	Loss:0.011243986	Accuarcy: 0.9981	global_step/sec:108.52587747573853
Epoch: 9	Loss:0.015978905	Accuarcy: 0.9971	global_step/sec:108.59890866279602
Epoch: 10	Loss:0.015191577	Accuarcy: 0.9973	global_step/sec:108.63755893707275
Epoch: 11	Loss:0.011812831	Accuarcy: 0.9977	global_step/sec:108.6318130493164
Epoch: 12	Loss:0.019150641	Accuarcy: 0.9966	global_step/sec:108.651113986969
Epoch: 13	Loss:0.012249673	Accuarcy: 0.9978	global_step/sec:108.60406231880188
Epoch: 14	Loss:0.015603679	Accuarcy: 0.9973	global_step/sec:108.60229063034058
Epoch: 15	Loss:0.011850546	Accuarcy: 0.9977	global_step/sec:108.6053581237793
Epoch: 16	Loss:0.011400858	Accuarcy: 0.998	global_step/sec:108.58378028869629
Epoch: 17	Loss:0.013340743	Accuarcy: 0.9974	global_step/sec:108.60283136367798
Epoch: 18	Loss:0.012514029	Accuarcy: 0.9978	global_step/sec:108.59445405006409
Epoch: 19	Loss:0.014338116	Accuarcy: 0.9973	global_step/sec:108.60498261451721
Epoch: 20	Loss:0.010791019	Accuarcy: 0.9979	global_step/sec:108.59331059455872
WARNING:tensorflow:From /home/work/user-job-dir/Non_Local_Net_ID0274_for_TensorFlow/Non_local_Net.py:330: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
WARNING:tensorflow:From /home/work/user-job-dir/Non_Local_Net_ID0274_for_TensorFlow/Non_local_Net.py:330: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
......................
	
```

## Reasoning/Verification Process


- 当前只能针对该工程训练出的checkpoint进行推理测试。
- 先PB转换成OM，此时的测试命令：atc --model=/home/HwHiAiUser/AscendProjects/Non_loca_net/input/model4.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/Non_loca_net/output/model4_32 --soc_version=Ascend310  --input_shape="input:32,224,224,1" --input_format=NHWC --log=error --out_nodes="output:0"
- 针对转换后的OM文件进行测试，此时的测试命令：./msame --model "/home/HwHiAiUser/AscendProjects/Non_loca_net/output/model4_32.om"  --input  "/home/HwHiAiUser/AscendProjects/Non_loca_net/test/input/bin/1_batch_0_32.bin" --output "/home/HwHiAiUser/AscendProjects/Non_loca_net/test/output/"  --outfmt TXT --loop 5
- 测试结束后会打印验证集的推测结果，如下所示
```
7 3 7 0 1 0 0 7 0 7 1 0 7 0 6 8 0 7 6 3 0 0 0 0 0 0 6 7 0 0 3 2
gt
7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1
```


## Result
- 训练性能(优化速度) batchsize=20, epoch=20

|           | NonLocalNet   |
|-----------|---------------|
| GPU(V100) | 平均每个Epoch560秒 |
| NPU       | 平均每个Epoch560秒 |


说明：从上表可以看出NPU上的优化速度远远超过GPU(V100)。


- 训练精度(准确率)


|            | NonLocalNet(accuracy) |
|------------|-----------------------|
| github(参考) | 最高99.95%大致处于99.90%左右  |
| GPU(V100)  | 最高99.87% 大致在99.75%左右  |
| Npu        | 最高99.81% 大致在99.75%左右  |

说明：测试的数据集是MINST，百分比越高代表准确率越好，可以看出GPU(V100)的准确率和NPU的准确率一样，但是NPU的优化速度远快于V100.


- Reference:

  - CVPR2018-[https://arxiv.org/pdf/1711.07971v1.pdf](https://arxiv.org/pdf/1711.07971v1.pdf)
  - [https://github.com/titu1994/keras-non-local-nets](https://github.com/titu1994/keras-non-local-nets)
  - [https://github.com/nnUyi/Non-Local_Nets-Tensorflow](https://github.com/nnUyi/Non-Local_Nets-Tensorflow)
  - [https://github.com/Tencent/tencent-ml-images](https://github.com/Tencent/tencent-ml-images)