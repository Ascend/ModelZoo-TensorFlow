## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Retrieval** 

**版本（Version）：1.1**

**修改时间（Modified） ：2022.05.22**

**大小（Size）：2.5M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于DELF(DEep Local Features)深度局部特征的图像检索**

## 概述
谷歌这篇paper所提出的方法DELF(DEep Local Features)是一种图像检索方法，可取代图像检索中其他的关键点检测和表达方法，获得更为准确的特征匹配和几何验证.

- 参考论文：[Hyeonwoo Noh, Andre Araujo, Jack Sim, Tobias Weyand, Bohyung Han. "Large-Scale Image Retrieval with Attentive Deep Local Features" arXiv:1612.06321](https://arxiv.org/abs/1612.06321#)

- 参考实现：[https://github.com/tensorflow/models/tree/master/research/delf](https://github.com/tensorflow/models/tree/master/research/delf)

- 适配昇腾 AI 处理器的实现：[https://gitee.com/huawei-fighter/ModelZoo-TensorFlow](https://gitee.com/huawei-fighter/ModelZoo-TensorFlow)

- 通过Git获取对应commit_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置

- 训练数据集（google landmarks v2数据集）：
   - 数据集类型：GLD-v2-clean
   - 样本数量：1,580,470
   - labels数量：81,313
- 测试数据集
   - query list：70 images
   - index list: 4993 images
   - ground truth：gnd_roxford5k.mat
- 训练超参
   - Train Batch size: 32
   - Max iters: 300000
   - Report interval: 500
   - Eval interval: 10000
   - Save interval: 25000
   - Initial lr: 0.01
   
## 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度
训练脚本开启混合精度.
npu session开启混合精度:
```
# Setup session
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True

## use NpuOptimizer
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

## set mix precision
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))
```
npu optimization增加loss scale
```
learning_rate = _learning_rate_schedule(global_step, max_steps, FLAGS.initial_lr)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9,
                                       use_nesterov=True,
                                       use_locking=True)
## added for enabling loss scale and mix precision
loss_scale_manager = FixedLossScaleManager(loss_scale=100, enable_overflow_check=False)
optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
```

## 训练环境准备
本机配置ModelArts训练相关参数，参考文档[Pycharm Toolkit训练](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha003/moddevg/tfmigr/atlasmprtgma_13_0003.html)。
当前模型支持的CANN镜像如表1所示。

**表1** [镜像列表](https://gitee.com/ascend/modelzoo/wikis/%E5%9F%BA%E4%BA%8EModelArts%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B%E6%A1%88%E4%BE%8B/ModelArts%E5%B9%B3%E5%8F%B0%20CANN%20%E8%87%AA%E5%AE%9A%E4%B9%89%E9%95%9C%E5%83%8F%E5%88%97%E8%A1%A8)

| 镜像名称 | Modelarts-Pycharm | 配套CANN版本 |
| :-----| :----- | :----- |
| ascend-share/5.1.rc1.alpha003_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0317 | https://modelarts-pycharm-plugin.obs.cn-north-1.myhuaweicloud.com/Pycharm-ToolKit-3.1.0.zip | 5.1 |

## 快速上手
1. 数据集准备: 下载cleaned google landmarks v2原始图像，用build_dataset.sh进行数据预处理，生成tf records格式的train和test数据集.
2. 在训练脚本中指定数据集路径data_path，直接训练即可. 如果采用ModelArts训练，参数data_path就是存在OBS的数据集路径.

## 模型训练
1. 从源码地址git clone到本地.
2. 训练脚本入口为modelarts_entry.py，配置训练数据集路径data_url和模型输出路径train_url.
3. modelarts_entry.py会自动调用npu_train.sh脚本.

| 参数 | 说明 |
| :-----| :----- |
| data_url | 数据集的OBS路径，对应训练脚本的data_path参数 |
| train_url | 模型与日志输出路径，对应训练脚本的output_path参数 |

3. npu_train.sh训练调用的参数为：
```
python3.7 ${code_dir}/train.py \
  --data_path=${dataset_path} \
  --output_path=${output_path} \
  --dataset_version=gld_v2_clean \
  --batch_size=32 \
  --max_iters=500000 \
  --report_interval=500 \
  --eval_interval=10000 \
  --save_interval=25000
```

## 模型推理
1. 测试脚本入口为npu_eval.sh

| 参数 | 说明 |
| :-----| :----- |
| data_path | 测试数据集,以及checkpoint的OBS主目录 |
| output_path | 输出日志的路径 |

2. npu_eval.sh训练调用的参数如下:
```
python3.7 ${code_dir}/eval.py \
  --data_path=${dataset_path} \
  --output_path=${output_path}
```
3. 结果checkpoint的OBS路径：obs://delf-training/eval_inputs/best_ckpts

## 脚本和示例代码
```
train.py               // train主程序 
npu_train.sh           // npu训练执行脚本
eval.py                // 测试主程序
npu_eval.sh            // npu测试执行脚本
build_image_dataset.py // 由原始数据创建训练用的tf records
build_test_dataset.py  // 由原始数据创建测试用的tf records
build_dataset.sh       // 创建数据集的shell脚本 
modelarts_entry.py     // modelarts执行训练入口程序
```

## 脚本参数
```
--batch_size                   训练的batch size
--max_iters                    训练的总steps数
--report_interval              每训练多少个step，打印耗时
--eval_interval                每训练多少个step，测试数据集
--save_interval                每训练多少个step，保存更优模型
--initial_lr                   网络基准learning rate
```

## 训练过程
1. 通过“模型训练”中的训练指令启动npu训练。
2. 训练脚本的模型存储OBS路径为output_path，训练过程中产生的log以及模型文件同步产生于output_path路径下。
```
I0521 23:13:25.118650 281472986218864 train.py:388] Train global steps 6500-6999:cost=60.840359926223755	desc_loss=10.8125	attn_loss=10.859375	reconstruction_loss=0.20434211194515228
I0521 23:14:25.174391 281472986218864 train.py:388] Train global steps 7000-7499:cost=59.826475381851196	desc_loss=10.5078125	attn_loss=10.6171875	reconstruction_loss=0.22455056011676788
I0521 23:15:24.513339 281472986218864 train.py:388] Train global steps 7500-7999:cost=59.10019540786743	desc_loss=10.9609375	attn_loss=10.9375	reconstruction_loss=0.26061391830444336
I0521 23:16:23.720234 281472986218864 train.py:388] Train global steps 8000-8499:cost=58.96684455871582	desc_loss=10.8515625	attn_loss=10.984375	reconstruction_loss=0.28897392749786377
I0521 23:17:23.630913 281472986218864 train.py:388] Train global steps 8500-8999:cost=59.678821325302124	desc_loss=10.5234375	attn_loss=10.7265625	reconstruction_loss=0.2510737180709839
I0521 23:18:23.498849 281472986218864 train.py:388] Train global steps 9000-9499:cost=59.63196301460266	desc_loss=10.5390625	attn_loss=10.703125	reconstruction_loss=0.29234328866004944
I0521 23:19:22.753770 281472986218864 train.py:388] Train global steps 9500-9999:cost=59.02104568481445	desc_loss=10.703125	attn_loss=10.7578125	reconstruction_loss=0.3136596977710724
I0521 23:19:22.754767 281472986218864 train.py:395] Eval start in global step:9999
2022-05-21 23:19:23.991188: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:777] The model has been compiled on the Ascend AI processor, current graph id is: 21
I0521 23:21:28.654571 281472986218864 train.py:411] Eval end in global step:9999, cost=125.89948081970215, desc_loss=10.542526109313965, desc_acc=0.0072, attn_loss=10.465998400878906, attn_acc=0.00725
```

## 推理/验证过程

1. 通过“模型推理”中的测试指令启动测试。
2. 针对该工程训练出的checkpoint进行推理测试。
3. 推理脚本的输入参数data_path配置为obs://delf-training/eval_inputs/，利用该路径下best_ckpts最新的.ckpt文件会进行推理。
4. 测试结束后会打印验证集的图像检索map，如下所示。
```
I0522 20:20:27.354708 281473293996400 saver.py:1284] Restoring parameters from /home/ma-user/modelarts/inputs/data_url_0/best_ckpts/model.ckpt-499999
2022-05-22 20:20:28.371955: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:777] The model has been compiled on the Ascend AI processor, current graph id is: 11
I0522 20:20:32.182304 281473293996400 eval.py:151] Restore from ckpt:/home/ma-user/modelarts/inputs/data_url_0/best_ckpts/model.ckpt-499999
Reading list of query images and boxes from dataset file...
Inference for 4993 image files in database.
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/DELF_ID2024_for_TensorFlow/eval.py:169: 

1it [00:18, 18.42s/it]
4it [00:18, 12.91s/it]
7it [00:18,  9.05s/it]
....
4968it [03:13, 28.88it/s]
4971it [03:13, 28.07it/s]
4974it [03:13, 27.81it/s]
4977it [03:13, 28.39it/s]
4980it [03:13, 28.59it/s]
4983it [03:13, 28.10it/s]
4986it [03:13, 28.22it/s]
4989it [03:13, 28.03it/s]
4993it [03:13, 28.84it/s]
4993it [03:13, 25.74it/s]
Inference for 70 image files for queries.

0it [00:00, ?it/s]
1it [00:00,  2.21it/s]
2it [00:00,  2.77it/s]
3it [00:00,  3.39it/s]
4it [00:00,  3.99it/s]
5it [00:01,  4.61it/s]
6it [00:01,  5.11it/s]
7it [00:01,  5.55it/s]
8it [00:01,  5.90it/s]
......
70it [00:10,  6.71it/s]
I0522 20:23:57.981321 281473293996400 eval.py:205] Eval cost in 205.72006487846375 seconds, MAP=0.8571428571428571
```

## NPU复现结果

1. 训练性能：每训练500个step，平均耗时为82s(GPU:79s)
2. 训练精度：loss收敛至0.62(GPU:0.58), accuracy收敛至0.72905(GPU:0.7813)

## 模型固化并转化为om文件
1. sess.run模式下保存graph pb. data_url为obs://delf-training/npu_results/best_ckpts/. train_url为固化pb的输出目录.
```
## [Freeze Graph]
shell_cmd = ("bash %s/npu_freeze.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))
```

2. 固化graph的输入是模型保存的**checkpoint路径**，存档OBS为obs://delf-training/npu_results/best_ckpts/model.ckpt-499999.index
. **生成的pb文件**地址是存档OBS路径：obs://delf-training/npu_results/pb_model/delf_model.pb.
3. 使用ATC模型转换工具，将上面步骤得到的pb模型转换成om离线模型，即可用于在昇腾AI处理器进行离线推理.
4. ATC环境搭建: 参见《CANN软件安装指南》 进行开发环境搭建，并确保开发套件包Ascend-cann-toolkit安装完成。该场景下ATC工具安装在“Ascend-cann-toolkit安装目录/ascend-toolkit/{version}/{arch}-linux/atc/bin”下。
本次转化时soc_version从文件/usr/local/Ascend/ascend-toolkit/latest/atc/data/platform_config/Ascend910A.ini中，得知--soc_version=Ascend910A 
5. 执行ATC转换脚本即可生成om文件: **生成的om文件**地址是存档OBS路径: obs://delf-training/npu_results/pb_model/tf_delf_model.om
  
```
atc --model=/home/TestUser08/work_xiangd/ModelZoo-TensorFlow/TensorFlow/contrib/cv/delf/DELF_ID2024_for_TensorFlow/npu_results/pb_model/delf_model.pb --framework=3 --output=/home/TestUser08/work_xiangd/ModelZoo-TensorFlow/TensorFlow/contrib/cv/delf/DELF_ID2024_for_TensorFlow/npu_results/pb_model/tf_delf_model --soc_version=Ascend910A
```