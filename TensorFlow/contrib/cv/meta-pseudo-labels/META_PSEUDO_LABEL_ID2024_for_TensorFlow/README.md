## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Semi Supervised Learning** 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.10.29**

**大小（Size）：1.9M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的半监督网络训练代码**

## 概述
谷歌这篇paper所提出的方法Meta Pseudo Labels是一种半监督学习方法（a semi-supervised learning ），或者说是self-training方法。Meta Pseudo Labels可以看成是最简单的Pseudo Labels方法的改进.

- 参考论文：[Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le. "Meta Pseudo Labels" arXiv:2003.10580](https://arxiv.org/abs/2003.10580#)

- 参考实现：[https://github.com/google-research/google-research/tree/master/meta_pseudo_labels](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels)

- 适配昇腾 AI 处理器的实现：[https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/meta-pseudo-labels/META_PSEUDO_LABEL_ID2024_for_TensorFlow](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/meta-pseudo-labels/META_PSEUDO_LABEL_ID2024_for_TensorFlow)

- 通过Git获取对应commit_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置

- 有标签训练数据集预处理（cifar10-4k数据集）：
   - 图像的输入尺寸：32*32
   - 图像输入格式：data_batch_1.bin  data_batch_2.bin  data_batch_3.bin	data_batch_4.bin  data_batch_5.bin
   - 随机裁剪图像尺寸
   - 随机水平翻转图像
   - 根据cifar10数据集通用的平均值和标准偏差对输入图像进行归一化
- 无标签训练数据集预处理（cifar10-4k数据集）：
   - 图像的输入尺寸：32*32
   - 图像输入格式：data_batch_1.bin  data_batch_2.bin  data_batch_3.bin	data_batch_4.bin  data_batch_5.bin
   - 随机裁剪图像尺寸
   - 随机水平翻转图像
   - 数据增强：cutout和distort操作
   - 根据cifar10数据集通用的平均值和标准偏差对输入图像进行归一化
- 测试数据集预处理（cifar10-4k数据集）
   - 图像的输入尺寸：32*32
   - 图像输入格式：test_batch.bin
   - 根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化
- 训练超参
   - Train Batch size: 64
   - Eval Batch size: 64
   - Uda data: 7
   - Num train steps: 300000
   - Mpl teacher lr: 0.05
   - Mpl teacher lr warmup steps: 5000
   - Mpl student lr: 0.05
   - Mpl student lr warmup steps: 5000
   - Momentum: 0.9
   - LR scheduler: cosine
   - Optimizer: MomentumOptimizer
   - Weight decay: 0.0005
   - Label smoothing: 0.15
   - Uda temp: 0.7
   - Uda threshold: 0.6
   - Uda weight: 8
   
## 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度
训练脚本开启混合精度.
npu session开启混合精度:
```
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True

# npu used
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

# 设置混合精度
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
# 设置混合精度黑名单
custom_op = npu_tf_config.update_custom_op_adv(custom_op, action="modify_mixlist")
# 打开浮点溢出开关
#custom_op = npu_tf_config.update_custom_op_adv(custom_op, action='overflow')
## 设置算子融合规则
custom_op = npu_tf_config.update_custom_op_adv(custom_op, action='fusion_switch')
tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))
```
npu optimization增加loss scale
```
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9,
                                       use_nesterov=True,
                                       use_locking=True)
## added for enabling loss scale and mix precision
loss_scale_manager = FixedLossScaleManager(loss_scale=10, enable_overflow_check=False)
optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
```

## 训练环境准备
本机配置ModelArts训练相关参数，参考文档[Pycharm Toolkit训练](https://support.huaweicloud.com/tfmigr-cann503alpha2training/atlasma_13_0004.html)。
当前模型支持的CANN镜像如表1所示。

**表1** [镜像列表](https://gitee.com/ascend/modelzoo/wikis/%E5%9F%BA%E4%BA%8EModelArts%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B%E6%A1%88%E4%BE%8B/ModelArts%E5%B9%B3%E5%8F%B0%20CANN%20%E8%87%AA%E5%AE%9A%E4%B9%89%E9%95%9C%E5%83%8F%E5%88%97%E8%A1%A8)

| 镜像名称 | Modelarts-Pycharm | 配套CANN版本 |
| :-----| :----- | :----- |
| ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1019 | https://modelarts-pycharm-plugin.obs.cn-north-1.myhuaweicloud.com/Pycharm-ToolKit-3.0.zip | 5.0.3 |

## 快速上手
1. 数据集准备: 下载cifar-10-binary.tar.gz，文件包含：
    batches.meta  data_batch_1  data_batch_2  data_batch_3  data_batch_4  data_batch_5  readme.html  test_batch
1. 在训练脚本中指定数据集路径data_path，直接训练即可. 如果采用ModelArts训练，参数data_path就是存在OBS的数据集路径.

## 模型训练
1. 从源码地址git clone到本地.
2. 训练脚本入口为modelarts_entry.py，配置训练数据集路径data_url和模型输出路径train_url. 
3. modelarts_entry.py会自动调用npu_train.sh脚本

| 参数 | 说明 |
| :-----| :----- |
| data_url | 数据集的OBS路径，对应训练脚本的data_path参数 |
| train_url | 模型与日志输出路径，对应训练脚本的output_path参数 |

3. npu_train.sh训练调用的参数为：
```
python3.7 ${code_dir}/main.py --data_path=${dataset_path} --output_path=${output_path} \
  --task_mode="train" \
  --master="/root/projects/meta-pseudo-labels-tf1-gpu/results/worker" \
  --dataset_name="cifar10_4000_mpl" \
  --model_type="wrn-28-2" \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --nouse_augment \
  --alsologtostderr \
  --running_local_dev \
  --image_size=32 \
  --num_classes=10 \
  --log_every=50 \
  --save_every=100 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --uda_data=7 \
  --weight_decay=5e-4 \
  --num_train_steps=300000 \
  --augment_magnitude=16 \
  --batch_norm_batch_size=256 \
  --dense_dropout_rate=0.2 \
  --ema_decay=0.995 \
  --label_smoothing=0.15 \
  --mpl_teacher_lr=0.05 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --mpl_student_lr=0.05 \
  --mpl_student_lr_wait_steps=1000 \
  --mpl_student_lr_warmup_steps=5000 \
  --uda_steps=5000 \
  --uda_temp=0.7 \
  --uda_threshold=0.6 \
  --uda_weight=8
```

## 模型推理
1. 测试脚本入口为npu_eval.sh

| 参数 | 说明 |
| :-----| :----- |
| data_path | 数据集的OBS路径 |
| output_path | OBS中checkpoint文件所在目录 |

2. npu_eval.sh训练调用的参数如下，需要把task_mode改为"eval":
```
python3.7 ${code_dir}/main.py --data_path=${data_path} --output_path=${output_path} \
  --task_mode="eval" \
  --master="/root/projects/meta-pseudo-labels-tf1-gpu/results/worker" \
  --dataset_name="cifar10_4000_mpl" \
  --model_type="wrn-28-2" \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --nouse_augment \
  --alsologtostderr \
  --running_local_dev \
  --image_size=32 \
  --num_classes=10 \
  --log_every=50 \
  --save_every=100 \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --uda_data=7 \
  --weight_decay=5e-4 \
  --num_train_steps=300000 \
  --augment_magnitude=16 \
  --batch_norm_batch_size=256 \
  --dense_dropout_rate=0.2 \
  --ema_decay=0.995 \
  --label_smoothing=0.15 \
  --mpl_teacher_lr=0.05 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --mpl_student_lr=0.05 \
  --mpl_student_lr_wait_steps=1000 \
  --mpl_student_lr_warmup_steps=5000 \
  --uda_steps=5000 \
  --uda_temp=0.7 \
  --uda_threshold=0.6 \
  --uda_weight=8
```
3. 结果checkpoint的OBS路径：obs://meta-training/best_ckpts

## 脚本和示例代码
```
augment.py           // 数据增强的各种操作
common_utils.py      // 获取learning_rate、optimizer等工具函数
data_utils.py        // 数据预处理生成tf.data
flag_utils.py        // 读取脚本输入参数
modeling_utils.py    // 神经网络的基本操作，比如卷积、池化等
modeling.py          // 神经网络的定义
training_utils.py    // 训练的train_step和测试的eval_step定义
main.py              // 程序入口，含train_gpu和eval_gpu
gpu_train.sh         // gpu训练执行脚本，task_mode="train"
npu_train.sh         // npu训练执行脚本，task_mode="train"
npu_eval.sh          // 推理测试执行脚本，task_mode="eval"
modelarts_entry.py   // modelarts执行训练入口程序
precision_tool       // 精度调优工具
requirements.txt     // python依赖包
```

## 脚本参数
```
--log_every                    每训练多少个step，打印耗时
--save_every                   每训练多少个step，测试数据集，并保存更优模型
--train_batch_size             训练的batch size
--eval_batch_size              测试的batch size
--uda_data                     无标签数据集扩大的倍数
--weight_decay                 权重衰减
--num_train_steps              训练的总steps数
--augment_magnitude            增强操作的量级
--dense_dropout_rate           网络全连接层的dropout概率
--ema_decay                    student网络参数的指数移动平均值
--label_smoothing              label smooth系数
--mpl_teacher_lr               teacher网络的基准learning rate
--mpl_teacher_lr_warmup_steps  teacher网络warmup的steps
--mpl_student_lr               student网络的基准learning rate
--mpl_student_lr_wait_steps    student网络在wait steps时不更新网络参数
--mpl_student_lr_warmup_steps  student网络warmup的steps
--uda_steps                    无标签数据loss的权重在uda_steps时达到最大                     
--uda_temp                     无标签数据预测概率时的softmax温度参数
--uda_threshold                无标签数据预测概率大于门限值时，才计算交叉上的loss
```

## 训练过程
1. 通过“模型训练”中的训练指令启动npu训练。
2. 训练脚本的模型存储OBS路径为output_path，训练过程中产生的log以及模型文件同步产生于output_path路径下。
```
I1106 19:03:30.052761 281472925466992 main.py:193] Train total cost in global steps 274200-274249:2.82422137260437
I1106 19:03:32.947063 281472925466992 main.py:193] Train total cost in global steps 274250-274299:2.863443374633789
I1106 19:03:32.947435 281472925466992 main.py:201] Eval start in global step:274299
I1106 19:03:34.136594 281472925466992 main.py:229] Eval end in global step:274299, cost=1.1886203289031982
I1106 19:03:34.136839 281472925466992 main.py:230] Eval end in global step:274299, teacher_top1=0.9418512658227848, teacher_top5=0.9966376582278481, teacher_loss=0.8245617830300633
I1106 19:03:34.136934 281472925466992 main.py:231] Eval end in global step:274299, student_top1=0.9517405063291139, student_top5=0.9981210443037974, student_loss=0.814969219738924
I1106 19:03:34.137013 281472925466992 main.py:232] Eval end in global step:274299, eemmmaa_top1=0.9528283227848101, eemmmaa_top5=0.9982199367088608, eemmmaa_loss=0.8119282782832279
I1106 19:03:50.980582 281472925466992 main.py:245] Saved Model for student in file:/home/ma-user/modelarts/outputs/train_url_0/ckpts/model.ckpt-274299, accuracy:0.9517405063291139
I1106 19:03:54.368419 281472925466992 main.py:193] Train total cost in global steps 274300-274349:3.345759391784668
I1106 19:03:57.297451 281472925466992 main.py:193] Train total cost in global steps 274350-274399:2.900482416152954
I1106 19:03:57.298314 281472925466992 main.py:201] Eval start in global step:274399
I1106 19:03:58.591202 281472925466992 main.py:229] Eval end in global step:274399, cost=1.2926883697509766
I1106 19:03:58.591523 281472925466992 main.py:230] Eval end in global step:274399, teacher_top1=0.9458069620253164, teacher_top5=0.9958465189873418, teacher_loss=0.8270031892800633
I1106 19:03:58.591629 281472925466992 main.py:231] Eval end in global step:274399, student_top1=0.9496637658227848, student_top5=0.9984177215189873, student_loss=0.8121106111550633
I1106 19:03:58.591711 281472925466992 main.py:232] Eval end in global step:274399, eemmmaa_top1=0.9533227848101266, eemmmaa_top5=0.998318829113924, eemmmaa_loss=0.8090109523338608
I1106 19:04:01.496418 281472925466992 main.py:193] Train total cost in global steps 274400-274449:2.874518871307373
I1106 19:04:04.414882 281472925466992 main.py:193] Train total cost in global steps 274450-274499:2.8899054527282715
I1106 19:04:04.415580 281472925466992 main.py:201] Eval start in global step:274499
I1106 19:04:05.785323 281472925466992 main.py:229] Eval end in global step:274499, cost=1.3695611953735352
I1106 19:04:05.785618 281472925466992 main.py:230] Eval end in global step:274499, teacher_top1=0.9439280063291139, teacher_top5=0.9961431962025317, teacher_loss=0.8251118720332279
I1106 19:04:05.785714 281472925466992 main.py:231] Eval end in global step:274499, student_top1=0.9484770569620253, student_top5=0.9979232594936709, student_loss=0.8250006180775317
I1106 19:04:05.785794 281472925466992 main.py:232] Eval end in global step:274499, eemmmaa_top1=0.9544106012658228, eemmmaa_top5=0.9982199367088608, eemmmaa_loss=0.8096722952927216
I1106 19:04:21.389029 281472925466992 main.py:249] Saved Model for ema in file:/home/ma-user/modelarts/outputs/train_url_0/ema_ckpts/model.ckpt-274499, accuracy:0.9544106012658228
```

## 推理/验证过程

1. 通过“模型推理”中的测试指令启动测试。
2. 针对该工程训练出的checkpoint进行推理测试。
3. 推理脚本的输入参数output_path配置为checkpoint所在的文件夹路径，利用该路径下最新的.ckpt文件会进行推理。
4. 测试结束后会打印验证集的top1 accuracy和top5 accuracy，如下所示。
```
I1108 09:12:55.062407 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/stem/conv2d/Conv2D:0                                                     (64, 32, 32, 32)
I1108 09:12:55.079802 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_1/residual/add:0                                                   (64, 32, 32, 32)
I1108 09:12:55.096908 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_2/residual/add:0                                                   (64, 32, 32, 32)
I1108 09:12:55.114213 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_3/residual/add:0                                                   (64, 32, 32, 32)
I1108 09:12:55.133295 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_4/residual/add:0                                                   (64, 32, 32, 32)
I1108 09:12:55.153911 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_5/residual/add:0                                                   (64, 16, 16, 64)
I1108 09:12:55.171566 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_6/residual/add:0                                                   (64, 16, 16, 64)
I1108 09:12:55.188975 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_7/residual/add:0                                                   (64, 16, 16, 64)
I1108 09:12:55.206370 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_8/residual/add:0                                                   (64, 16, 16, 64)
I1108 09:12:55.226791 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_9/residual/add:0                                                   (64, 8, 8, 128)
I1108 09:12:55.244565 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_10/residual/add:0                                                  (64, 8, 8, 128)
I1108 09:12:55.261770 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_11/residual/add:0                                                  (64, 8, 8, 128)
I1108 09:12:55.279340 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/block_12/residual/add:0                                                  (64, 8, 8, 128)
I1108 09:12:55.285640 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/head/global_avg_pool:0                                                   (64, 128)
I1108 09:12:55.290395 281473413121488 modeling_utils.py:78] ema/model/wrn-28-2/head/dense/bias_add:0                                                    (64, 10)
INFO:tensorflow:Restoring parameters from /root/projects/meta-pseudo-labels-tf1-gpu/results/ema_ckpts/model.ckpt-274499  
I1108 09:12:55.411040 281473413121488 saver.py:1284] Restoring parameters from /root/projects/meta-pseudo-labels-tf1-gpu/results/ema_ckpts/model.ckpt-274499
I1108 09:12:55.449819 281473413121488 main.py:283] Restore from ema ckpt:/root/projects/meta-pseudo-labels-tf1-gpu/results/ema_ckpts/model.ckpt-274499
I1108 09:13:24.130415 281473413121488 main.py:299] Eval cost in 1.203519706726074 seconds; top1=0.9531106012658228; top5=0.9970110759493671
```

## NPU精度复现结果
与GPU对比结果总结如下：
* 完成GPU的复现，预测精度0.9531，超过网上任何用GPU实现版本的预测精度（网上最高精度为0.9467）。官方实现用TPU，无参考意义。

* 完成NPU的复现，预测精度0.9532，与GPU持平。训练和推理性能均是GPU的1.5倍以上。

详细数据如下：
1. 完成gpu复现
----三次重复实验
----测试集准确率为0.9531~0.9532
----训练集loss收敛至0.30
2. 完成gpu迁移
----Session改造
----关闭tf.device算子
----增加混合精度loss scale
----溢出算子保持原有精度
----dropout算子迁移
3. 完成npu功能打通
----ModelArts输入与输出配置
----浮点异常检测工具使用
----关闭SoftmaxArgMaxValueONNXFusionPass融合算子
----找到最佳loss scale初始值
----测试集准确率达标0.9532， 与GPU预测精度持平
----训练集loss收敛至0.2847
4. gpu训练性能
----每训练50个step，13.5686s
5. gpu推理性能
----测试10000个图片，2.08354s
6. npu训练性能
----每训练50个step，2.95s， 是GPU训练性能的4.6倍
7. npu推理性能
----测试10000个图片，1.21s， 性能提升1.71倍

## 模型固化并转化为om文件 
1. sess.run模式下保存graph pb. modelarts_entry.py打开freeze模式，注释掉训练入口那一行. 其中modelarts插件需设置Data Path in OBS为: /meta-training/best_ckpts/. OBS Path为固化pb的输出目录.
```
## [Training]
#shell_cmd = ("bash %s/npu_train.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))
## [Freeze Graph]
shell_cmd = ("bash %s/npu_freeze.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))
```

2. 固化graph的输入是模型保存的**checkpoint路径**，存档OBS为obs://meta-training/best_ckpts/model.ckpt-274499. **生成的pb文件**地址是存档OBS路径：obs://meta-training/pb_model/meta_pseudo_labels.pb
3. 使用ATC模型转换工具，将上面步骤得到的pb模型转换成om离线模型，即可用于在昇腾AI处理器进行离线推理.
4. ATC环境搭建. 参见《CANN软件安装指南》进行开发环境搭建，并确保开发套件包Ascend-cann-toolkit安装完成。该场景下ATC工具安装在“Ascend-cann-toolkit安装目录/ascend-toolkit/{version}/{arch}-linux/atc/bin”下。soc_version查看方法参见：https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0061.html
本次转化时soc_version从文件/usr/local/Ascend/ascend-toolkit/5.0.3.alpha003/atc/data/platform_config/Ascend910A.ini中，得知--soc_version=Ascend910A
5. 执行ATC转换脚本即可生成om文件: sh convert_atc.sh. **生成的om文件**地址是存档OBS路径: obs://meta-training/pb_model/tf_meta_pseudo_labels.om
```
export PATH=/root/anaconda3/envs/delf/bin:$PATH
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/5.0.3.alpha003/atc/lib64/stub
atc --model=/root/projects/meta-pseudo-labels-tf1-gpu/pb_model/meta_pseudo_labels.pb --framework=3 --output=/root/projects/meta-pseudo-labels-tf1-gpu/pb_model/tf_meta_pseudo_labels --soc_version=Ascend910A --input_shape="input:1,32,32,3"
```