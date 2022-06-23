# MixMatch - A Holistic Approach to Semi-Supervised Learning

参考文献: "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" 

参考实现：[https://github.com/google-research/mixmatch](https://github.com/google-research/mixmatch)
## 基本信息


- 框架: TensorFlow 1.15.0


- 处理器: 昇腾910


- 编译器： Python 3.7


- 描述: 基于Tensorflow框架的mixmatch方法的实现，此方法仅用少量的标记数据，就使半监督学习的预测精度逼近监督学习。

## 默认配置

- Dataset: cifar10.1@250-5000
- arch: resnet
- batch: 64                         
- beta: 0.5
- ema: 0.999
- filters: 32
- lr: 0.002
- nclass: 10
- repeat: 4
- scales: 3
- w_match: 100.0
- wd: 0.02

## 下载数据集

	```bash

	export ML_DATA="path to where you want the datasets saved"
	# Download datasets
	CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
    # Create semi-supervised subsets
    for seed in 1 2 3 4 5; do
        for size in 250 500 1000 2000 4000; do      
        	CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord 
    	done
    done

## 运行
### 运行环境
pip install -r requirements.txt


- absl-py
- easydict
- cython
- numpy
- tqdm

### 运行示例

For example, training a mixmatch with 32 filters on cifar10 shuffled with `seed=3`, 250 labeled samples and 5000
validation samples:
	
	CUDA_VISIBLE_DEVICES=0 python mixmatch.py --filters=32 --dataset=cifar10.3@250-5000 --w_match=75 --beta=0.75


## 训练过程
### 训练路径设置：

	# 在ModelArts容器创建数据存放目录
	data_dir = "/cache/dataset"
	os.makedirs(data_dir)

	# OBS数据拷贝到ModelArts容器内
	mox.file.copy_parallel(FLAGS.data_url, data_dir)
	# 在ModelArts容器创建训练输出目录
	model_dir = "/cache/result"
	os.makedirs(model_dir)

	# 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
    mox.file.copy_parallel(model_dir, FLAGS.train_url)
    mox.file.copy_parallel('/var/log/npu/', FLAGS.train_url)
 

### 训练精度设置

	# fp_32
	custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
	custom_op.name = "NpuOptimizer"
	custom_op.parameter_map["dynamic_input"].b = True
	custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
	custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
	config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
	config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    #loss scale
	loss_manager = loss_xe + w_match * loss_l2u
	train_op = create_optimizer(loss_manager, lr, "adam")
        if FLAGS.use_fp16 and (FLAGS.loss_scale not in [None, -1]):
            opt_tmp = train_op
            if FLAGS.loss_scale == 0:
                loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                                       decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            elif FLAGS.loss_scale >= 1:
                loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.loss_scale)
            else:
                raise ValueError("Invalid loss scale: %d" % FLAGS.loss_scale)

            opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
        train_op = opt.minimize(loss_manager, colocate_gradients_with_ops=True)

### 训练精度结果
- npu测试精度

--filters=32 --dataset=cifar10.1@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  87.32  86.87

--filters=32 --dataset=cifar10.2@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  87.44  87.01

--filters=32 --dataset=cifar10.3@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  87.92  87.35

- gpu测试精度

--filters=32 --dataset=cifar10.1@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  88.52  87.90
--filters=32 --dataset=cifar10.2@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  88.36  87.39
--filters=32 --dataset=cifar10.3@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  87.34  86.52


- 论文中的精度要求
  cifar10 250models error rate of 11.08±0.87%

## 代码及日志链接

obs://mixmatch-npu