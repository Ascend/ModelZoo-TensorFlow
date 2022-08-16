执行结果打屏信息：

```
Iteration 58100: 0.15, 1.0
Iteration 58200: 0.15, 1.0
Iteration 58300: 0.15, 1.0
Iteration 58400: 0.15, 1.0
Iteration 58500: 0.15, 1.0
Validation results: 0.15, 0.18666667
Totally results: , 0.24025640706730705
Iteration 58600: 0.25, 1.0
Iteration 58700: 0.25, 1.0
Iteration 58800: 0.25, 1.0
Iteration 58900: 0.2, 1.0
Iteration 59000: 0.25, 1.0
Validation results: 0.35, 0.19666666
Totally results: , 0.23988700241355573
Iteration 59100: 0.05, 1.0
Iteration 59200: 0.05, 1.0
Iteration 59300: 0.05, 1.0
Iteration 59400: 0.05, 1.0
Iteration 59500: 0.05, 1.0
Validation results: 0.3, 0.26333332
Totally results: , 0.24008403029762396
Iteration 59600: 0.15, 1.0
Iteration 59700: 0.15, 1.0
Iteration 59800: 0.15, 1.0
Iteration 59900: 0.15, 1.0
```

执行任务OBS链接：

npu训练在裸机上执行，无obs链接,裸机输出已上传至obs：
```
obs://mamlnpu/logs_0804/logs/
```
裸机屏幕输出：
```
obs://mamlnpu/logs_0804/train_log_0804.log
```

数据集OBS链接：

```
obs://mamlnpu/data/
```

Ascend NPU INFO NOTICE：INFO, your task have used Ascend NPU, please check your result.

```
2022-08-10 16:18:39.920619: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 1
88
logs/miniimagenet1shot//cls_5.mbs_4.ubs_1.numstep5.updatelr0.01hidden32maxpoolbatchnorm/model34000
34000
Restoring model weights from logs/miniimagenet1shot//cls_5.mbs_4.ubs_1.numstep5.updatelr0.01hidden32maxpoolbatchnorm/model34000

```

Final：

gpu训练日志文件gpu_log.log，精度为23.7%；npu训练日志文件train_log0804.log，精度为24.0%。

|               | GPU   | NPU   |
| ------------- | ----- | ----- |
| mini imagenet | 23.7% | 24.0% |

启动命令：
```
python3 main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True
```