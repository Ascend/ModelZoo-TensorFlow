推理情况表   
| 模型 |数据集| 输入shape | 输出0shape | 输出1shape | 推理时长 | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- | -- |
| realnvp | cifar10 | `3*32*32*12` | `3*32*32*12` | `1*12`| 81.11/12张 | 3.38 | 3.49 |


#### 1. 原始模型
[Download here](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=pAIGnhlI4xlSWI2+DJxVL224JiA0ijz4Uqh5BUNrCGtJC3n+wZjKsgf/mUym/zUwH4RdpaYAJasTkdls6/Yo46qI9EnhA9juLjmeG0T0ocbag5lhku2xKCIaIpiznMxd1V1RSqkfqvbekVp34lrtSPP5BjhjHCu8T/vklHELmqfKbMlayEOuvze0CqOJJma5BNKaaQtwI3V9LLGDgUQhyNjT4hU7x8FrIKiMHSKM2vG+ecWzlINBLp6JNl6bQvk0u4VOmWojIpA15pNPlwA1KqNfXI0lySltXtFMSJqhCEm1cQj2eoRQdYUIBz8NT6z2OCV5ojX0csfPRfLjknLwpi3ccsYjjfnJRTtXR4UFOn9+R6vesn2BDOX5I3oR+BKwKm4KsOj/xA2TiWOTrBRxu8gqfP4v/oMr+UsZCOT49WgcAYJsbsbgaeAiQ6lfY36ShYWVPwTbydW/I0op98GSlKVP5qehqyvqTnKEZX5C2/Fdo5vJpbhbCZQnylkGjPJdr+jN78PTfmSMgn3zp6Sqh7sCR1Qc1attCsv3ElQazBE=). (pwd: 666666)  找到对应文件夹，下载对应的`ckpt`，并转成pb模型。

#### 2. 转om模型
##### 方法一：

执行atc转换命令：

```sh
atc --model=/home/HwHiAiUser/project/realnvp.pb --framework=3 --output=/home/HwHiAiUser/projects/realnvp_1609250494.4331021 --soc_version=Ascend310 --log=info
```

由于在生成pb文件时已指定输出节点，因此此处不需要声明输出。

##### 方法二：

可以直接找到realnvp_1609250494.4331021.om文件进行推理。[Download here](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=pAIGnhlI4xlSWI2+DJxVL224JiA0ijz4Uqh5BUNrCGtJC3n+wZjKsgf/mUym/zUwH4RdpaYAJasTkdls6/Yo46qI9EnhA9juLjmeG0T0ocbag5lhku2xKCIaIpiznMxd1V1RSqkfqvbekVp34lrtSPP5BjhjHCu8T/vklHELmqfKbMlayEOuvze0CqOJJma5BNKaaQtwI3V9LLGDgUQhyNjT4hU7x8FrIKiMHSKM2vG+ecWzlINBLp6JNl6bQvk0u4VOmWojIpA15pNPlwA1KqNfXI0lySltXtFMSJqhCEm1cQj2eoRQdYUIBz8NT6z2OCV5ojX0csfPRfLjknLwpi3ccsYjjfnJRTtXR4UFOn9+R6vesn2BDOX5I3oR+BKwKm4KsOj/xA2TiWOTrBRxu8gqfP4v/oMr+UsZCOT49WgcAYJsbsbgaeAiQ6lfY36ShYWVPwTbydW/I0op98GSlKVP5qehqyvqTnKEZX5C2/Fdo5vJpbhbCZQnylkGjPJdr+jN78PTfmSMgn3zp6Sqh7sCR1Qc1attCsv3ElQazBE=). (pwd: 666666)

#### 3. 编译msame推理工具

参考https://gitee.com/ascend/tools/tree/master/msame，编译出msame推理工具。


#### 4. 执行推理

##### 4.1 下载预处理后的cifat-10-test数据集

[Download here](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=pAIGnhlI4xlSWI2+DJxVL224JiA0ijz4Uqh5BUNrCGtJC3n+wZjKsgf/mUym/zUwH4RdpaYAJasTkdls6/Yo46qI9EnhA9juLjmeG0T0ocbag5lhku2xKCIaIpiznMxd1V1RSqkfqvbekVp34lrtSPP5BjhjHCu8T/vklHELmqfKbMlayEOuvze0CqOJJma5BNKaaQtwI3V9LLGDgUQhyNjT4hU7x8FrIKiMHSKM2vG+ecWzlINBLp6JNl6bQvk0u4VOmWojIpA15pNPlwA1KqNfXI0lySltXtFMSJqhCEm1cQj2eoRQdYUIBz8NT6z2OCV5ojX0csfPRfLjknLwpi3ccsYjjfnJRTtXR4UFOn9+R6vesn2BDOX5I3oR+BKwKm4KsOj/xA2TiWOTrBRxu8gqfP4v/oMr+UsZCOT49WgcAYJsbsbgaeAiQ6lfY36ShYWVPwTbydW/I0op98GSlKVP5qehqyvqTnKEZX5C2/Fdo5vJpbhbCZQnylkGjPJdr+jN78PTfmSMgn3zp6Sqh7sCR1Qc1attCsv3ElQazBE=). (pwd: 666666) 数据集在out_reshape文件夹中，已经按照12张图片为一组进行划分。

如果希望自行预处理数据，请执行cifar_cut_bin.py文件。

##### 4.2 执行推理
执行以下命令：

```
msame --model realnvp.om --input out_reshape --output output_folder --outfmt TXT
```

其中，msame定义到自己的msame文件，realnvp.om定义到自己的om文件路径，out_reshape是4.1中的数据集，output_folder请指定一个空文件夹用于存放结果。以下是一种常见的命令书写格式：

```
/home/HwHiAiUser/AscendProjects/resnext101-om/om/tools/msame/out/msame --model /home/HwHiAiUser/AscendProjects/rvp/realnvp_1609250494.4331021.om --input /home/HwHiAiUser/AscendProjects/realnvp/out_reshape/ --output /home/HwHiAiUser/AscendProjects/rvp/msame_out_reshape/ --outfmt TXT
```

如果出现环境变量错误，请自行根据服务器的路径进行CMakeFile文件的修改。例如，你可能需要设置以下命令：

```
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/20.2.alpha001/x86_64-linux
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/20.2.alpha001/x86_64-linux/acllib/lib64/stub
```

执行结果大致如下（你的结果应与此略有不同）：

```log
******************************
Test Start!
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] malloc buffer for mem , require size is 21309952
[INFO] malloc buffer for weight,  require size is 23986176
[INFO] load model /home/HwHiAiUser/modelzoo/realnvp_1609250494.4331021/device/realnvp_1609250494.4331021.om success
[INFO] create model description success
[INFO] create model output success
/home/HwHiAiUser/project/msame_out_reshape//20210406_203137
[INFO] start to process file:/home/HwHiAiUser/project/our_reshape//cifar_test0.bin
[INFO] model execute success
Inference time: 82.243ms
[INFO] output data success
[INFO] start to process file:/home/HwHiAiUser/project/our_reshape//cifar_test1.bin
[INFO] model execute success
Inference time: 81.084ms
[INFO] output data success
.............................................
[INFO] start to process file:/home/HwHiAiUser/project/our_reshape//cifar_test99.bin
[INFO] model execute success
Inference time: 81.122ms
[INFO] output data success
Inference average time : 81.11 ms
Inference average time without first time: 81.11 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

##### 4.3 精度验证

输出文件有两个（参考res_reshape文件夹），分别表示拟合后的图片数据（输出0）以及sum_det_jacobian的值（输出1）。

执行时请指定输出文件夹，文件夹路径中必须含有最后一个’/'，例如：

```
loss-check.py "../real-nvp-info/res_reshape/"
```

对输出文件执行loss-check.py文件即可获取新的精度信息。执行结果如下：

```
Avg Loss: 3.3797707058482445
```



