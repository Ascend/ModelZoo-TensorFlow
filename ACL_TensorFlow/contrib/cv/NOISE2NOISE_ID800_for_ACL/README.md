# 模型概述

详情请看**NOISE2NOISE_ID800_for_TensorFlow** README.md *概述*

#  数据集

- bsd300
- kodak
- ixi_valid.pkl

1、用户自行准备好数据集。

2、数据集的处理可以参考 **NOISE2NOISE_ID800_for_TensorFlow** README.md *概述*  "模型来源"

# pb模型

详情请看**DENSEDEPTH_ID806_for_TensorFlow** README.md *模型固化*

链接：https://pan.baidu.com/s/1V2dsBNQFWWE836FinouvOw 
提取码：4521

# om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:

**Noise2Noise 网络**：

```
atc --model=pb/test.pb --input_shape="input:1,3,512,768" --framework=3 --output=om/test --soc_version=Ascend910A --input_format=NCHW
```

**Noise2Noise MRI 网络**：

```
atc --model=pb/test_mri.pb --input_shape="input:1,255,255" --framework=3 --output=om/test_mri --soc_version=Ascend910A
```

具体参数使用方法请查看官方文档。

链接：https://pan.baidu.com/s/1WmF2wEwKxR_vTK8CJdoRPQ 
提取码：4521

# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 数据集转换bin

**Noise2Noise 网络**：

```
python img2bin.py --input=datasets/kodak --output=bin/kodak --noise=gaussian --width=768 --height=512
```

参数解释：

```
--input 	图片位置，文件夹或是单张图片
--output	bin文件位置，文件夹
--width		图片reshape：width
--height	图片reshape：height
--noise     添加噪声 gaussian or poisson
```

**Noise2Noise MRI 网络**：

```
python pkl2bin_mri.py --input=datasets/ixi_valid.pkl --output=/bin/ixi_valid --bs=1
```

## 推理

可参考如下命令 msame.sh：

**Noise2Noise 网络**：

```
msame --model=om/n2n_gaussian_h512_w768_910A.om --input=bin/kodak --output=result/bin/kodak --outfmt=BIN
```

```
...
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success.
Inference average time : 2.62 ms
Inference average time without first time: 2.56 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

**Noise2Noise MRI 网络**：

```
msame --model=om/mri.om --input=bin/ixi_valid --output=result/bin/ixi_valid --outfmt=BIN
```

```
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success.
Inference average time : 0.89 ms
Inference average time without first time: 0.89 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

注：result/bin/kodak、result/bin/ixi_valid需要提前创建

## 推理结果后处理

## 测试精度

**Noise2Noise 网络**：

```
python offline_infer_acc.py --bs=1 --bin_dir=result/bin/kodak/<folder>  --dataset=datasets/kodak --width=768  --height=512
```

参数解释：

```
--bs 	推理批次 默认：1
--bin_dir 使用msame推理后生成的bin文件位置 默认：bin/image
--dataset 验证集位置 默认：dataset/kodak
--width   reshape验证集图片：width
--height  reshape验证集图片：height
```

```
bin images data loaded
Average PSNR: 33.11
```

注：推理前会将图片进行填充，验证结果可能会与使用原始图片进行验证的结果存在差异。

**Noise2Noise MRI 网络**：

```
python offline_infer_acc_mri.py --bs=1 --bin_dir=result/bin/ixi_valid/<folder>  --dataset=datasets/ixi_valid.pkl 
```

```
[info] Start changing pkl to bin...
[info] Loading dataset from ../datasets/ixi_valid.pkl
[info] Bernoulli probability at edge = 0.02500
[info] Average Bernoulli probability = 0.10477
[info] test image:1000
[info] test_db_clamped: 29.53
```

开启：--post_op=fspec

```
python offline_infer_acc_mri.py --bs=1 --bin_dir=result/ixi_valid/<folder>  --dataset=datasets/ixi_valid.pkl --post_op=fspec
```

```
[info] Start changing pkl to bin...
[info] Loading dataset from ../datasets/ixi_valid.pkl
[info] Bernoulli probability at edge = 0.02500
[info] Average Bernoulli probability = 0.10477
offline_infer_acc_mri.py:238: ComplexWarning: Casting complex values to real discards the imaginary part
  denoised = denoised.astype(np.float32)  # Shift back and IFFT.
[info] test image:1000
[info] test_db_clamped: 32.27
```

## 推理样例展示

**Noise2Noise 网络**：

```
python bin2image.py --input=result/bin/kodak/<folder> --output=result/img/kodak --width=768 --height=512 --bs 1
```

参数解释：

```
--input 推理所得的bin文件或文件夹 默认：result/bin/kodak
--output 生成的展示图片的保存位置 默认：result/img/kodak
--bs 推理所得的bin文件包含的图片批次 默认：1 注：bs应当与bin文件批次大小相同	
--width bin文件对应的width
--height bin文件对应的height
```



<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202110090008981.png" alt="kodim01_output_0" style="zoom:50%;" />

<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202110090009298.png" alt="kodim04_output_0" style="zoom:50%;" />

**Noise2Noise MRI 网络**：

```
python bin2image.py --input=result/ixi_valid/<folder> --output=img/ixi_valid --width=255 --height=255 --bs 1
```

<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202110090017044.png" alt="img00000" style="zoom:50%;" />

注：mri图片未经傅里叶(逆)变换处理。
