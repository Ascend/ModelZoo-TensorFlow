## 推理情况表    

| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度MIoU | 目标精度MIoU |
|--|--|--|---| -- | --| -- |
| resnest | Citscapes  | `1*1024*2048*3` | `1*1024*2048*21`  | 2232.43ms | 66.45% | 63.38%|   


## 1、原始模型 

[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyQr/Jp/0EZDg6mZjMV6v3za+27+6baq0u25MQxnMTWQ/blfe3uSOORJNza15OzPHxIS8Luy+AjC61/WJdLIYpMwW5Mp9geuTpcUhX8FmYdTRgUJK0ZOCoKZ/3jWz7xRcTXHSobGdoPmNiZIuuL2dVb/iB9skiUIrqIL0OVYGNP9toFswzeapeF47oYzFJGITt3mcmFeh8d8UMjSxQr62wW14x4RlRiOu5hCw40thTlqtv1RunJYH0Mj16n5ixZX+jXlhGEfQ2JYy/ed6BOFseJDkR841B+8rLJJglOsJ0e7dFxeVf0Seqj8wkgGtqB6ig7OymUuDoo0/jAshf7sECn7Jk2y20+LrEWno4+TjFxLbLblRilhrAHOPsdIsgAtc5fDGpjrbXz2zHyAaDD9G7dbsMg8dLALSqz/XsBRYkLBx/JeUx9JAkFFrAkYOtuZe/GzYV7kyEoP1kyvUnOffC1moHIRq0IGQ1TIXf0d2hOA8ZsvUMMBumCSKLD4nz0TONzbFIaDqE/wmDXWXcnCXjoiICD77ffOgbQItGTWNZGYJ7gK0TLo0VsrnhhF0vcef22e8pRIlJHbsAGayGdaK2wngWkeYAMPUY7CWn0mGpsXD2nSn67ZH1FLHBoPNtsOLmQ==)(提取码为666666), 找到对应文件夹,下载对应的`ckpt`，使用`convert.py`脚本转成pb模型。


## 2、转om模型

[obs链接](https://huaweiinference.obs.cn-north-4.myhuaweicloud.com:443/resnest/resnest.om?AccessKeyId=V5QSYDN1WFOA7YZRA5Y2&Expires=1659231717&Signature=LOw7Qj5t5VPoxZjtczRMyVOO%2Bq8%3D)(提取码为666666),找到对应的om的文件


atc转换命令参考：

```sh
atc --model=resnest.pb  --framework=3 --input_shape="inputx:1,1024,2048,3" --output=./resnest --out_nodes="resize/ResizeBilinear:0" --soc_version=Ascend310 --precision_mode=allow_fp32_to_fp16
```


### 特别注意  

这里面多加了一个`precision_mode`   


## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：  


### 4.1 下载预处理后的Cityscapes val数据集

[obs链接](https://public-dataset.obs.cn-north-4.myhuaweicloud.com:443/resnest/cityscapes_val.tfrecords?AccessKeyId=V5QSYDN1WFOA7YZRA5Y2&Expires=1659156639&Signature=DRFZDNay0HeFlk%2Bk6SGP%2B1LIBUk%3D)(提取码为666666) 找到`Citscapes.val.tfrecord` 文件   

```sh
python3 dataprepare.py /root/310/datasets/cityscapes_val.tfrecords /root/310/datasets/bin 
# or
bash start_data2bin.sh
```

```python 
# 数据预处理部分
parsed = tf.io.parse_single_example(tfrecord_file, features)
image = tf.decode_raw(parsed['image'], tf.uint8)
image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
label = tf.decode_raw(parsed['label'], tf.uint8)
label = tf.reshape(label, [parsed['height'], parsed['width'],1])
combined = tf.concat([image, label], axis=-1)
combined = tf.random_crop(combined,(img_H, img_W,4))
image = combined[:,:,0:3]
label = combined[:,:,3:4]
image = tf.cast(image, tf.float32)
label = tf.cast(label, tf.int64)
```

### 4.3 执行推理和精度计算 

执行`bash start_inference.sh`脚本

```log
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success.
[INFO] start to process file:/root/310/datasets/bin/resnest/data/98.bin
[INFO] model execute success
Inference time: 2232.54ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success.
[INFO] start to process file:/root/310/datasets/bin/resnest/data/99.bin
[INFO] model execute success
Inference time: 2232.43ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success.
Inference average time : 2232.41 ms
Inference average time without first time: 2232.44 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
/root/310/datasets/bin/resnest/inference
inference_path /root/310/datasets/bin/resnest/inference/20210728_121825/
100%█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [05:35<00:00, 1.49it/s]
[0.97165736 0.79040782 0.89897331 0.41002983 0.45321966 0.60396467
0.6397655 0.74175404 0.90906697 0.57525981 0.9359027 0.78225702
0.53072048 0.92656564 0.440881 0.61461404 0.21829686 0.44797544
0.73470289]
>>>>> 共 500 测试样本    MIoU: 0.664527 0.9465961548353052 0.7354460680985017 0.6645271075582194 0.9023151962914528
```