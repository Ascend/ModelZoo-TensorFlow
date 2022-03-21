## 模型功能

图像分类

## 原始模型

参考论文：

[Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).](https://arxiv.org/pdf/1409.1556.pdf)

参考模型：

https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/VGG16_for_TensorFlow

https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/VGG19/VGG19_%20ID0374_for_TensorFlow

pb文件下载地址 :
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=gGnnL/JlT6iUNt4JBVjGkgHvmWaqKoA6ZS6x9rFZTtWoT7J4+CdgnwskVAsXefg0Xszwvc6pDQnBlu62eCXs3OZ99F5vspJP916F4Rrml7ujQ4XI8+GkD9se/I7tSAXP6zFZLzO9p0IeZF85Xnuy01tvr3aBmwWfyFOSMBq2i1dmJqUzVXZVw61aSNysZCNLbCqJrzpXs1GnXzFzA88oqg7OMfrOwqCqqd66usiVU8bVGU46TvWGWoALDlzRasRFHJgP0xiUmmXa09gGVdgJNmms+8jK3M2z/lwU0BhgWbe8wTb9LerXUCl/4ZUrSacACuHkoJQky1JmVtz9nqBvLFnoV96aMvAMDtd2Voup9dp9owqV5mBF0/FZzjU6nRPNr8aIwUSiO7JwZDo/6SnlVQkEcxdO3MmokRHaYQ4OgjRGZ0lCIAQ0r6L9WuK6/xOQ4952bbbIR6axc5KayI1lLJyqUlfO63oSgHZmM7Gd7Rrsrkbb7LclxsrrTZBWLD39CiGK+dqRT0bQ9kMiXDHqW2wygKokl5KIjOOGegfYe8c=

​		提取码:
​		123456


## om模型

om模型下载地址：

https://vgg19-acl.obs.cn-north-4.myhuaweicloud.com:443/vgg19_tf_310.om?AccessKeyId=Z5KDYQHVBENJIDBUH7YD&Expires=1661511270&Signature=GWIODfvajxAjI2kyl9lSG/GmOe0%3D

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=pb_path --framework=3 --output=om_path --soc_version=Ascend310 --input_format=NHWC --output_type=FP32 \
     	--input_shape="input:-1,224,224,3" \
        --out_nodes="dense_2/BiasAdd:0" \
        --precision_mode=allow_mix_precision  
```

## 数据集准备

imagenet2012原始验证集中的图像数据转换为bin文件可参考如下代码：

```
def normalize(inputs):
    imagenet_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    imagenet_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    imagenet_mean = tf.expand_dims(tf.expand_dims(imagenet_mean, 0), 0)
    imagenet_std = tf.expand_dims(tf.expand_dims(imagenet_std, 0), 0)
    inputs = inputs - imagenet_mean
    inputs = inputs * (1.0 / imagenet_std)
    return inputs
for file in os.listdir(datasetA_dir):
    print(file)
    if file.endswith('.JPEG'):
        src_testA = datasetA_dir + "/" + file
        print("start to process %s" % src_testA)
        #预处理
        image = tf.io.read_file(filename=src_testA)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, size=(256, 256))
        image = tf.image.central_crop(image, 224.0 / 256)
        image = normalize(image)
        with tf.Session() as sess:
            image_numpy = image.eval()
        image_numpy.tofile(dstA_path+"/" + file+".bin"
```



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行性能测试。



## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame  --model "/home/HwHiAiUser/omfile/vgg19_tf_310.om"  --input "/home/HwHiAiUser/omfile/bin"  --output "/home/HwHiAiUser/omfile/out/"  --outfmt TXT
```

```
Inference average time : 9.76 ms
Inference average time without first time: 9.76 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能为 9.7ms

## 精度测试

执行精度对比文件：

```
python3 compare.py
```

最终精度：

```
Totol pic num: 49899, Top1 accuarcy: 0.7289
```





