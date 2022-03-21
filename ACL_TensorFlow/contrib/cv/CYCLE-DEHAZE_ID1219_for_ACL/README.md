# Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing

&emsp;&emsp;原始模型参考[github链接](https://github.com/engindeniz/Cycle-Dehaze)，本项目在迁移到NPU的基础上实现离线推理。

## Requirments

* TensorFlow 1.15
* Python 3.7.5
* MATLAB 
* Ascend 910、Ascend 310
* GPU运行平台: Tesla V100
* NPU运行平台: ModelArts
* 依赖的其他网络: vgg16.npy
* Dataset: NYU-Depth

## 代码路径解释

```
CYCLE-DEHAZE_ID1219_for_ACL
|_
  |-- data                   ----存放数据(.png)
      |-- allData            ----存放数据集中所有图片
          |-- clearImage     ----存放数据集中所有清晰的图片
          |-- hazyImage      ----存放数据集中所有有雾的图片
      |-- trainData          ----存放用于训练的图片，由allData中的图片划分而来
          |-- clearImage   
          |-- hazyImage
      |-- testData           ----存放用于测试的图片，由allData中的图片划分而来
          |-- hazyImage
          |-- groundtruth
          |-- model_predict  ----测试时生成，存放模型输出的去雾图片
      |-- tfrecords          ----存放训练时使用的tfrecords文件
  |-- boot_modelarts.py      ----modelarts平台启动文件
  |-- checkpoints            ----训练时生成，保存ckpt文件
  |-- pretrained             ----训练时生成，保存pb文件
  |-- build_data.py
  |-- cal_metrics.py         
  |-- convertHazy2GT.sh
  |-- discriminator.py 
  |-- export_graph.py
  |-- generator.py
  |-- help_modelarts.py
  |-- inference.py           ----推理时实际输出结果经过可视化产生最终结果
  |-- inference_outnodes.py  ----离线推理时产生实际输出结果
  |-- inference_outnodes.sh
  |-- laplacian.m
  |-- model.py
  |-- model_convert.sh       ----pb模型转om模型
  |-- ops.py
  |-- reader.py
  |-- resize_im.m
  |-- shuffle_copy.py
  |-- test.sh                ----离线测试启动脚本
  |-- train_gpu.py    
  |-- train_npu.py         
  |-- train_npu.sh           ----npu训练启动脚本
  |-- train_gpu.sh           ----gpu训练启动脚本
  |-- utils.py
  |-- vgg16.npy              ----需自行下载放在当前位置
  |-- vgg16.py
  |-- LICENSE
  |-- README.md

```
## pb模型生成
```python
python3 export_graph.py --checkpoint_dir checkpoints/Hazy2GT \
                        --model_dir models \
                        --XtoY_model Hazy2GT.pb \
                        --YtoX_model GT2Hazy.pb \
                        --image_size1 256 \
                        --image_size2 256
```

## om模型生成
&emsp;&emsp;在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:
```sh
sh model_convert.sh
```
&emsp;&emsp;推理示例使用的pb模型以及om模型[在此获取](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=65Ix3OHMKsytrZjXu2zhHbZHoHxc0iGrD6gYXh73/cB/heZFMjxUt3yUV3iMa3QazhGeSS3JgziE14UPcT63/47aTSkTDJUIQx7QIrgnnMVcrq//lADkOzdpu1MChTk6SbHGBy+hR6KigVetacNEhmOy+TqPj6JrJGQKXX1W5yDn418GbfHHuX1wHdbhiCxaXNLduOCEu6CxJ2Wib/uFV8t6/ukjHt6I6TmnSP98ZUZ7bY9Hd8dyLCBjdaaqRFlepCqd0c4EjuT98h26+pK+clGFHM7QJfg4N/LPdge5Fq/vUSM3vzHZAmPxcSTmnLQ/wLId+bhQzrdB7V7CVwCK7uYZ4RPN3G4ymEEMbHCEL51gVha4WRAt4u+FA00+ZyIu1hDRZhjLZINSx9i8tOyATsD0jWfy23cGYwoSu3lzxwAJzuEFgIi5U9pIpoXeBwE1rL1GwySg2HkSglgECZtq+gfp0fZnHftsG7Jy7Fb1l1ZCNKtsfzQoRQ1H6MILkWaDD4/sR3NNTh63G1YVaUeJ/1WxqUqm1IJyaleRIJU5mB8M6O3YPrN/pJ8EqNJY31SGKCkCtoAFfT0GVOQBPFFkL+SJmWPUj89q4LK8RxyKN0/dUzC1UsAAs+Mk+h/NjlafcYKq0Oc7Nl55edzhwFeb0ZZ0pONKNwCLdn9EYvDDBf0=)（提取码111111）

## 使用msame工具推理
### 1.数据集转换bin
&emsp;&emsp;参考[img2bin](https://gitee.com/ascend/tools/tree/master/img2bin)实现图片数据集到bin的转换，转换命令如下：
```sh
python3 img2bin.py -i /home/om_inference/hazyImage \
                   -w 256 \
                   -h 256 \
                   -f RGB \
                   -a NHWC \
                   -t float32 \
                   -o /home/om_inference/hazyImage_bin_output/out1
```
### 2.推理
&emsp;&emsp;参考[msame](https://gitee.com/ascend/tools/tree/master/msame)配置好msame离线推理工具，进入msame文件夹，执行推理命令如下：
```sh
./msame --model "/home/pb2om/om4/om4.om" \
        --input "/home/om_inference/hazyImage_bin_output/out1" \
        --output "/home/om_inference/om_inf_bin/inf1/" \
        --outfmt TXT
```
### 3.推理后处理
&emsp;&emsp;由于CYCLE-DEHAZE网络模型的实际输出节点为"G_9/output/MirrorPad"，此节点的输出结果是一个无法直接转换为图片的非可视化的结果。因此，在精度对比时我们直接比较pb模型和om模型在对同样的输入数据进行推理时此节点输出结果的差异。
&emsp;&emsp;pb模型对图片数据集进行推理时执行以下命令：
```sh
sh inference_outnodes.sh offline_inference/hazyImage offline_inference/pb_output models/Hazy2GT-bs2-dataset-80000.pb
```
&emsp;&emsp;进行1、2中的步骤将图片数据集转换为bin文件并使用om模型进行推理并得到结果。
&emsp;&emsp;之后执行以下命令计算使用pb模型推理和om模型推理时输出节点输出值的差异，命令行的第一个参数为存放pb模型推理结果的文件夹，命令行的第二个参数为存放om模型推理结果的文件夹：
```sh
python3 cal_difference.py offline_inference/pb_output offline_inference/om_output
```
### 4.推理精度对比
&emsp;&emsp;离线推理使用的图片数据集示例及其对应的bin文件[点击此处](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=65Ix3OHMKsytrZjXu2zhHbZHoHxc0iGrD6gYXh73/cB/heZFMjxUt3yUV3iMa3QazhGeSS3JgziE14UPcT63/47aTSkTDJUIQx7QIrgnnMVcrq//lADkOzdpu1MChTk6SbHGBy+hR6KigVetacNEhmOy+TqPj6JrJGQKXX1W5yDn418GbfHHuX1wHdbhiCxaXNLduOCEu6CxJ2Wib/uFV8t6/ukjHt6I6TmnSP98ZUbSxPO1Pd1sfNSNwNYu/5a6OqEXdpW6bjCeQSbJxsmBN7c68AX1OJSXaBgJa/FpvIW2PGjT+GB7h7+gH9ZFA00lJkzViiPmth/rU949phLGBYlexcGCFi5v6BA65xwHtCoki8sRsJ0OSACYLnDf3HKjvZ7ldYwER978A4g3pVdx8SfJbmwsFYRUsnKnF3wpw8+YD8UZ06ls6NlSqrxjhOTF0s3qANM1fettvydEK04FioD50YsbHeOih2bKfsq+cERoi/z5tW7pk59K4jFxxKv+lNtWOXMDAteuPZjmY63dO/vy6HPX3jzv85z7wIf6HDPartT6ggUozEd8/MPI7sEO0SSmEfX27oX2fERlcXbI+wFUtc6mDs4tAjC7VrtKqN8GxE7xdZXZJr1FE8c1s3MFvYbGwIRNkOrQlJhePO5aWg==)获取（提取码111111），pb模型以及om模型推理的结果[点击此处](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=65Ix3OHMKsytrZjXu2zhHbZHoHxc0iGrD6gYXh73/cB/heZFMjxUt3yUV3iMa3QazhGeSS3JgziE14UPcT63/47aTSkTDJUIQx7QIrgnnMVcrq//lADkOzdpu1MChTk6SbHGBy+hR6KigVetacNEhmOy+TqPj6JrJGQKXX1W5yDn418GbfHHuX1wHdbhiCxaXNLduOCEu6CxJ2Wib/uFV8t6/ukjHt6I6TmnSP98ZUZIqUZTpF+JlK8Xu0YrcJe/6PSFoIG+9FNz7NuO9Gmffi2QAYwqg0s0zRV14c/25YTAdQMeJ1Cuq/oUWtx9ejjjlbt81gJ98APSC4KxKu4UhseyzBru8lwlOjbxjfZlYpQc/EUStKEcncPhinEvS0kpAsY6it6n2dpjldvmIBlmzHFzXOUkTUqTKNOSHTgMtIyao8HATst8/HR4P2crAZ0dTAsZqitR7JzYCZc+kQXEbwUucGQJjl7/3TxYjSm06scuBilDVRiCrtsa0Au8/aHpgcuoJ0tUHDf6Zpk4zMQUFMNV/HTQNpxM+ac5aUHf3ZUm3bK/FyaSrcO/KYI047dVsUf0eVTnfGTPf/aKuR33tOLn7xZK/Pn85tK3F/icRzbh28VkUw1ih6lPKGc/HrmnPb/egTLyyl4ZCpJ/eI3+6w==)获取（提取码111111）。
&emsp;&emsp;精度对比的结果如下图所示，可见om模型推理与pb模型推理时输出节点输出值之间的差异相对于pb模型推理时输出值的百分比为1.17%，即差异可以忽略不计，满足精度要求。
<div align="center">
    <img src="figure/precision_difference.png" width="450" height="160">
  </div>

### 5.推理性能对比
* GPU离线推理
  GPU离线推理单张图片约耗时3107ms。
  <div align="center">
    <img src="figure/gpu_offline_inference.png" width="450" height="80">
  </div>
* Ascend310离线推理
  NPU离线推理单张图片约耗时1169ms。
  <div align="center">
    <img src="figure/Ascend310_offline_inference.png" width="450" height="130">
  </div>


