推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| dpn | voc2012 | `9*512*512*3` | `9*64*64*21`  | 255ms~ | 44.33% | 44%|

## 1、原始模型
[下载地址](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOI+eCEb0b/jKJonh9bds90Vn0FstyWcdGBkm8QwQ1HQXEPxDlDurbd/mWD1rpND1DbpRz4NFe7yyFBRXVFinInzSuYrxxN0xEW4WT2cduQKDFnLzm5/weUI88S5sghr9M67rCIegjz9620sx2bhbdow+4xBQtlVk6ZJqmwMX9hIclYyDKeCqsC/ZsUdpxocbikK2eNN+Xq14vz8WsuWEl7/SfRH2uucjV1P1pk1QCo9cKxe323VqZvJ+thyJBixyRQxi9MpguK4QAPXtqW3meo6M+9EuUV4Wa9ccvO0Cjsk5J+E4UGxUs2LxdlUOnaiqvHBIJRzeJRA7iHecVtqmwDPYTkos4Z5K3l6ZeuYg4rMwGwMWmm8Kj1d/RHFcQ+1hYHSLR0ZAJ0LHPrQ0yTPg4yZq99dCB0G51x82LD+L1zmijK3F9bsU1BT4pP9tiF9WJhLEB9Tk9KqLhpmQKhyEA3mg==), 找到对应文件夹,下载对应的`ckpt`，使用该文件夹下面的`convert.py`脚本转成pb模型。(提取码为 66666) 

## 2、转om模型
[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOI+eCEb0b/jKJonh9bds90Vn0FstyWcdGBkm8QwQ1HQXEPxDlDurbd/mWD1rpND1DbpRz4NFe7yyFBRXVFinInzSuYrxxN0xEW4WT2cduQKDFnLzm5/weUI88S5sghr9M6Xdys9V8cewHh2+nWEQMUuBpXpRyr88hEB4QwTR/PHjy1mxJJaFbgKv6ri1NNWrTXv6CUKEg8rxFMjtwTOBJb830Wm65J48X/gtnr8UazHZZyfDpi6QxerhFSx+SOjJ9yACZ0qnyN5MsTZRLh5765I38rB5kAInGogZvXTb66wdstLdOPIQaakpH/HvFzQ4e0/6y+pUiSys69szNHNZAXsxF7WUtntnuv+zRDaKYGy7JVgBh3AqcDxpMwXnWkl/NW4tzMfua+ItFA4In9LAG4efkxaXDGqjnIJW/Js2FZslETm5cpwykol9laf3v6FvPGPFDDciMzM4EdW1Py5WqgAw==
)  (提取码为666666),找到对应的om的文件

atc转换命令参考：

```sh
atc --model=dpn.pb  --framework=3 --input_shape="inputx:9,512,512,3" --output=./dpn --out_nodes="upsample/Conv_2/Relu:0" --soc_version=Ascend310
```

注意： dpn需要使用3.3.0.alpha001及以上 CANN toolkit才能实现模型转换。3.3.0 CANN toolkit下载地址为： https://www.hiascend.com/software/cann/community 

环境设置参考： 
```shell
export driver_home=/usr/local/Ascend
export install_path=${driver_home}/ascend-toolkit/latest
export ASCEND_HOME=${install_path}
export DDK_PATH=${install_path}
export ASCEND_AICPU_PATH=${install_path}/x86_64-linux
export ASCEND_TENSOR_COMPLIRE_INCLUDE=${install_path}/atc/include
export TOOLCHAIN_HOME=${install_path}/toolkit
export PATH=/usr/local/python3.7.5/bin:${install_path}/toolkit/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:${install_path}/fwkacllib/ccec_compiler/bin:${install_path}/fwkacllib/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/:${install_path}/acllib/lib64:${install_path}/atc/lib64:${install_path}/fwkacllib/lib64:${driver_home}/driver/lib64:${driver_home}/add-ons:${LD_LIBRARY_PATH}
export PYTHONPATH=${install_path}/pyACL/python/site-packages:${install_path}/fwkacllib/python/site-packages:${install_path}/fwkacllib/python/site-packages/auto_tune.egg:${install_path}/fwkacllib/python/site-packages/schedule_search.egg:${install_path}/toolkit/python/site-packages:${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:${install_path}/opp/op_impl/built-in/ai_core/tbe:${install_path}/toolkit/latest/acllib/lib64:${PYTHONPATH}
export ASCEND_OPP_PATH=${install_path}/opp
export NPU_HOST_LIB=${install_path}/acllib/lib64/stub
export SOC_VERSION=Ascend310
```

## 3、编译最新的msame推理工具

编译最新的msame工具前，由于更新了CANN toolkit ，需要同步更新驱动. 驱动下载地址为 https://www.hiascend.com/software/cann/community
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

### 4.1 下载数据集

[obs链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=NOinLpNll/SKHwn8wXpeyRNbzJuYebMEGXyQJ/hb0naBiXj5UjE2P6zqxoGe5d7Obh95/GnEoOCpENkubigHInNGtyvFMwcidFvYzKq4SokTLsR46FTRbbrH99u9lQOI+eCEb0b/jKJonh9bds90Vn0FstyWcdGBkm8QwQ1HQXEPxDlDurbd/mWD1rpND1DbpRz4NFe7yyFBRXVFinInzSuYrxxN0xEW4WT2cduQKDFnLzm5/weUI88S5sghr9M65bj17CbuxacySCeu2AyV8SOtaoVOSNzTL624qj9dO1dHlnXrtZIBvUkr7EX8oDTog3IEaorPwn5YuBwAGh7SBVxfFQ9LB8JpCMiWMT8qK7MEMYLfE2O1Txbzr/nc7rhvamsmgxZUioyuArwzLi23Fd16v64sk7rQ1KPygkSwY0/fo7tcbljdroGhHgN1B8VbVHP87voR/St6Quk+HZZXWkL5oPBYflpt512hWkLdFIH+ddm2x/FnuYEMREfrbI33IMybARzgHvYreythXA7P2StuBBB+xXQp9XxvkBISPWfaL9m8m1WL+9TpSBubtff8T3mL6TuW683XAQxYH0EPx4+yGdCk2HXkD0OU6ph1UUs=)  (提取码为666666) 找到`dpn.tfrecord` 文件   

### 4.2 预处理

使用double_unet_data_preprocess.py脚本,设定好路径后,
执行shell脚本: `./dpn_data_preprocess.sh` 生成bin文件  

```sh
python3 dpn_data_preprocess.py /root/dataset/dpnval.tfrecords  /root/dataset/bin/dpn
```

```python
def _augmentation(image, label, mask, h, w):
    image = tf.image.resize_images(image, size=[512, 512], method=0)
    image = tf.cast(image, tf.uint8)
    label = tf.image.resize_images(label, size=[64, 64], method=1)
    mask = tf.image.resize_images(mask, size=[64, 64], method=1)
    # 随机翻转
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
        mask = tf.image.flip_left_right(mask)
    return image, label, mask

def _preprocess(image, label, mask):
    image = image - [122.67891434, 116.66876762, 104.00698793]
    image = image / 255.
    return image, label, mask
```

### 4.3 执行推理和精度计算

然后执行'./start_inference.sh'脚本
```log
[INFO] start to process file:/root/dataset/bin/dpn/data/97.bin
[INFO] model execute success
Inference time: 2302.25ms
[INFO] output data success
[INFO] start to process file:/root/dataset/bin/dpn/data/98.bin
[INFO] model execute success
Inference time: 2302.46ms
[INFO] output data success
[INFO] start to process file:/root/dataset/bin/dpn/data/99.bin
[INFO] model execute success
Inference time: 2301.66ms
[INFO] output data success
Inference average time : 2302.18 ms
Inference average time without first time: 2302.18 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
******************************
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
[INFO]    推理结果生成结束
>>>>>  共 1449 测试样本 	 MIoU: 0.443261

```
