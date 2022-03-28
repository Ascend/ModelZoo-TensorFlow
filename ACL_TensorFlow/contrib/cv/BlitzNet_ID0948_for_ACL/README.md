
## 推理过程<a name="section1589455252218"></a>
环境
- Tensorflow 1.15
- python 3.7

1.  ckpt文件

- ckpt文件下载地址:
  
  https://sharegua.obs.cn-north-4.myhuaweicloud.com:443/checkpoint65.zip?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1667698491&Signature=Ltfv5%2B5VbaFSklW3pI6W6oTh73A%3D
  
    通过freeze_graph.py转换成pb文件bliznet_tf_310.pb
  
- pb文件下载地址:
  
  https://sharegua.obs.myhuaweicloud.com:443/bliznet_tf_310.pb?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1667656586&Signature=JhBRfk5dpeDFE%2BPy1jQg6Q4mvHY%3D

2.  om模型

- om模型下载地址:
  
  https://sharegua.obs.myhuaweicloud.com:443/bliznet_tf_310.om?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1667656644&Signature=Z7DyzKRGPd27pYipfD2Ke/KSGAo%3D

    使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/home/HwHiAiUser/atc/bliznet_tf_310.pb --framework=3 --output=/home/HwHiAiUser/atc/bliznet_tf_310 --soc_version=Ascend310 \
        --input_shape="input:1,300,300,3" \
        --log=info \
        --out_nodes="concat_1:0;concat_2:0;ssd_2/Conv_7/BiasAdd:0"      
```

3.  使用msame工具推理
    
    参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

    获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。
    
    msame推理可以参考如下指令:
```
./msame --model "/home/HwHiAiUser/msame/bliznet_tf_310.om" --input "/home/HwHiAiUser/msame/data" --output "/home/HwHiAiUser/msame/out/" --outfmt TXT
```
- 将测试集数据转为bin文件:
```
  imageToBin.py
```

- 测试数据bin文件下载地址:
  
  https://sharegua.obs.cn-north-4.myhuaweicloud.com:443/img.zip?AccessKeyId=UC40X3U4Z2RUPSTV8ADH&Expires=1667698452&Signature=f3aLaUdPnodF8PKtCaI5Ox4wb6c%3D
  

4.  性能测试
    
    使用testBliznetPb_OM_Data.py对推理完成后获得的txt文件进行测试

<h2 id="精度测试">精度测试</h2>

训练集：VOC12 train-seg-aug

测试集：VOC12 val

|    | mIoU |  mAP |
| ---------- | -------- | -------- |
| 论文精度 | 72.8       | 80.0 |
| GPU精度32 | 72.8       | 80.0 |
| GPU精度16 | 72.0       | 78.3 |
| NPU精度   | 70.1       | 77.6 |
| 推理精度   | 70.1       | 77.6 |