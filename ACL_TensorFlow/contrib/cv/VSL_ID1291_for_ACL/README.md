# 模型功能

VSL是一种变分形状学习者，这是一种生成模型，可以在无监督的方式下学习体素化的三维形状的底层结构，通过使用`skip-connections`，该模型可以成功地学习和推断一个潜在的、层次的对象表示。此外，VSL模型还可以很容易的生成逼真的三维物体。该生成模型可以从2D图像进行端到端训练，以执行单图像3D模型检索。实验结果表明，改进后的算法具有定量和定性两方面的优点。 

- 参考论文：

    [Learning a Hierarchical Latent-Variable Model of 3D Shapes](https://arxiv.org/abs/1705.05994) 
    
    对于更详细的结果，可以参考[项目主页](https://shikun.io/projects/variational-shape-learner)


# om模型转换
在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1

atc --model=/home/HwHiAiUser/AscendProjects/VSL/modelnet10.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/VSL/modelnet10 --soc_version=Ascend310 --input_shape="Placeholder:100,30,30,30,1" --log=info --out_nodes="latent_feature:0"
```
请从此处下载[pb模型](https://canntf.obs.myhuaweicloud.com:443/vsl_zwt/vsl/pb_om/modelnet10.pb?AccessKeyId=NLVKVVAQHOUIA7ROJBEZ&Expires=1670766198&Signature=H4GOMDBr7ak8HGXRT4S03K/rJDc%3D)

请从此处下载[om模型](https://canntf.obs.myhuaweicloud.com:443/vsl_zwt/vsl/pb_om/modelnet10.om?AccessKeyId=NLVKVVAQHOUIA7ROJBEZ&Expires=1670766284&Signature=OuArsad0gLTjPmXi%2BPM4BbJUMYI%3D)


# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。


## 1.数据集转换bin
原始数据集[链接](https://gitee.com/link?target=https%3A%2F%2Fwww.dropbox.com%2Fsh%2Fba350678f7pbwx8%2FAAC8-2X1p4BiOKlyYuuxFcDBa%3Fdl%3D0)

下载好原始数据集后，将其保存在dataset目录下，并执行`mat2bin.py`文件将mat数据转换为推理需要的[bin数据](https://canntf.obs.myhuaweicloud.com:443/vsl_zwt/vsl/pb_om/ModelNet10.bin?AccessKeyId=NLVKVVAQHOUIA7ROJBEZ&Expires=1670766310&Signature=y6r72GUZU006/cy8PoLPUC9zHEI%3D)


## 2.推理

使用msame推理工具，发起推理测试，推理命令如下：

```bash
cd /home/HwHiAiUser/AscendProjects/tools/msame/out

./msame --model "/home/HwHiAiUser/AscendProjects/VSL/modelnet10.om" --input "/home/HwHiAiUser/AscendProjects/VSL/ModelNet10.bin" --output "/home/HwHiAiUser/AscendProjects/VSL/om/" --outfmt TXT --loop 1
```

## 3.推理结果

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/AscendProjects/VSL/modelnet10.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/AscendProjects/VSL/out/modelnet10//20211215_215629
[INFO] start to process file:/home/HwHiAiUser/AscendProjects/VSL/ModelNet10.bin
[INFO] model execute success
Inference time: 156.002ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 156.002000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

将推理生成的txt文件下载下来并存入`dataset`目录下，用于之后的精度测试。

## 4.精度测试
执行`calculate_accuracy.py`文件用于对推理结果进行精度测试。测试结果如下：
```
Shape classification: test: 0.8800
```
说明：pb模型在测试集上在线推理的精度为0.8933，与离线推理的精度接近。