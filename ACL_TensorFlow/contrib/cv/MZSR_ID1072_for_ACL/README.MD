## ## 模型功能

图片超分辨率

## pb模型

```
python freeze_graph.py  --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath output/ --kernelpath Input/g20/kernel.mat  --ck2pb
```
需要前往作者仓库提前下载Input/g20/Set5/、GT/Set5/、Input/g20/kernel.mat下的文件(https://github.com/JWSoh/MZSR)且在freeze_graph.py文件中修改自己生成的模型路径
相关结果文件获取：

链接：https://pan.baidu.com/s/1ZJvpUE_rArECei2bgw3AnQ 
提取码：jr85

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model="/root/MZSR/frozen_model.pb" --framework=3 --output="/root/MZSR/mzsr_model_all" --soc_version=Ascend310 --input_shape="input:1,-1,-1,3" --dynamic_dims="344,228;256,256;280,280;288,288;512,512"  --log=info --out_nodes="MODEL_1/output:0" --input_format=ND
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

## 数据集转换bin

```
python freeze_graph.py  --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath output/ --kernelpath Input/g20/kernel.mat  --png2bin
```

## 推理测试

使用msame推理工具，参考如下命令，发起推理测试： 

```
./msame --model "/root/MZSR/mzsr_model_all.om"  --output "/root/MZSR/" --outfmt TXT --loop 1 --input "/root/MZSR/Input/woman.bin" --dymDims "input:1,344,228,3" 
```

对单张图片进行推理输出

根据图片改变--input和--dymDims中的值，其中dymDims和GT图片shape一致

output为输出推理结果的路径

计算精度
output_dir为推理得到的输出文件路径
data_dir为验证数据集的路径

将生产的txt文件拷贝到项目中output/下执行以下命令

```
python freeze_graph.py  --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath output/ --kernelpath Input/g20/kernel.mat  --om2test
```

## 推理精度（psnr）

|baby|bird|butterfly|head|woman|
|:--:|:--:|:--:|:--:|:--:|
|37.09|38.58|31.88|35.00|34.20|
