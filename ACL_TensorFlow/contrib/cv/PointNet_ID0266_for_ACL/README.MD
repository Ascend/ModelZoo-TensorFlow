## 1、原始模型

参考训练工程，训练网络生成ckpt模型，并转成pb模型
```
https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/pointnet/PointNet_ID0266NRE_for_TensorFlow
```
可通过链接 [ckpt](https://pan.baidu.com/s/1F6FXxYVCZ8FJoEC2IYSWpQ)
获取模型ckpt文件（提取码：7777）

可通过链接 [pb](https://pan.baidu.com/s/10yT_IgIlQ0lwXE7r4zSEJQ)
获取模型pb文件（提取码：7777）


## 2、pb转om模型

atc转换命令参考：
```
atc --model=./pointnet.pb \
    --framework=3 \
    --output=./pointnet \
    --soc_version=Ascend310 \
    --input_shape="pointclouds_ph:1,3000,3;input_label_ph:1,16"
```
说明：上述命令中，指定的推理硬件为Ascend310，om模型推理时要在对应的硬件上运行；文件路径请根据情况配置。

可通过链接 [om](https://pan.baidu.com/s/1lcZKyqpI1mVFyQfB2kS6mA)
获取模型om文件（提取码：7777）


## 3、数据转换bin文件

数据转换成bin文件，可使用numpy.array的tofile函数，对应模型输入，将输入分别保存在不同的文件夹中，每个文件夹中包含多条测试样例，相同测试样例的不同输入文件夹下的文件名相同。可参照示例中的数据处理方式：
```
python3 evaluate.py --phase data2bin
```


## 4、编译msame推理工具

参考 https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 5、使用msame进行离线打通

使用msame推理工具，参考如下命令，发起推理性能测试： 
```
./msame --model "./pointnet.om" --input "./batch_data,./cur_label_one_hot" --output "./out/om_out/" --outfmt BIN
```

推理测试成功，会输出如下log信息：
```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/pointnet/om/pointnet.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/pointnet/om/out//20211222_17_52_13_963238
[INFO] start to process file:/home/HwHiAiUser/pointnet/om/batch_data/0.bin
[INFO] start to process file:/home/HwHiAiUser/pointnet/om/cur_label_one_hot/0.bin
[INFO] model execute success
Inference time: 8.227ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/HwHiAiUser/pointnet/om/batch_data/1.bin
[INFO] start to process file:/home/HwHiAiUser/pointnet/om/cur_label_one_hot/1.bin
[INFO] model execute success
Inference time: 8.157ms
...
```

## 6、ShapeNet数据集上精度测试：

### 6.1 数据集准备

1. 模型训练使用**ShapeNet**数据集，详见
https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/pointnet/PointNet_ID0266NRE_for_TensorFlow

2. 数据集下载后，放入相应目录下，在数据预处理脚本中指定数据集路径，可正常使用。

### 6.2 数据预处理

下载好的数据集放在任意目录，执行预处理脚本生成bin文件：
```
python3 evaluate.py --phase data2bin
```

### 6.3 执行推理
```
./msame --model "./pointnet.om" --input "./batch_data,./cur_label_one_hot" --output "./out/om_out/" --outfmt BIN
```

### 6.4 精度测试
1. 参见requirements.txt配置测试所需依赖库
    ```
    tensorflow==1.15.0
    numpy>=1.18.1
    ```

2. 推理精度测试
    ```
    python3 evaluate.py --phase evaluate
    ```
   ShapeNet测试数据上
   - om模型推理的精度(IoU)结果: 83.792
   - pb模型推理精度(IoU)结果: 83.792
    

5. NPU与GPU推理精度对比
   
   ShapeNet测试数据上
   - NPU上om模型推理的精度(IoU)结果: 83.792
   - GPU上pb模型推理的精度(IoU)结果: 83.791
