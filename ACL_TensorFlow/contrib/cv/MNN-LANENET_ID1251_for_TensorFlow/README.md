## 模型功能

 车道线检测




## pb模型

```

python3.7 ckpt2pb.py
```

convert_ckpt_into_pb_file（）函数，pb模型 PATH=./pretrained_model/eval.pb

## om模型


使用ATC模型转换工具进行模型转换时可以参考如下指令:


```

atc --model=/usr/pb2om/eval.pb

      --framework=3 

      --output=/usr/pb2om/frozen

      --soc_version=Ascend910 

      out_nodes="lanenet/binary_seg_out:0;lanenet/instance_seg_out:0"

      --input_shape="input_tensor:1,256,512,3"

      --input_format=NHWC
```



## 使用msame工具推理


参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。



## 数据集转换bin

```

python3.7 eval_pb.py
```

freeze_graph_test（）函数 img_feed转test_img.bin

## 推理测试


使用msame推理工具，参考如下命令，发起推理测试：
 

```

./msame --model "/home/test_user05/pb2om/frozen.om" 

             --input "/home/test_user05/pb2om/test_img.bin" 

             --output "/home/test_user05/pb2om/" 

             --outfmt BIN

             --loop 1
```


## 脚本和示例代码

代码：链接：链接：https://pan.baidu.com/s/1oqdEQGH8ixPv4tNMBRExfQ?pwd=6666 
提取码：6666 


输入数据：链接：https://pan.baidu.com/s/1bwZR3FfhP18mLMa44781Gw?pwd=0000 提取码：0000

├── eval_data


1.pb预测

├── eval_pb.py                          //主代码

├── lanenet_model                                    

    ├── lanenet.py                      //LANENET模型

├── README.md                   //代码说明文档



2.om预测

├── eval_om.py                          //主代码

├── lanenet_model                                    

    ├── lanenet.py                      //LANENET模型

├── README.md                   //代码说明文档



3.ckpt转pb

├── ckpt2pb.py



##推理输出计算精度

```

python3.7 eval_om.py
```

## 推理精度
定量指标采用准确率，精度为0.965，精度达标。
定性指标采用论文中可视化binary segmentation和instance segmentation，输出在eval_output下

| gpu   | npu  |原论文 |推理   |
|-------|------|-------|-------|
|   96.5  |  96.5  |   96.4   |96.5 |



## 推理性能
inference time.jpg，性能达标