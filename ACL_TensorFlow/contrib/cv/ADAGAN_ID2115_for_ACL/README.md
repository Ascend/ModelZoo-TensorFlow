## 模型功能

 解决模式缺失问题




## pb模型

```

python3.7 ckpt_pb.py
```

   freeze_graph2（）函数，pb模型及相关文件在百度网盘中,链接：https://pan.baidu.com/s/1CQ6iDR1Qg7pt8B-WJW7-5g 
提取码：6666

## om模型


使用ATC模型转换工具进行模型转换时可以参考如下指令:


```
atc --model=/usr/model_test/frozen_model.pb 

       --framework=3
 
      --output=/usr/model_test/frozen_model

       --soc_version=Ascend310 

       --out_nodes="GENERATOR/output:0" 

       --input_shape "noise_ph:5000,5"

```



## 使用msame工具推理


参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。



## 数据集转换bin


```
python3.7 ckpt_pb.py
```

freeze_graph_test（）函数 noise.npy转noise.bin

## 推理测试


使用msame推理工具，参考如下命令，发起推理测试：
 

```
./msame --model "/usr/model_test/frozen_model.om"
            
--input "/usr/model_test/noise.bin" 

            --output "/usr/model_test/output/"
            
--outfmt TXT  

            --loop 1

```


## 脚本和示例代码
1.pb预测

├── adagan_gmm_pb.py                              //主代码

├── README.md                                     //代码说明文档

├── adagan_pb.py                                  //权重更新

├── gan_pb.py                                     //gan网络模型

├── metrics.md                                    //plot图

2.om预测

├── adagan_gmm_om.py                             //主代码

├── README.md                                    //代码说明文档

├── adagan_om.py                                 //权重更新

├── gan_om.py                                    //gan网络模型

├── metrics.md                                   //plot图 

3.转pb

├── ckpt_pb.py                                   //冻结pb、转bin文件 

##推理输出计算精度


```
python3.7 adagan_gmm.py
```
out.npy与om推理出的数据处理相同

## 推理精度
0.00-0.01，在推理精度文件中，模型是第十次迭代获取的，此时的生成器权重偏向于为被覆盖的部分，而此时已实现全覆盖，覆盖率为1，生成器新生成的fake点将不会覆盖real点，计算的值为0-0.01，所以其精度达标。

| gpu   | npu  |原论文 |推理   |
|-------|------|-------|-------|
|   1   |  1   |   1   |0.99-1 |