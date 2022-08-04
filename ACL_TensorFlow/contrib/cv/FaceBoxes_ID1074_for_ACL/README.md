
## 1、原始模型
运行ck2pb.py将训练好的ckpt模型转换为pb模型，其中modelfile用于指定ckpt模型的位置

## 2、转om模型

atc转换命令参考：

```sh
atc --model=./faceboxes.pb --framework=3 --output=./faceboxes_base310 --soc_version=Ascend310         --input_shape="image_tensor:1,1024,1024,3"         --log=info          --out_nodes="nms/map/TensorArrayStack/TensorArrayGatherV3:0;nms/map/TensorArrayStack_1/TensorArrayGatherV3:0;nms/map/TensorArrayStack_2/TensorArrayGatherV3:0"
```


## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具




## 4、数据集准备
测试使用的是[FDDB数据集](http://vis-www.cs.umass.edu/fddb/index.html#download)，将数据集下载后解压

运行img2bin.py将FDDB数据库中的图片转换为bin文件，需要自行调整文件路径。

运行createellipselist.py生成ellipseList.txt和faceList.txt，需要自行调整文件路径。


## 5、执行推理

  
执行命令 
```sh
./msame --model "/home/test_user04/model_base310.om" --input "/home/test_user04/inference_data" --output "/home/test_user04/" --outfmt TXT  --outputSize "10000,10000,10000"
```


## 5、性能

![输入图片说明](time.png)


## 6、精度计算
![输入图片说明](roc.png)
