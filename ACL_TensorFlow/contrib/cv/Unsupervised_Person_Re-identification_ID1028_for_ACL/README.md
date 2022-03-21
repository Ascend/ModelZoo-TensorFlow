## 离线推理
### 1、原始模型转PB模型
```
python3 export_test.py
```
生成的PB模型为：
```
链接：https://pan.baidu.com/s/1fL_89xnTf7nj7BcSgA7rUQ 
提取码：p05x
```
### 2、PB模型转OM模型
```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc  \
     --model=/home/data/dataset/wxm/tensorflow_model/model.pb \
     --framework=3 --output=/home/data/dataset/wxm/tensorflow_model/net_test \
     --soc_version=Ascend310 \
     --input_shape="input_1:1,224,224,3" \
     --log=info --out_nodes="avg_pool/AvgPool"
```
使用ATC模型转换工具转换PB模型到OM模型
### 3、数据预处理
特别注意，数据预处理需要分别对bounding_box_test与query两个文件夹均执行以下操作：
```
使用git命令方式获取img2bin脚本：
在命令行中：$HOME/AscendProjects目录下执行以下命令下载代码。
git clone https://gitee.com/ascend/tools.git
```
```
进入img2bin脚本所在目录
cd $HOME/AscendProjects/img2bin
```
```
执行
python3 img2bin.py -i /home/data/dataset/unsupervised/dataset/Market/query/ -w 224 -h 224 -f BGR -a NHWC -t float32 -m [104,117,123] -c [1,1,1] -o /home/data/dataset/wxm/out/market_query2
```
过程如下图
![数据预处理](picture/p1.png)

### 4、准备msame推理工具
参考[msame](https://gitee.com/ascend/tools/tree/master/msame)
### 5、om模型推理性能精度测试
#### 推理性能测试
##### 无输入版本
使用如下命令进行性能测试：
```
msame ./msame --model "/home/HwHiAiUser/wxm/net_base.om" --output "/home/HwHiAiUser/wxm/out" --outfmt TXT --loop 100
```
测试结果如下：
![无输入版本1](picture/p2.png)
![无输入版本2](picture/p3.png)

##### 输入待处理的文件夹
使用如下命令进行性能测试：
```
msame ./msame --model "/home/HwHiAiUser/wxm/net_test.om" --input "/home/HwHiAiUser/wxm/input/market_test2/" --output "/home/HwHiAiUser/wxm/out" --outfmt TXT --loop 1
```
测试结果：
```
Inference average time : 2.41 ms
Inference average time without first time: 2.41 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
```
![输入待处理的文件夹1](picture/p4.png)
![输入待处理的文件夹2](picture/p5.png)

#### 推理精度测试
使用如下命令进行精度测试：
```
python3 evaluate_offline.py --query_path=/home/HwHiAiUser/wxm/out/query/ --test_path=/home/HwHiAiUser/wxm/out/test/
```


