## CSPNet 离线推理步骤

### Step0: 下载所需文件,安装依赖库
#### checkpoint 文件地址：
> obs://cann-id0891/ACL_Tensorflow/checkpoint/

#### bin 文件地址：
> obs://cann-id0891/ACL_Tensorflow/data/

根据requirements文件安装后处理依赖库.

**注**:X_test.bin 和Y_test.bin 由 cifar-10数据集中的 test_batch 经过预处理得到,可使用make_bin.py脚本自行转换.

### Step1: 将 checkpoint 文件固化为 pb 文件：
将 convert_pb.py 中的 input_checkpoint 变量值改为 checkpoint 文件路径,
运行 convert_pb.py, 固化生成 pb 文件:  
cspdarknet53.pb



### Step2： 将 pb 文件转换为 om 文件：
需要使用 atc 工具.
```
atc --model=./model/pb/cspdarknet53.pb --framework=3 \
	--output=./model/om/cspdarknet53 --soc_version=Ascend310 \
	--input_shape="Placeholder:10000,32,32,3"
```

### Step3 进行推理：
需要使用 msame 工具.
```
./msame --model ./model/om/cspdarknet53.om \
	--input ./data/Xtest.bin \
	--output ./output --outfmt BIN
```

### Step4 进行后处理：
需要使用 Step3 中推理得到的 cspdarknet53_output_0.bin
将 process.py 中 Y_estimation,Y_truth 变量值中的地址改为对应文件地址(修改变量值中的时间戳).

```
对应关系:
Y_estimation -> cspdarknet53_output_0.bin
Y_truth -> Ytest.bin
```
运行 process.py,得推理精度.

## CSPDarknet53 离线推理精度与推理时间
在Ascend 310进行推理, 含有10000条样本的bin文件，  
### 推理时间：  
8856.96ms, 每个样本平均0.886ms  
### 推理精度:  
0.6929, 即69.29%.  
对比GPU推理精度：  
0.7028, 即70.28%,达到精度要求
