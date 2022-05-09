## UGATIT 离线推理步骤

### Step0: 下载所需文件,安装依赖库
#### checkpoint 文件地址：
> obs://cann-id0722/ACL_Tensorflow/checkpoint/

#### bin 文件地址：
> obs://cann-id0722/ACL_Tensorflow/data/testA
> obs://cann-id0722/ACL_Tensorflow/data/testB

根据requirements文件安装后处理依赖库.

**注**:bin文件 由图像文件经过预处理得到,可使用make_bin.py脚本自行转换.

### Step1: 将 checkpoint 文件固化为 pb 文件：
将 convert_pb.py 中的 input_checkpoint 变量值改为 checkpoint 文件路径,
运行 convert_pb.py, 固化生成 pb 文件:  
UGATIT_AtoB.pb   
UGATIT_BtoA.pb  


### Step2： 将 pb 文件转换为 om 文件：
需要使用 atc 工具.

转换UGATIT_AtoB.pb命令:
```
atc --model=./model/pb/UGATIT_AtoB.pb --framework=3 \
	--output=./model/om/UGATIT_AtoB --soc_version=Ascend310 \
	--input_shape="test_domain_A:1,256,256,3" --precision_mode="force_fp32"  
```

转换UGATIT_BtoA.pb命令:
```
atc --model=./model/pb/UGATIT_BtoA.pb --framework=3 \
	--output=./model/om/UGATIT_BtoA --soc_version=Ascend310 \
	--input_shape="test_domain_B:1,256,256,3" --precision_mode="force_fp32"  
```


### Step3 进行推理：
需要使用 msame 工具.
使用UGATIT_AtoB.om推理命令:
```
./msame --model ./model/om/UGATIT_AtoB.om \
	--input ./data/testA/female_13138.bin \
	--output ./output/AtoB --outfmt BIN
```

使用UGATIT_BtoA.om推理命令:
```
./msame --model ./model/om/UGATIT_BtoA.om \
	--input ./data/testB/0000.bin \
	--output ./output/BtoA --outfmt BIN
```

### Step4 进行后处理：
需要使用 Step3 中推理得到的 UGATIT_output_0.bin
将 process.py 中 fake变量值中的地址改为对应文件地址(修改变量值中的时间戳).

运行 process.py,得推理图片结果.

## UGATIT离线推理精度与推理时间
在Ascend 310进行推理, 含有1张图片样本的bin文件，  
### 推理时间：  
2126.14ms 
### 推理结果
与GPU推理结果相同,达到精度要求
