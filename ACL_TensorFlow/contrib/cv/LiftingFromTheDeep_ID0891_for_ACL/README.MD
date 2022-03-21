## LiftingFromTheDeep 离线推理方法

### Step0: 下载所需文件,安装依赖库
#### checkpoint 文件地址：
> obs://cann-id0891/ACL_Tensorflow/checkpoint/

#### bin 文件地址：
> obs://cann-id0891/ACL_Tensorflow/input/

#### prob_model_params 文件地址：
> obs://cann-id0891/ACL_Tensorflow/packages/

根据requirements文件安装后处理依赖库.

**注**:b_image.bin 由 test_image.jpg 经过预处理得到,可使用make_bin.py脚本自行转换. 请将 bin 文件和 jpg 图片文件一同放入 input 文件夹中.  
请将 prob_model_params.mat 文件放入packages文件夹中.

### Step1: 将 checkpoint 文件固化为 pb 文件：
将 convert_pb.py 中的 input_checkpoint 变量值改为 checkpoint 文件路径,
运行 convert_pb.py, 固化生成两个 pb 文件：
hmap.pb 和 posenet.pb

### Step2： 将 pb 文件转换为 om 文件：
需要使用 atc 工具.

**注**:由于有两个 pb 文件,因此要进行两次转换.

#### Step2.1 hamp.pb 转 hmap.om:

```
atc --model=./checkpoint/hmap.pb --framework=3 \
	--output=./checkpoint/om/hmap --soc_version=Ascend310 \
	--input_shape="CPM/Placeholder:1,368,654,3"
```


#### Step2.2 posenet.pb 转 posenet.om:

```
atc --model=./model/pb/posenet.pb --framework=3 \
    --output=./model/om/posenet --soc_version=Ascend310 \
    --input_shape="CPM/Placeholder_2:4,368,368,3;CPM/Placeholder_3:4,368,368,1"
```


### Step3 进行第一次推理：
使用 hmap.om 模型进行推理, 输入为 b_image.bin, 输出一个 bin 文件(hmap_output_0.bin).

```
./msame --model ./om/hmap.om \
	--input b_image.bin \
	--output ./output --outfmt BIN
```


### Step4 进行第一次后处理：
需要使用 Step3 中推理得到的 hmap_output_0.bin, 以及 b_image.bin  
将 process_hmap.py 中 b_image, heatmap_person_large 变量值中的地址改为对应文件地址(修改变量值中的时间戳).  
运行 process_hmap.py  
生成两个 bin 文件: b_pose_image.bin 和 b_pose_cmap.bin

### Step5 进行第二次推理：

使用 posenet.om 模型进行推理，输入为 b*pose_image.bin 和 b_pose_cmap.bin.
输出两个 bin 文件. (posenet_output_0.bin 和 posenet_output_1.bin)  

**注**: 注意将下列 msame 命令中om模型和bin文件路径改为文件所在路径,**input 参数值中逗号前后均不能有空格**


```
./msame --model './om/posenet.om' \
	--input './input/b_pose_image.bin,./input/b_pose_cmap.bin' \
	--output '/root/LiftingFromTheDeep/output' \
	--outfmt BIN
```

### Step6 进行第二次后处理：
需要使用 Step3 中推理得到的 hmap_output_0.bin,以及 Step5 中推理得到的 posenet_output_0.bin 和 posenet_output_1.bin

将 process_posenet.py 中 heatmap_person_large,pred_likelihood, pred_2d_pose 变量值中的地址改为对应文件地址(修改变量值中的时间戳).

```
对应关系:
heatmap_person_large -> hmap_output_0.bin
pred_likelihood -> posenet_output_0.bin (224 Bytes)
pred_2d_pose -> posenet_output_1.bin (896 Bytes)
```

运行 process_posenet.py, 可在result文件夹中找到结果.


## LiftingFromTheDeep 离线推理精度与推理时间
该模型为生成模型, 推理单张图片并不具有精度可供量化.  
在Ascend 310进行推理,   
第一次推理:
Inference average time: 57.324000 ms  
第二次推理:
Inference average time: 441.575000 ms

## 推理效果：
### 输入图片:
![test image](./input/test_image.jpg)

### 输出图片：
![2d result](./result/result2d.jpg)
![3d result](./result/result3d_0.jpg)


