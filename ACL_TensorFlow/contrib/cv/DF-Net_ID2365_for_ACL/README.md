# DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency
## 离线推理
### 1. 原始模型转pb

权重文件obs地址：obs://cann-id2365/inference/ckpt/

深度网络:

设置第10行ckpt_path为以上权重文件下载到的存放地址，运行以下命令即可生成pb文件：depth.pb
```
python3 ckpt2pb_depth.py  
```

光流网络:

设置第9行ckpt_path为以上权重文件下载到的存放地址，运行以下命令即可生成pb文件：flow.pb
```
python3 ckpt2pb_flow.py
```

已转换好的pb模型，obs地址： 

obs://cann-id2365/inference/depth.pb 

obs://cann-id2365/inference/flow.pb

### 2. pb转om模型

使用atc模型转换工具转换pb模型到om模型

深度网络： 
```
atc --model=depth.pb --framework=3 --output=depth_om --soc_version=Ascend310 --input_shape="input:1,160,576,3" --log=info --out_nodes="output:0" 
```
光流网络：
```
atc --model=flow.pb --framework=3 --output=flow_om --soc_version=Ascend310 --input_shape="input:1,2,384,1280,3" --log=info --out_nodes="output:0"
```

转换好的OM模型，obs地址：  

obs://cann-id2365/inference/depth_om.om 

obs://cann-id2365/inference/flow_om.om

### 3. 数据处理

将输入的测试图片做与ckpt测试时相同预处理，再转化为BIN格式。

深度网络：

原始JPG数据的obs地址：obs://cann-id2365/dataset/KITTI/raw/data/

设置第10行的dataset_dir，表示原始JPG数据的路径。运行以下命令，在dataset_dir路径下生成bin格式的测试数据。
```
python3 tobin_depth.py
```

光流网络：

原始JPG数据的obs地址：obs://cann-id2365/dataset/data_scene_flow_2015/training/

设置第61行的dataset_dir，表示原始JPG数据的路径。运行一下命令，在dataset_dir路径下生成bin格式的测试数据。
```
python3 tobin_flow.py
```
生成的输入数据bin文件，obs地址：
 
obs://cann-id2365/inference/dataset/

### 4. 准备msame推理
参考[msame](https://gitee.com/ascend/modelzoo/wikis/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E5%B7%A5%E5%85%B7msame%E4%BD%BF%E7%94%A8%E6%A1%88%E4%BE%8B)

### 5. om模型推理

使用如下命令进行性能测试：

深度网络（以序列0002为例，有多个序列需要运行多次）：
```
./msame --model /root/DF-Net/depth_om.om --input /root/DF-Net/dataset/depth/bin/data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/ --output /root/DF-Net/depth_output/0002
```

测试结果如下：

```
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 13.52 ms
Inference average time without first time: 13.51 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```
深度网络单张图片的平均推理时间为13.52 ms

光流网络：
```
./msame --model /root/DF-Net/flow_om.om --input /root/DF-Net/dataset/flow/image_2_bin/ --output /root/DF-Net/flow_output
```
测试结果如下：
```
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 128.66 ms
Inference average time without first time: 128.66 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```
光流网络单张图片的平均推理时间为128.66 ms

### 6. om精度测试

深度网络：
```
python3 bintoacc_depth.py --kitti_dir $/dataset/KITTI/raw/data/ --pred_bin_file $/depth_output/
```
测试结果如下：
|      | abs_rel | sq_rel | rms    | log_rms | a1     | a2     | a3   |
|------|---------|--------|--------|---------|--------|--------|------|
| 原论文  | 0.1452  | 1.2904  | 5.6115  | 0.2194  | 0.8114  | 0.9394  | 0.9767  |
| GPU  | 0.1214  | 0.7413 | 4.7026 | 0.1898  | 0.8563 | 0.9585 | 0.9834 |
| 离线推理 | 0.1214  | 0.7413 | 4.7025 | 0.1898  | 0.8564 | 0.9585 | 0.9834 |


光流网络：

第13行设置dataset_dir：$/dataset/data_scene_flow_2015/training/
```
python3 bintoacc_flow.py
```

测试结果如下：
|      | endpoint_error | f1 score |
|------|----------------|----------|
| 原论文  |  7.4482        | 0.2695    |
| GPU  | 7.9336         | 0.2321   |
| 离线推理 | 7.9371         | 0.2322   |

 
离线推理精度达标  
 
om推理输出bin文件，obs地址： 

obs://cann-id2365/inference/depth_output/

obs://cann-id2365/inference/flow_output/
