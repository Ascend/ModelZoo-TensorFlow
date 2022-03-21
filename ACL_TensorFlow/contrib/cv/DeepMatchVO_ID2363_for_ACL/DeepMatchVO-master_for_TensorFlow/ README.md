# DeepMatchVO

原始模型模型链接：https://github.com/hlzz/DeepMatchVO

模型功能：采用自监督的方法实现视觉里程计，在一定程度上效果好于传统单目方法

## 离线推理

### 1、原始ckpt模型转换为pb

```
python freeze.py --ckpt_file='/home/DeepMatchVo_ID2363_for_TensorFlow/ckpt-yyw/model-258000' --pb_file='/home/DeepMatchVo_ID2363_for_TensorFlow/yyw_npu_258000.pb'
```
ckpt_file是训练好的ckpt文件，pb_file是生成pb文件

ckpt的obs链接：obs://obsdeepmatchvo/ckpt_yyw_258000_npu/

（绝大多数推理文件文件都可以在obs://obsdeepmatchvo目录下选择下载，其他原始数据文件obs://obsdeepmatchvo/root/kitti/在中）

### 2、pb转化为om模型

```
atc --model=/home/HwHiAiUser/AscendProjects/DeepMatchVO/pb_model/yyw_npu_258000.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/DeepMatchVO/om_yyw_npu_258000 --soc_version=Ascend310 --input_shape="data_loading/sub:4,128,416,3;data_loading/sub_1:4,128,416,6" --log=info --out_nodes="pose_and_explainability_prediction/pose_exp_net/pose/mul:0"
```
此过程在推理服务器上实现 需要提前source环境变量、设置输出日志打屏等，input_shape是根据netron打开pb模型在所得

pb的obs链接：obs://obsdeepmatchvo/model_om&pb/

om的obs链接：obs://obsdeepmatchvo/model_om&pb/


### 3、数据处理

```
python pre_data.py --test_seq=09 --dataset_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/dataset/'   --seq_length=3 --concat_img_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/generatetestimage' --bin_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/bin/'
```
如果只验证推理过程，不用下载全部文件

dataset_dir可以只存放dataset/sequences/09的times.txt和calib.txt文件（其他文件是训练时候使用）

concat_img_dir可以只存放09序列

bin_dir会生成两个文件夹，分别对应tgt和src

dataset的obs链接：obs://obsdeepmatchvo/root/kitti/dataset/

concat_img_dir的obs链接：obs://obsdeepmatchvo/root/kitti/generatetestimage/

bin的obs链接：obs://obsdeepmatchvo/bin/

### 4、msame推理

```
./msame --model /home/HwHiAiUser/AscendProjects/DeepMatchVO/om_yyw_npu_258000.om --input /home/HwHiAiUser/AscendProjects/DeepMatchVO/bin/tgt/,/home/HwHiAiUser/AscendProjects/DeepMatchVO/bin/src/ --output /home/HwHiAiUser/AscendProjects/DeepMatchVO/output_yyw_npu258000 --outfmt TXT --loop 2
```

```
Inference average time : 2.18 ms
Inference average time without first time: 2.18 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

msame需要自行编译，如果服务器上有output/msame可直接使用

model为步骤2生成的om模型

input步骤3生成的bin

output输出txt格式结果，结果存放在一个带有时间的文件夹内2022119_16_57_44_942884，每个batch推理时间为2.18ms

结果的的obs链接：obs://obsdeepmatchvo/2022119_16_57_44_942884/

### 4、结果预处理

```
python pre_result.py --test_seq=09 --dataset_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/dataset/' --output_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/output_om_npu258000/09/'  --seq_length=3 --concat_img_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/kitti/generatetestimage' --result_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/output_yyw_npu258000/2022119_16_57_44_942884/' 
```
结果预处理，output_dir是最终输出，result_dir是步骤3输出文件夹

### 5、评估结果

```
python kitti_eval/eval_pose.py --gtruth_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/DeepMatchVO-master_for_TensorFlow/kitti_eval/pose_data/ground_truth/seq3/09/' --pred_dir='/home/DeepMatchVo_ID2363_for_TensorFlow/output_om_npu258000/09/'
```

```
ATE mean: 0.0128, std: 0.0069
```

--pred_dir是步骤四生成的结果，最终结果打屏显示，推理结果和原始npu_ckpt_258000一样，精度达标

gtruth_dir的obs链接：obs://obsdeepmatchvo/pose_data/ground_truth/


|   | 论文  |  作者给出258000 |  GPU复现240000  |  NPU复现140000 |  GPU复现258000 | NPU复现258000  |  推理模型npu258000 |
|---|---|---|---|---|---|---|---|
| seq09 ATE  | 0.0089  |0.0091   |0.0116   |0.0115   |0.0111   |0.0128   | 0.0128  |
| seq09 std  | 0.0054  |0.0055   |0.0074   |0.0072   |0.0069   |0.0069   | 0.0069  |

整体来看GPU、NPU、NPU的推理三者效果一致 与原作者论文和原作者给出的结果有一定差距

