# GeoNet

## 离线推理

### 1、原始ckpt模型转换为pb

```
python freeze.py 
```
其中64,65行可修改ckpt和pb文件的路径。

ckpt的obs链接：obs://cann-id2357/tuili/ckpt/

pb的obs链接：obs://cann-id2357/tuili/pb/


### 2、pb转化为om模型

```
atc --model=/home/HwHiAiUser/AscendProjects/GeoNet/pb/model-185000.pb --framework=3 --output=/home/HwHiAiUser/AscendProjects/GeoNet/om/om_185000 --soc_version=Ascend310 --input_shape="sub:1,128,416,3;sub_1:1,128,416,6" --log=info --out_nodes="pose_net/mul:0"

```
om的obs链接：obs://cann-id2357/tuili/om/


### 3、数据处理

```
python geonet_main.py --mode=data_pre --dataset_dir=/home/yhm/stageone/test/ --batch_size=4 --seq_length=3 --pose_test_seq=9 --bin_dir=/home/yhm/bin/

```

注意需要在bin_dir下手动生成两个文件夹，分别命名为src和tgt。否则会报错，显示找不到文件夹。

dataset的obs链接：obs://cann-id2357/dataset/stageone/test/

bin文件的obs链接：obs://cann-id2357/tuili/bin/

### 4、msame推理

```
 /home/HwHiAiUser/AscendProjects/SID/tools/msame/out/msame --model /home/HwHiAiUser/AscendProjects/GeoNet/om/om_185000.om --input /home/HwHiAiUser/AscendProjects/GeoNet/bin/tgt,/home/HwHiAiUser/AscendProjects/GeoNet/bin/src --output /home/HwHiAiUser/AscendProjects/GeoNet/output/ --outfmt TXT --loop 2

```
运行结果：
![输入图片说明](msame.png)
```
Inference average time : 1.01 ms
Inference average time without first time: 1.01 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

output输出txt格式结果，结果存放在一个带有时间的文件夹内2022211_16_34_38_219337，每个batch推理时间为1.01ms。

推理使用了不同迭代次数的模型，以下是文件夹名称与模型对应关系：
```
2022211_16_37_48_402205------------160000
2022211_16_36_59_8290 -------------170000
2022211_14_49_25_779584 -----------180000
2022211_16_34_38_219337 -----------185000
2022211_16_39_17_805299------------190000
```

om推理结果的obs链接：obs://cann-id2357/tuili/om_result/

### 4、结果预处理

```
 python geonet_main.py --mode=result_pre --dataset_dir=/home/yhm/stageone/test/ --batch_size=4 --seq_length=3 --pose_test_seq=9 --result_dir=/home/yhm/om_result/2022211_16_34_38_219337 --output_dir=/home/yhm/output/tuili_185000/
 
```
结果预处理，output_dir是最终输出，result_dir是步骤3输出文件夹。

最终输出的obs链接：obs://cann-id2357/tuili/output/

### 5、评估结果

```
 python kitti_eval/eval_pose.py --gtruth_dir=/home/yhm/stageone/eval/09_snippets/ --pred_dir=/home/yhm/output/tuili_185000/

```

```
ATE mean: 0.0174, std: 0.0139
```

--pred_dir是步骤四生成的结果，最终结果打屏显示，推理结果和原始npu_ckpt_185000接近，精度达标

gtruth_dir的obs链接：obs://cann-id2357/dataset/stageone/eval/09_snippets/
![输入图片说明](result.png)




