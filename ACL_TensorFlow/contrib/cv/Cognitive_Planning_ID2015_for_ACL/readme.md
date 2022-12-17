#  **Cognitive_Planning**

## 离线推理

### 1. 原始模型转pb

```
python ./ckpt2pb.py  
```



### 2. pb转om模型

使用atc模型转换工具转换pb模型到om模型。

```
atc --model=/home/test_user07/Cognitive/pb/placeholder_protobuf.pb --framework=3 --output=/home/test_user07/Cognitive/om/cognitiveplanning_acc --soc_version=Ascend910 --input_shape="taskdata:8,20,64,64,90;taskdata_1:8,20,5;taskdata_2:8,20,8" --log=info --out_nodes="policy/Reshape_3:0"
```



### 3. 数据处理

对输入的数据进行处理，将数据变为BIN格式。在原环境中重新运行此py文件，此py文件添加了将数据处理为bin格式的相关代码。

```
python train_supervised_active_vision.py   --mode=train   --logdir=checkpoint   --modality_types=det   --batch_size=8   --train_iters=800000   --lstm_cell_size=2048   --policy_fc_size=2048   --sequence_length=20   --max_eval_episode_length=100   --test_iters=194   --gin_config=envs/configs/active_vision_config.gin   --gin_params="ActiveVisionDatasetEnv.dataset_root='AVD_Minimal'"   --logtostderr
```

生成taskdata_0.bin、taskdata_1.bin、taskdata_2.bin三个bin文件。

### 4. pb推理

```
python ./pb_inference.py
```

得到网络的输出结果

### 5. om模型推理

使用如下命令进行性能测试：

```
./msame --model "/home/test_user07/Cognitive/om/cognitiveplanning_acc.om" --input "/home/test_user07/Cognitive/bin/taskdata_0.bin,/home/test_user07/Cognitive/bin/taskdata_1.bin,/home/test_user07/Cognitive/bin/taskdata_2.bin" --output "/home/test_user07/Cognitive/outs/" --outfmt TXT  --loop 1
```

出现如下错误：

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/test_user07/Cognitive/om/cognitiveplanning_acc.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
EH0001: Value [0] for [size] is invalid. Reason: size must be greater than zero.

[ERROR] can't malloc buffer, size is 0, create output failed
[ERROR] create model output failed
[INFO] unload model success, model Id is 1
[ERROR] Sample process failed
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

由于pb推理可以成功，om推理出现问题，考虑工具还不支持静态输入，动态输出的相关问题，先提交于此，后面再做相关修改。

