 # ATC Fpointnet


- references：

    ["Frustum PointNets for 3D Object Detection from RGB-D Data"](https://arxiv.org/pdf/1711.08488.pdf)


- training model：
    
    [Fpointnet_ID1470_for_TensorfFow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Fpointnet_ID1470_for_TensorfFow)


# 1. ckpt to pb

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

[ckpt](https://pan.baidu.com/s/1U7qJ4cYOokd8cRAmeezxHg)
Password:ms1m

# 2. pb to om
Command:
```
atc --input_shape="Placeholder:32,1024,4;Placeholder_1:32,3;Placeholder_3:32,3;Placeholder_9:32,512,2" --input_format=ND --output="fpointnet" --soc_version=Ascend310 --framework=3 --model="./fpointnet.pb"  --log=info 
```
[Pb](https://pan.baidu.com/s/1UzZkdb_1rkEOdNwWx50JFw)
Password:z8ml

[OM](https://pan.baidu.com/s/1Q-GDPUCVJ6-ZyuC7Xp_QEw)
Password:zzls

# 3. compile msame
Reference to https://gitee.com/ascend/tools/tree/master/msame, compile **msame** 

Compile msame command:
```bash
. /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/runtime/lib64/stub
cd $HOME/AscendProjects/tools/msame/
./build.sh g++ $HOME/AscendProjects/tools/msame/out

```

# 4. inference
Inference command:
```bash
cd $HOME/AscendProjects/tools/msame/out
model_path=/home/HwHiAiUser/AscendProjects/Fpointnet/fpointnet.om
output_path=/home/HwHiAiUser/AscendProjects/Fpointnet
./msame --model ${model_path} --output ${output_path}
```



Part of **Inference sys output**:
```bash
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./fpointnet.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
.//2022117_16_37_5_559775
[INFO] model execute success
Inference time: 29.151ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 29.151000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```



