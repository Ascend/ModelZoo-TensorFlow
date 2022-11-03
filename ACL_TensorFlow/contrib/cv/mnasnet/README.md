# ATC mnasnet


- references：

    ["Platform-Aware Neural Architecture Search for Mobile"](https://arxiv.org/abs/1807.11626)


- training model：
    
    [MnasNet_ID0728_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MnasNet_ID0728_for_TensorFlow)


# 1. ckpt to pb

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

[ckpt](https://pan.baidu.com/s/1-E3SQAxShCYcIVdkxbg19w)
Password:e3el

# 2. pb to om
Command:
```
atc --model=./mnasnet.pb  --framework=3  --input_shape="input1:1,224,224,3" --output=./mnasnet  --soc_version=Ascend910" 
```
[Pb](https://pan.baidu.com/s/1fUGFDZxi-6iit56PGN7sKg)
Password:qcvn

[OM](https://pan.baidu.com/s/1Z6IqgDpjC3h4sqhcX9ej8g)
Password:vghg

# 3. compile masame
Reference to https://gitee.com/ascend/tools/tree/master/msame, compile **msame** 

Compile masame command:
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
cd /root/msame/out
batchSize=64
model_path=/home/HwHiAiUser/AscendProjects/SparseNet/freezed_SparseNet_batchSize_${batchSize}.om
input_path=/home/HwHiAiUser/AscendProjects/SparseNet/test_bin_batchSize_${batchSize}
output_path=/home/HwHiAiUser/AscendProjects/SparseNet/output
./msame --model ${model_path} --input ${input_path} --output ${output_path} --outfmt TXT
```



Part of **Inference sys output**:
```bash
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./mnasnet.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
./output11/2022112_17_42_45_913229
[INFO] start to process file:./pic.bin
[INFO] model execute success
Inference time: 1.302ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 1.302000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```


Part of **Inference sys output**:
```bash
0.000487804 0.000569344 0.000521183 0.0006423 0.000464678 0.00140095 0.000912189 0.000928402 0.00101662 0.000784874 0.000334501 0.000647545 0.000609398 0.000686646 0.000246763 0.000668049 0.000214338 0.000707626
```
