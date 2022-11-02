# ATC mnasnet
Platform-Aware Neural Architecture Search for Mobile  [[PDF](https://arxiv.org/abs/1807.11626)]

Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
# 1. original model

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

[ckpt](https://pan.baidu.com/s/1-E3SQAxShCYcIVdkxbg19w)
Password:e3el

# 2. pb to om
Command:
```
atc --model=./mnasnet.pb  --framework=3  --input_shape="input1:1, 224, 224, 3" --output=./mnasnet  --soc_version=Ascend910" 
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
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/HwHiAiUser/AscendProjects/SparseNet/test_bin_batchSize_64/110_batch_6976_7040.bin
[INFO] model execute success
Inference time: 235.143ms
```


Part of **Inference sys output**:
```bash

```
