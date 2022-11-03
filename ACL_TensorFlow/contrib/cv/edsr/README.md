# ATC EDSR


- references：

    ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/abs/1707.02921)


- training model：
    
    [EDSR_ID1263_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/EDSR_ID1263_for_TensorFlow)


# 1. ckpt to pb

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

[ckpt](https://pan.baidu.com/s/1-E3SQAxShCYcIVdkxbg19w)
Password:e3el

# 2. pb to om
Command:
```
atc --model=./mnasnet.pb  --framework=3  --input_shape="input1:1,224,224,3" --output=./mnasnet  --soc_version=Ascend910" 
```
[Pb](https://pan.baidu.com/s/1YhB_1zjYb2dz_h8P_kIGUQ)
Password:m6mx

[OM](https://pan.baidu.com/s/1mKV8wkUBz3KiF8hpxUh9mA)
Password:zdo1

# 3. compile msame
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
cd $HOME/AscendProjects/tools/msame/out
model_path=/home/HwHiAiUser/AscendProjects/MnasNet/mnasnet.om
input_path=/home/HwHiAiUser/AscendProjects/MnasNet/pic.bin
output_path=/home/HwHiAiUser/AscendProjects/MnasNet/output
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
./output/2022113_9_31_13_612340
[INFO] start to process file:./pic.bin
[INFO] model execute success
Inference time: 1.359ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 1.359000 ms
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
0.000629902 0.000514507 0.000611782 0.000699043 0.000445127 0.00120354 0.00102234 0.00104713 0.0011034 0.000992775 0.000550747 0.00101948 0.00100136 0.000835419 0.000398874 0.000741005 0.000406742 0.00107861
```
