# ATC EDSR


- references：

    ["Enhanced Deep Residual Networks for Single Image Super-Resolution"](https://arxiv.org/abs/1707.02921)


- training model：
    
    [EDSR_ID1263_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/EDSR_ID1263_for_TensorFlow)


# 1. ckpt to pb

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

 **Note** ：Before the model transformation, the **run.py** and **edsr.py** files in the training code need to be replaced with those in the current directory.

[ckpt](https://pan.baidu.com/s/17vlOrwXbygdce8l8OHErCA)
Password:x0j4

# 2. pb to om
Command:
```
atc --model=./edsr.pb --framework=3 --input_shape="LR:1,48,48,3" --output=./edsr --soc_version=Ascend310
```
[Pb](https://pan.baidu.com/s/1vxU_Q3qorOlPvDJYQ9tTsw)
Password:1wbo

[OM](https://pan.baidu.com/s/1QeQSRdckigMAbBRnXooRJA)
Password:djqe

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
model_path=/home/HwHiAiUser/AscendProjects/EDSR/edsr.om
input_path=/home/HwHiAiUser/AscendProjects/EDSR/testpic.bin
output_path=/home/HwHiAiUser/AscendProjects/EDSR/output
./msame --model ${model_path} --input ${input_path} --output ${output_path} --outfmt TXT
```



Part of **Inference sys output**:
```bash
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./edsr.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
./output/2022113_15_10_1_472972
[INFO] start to process file:./testpic.bin
[INFO] model execute success
Inference time: 20.962ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 20.962000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```
[Inference Result](https://pan.baidu.com/s/1PrIrKap_V0C_qe_bLNC7bA)
Password:y2ic



