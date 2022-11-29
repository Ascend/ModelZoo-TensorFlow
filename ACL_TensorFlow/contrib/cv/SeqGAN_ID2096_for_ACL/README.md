# ATC SeqGAN


- references：

    ["SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient"](https://arxiv.org/abs/1609.05473)


- training model：
    
    [SeqGAN_ID2096_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/SeqGAN_ID2096_for_TensorFlow)


# 1. ckpt to pb

download **ckpt** and use **ckpt2pb.py** process ckpt to pb.

[ckpt](https://pan.baidu.com/s/1hTrdrkppaHT_onXrrTMwFg)
Password:lzqe

# 2. pb to om
Command:
```
atc --model=seqgan.pb --framework=3 --input_format="ND"  --input_shape="Placeholder:64,20" --output=seqgan --soc_version=Ascend310 
```
[Pb](https://pan.baidu.com/s/1zudcMVdDYCDN5q4LDrDXrg)
Password:yh6q

[OM](https://pan.baidu.com/s/1HJZhBZKBX2W6i6daaMKuuw)
Password:9p09

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
model_path=/home/HwHiAiUser/AscendProjects/SeqGAN/seqgan.om
output_path=/home/HwHiAiUser/AscendProjects/SeqGAN
./msame --model ${model_path} --output ${output_path} --outputSize 25600000
```



Part of **Inference sys output**:
```bash
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./seqgan.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
.//2022118_17_18_37_136806
[INFO] model execute success
Inference time: 181.534ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 181.534000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

