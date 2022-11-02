# ATC mnasnet


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
Reference to https://gitee.com/ascend/tools/tree/ccl/msame, compile **msame** 

Compile masame command:
```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/acllib/lib64/stub
cd /root/msame/
bash build.sh g++ /root/msame/out
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



[Inference Sys Output](https://pan.baidu.com/s/1J0rwcydSh5f_bpq_Fvjpog)
Password:hfb0

Part of **Inference sys output**:
```bash
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:/home/HwHiAiUser/AscendProjects/SparseNet/test_bin_batchSize_64/110_batch_6976_7040.bin
[INFO] model execute success
Inference time: 235.143ms
```

[Inference Result](https://pan.baidu.com/s/1J0rwcydSh5f_bpq_Fvjpog)
Password:x88i

# 5. calculate Top-1 error

get **inference result** and use **calculate_kpi.py** get the **top-1 err**.

Top-1 error:
```bash
OM-Top1-err: 0.0643
GPU-Top1-err: 0.050
NPU-Top1-err: 0.048
```

# SparseNet implementation


```
@article{DBLP:journals/corr/abs-1801-05895,
  author    = {Ligeng Zhu and
               Ruizhi Deng and
               Michael Maire and
               Zhiwei Deng and
               Greg Mori and
               Ping Tan},
  title     = {Sparsely Aggregated Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1801.05895},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.05895},
  archivePrefix = {arXiv},
  eprint    = {1801.05895},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1801-05895},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
