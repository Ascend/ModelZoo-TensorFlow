# ATC SparseNet
Sparsely Aggregated Convolutional Networks [[PDF](https://arxiv.org/abs/1801.05895)]

[Ligeng Zhu](https://lzhu.me), [Ruizhi Deng](http://www.sfu.ca/~ruizhid/), [Michael Maire](http://ttic.uchicago.edu/~mmaire/), [Zhiwei Deng](http://www.sfu.ca/~zhiweid/), [Greg Mori](http://www.cs.sfu.ca/~mori/), [Ping Tan](https://www.cs.sfu.ca/~pingtan/)

# 1. test dataset to bin
[CIFAR10](https://pan.baidu.com/s/1drCJNhNs5Ek6Mm92TGfEYw)
Password:zmof

[Test Dataset](https://pan.baidu.com/s/1-KBREYkGgBfr9yV96O4xkw)
Password:on6n

<table> 
<th>Data augmentation: 
</th>
<td>standard
<td>mirroring
<td>shifting
</table>

<table> 
<th>Preprocessing: </th>
<td>normalize the data by the channel mean and standard deviation
</table>

download **Test Dataset** and use **img2bin.py** process dataset to bin file.

[Test Dataset Bin File](https://pan.baidu.com/s/1tQxnY7MF75CYoS9ZhgaSKA)
Password:dcjn

# 2. pb to om
Command:
```bash
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/5.0.4.alpha002/acllib/lib64/stub
pb_path=/home/HwHiAiUser/AscendProjects/SparseNet/pb/SparseNet_freeze.pb
batchSize=64
output_om_path=/home/HwHiAiUser/AscendProjects/SparseNet/freezed_SparseNet_batchSize_${batchSize}
atc --model=${pb_path} --framework=3 --output=${output_om_path}  --soc_version=Ascend310  --input_shape="input:${batchSize},32,32,3" --input_format=NHWC --log=debug --debug_dir=/home/HwHiAiUser/AscendProjects/SparseNet/debug_info --out_nodes="output:0" 
```
[Pb](https://pan.baidu.com/s/17m7o1BUAkdOuGKTd2_SZrg)
Password:mvqu

[OM](https://pan.baidu.com/s/1NczbTg7XPzhjsdj-AnyvAQ)
Password:wtp9

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
