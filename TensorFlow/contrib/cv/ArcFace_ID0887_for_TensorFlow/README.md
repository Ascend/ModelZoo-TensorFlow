## 模型功能

人脸识别

## 原始模型

参考实现 ：

https://github.com/luckycallor/InsightFace-tensorflow

## 训练命令
```
code_dir=/
data_dir=/
result_dir=/

current_time=`date "+%Y-%m-%d-%H-%M-%S"`
python3.7 ${code_dir}/train_npu.py \
        --input_dir=${data_dir} \
        --result=${result_dir} \
        --code_dir=${code_dir} \
        --chip='npu' \
        --platform='apulis' \
        --npu_profiling=False  2>&1 | tee ${result_dir}/${current_time}_train_npu.log
```
## 推理命令
```
code_dir=/
data_dir=/
result_dir=/
model_path=/

current_time=`date "+%Y-%m-%d-%H-%M-%S"`
python3.7 ${code_dir}/evaluate_npu.py \
        --input_dir=${data_dir} \
        --result=${result_dir} \
        --obs_dir=${obs_url} \
        --model_path=${model_path} \
        --code_dir=${code_dir}  2>&1 | tee ${result_dir}/${current_time}_train_npu.log
        
```

## pb模型
```
code_dir=/
model_path=/
result=/

current_time=`date "+%Y-%m-%d-%H-%M-%S"`
python3.7 ${code_dir}/freeze_graph.py \
          --code_dir=${code_dir} \
          --model_path=${model_path} \
          --result=${result} 2>&1 | tee ${result}/${current_time}_freeze.log

```

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=pb_path --framework=3 --output=om_path --soc_version=Ascend310 --input_format=NHWC --output_type=FP32 \
        --input_shape="input:1,112,112,3" \
        --log=info \
        --out_nodes="embd_extractor/BatchNorm_1/Reshape_1:0" \
        --precision_mode=allow_mix_precision
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

## 数据集转换bin
data_dir为验证数据集的路径
data_bin_dir为read2bin.py转化的bin数据集的路径

```
python3.7 read2bin.py --read_dir=${data_dir} --save_path=${data_bin_dir}
```

## 推理测试

使用msame推理工具，参考如下命令，发起推理测试： 

```
./msame --model om_path --input ${data_bin_path} --output ${output_path} 

```

对多个验证集进行推理输出

output_dir为输出推理结果的路径

data_bin_dir为read2bin.py转化的bin数据集的路径

inference_dir为推理脚本inference.sh的路径
```
python3.7 get_embd.py --output_dir=${output_dir} --data_bin_dir=${data_bin_dir} --inference_dir=${inference_dir}
```

计算精度
output_dir为推理得到的输出文件路径
data_dir为验证数据集的路径
```
python3.7 evaluate_310.py --output_dir=${output_dir} --data_dir=${data_dir}
```

## npu训练精度

|lfw|calfw|cplfw|agedb_30|cfp_ff|cfp_fp|vgg2_fp|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|99.583%|94.983%|89.583%|95.567%|99.671%|96.157%|94.920%|


## 训练性能对比

batchsize=32:

|platform|npu|gpu|
|:----:|:----:|:----:|
|FPS|460|290|