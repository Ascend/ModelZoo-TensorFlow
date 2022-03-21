# Roberta-SQuAD离线推理

# 数据准备

URL:
[OBS](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OxyGnOefBaHkS3hOfHnbzE40RT3WmYc8akmb97WtrOa9psNZUNsYqHgFQ2V7SmyUyXVHQxIpJ4gFrxRA0502wE+NOBV1TNwcGJJw8ISaIBLfuz5RWu1KiNLAVBFAiltOfe2h4LeMjZByPmgP/2ehO+ggr6oQXjmB9Ew55SBJ1dIATdfvvBDQg0xWv6tF1EDz2AoMjPMr4EtgjiYyIPGFgu/nirlEV7DM9lIJN4KuHkG4O4z/bIeNsb0W53Pjgmz9mxQ34q6yDvrM+D8dse6yxjkgZJ8vcqQLsR9INuZYKyPwEsA8hKIj/s+leIYGxK7qVmUSi8n7ifCjCr0BQ38XEenH+OcJ8JCmTzk/PopV9X9KDqYAGPRZ9X7wB3/bzr+f7SSNwqPFOTzZxRIM1L8MAHjYv5Lt/pnvegTKtbuOyli3hLMjEEYpDWBBCPfxYyu2Ts9Z0s/SpDpwpHDOssQAAw44onlZj3UBDhp1koG6nSjrVu5mfFuEBsgPPOBV9SJR3KMCWgd/FGlQwuA/gSalfA==)

提取码:
111111

# STEP1:训练模型转PB文件

修改模型配置文件路径bert_config_file，pb文件输出路径output_dir，训练好的模型文件路径ckpt_dir后，运行frzee_graph.sh

```
python3 freeze_graph.py \
  --bert_config_file=/home/test_user02/lil/model/Roberta-large/config.json \
  --output_dir=/home/test_user02/lil/output/msame/squadv2 \
  --ckpt_dir=/home/test_user02/lil/data/ckpt/squadv1/model.ckpt-43800 \
  --max_seq_length=512
```

# STEP2:PB文件转OM文件

检查环境中ATC工具环境变量，设置完成后，修改PB文件路径PB_DIR，运行run_om.sh

```
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/te:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:${LD_LIBRARY_PATH}

PB_DIR=/home/test_user02/lil/output/msame/squadv2

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$PB_DIR/roberta.pb \
        --framework=3 \
        --output=$PB_DIR/roberta \
        --soc_version=Ascend910 \
        --input_shape="input_ids:1,512;input_mask:1,512;segment_ids:1,512" \
        --log=info \
        --out_nodes="logits:0"
```

# STEP3:数据转bin文件

修改文件输出路径BASE_DIR，tf_record文件路径TF_DIR后，运行convert_bin.sh

```
BASE_DIR=/home/test_user02/lil/output/msame/squadv2 #原始输入文件夹
TF_DIR=/home/test_user02/lil/data/SQuAD/eval/squadv2 #tf文件路径

#多输入的.bin格式数据
input_id_path=$BASE_DIR/input_ids
input_mask_path=$BASE_DIR/input_masks
segment_id_path=$BASE_DIR/segment_ids

if [ ! -d ${input_id_path} ]; then
  mkdir ${input_id_path}
fi
if [ ! -d ${input_mask_path} ]; then
  mkdir ${input_mask_path}
fi
if [ ! -d ${segment_id_path} ]; then
  mkdir ${segment_id_path}
fi

python3.7 convert_bin.py --base_dir ${BASE_DIR} --tf_dir ${TF_DIR} --max_seq_length=512
```

# STEP4:msame离线打通

安装好[msame]([tools: Ascend tools - Gitee.com](https://gitee.com/ascend/tools/tree/master/msame))，修改msame安装路径，bin文件路径BASE_DIR后，运行run_msame.sh

```
BASE_DIR=/home/test_user02/lil/output/msame/squadv2

input_id_path=$BASE_DIR/input_ids
input_mask_path=$BASE_DIR/input_masks
segment_id_path=$BASE_DIR/segment_ids
ulimit -c 0
/home/test_user02/lil/tools/msame/out/msame --model $BASE_DIR/roberta.om \
  --input ${input_id_path},${input_mask_path},${segment_id_path} \
  --output $BASE_DIR \
  --outfmt TXT
```

注意，msame生成的推理文件夹是根据时间命名的，类似于20210916_233440这样的格式，需要自己检查路径，在后续精度验证的步骤中修改。

## 推理性能

### SQuADv1.1

```
Inference time: 12.818ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 12.93 ms
Inference average time without first time: 12.93 ms
```

### SQuADv2.0

```
Inference time: 13.033ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 12.88 ms
Inference average time without first time: 12.88 ms
```

# STEP5:验证精度

## SQuADv1.1

修改推理文件路径rootdir，模型配置文件路径BERT_BASE_DIR，预测文件路径SQUAD_DIR，idx_file是步骤3转换数据bin过程产生的，路径需保持一致。

```
BASE_DIR=/home/test_user02/lil/output/msame/squadv2
BERT_BASE_DIR=/home/test_user02/lil/model/Roberta-large
SQUAD_DIR=/home/test_user02/lil/data/SQuAD/data

python3 evaluate.py \
  --rootdir=$BASE_DIR/20210916_154109 \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --idx_file=$BASE_DIR/idx.txt \
  --output_dir=$BASE_DIR \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --max_seq_length=512 \
  --version_2_with_negative=False

python3 $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $BASE_DIR/predictions.json
```

结果如下

```
{"exact_match": 83.41532639545885, "f1": 90.6048058023419}
```

## SQuADv2.0

修改推理文件路径rootdir，模型配置文件路径BERT_BASE_DIR，预测文件路径SQUAD_DIR，idx_file是步骤3转换数据bin过程产生的，路径需保持一致。

```
BASE_DIR=/home/test_user02/lil/output/msame/squadv2
BERT_BASE_DIR=/home/test_user02/lil/model/Roberta-large
SQUAD_DIR=/home/test_user02/lil/data/SQuAD/data

python3 evaluate.py \
  --rootdir=$BASE_DIR/20210917_095250 \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --idx_file=$BASE_DIR/idx.txt \
  --output_dir=$BASE_DIR \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --max_seq_length=512 \
  --version_2_with_negative=True

python3 $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $BASE_DIR/predictions.json \
  --na-prob-file $BASE_DIR/null_odds.json
```

结果如下

```
{
  "exact": 79.84502653078414,
  "f1": 83.47988017695722,
  "total": 11873,
  "HasAns_exact": 77.17611336032388,
  "HasAns_f1": 84.45624449072442,
  "HasAns_total": 5928,
  "NoAns_exact": 82.50630782169891,
  "NoAns_f1": 82.50630782169891,
  "NoAns_total": 5945,
  "best_exact": 80.37564221342542,
  "best_exact_thresh": -4.1171999999999995,
  "best_f1": 83.78534952815988,
  "best_f1_thresh": -3.99024
}
```



