# 概述

Roberta是一个Transformer模型，它以自监督的方式在大量英文数据上进行了预训练。预训练过程中，以掩码语言模型为训练目标。取一个句子，模型随机屏蔽输入句子15%的单词，让模型去预测被屏蔽掉的单词。这种方式不同于一个接一个地看到单词的传统循环神经网络，在内部就屏蔽掉了后来要预测的单词，因此Roberta预训练模型允许模型学习句子的双向表示，提取到的特征可用于不同类型的下游任务。

+ 参考论文

  [ RoBERTa: A Robustly Optimized BERT Pretraining Approach (arxiv.org)](https://arxiv.org/abs/1907.11692)

+ 参考实现
  + NVIDIA:https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling
  + Google:https://github.com/google-research/bert

# 准备数据

论文预训练语料共160G，包括五个数据集。

* BookCorpus
* English Wikipedia
* CC-News
* OpenWebText
* Stories

SQuAD问答任务包含两个版本，v1.1与v2.0.

* SQuAD

预训练模型权重参考

+ Roberta-base
+ Roberta-large

已经处理好的部分数据集及预训练模型权重，请用户自行下载。

+ URL:
  [OBS](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=OxyGnOefBaHkS3hOfHnbzE40RT3WmYc8akmb97WtrOa9psNZUNsYqHgFQ2V7SmyUyXVHQxIpJ4gFrxRA0502wE+NOBV1TNwcGJJw8ISaIBLfuz5RWu1KiNLAVBFAiltOfe2h4LeMjZByPmgP/2ehO+ggr6oQXjmB9Ew55SBJ1dIATdfvvBDQg0xWv6tF1EDz2AoMjPMr4EtgjiYyIPGFgu/nirlEV7DM9lIJN4KuHkG4O4z/bIeNsb0W53Pjgmz9mxQ34q6yDvrM+D8dse6yxjkgZJ8vcqQLsR9INuZYKyPwEsA8hKIj/s+leIYGxK7qVmUSi8n7ifCjCr0BQ38XEenH+OcJ8JCmTzk/PopV9X9KDqYAGPRZ9X7wB3/bzr+f7SSNwqPFOTzZxRIM1L8MAHjYv5Lt/pnvegTKtbuOyli3hLMjEEYpDWBBCPfxYyu2Ts9Z0s/SpDpwpHDOssQAAw44onlZj3UBDhp1koG6nSjrVu5mfFuEBsgPPOBV9SJR3KMCWgd/FGlQwuA/gSalfA==)

  提取码:
  111111

# 模型训练

## 预训练

首先生成训练数据，修改BookCorpus_DIR与BERT_BASE_DIR后，运行create_pretraining_data.sh

```
BookCorpus_DIR=/home/TestUser03/bupt_lil/Data/BookCorpus
BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large

python3 create_pretraining_data.py \
  --input_file=$BookCorpus_DIR/test_bookscorpus.txt \
  --output_file=$BookCorpus_DIR/pretrain_data \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

修改BERT_BASE_DIR、BookCorpus_DIR、OUT_DIR后，运行run_pretraining.sh开始预训练。

```
export JOB_ID=10086
export ASCEND_DEVICE_ID=0

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
BookCorpus_DIR=/home/TestUser03/bupt_lil/Data/BookCorpus
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/pretrain

#start exec
python3.7 run_pretraining.py \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_warmup_steps=100 \
  --num_train_steps=100000 \
  --optimizer_type=adam \
  --manual_fp16=True \
  --use_fp16_cls=True \
  --input_files_dir=$BookCorpus_DIR/pretrain_data \
  --eval_files_dir=$BookCorpus_DIR/eval_data \
  --npu_bert_debug=False \
  --npu_bert_use_tdt=True \
  --do_train=True \
  --num_accumulation_steps=1 \
  --npu_bert_job_start_file= \
  --iterations_per_loop=100 \
  --save_checkpoints_steps=10000 \
  --npu_bert_clip_by_global_norm=False \
  --distributed=False \
  --npu_bert_loss_scale=0 \
  --output_dir=$OUT_DIR \
  --out_log_dir=$OUT_DIR/loss
```

## 下游任务

### SQuADv1.1

修改BERT_BASE_DIR、SQUAD_DIR、OUT_DIR后，运行run_squadv1.1.sh开始微调并验证。

```
export JOB_ID=10086
export ASCEND_DEVICE_ID=0

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
SQUAD_DIR=/home/TestUser03/bupt_lil/Data/SQuAD/data
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/v1-384

#rm -rf output

python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/tf_model/roberta_large.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=4 \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --version_2_with_negative=False


python3 $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUT_DIR/predictions.json
```

精度

```
{"exact_match": 83.56669820245979, "f1": 90.73758858437041}
```



### SQuADv2.0

修改BERT_BASE_DIR、SQUAD_DIR、OUT_DIR后，运行run_squadv2.0.sh开始微调并验证。

```
export JOB_ID=10086
export ASCEND_DEVICE_ID=0

BERT_BASE_DIR=/home/TestUser03/bupt_lil/Model/Roberta-large
SQUAD_DIR=/home/TestUser03/bupt_lil/Data/SQuAD/data
OUT_DIR=/home/TestUser03/bupt_lil/Output/Roberta/modelzoo/v2-512

#rm -rf output

python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.json \
  --merges_file=$BERT_BASE_DIR/merges.txt \
  --bert_config_file=$BERT_BASE_DIR/config.json \
  --init_checkpoint=$BERT_BASE_DIR/tf_model/roberta_large.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=4 \
  --predict_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --version_2_with_negative=True


python3 $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $OUT_DIR/predictions.json \
  --na-prob-file $OUT_DIR/null_odds.json
```

精度

```
{
  "exact": 79.28914343468374,
  "f1": 82.8272303206072,
  "total": 11873,
  "HasAns_exact": 77.20985155195682,
  "HasAns_f1": 84.29617165934029,
  "HasAns_total": 5928,
  "NoAns_exact": 81.36248948696384,
  "NoAns_f1": 81.36248948696384,
  "NoAns_total": 5945,
  "best_exact": 80.08927819422219,
  "best_exact_thresh": -5.314453125,
  "best_f1": 83.35610221328716,
  "best_f1_thresh": -4.283203125
}

```

# GPU/NPU精度与性能对比

本案例中GPU特指NVIDIA Tesla V100

## 性能

预训练阶段只有性能，精度在下游任务中体现。

单位global_step/sec，表示全局步数/秒.

|           | GPU  | NPU  |
| --------- | ---- | ---- |
| pretrain  | 1.04 | 3.12 |
| SQuADv1.1 | 1.61 | 3.46 |
| SQuADv2.0 | 1.61 | 3.68 |

## 精度

### SQuADv1.1

GPU：

```
{"exact_match": 83.6329233680227, "f1": 90.69222754254643}
```

NPU：

```
{"exact_match": 83.56669820245979, "f1": 90.73758858437041}
```

### SQuADv2.0

GPU：

```
{
  "exact": 80.16508043459952,
  "f1": 83.4364847051163,
  "total": 11873,
  "HasAns_exact": 77.76653171390014,
  "HasAns_f1": 84.31872181238975,
  "HasAns_total": 5928,
  "NoAns_exact": 82.55677039529016,
  "NoAns_f1": 82.55677039529016,
  "NoAns_total": 5945,
  "best_exact": 80.76307588646509,
  "best_exact_thresh": -5.044147253036499,
  "best_f1": 83.78073103615503,
  "best_f1_thresh": -3.0552178025245667
}

```

NPU：

```
{
  "exact": 79.28914343468374,
  "f1": 82.8272303206072,
  "total": 11873,
  "HasAns_exact": 77.20985155195682,
  "HasAns_f1": 84.29617165934029,
  "HasAns_total": 5928,
  "NoAns_exact": 81.36248948696384,
  "NoAns_f1": 81.36248948696384,
  "NoAns_total": 5945,
  "best_exact": 80.08927819422219,
  "best_exact_thresh": -5.314453125,
  "best_f1": 83.35610221328716,
  "best_f1_thresh": -4.283203125
}
```

