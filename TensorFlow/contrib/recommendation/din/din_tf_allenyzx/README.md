

# DIN

This is a TenforFlow implementation of DIN on Ascend 910 for CTR prediction task, as described in paper:

Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Han Zhu, Ying Fan, Na Mou, Xiao Ma, Yanghui Yan, Xingya Dai, Junqi Jin, Han Li, Kun Gai. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf). arXiv preprint arXiv:1706.06978, 2018.

## Requirements

- Tensorflow 1.15
- Python 3.7.5
- Ascend 910

## Usage

Use  Amazon-Electronic Dataset，and take `bash preprocess_data.sh` when you use model training.

When use Ascend 910：

```shell
bash train_npu.sh
```

When use GPU or CPU：
```shell
bash train.sh
```

## Result

| Device        | Activation | Model Loss | Val AUC | Val Accuracy |
| ------------- | ---------- | ---------- | ------- | ------------ |
| Ascend 910    | PRelu      | 0.49068    | 0.83741 | 0.76009      |
|               | Dice       | 0.49254    | 0.835   | 0.76092      |
| GPU（2080Ti） | PRelu      | 0.49101    | 0.83678 | 0.75965      |
|               | Dice       | 0.48509    | 0.83057 | 0.75514      |

## Offline Inference

if you want to start offline infrence , only use it：

```shell
bash start_inference.sh
```

This script contains the following content：

```shell
embed_dim=8		# din model embedding_dim
maxlen=40			# din model params
file=../raw_data/remap.pkl	# dataset preprocess result
input_checkpoint=../checkpoint/model.ckpt-15001	# checkpoint path
output_graph=../checkpoint/frozen_model.pb	# pb path

mkdir raw_data/
rm -rf raw_data/*

# create test dataset
python3 test_preprocess.py \
--file=${file} \
--embed_dim=${embed_dim} \
--maxlen=${maxlen}

# create pb model
python3 create_pb.py \
--maxlen=${maxlen} \
--input_checkpoint=${input_checkpoint} \
--output_graph=${output_graph}

# inference
python3 inf_acc.py --pb_path=${output_graph}
```

Among then，Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as：

```python
# create pb model
python3 create_pb.py \
--maxlen=40 \
--input_checkpoint=../checkpoint/model.ckpt-15001 \
--output_graph=../checkpoint/frozen_model.pb
```

