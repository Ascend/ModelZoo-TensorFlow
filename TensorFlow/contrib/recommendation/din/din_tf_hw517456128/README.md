# DIN

This is a TenforFlow implementation of DIN on Ascend 910 for CTR prediction task, as described in paper:

Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Han Zhu, Ying Fan, Na Mou, Xiao Ma, Yanghui Yan, Xingya Dai, Junqi Jin, Han Li, Kun Gai. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf). arXiv preprint arXiv:1706.06978, 2018.

This implementation achieves 0.656 accuracy. The performance on GPU is 0.657.



## Requirements

- Tensorflow 1.15
- Python 3.7
- Ascend 910

## Usage

```
python3 train.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--epoch 3 \
	--batch_size 1024 \
	--learning_rate 0.01 \
	--maxlen 100 \
	--model_type DIN
```

## Inference

Offline inference on Ascend 310 enviroment

Download the pb file, the om file and bin data from: https://share.weiyun.com/U1atoHd3

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=din.pb \ 
	--framework=3 \
	--output=din_base \
	--soc_version=Ascend310 \
	--input_shape="Inputs/mid_his_batch_ph:1024,100;Inputs/cat_his_batch_ph:1024,100;Inputs/uid_batch_ph:1024;Inputs/mid_batch_ph:1024;Inputs/cat_batch_ph:1024;Inputs/mask:1024,100" \
	--out_nodes="output:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model din_base.om \
	--input "bin_data/mid_his,bin_data/cat_his,bin_data/uids,bin_data/mids,bin_data/cats,bin_data/mid_mask" \
	--output ./out \
	--outfmt BIN
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference accuracy on Ascend 310 is 0.656.