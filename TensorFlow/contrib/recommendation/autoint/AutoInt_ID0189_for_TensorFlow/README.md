# AutoInt

This is a TenforFlow implementation of AutoInt on Ascend 910 for CTR prediction task, as described in paper:

Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang and Jian Tang. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf). arXiv preprint arXiv:1810.11921, 2018.

This implementation achieves around 0.8400 AUC and 0.3828 Logloss on the MovieLens-1M dataset. The original paper reports 0.8456 AUC and 0.3797 Logloss.

## Requirements

- Tensorflow 1.15
- Python 3.7
- Ascend 910 (Image path: swr.cn-north-4.myhuaweicloud.com/ascend-share/3.3.0.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_0412)

## Usage

```
python3 train.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--epoch 30 \
	--batch_size 1024 \
	--learning_rate 0.001 \
	--field_size 7
```

## Inference

Offline inference on Ascend 310 enviroment

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=autoint.pb \
	--framework=3 \
	--output=autoint_base \
	--soc_version=Ascend310 \
	--input_shape="feat_index:4096,6;feat_value:4096,6;genre_index:4096,6;genre_value:4096,6" \
	--out_nodes="pred:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model autoint_base.om \
	--input "bin_data/feat_index,bin_data/feat_value,bin_data/genre_index,bin_data/genre_value" \
	--output ./output \
	--outfmt BIN
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference AUC on Ascend 310 is 0.8401 and Logloss is 0.3838.