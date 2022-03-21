# TextING

The code for ACL2020 paper Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks [(https://arxiv.org/abs/2004.13826)](https://arxiv.org/abs/2004.13826), implemented in Tensorflow on Ascend 910 environment.

# Usage 

Substitute mat_mul.py from https://gitee.com/ascend/modelzoo/issues/I28H6W

Start training and inference as:

```
python3.7 train.py \
	--dataset mr \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_DATA \
	--learning_rate 0.005 \
	--epochs 50 \
	--batch_size 1024 \
	--hidden 96
```

The reported result from the original paper is 79.8, and this implementation achieves around 79.4

# Inference

Offline inference on Ascend 310 enviroment

Download the pb file, the om file and bin data from: https://share.weiyun.com/NegZpgBT

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=texting.pb \
	--framework=3 \
	--output=texting_base \
	--soc_version=Ascend310 \
	--input_shape="support:3554,46,46;features:3554,46,300;mask:3554,46,1" \
	--out_nodes="output:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model texting_base.om \
	--input "bin_data/support/a.bin,bin_data/features/a.bin,bin_data/mask/a.bin" \
	--output ./output \
	--outfmt TXT \
	--loop 1
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference accuracy on Ascend 310 is 79.3.
