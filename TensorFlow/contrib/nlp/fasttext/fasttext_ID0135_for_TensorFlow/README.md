# Fasttext 
Bag of Tricks for Efﬁcient Text Classification
(Armand Joulin, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. 2017. In Proceedings of EACL.)

Python Tensorflow Implementation on Ascend 910 environment

Image Path : swr.cn-north-4.myhuaweicloud.com/ascend-share/3.3.0.alpha001_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_0306

# Datasets

AG_news

dataset can be  preprocessed according to GPU reference.

# Results

NPU  Top1 accuracy: 95.37  Top3 accuracy: 99.92  
GPU  Top1 accuracy: 91.26  Top3 accuracy: 99.74

# Usage

```
python3 main.py
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--embedding_dim 10 \
	--num_epochs 5 \
	--batch_size 4096 \
	--dropout 0.5 \
	--top_k 3
```

# Inference


Offline inference on Ascend 310 enviroment

Download the pb file, the om file and bin data from: https://share.weiyun.com/btkOzrk1

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=gat.pb \
	--framework=3 \
	--output=fasttext_base \
	--soc_version=Ascend310 \
	--input_shape="input:7600,126;weights:7600,126" \
	--out_nodes="prediction:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model fasttext_base.om \
	--input "bin_data/input/a.bin,bin_data/weights/a.bin" \
	--output ./output \
	--outfmt TXT \
	--loop 1
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference accuracy on Ascend 310 is 91.76, top3 accuracy is 99.62.
