# Autorec

Autorec : Autoencoder meets Collaborative Filtering
(Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th International Conference on World Wide Web (pp. 111-112). ACM

TensorFlow Implementation for I-AutoRec on Ascend 910 environment



Reported ml-1m RMSE from the original paper : 0.831; from this implementation : 0.809. dataset can be  preprocessed according to GPU reference.

## Usage

```
python3 main.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--hidden_neuron 1024 \
	--train_epoch 200 \
	--batch_size 256
```

## Inference

Offline inference on Ascend 310 enviroment

Download the pb file, the om file and bin data from: https://share.weiyun.com/3tbDYqeL

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=autorec.pb \
	--framework=3 \
	--output=autorec_base \
	--soc_version=Ascend310 \
	--input_shape="input_R:6040,3952" \
	--out_nodes="output:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model autorec_base.om \
	--input "bin_data/input_R/a.bin" \
	--output ./output \
	--outfmt BIN \
	--loop 1
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference RMSE on Ascend 310 is 0.808.
