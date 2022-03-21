# GAT
Graph Attention Networks (Veličković *et al.*, ICLR 2018): [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for Cora;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` contains a pre-trained Cora model;
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);
    * preprocessing utilities for the PPI benchmark (`process_ppi.py`).

Finally, `execute_cora.py` puts all of the above together and may be used to execute a full training run on Cora. The reported result from original paper is 83.0&plusmn;0.7%. This implemetation achieves around 82.9%.

## Dependencies

The script has been tested running under Python 3.7 Ascend 910 environment, with the following packages installed (along with their dependencies):

- `numpy`
- `scipy`
- `networkx`
- `tensorflow`

## Usage

```
python3 execute_cora.py \
	--data_url PATH_TO_DATA \
	--train_url PATH_TO_OUTPUT \
	--batch_size 1 \
	--nb_epochs 200 \
	--lr 0.005 \
	--l2_coef 0.0005
```

## Inference

Offline inference on Ascend 310 enviroment

Download the pb file, the om file and bin data from: https://share.weiyun.com/yyOFMBKL

Freeze the ckpt file into pb file with ./offline_inference/create_pb.py as:

```
python3 create_pb.py
```

Create the om file as:

```
/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc 
	--model=gat.pb \
	--framework=3 \
	--output=gat_base \
	--soc_version=Ascend310 \
	--input_shape="ftr_in:1,2708,1433;bias_in:1,2708,2708" \
	--out_nodes="output:0"
```

Infer with the om file on Ascend 310:

```
./msame	--model gat_base.om \
	--input "bin_data/ftr_in/a.bin,bin_data/bias_in/a.bin" \
	--output ./output \
	--outfmt TXT \
	--loop 1
```

Change the PATH in ./offline_inference/inf_acc.py to the output path:

```
python3 inf_acc.py
```

The inference accuracy on Ascend 310 is 82.3.

## License
MIT
