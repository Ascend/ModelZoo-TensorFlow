Author: Tao Wu


# Variational Graph Auto-Encoder 


Variational Graph Auto-Encoder (VGAE) was proposed by [T. N. Kipf and M. Welling](https://arxiv.org/abs/1611.07308) for link prediction.
The code in this repository is adapted from the author's [original implementation](https://github.com/tkipf/gae) to run on [Ascend platform](https://www.hiascend.com/).


## Usage

### Prepare datasets

- Download datasets from https://github.com/kimiyoung/planetoid/tree/master/data

- Move the files to the subdirectory `./data/`
```
├── data
│   ├── ind.citeseer.allx
│   ├── ind.citeseer.ally
│   ├── ind.citeseer.graph
│   ├── ind.citeseer.test.index
│   ├── ind.citeseer.tx
│   ├── ind.citeseer.ty
│   ├── ind.citeseer.x
│   ├── ind.citeseer.y
│   ├── ind.cora.allx
│   ├── ind.cora.ally
│   ├── ind.cora.graph
│   ├── ind.cora.test.index
│   ├── ind.cora.tx
│   ├── ind.cora.ty
│   ├── ind.cora.x
│   ├── ind.cora.y
│   ├── ind.pubmed.allx
│   ├── ind.pubmed.ally
│   ├── ind.pubmed.graph
│   ├── ind.pubmed.test.index
│   ├── ind.pubmed.tx
│   ├── ind.pubmed.ty
│   ├── ind.pubmed.x
│   ├── ind.pubmed.y
│   ├── trans.citeseer.graph
│   ├── trans.citeseer.tx
│   ├── trans.citeseer.ty
│   ├── trans.citeseer.x
│   ├── trans.citeseer.y
│   ├── trans.cora.graph
│   ├── trans.cora.tx
│   ├── trans.cora.ty
│   ├── trans.cora.x
│   ├── trans.cora.y
│   ├── trans.pubmed.graph
│   ├── trans.pubmed.tx
│   ├── trans.pubmed.ty
```

### Prepare environment

- On CPU or GPU:

  Create [conda](https://docs.conda.io/en/latest/miniconda.html) environment:
  ```
  conda create -n vgae_tf python=3.7
  conda activate vgae_tf
  ```

  Install required dependencies via Conda:
  ```
  conda install 'tensorflow-gpu=1.15' 'scipy<1.7' matplotlib networkx scikit-learn
  ```
   
- On NPU:

  Install [Python 3.7.5](https://www.python.org/downloads/release/python-375/), [Tensorflow 1.15](https://www.tensorflow.org/install), [Ascend CANN](https://www.hiascend.com/software/cann/community) and [Ascend Tensorflow Adaptor](https://www.hiascend.com/software/ai-frameworks/community) (tested with CANN version 5.0.2.alpha005). See [installation guide](https://support.huaweicloud.com/intl/en-us/instg-cli-cann502/atlasdeploy_03_0086.html).
  
  Install additional dependencies via pip:
  ```
  python3.7 -m pip install -r requirements.txt
  ```
  
- The code is tested on Ubuntu 18.04 with x86_64 or aarch64 architecture. Adaptation may be required for a different OS. 

  
### Run training

Train VGAE for {Cora,Citeseer} on {CPU,GPU,NPU}:
```
bash scripts/train_{cora,citeseer}_{cpu,gpu,npu}.sh
```

Results for Cora on NPU:
```
ep= 196 |train_loss= 0.457458 |train_acc= 0.531970 |val_roc= 0.933258 |val_ap= 0.933996 |epochs_per_second= 0.310577
ep= 197 |train_loss= 0.459309 |train_acc= 0.530618 |val_roc= 0.933901 |val_ap= 0.934589 |epochs_per_second= 0.309788
ep= 198 |train_loss= 0.460188 |train_acc= 0.527331 |val_roc= 0.933800 |val_ap= 0.934019 |epochs_per_second= 0.309006
ep= 199 |train_loss= 0.458782 |train_acc= 0.529176 |val_roc= 0.932969 |val_ap= 0.933148 |epochs_per_second= 0.308232
ep= 200 |train_loss= 0.458892 |train_acc= 0.526543 |val_roc= 0.932470 |val_ap= 0.932991 |epochs_per_second= 0.307466
Test ROC score: 0.923220
Test AP score: 0.924890
```

### Run inference
Running inference requires first generating the offline model from the model obtained after training

- Obtaining original trained model file and input data files

The input dataset is generated during the training phase. At the end of the training, an archive (`data/inference_files-DATASET.tar.gz` where `DATASET` is either `cora` or `citeseer`) containing the model file (`constant_graph_DATASET.pb`) and the input data (`inputs_DATASET.npy`). The user then needs to transfer that archive from the training platform to the inference platform, in the same directory (`data/`). The archive will be automatically decompressed during offline model generation. One should note that input data files generated after two training jobs are not identical. This is due a randomness within Cora dataset. Hence, it is mandatory to use the data files and model file generated by the same training job. Otherwise, the inference accuracy will not be similar to that of the training.

- Offline model generation and model inference (Cora)
```
# Generating offline model
bash scripts/generate_om_cora.sh
# Model inference
bash scripts/run_inference_cora.sh
```

- Offline model generation and model inference (Citeseer)
```
# Generating offline model
bash scripts/generate_om_citeseer.sh
# Model inference
bash scripts/run_inference_citeseer.sh
```
