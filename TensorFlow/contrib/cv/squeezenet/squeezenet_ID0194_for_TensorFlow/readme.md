# SqueezeNet
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size: [https://arxiv.org/abs/1602.07360](https://arxiv.org/abs/1602.07360)
Iandola F N, Han S, Moskewicz M W, et al.

## Overview

Here we provide the implementation of Squeezenet in TensorFlow. The repository is organised as follows:


- `data/` contains the necessary dataset tfrecord genetator and tfrecord reader;
- `utils/` contains the util of the SqueezeNet network ;
- `pre_trained/` contains a pre-trained SqueezeNet model;
- `models.py` contains the implementation of the SqueezeNet network ;
- `config.py` contains the config of the SqueezeNet network ;



Finally, `train_npu.py` puts all of the above together and may be used to execute a full training run on Tint image net.
The reported result from paper is TOP-1 Accuracy 57.5% and TOP-5 Accuracy 80.3%. The model we reproduce can achieve TOP-1 Accuracy 57.4% and TOP-5 Accuracy 80.1%  train on ImageNet

datasource: obs://ma-iranb/imagenet
copy data to your machine ,and change config.py line 41 to your data source 
use modelarts-image: swr.cn-north-4.myhuaweicloud.com/ascend-share/c76_tensorflow-mox-ascend910-training-euleros2.8-py3.7.5:1.15.0-2.0.8_0225

## Dependencies

The script has been tested running under Python 3.7 Ascend 910 environment, with the following packages installed (along with their dependencies):

- `tensorflow - 1.15`
- `easydict`

dataset TFRECORD  obs path: `obs://ma-iranb/data/squeezenet/`
download pretrained model : obs://ma-iranb/code/squeeze-new/data/squeezenet_weights_tf_dim_ordering_tf_kernels.h5
then put it to ./data/squeezenet_weights_tf_dim_ordering_tf_kernels.h5

## Usage

```
python3 train_npu.py
```


## License
MIT