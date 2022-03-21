# RetinaFace

[Deng, Jiankang, et al. "Retinaface: Single-shot multi-level face localisation in the wild." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.](
    https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html
    )

## Overview

Here we provide the implementation of the RetinaFace in TensorFlow-1.15. The codes include two versions, which can be executed on both GPU and NPU.
+ `configs/`
    > This folder contains the different versions of configs, which are used to train the RetinaFace network. The configs include the necessary parameters used in both training and testing. 
+ `modules/`
    > This folder contains the ***loss function, backbone, learning rate scheduler, dataset processing and utils***.
    - `resnet_v1/`
        > This folder contains the code of **ResNet** built by **slim** package.
+ `pretrain_models/`
    > This folder contains the pretrained **ResNet** models. (Models Link: obs://snowball-bucket/Snowball_Retinaface/pretrain_models)
+ `widerface_evaluate/`
    > This folder contains the codes whcih are used to evaluate the model on GPU.
    - `ground_truth/`
        > The folder contains the ground truth data in **.mat** files.
    - `build/`
        > The folder contains the **.o** file, which is built by the **.c** files in `widerface_evaluate/` to make the evaluation processing faster.
+ `widerface_evaluate_npu/`
    > This folder contains the codes whcih are used to evaluate the model on NPU.
    - `ground_truth/`
        > The folder contains the ground truth data in **.mat** files.

The result of **ResNet50** with **Mxnet framwork** from original paper is **Medium: 93.89%**. This implemetation achieves around **93.19%** on GPU.

## Dependencies

The script has been trained with Python 3.7 Ascend 910 environment, with the following packages installed (the requirments are also can be found in **snowball_requirments**):
+ numpy
+ opencv-python
+ PyYAML
+ tqdm
+ Pillow
+ Cython
+ ipython

## Usage
```
python3 slim_train_npu.py\
        --model 'res50'\
        --log_level '3'\
        --pretrain_path 'pretrain_models'
```

> Info: 
> + If you want to train by yourself, you are supposed to change the config files first.
> + Data OBS Path: obs://snowball-bucket/RetinaFace_Data
> + Models Link: obs://snowball-bucket/Snowball_Retinaface/pretrain_models