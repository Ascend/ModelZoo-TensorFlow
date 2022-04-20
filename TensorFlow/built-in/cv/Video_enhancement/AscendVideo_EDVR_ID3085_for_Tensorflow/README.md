English | [中文](README_zh-cn.md)

# Ascend Video Processing

This repository implements a Video Processing & Enhancement Framework on Ascend Platform, aiming to lower the barrier of implementing and deploying video restoraton and enhancement models, help users to build their own processing pipelines.

We also provide some open-source enhancement models, as samples about how to use this framework for efficient training, online inference, and offline inference on Ascend platform. One can see the [src/networks/edvr.py](src/networks/edvr.py) file contains the classic video super-resolution model [EDVR](http://arxiv.org/abs/1905.02716)  that can be trained and evaluated in tensorflow on Ascend NPU platform, which includes a ``deformable convolution`` operator implenmented exclusively on NPU. As well, we'll provide an example on how to inference with EDVR OM (offline models) on Ascend, and how to build a naive processing pipeline with video in and video out.

## Environment

- python3.7
- training & online inference (with checkpoint file or PB file)
    - Ascend 910 or Ascend 710
- offline inference (with OM)
    - Ascend 310 or Ascend 710
## Requirements

- tensorflow==1.15
- opencv-python
- yacs
- tqdm

## Customize Model
To construct your own model and fit the framework, you should define the model with base class ``src.networks.base_model.Base``, put it in ``src/networks`` folder, and that's it:

```python
from src.networks.base_model import Base

class YOUR_MODEL(Base):
    pass
```

You can use your customized model by setting ``cfg.model.name=YOUR_MODEL`` in ``configs/models/YOUR_MODEL.py``. All the model details, training and inference details can as well be configured in this file, which will overide the default config terms in [src/utils/defaults.py](src/utils/defaults.py).

## Training

Enter the repository folder:

```sh
cd AscendVideo
```

Modify the [scripts/env.sh](scripts/env.sh) to make sure you can import ``npu_bridge`` python package:
```sh
source scripts/env.sh
python3 -c "import npu_bridge"
```

Run training on a single device 0 with the configuration ``configs/models/YOUR_MODEL.py``:

```sh
# On a single device 0
bash scripts/run_train.sh 0 configs/models/YOUR_MODEL.py
```

Run training on two devices 1,2 with the configuration ``configs/models/YOUR_MODEL.py``:

```sh
# On multiple devices, e.g., 1,2
bash scripts/run_train.sh 1,2 configs/models/YOUR_MODEL.py
```

## Inference
Once you have trained the model, the checkpoint files will be saved in the output directory (specified by ``cfg.train.output_dir``). Each checkpoint consists of three files:
- ``YOUR_MODEL-10000.data****``
- ``YOUR_MODEL-10000.meta``
- ``YOUR_MODEL-10000.index``

Suppose the video frames lies in ``/path/to/frames``, where each frame is indexed following some pattern like: ``0001.png``, ``0002.png``, etc. It is easy to do inference with:

```bash
bash scripts/run_inference.sh 0 configs/models/YOUR_MODEL.py /path/to/YOUR_MODEL-10000 /path/to/frames
```

The inference result will be saved in ``/path/to/frames_YOUR_MODEL``.

### Freeze ckpt to PB
Checkpoint to PB file.
```shell
bash scripts/run_freeze.sh ckpt config.yaml
```



