[English](README.md) | 中文

# 昇腾视频增强
本仓库为基于昇腾平台的视频处理、修复和增强框架，目的是降低相关领域的开发者在昇腾平台上进行算法和模型研究、实现和部署模型的门槛，提升算法迭代的效率，帮助开发者建立自己的视频修复和增强流程。

本仓库会提供若干开源增强模型的样例，来引导开发者如何使用本框架和昇腾平台进行高效的训练、在线推理和离线推理等。本仓库支持多种不同的视频处理任务，如去噪、超分、插帧、HDR、人脸增强以及其他可以用模型或者非模型来完成的视频处理任务。开发者可参考[src/networks/edvr.py](src/networks/edvr.py)文件，来了解一个经典的视频超分模型[EDVR](http://arxiv.org/abs/1905.02716)是如何在昇腾平台上进行搭建、训练和评估的。

EDVR模型包含了一个特殊的算子``deformable_convolution``（可变卷积），昇腾平台有一个独占性的实现，对tensorflow上可变卷积的计算进行了一定优化。同时，我们会提供EDVR离线模型（Offline Model）以供开发者参考，如何在昇腾平台上进行离线推理，以及如何建立自己的视频增强端到端流程。

## 环境
- Python版本：python3.7
- 训练和在线推理硬件：Ascend 910

## 依赖项
- tensorflow==1.15
- opencv-python
- yacs
- tqdm

## 自定义模型
添加自定义模型比较简单，只需要继承``src.networks.base_model.Base``类创建一个新的模型类，将其放在``src/networks``下即可：

```python
from src.networks.base_model import Base

class YOUR_MODEL(Base):
    # Define your own structure.
    pass
```

然后通过``configs/models/YOUR_MODEL.py``作为配置文件来来调用自定义模型即可：
```yaml
model:
    name: YOUR_MODEL
# other configurations
```
该文件也可以配置模型结构或是训练、推理策略，程序将在[src/utils/defaults.py](src/utils/defaults.py)的基础上进行覆盖该配置文件。

## 训练
进入目录：

```sh
cd AscendVideo
```

根据硬件需要对环境变量文件[scripts/env.sh](scripts/env.sh)进行修改：
```sh
vim scripts/env.sh

# 修改对应的环境变量，确保能import npu_bridge
```

在0号NPU设备上使用``configs/models/YOUR_MODEL.py``配置文件进行训练：

```sh
# On a single device 0
bash scripts/run_train.sh 0 configs/models/YOUR_MODEL.py
```

在1,2两个NPU设备上使用``configs/models/YOUR_MODEL.py``进行多卡训练:

```sh
# On multiple devices, e.g., 1,2
bash scripts/run_train.sh 1,2 configs/models/YOUR_MODEL.py
```

## 推理
训练完成之后，输出目录（由``cfg.train.output_dir``指定）下会生成定间隔保存的checkpoint文件，例如:
- ``YOUR_MODEL-10000.data****``
- ``YOUR_MODEL-10000.meta``
- ``YOUR_MODEL-10000.index``

给定任意输入视频帧路径``/path/to/frames``，每一帧按顺序编号为``0001.png``,``0002.png``以此类推，则只需要使用如下命令即可进行在线推理：

```bash
bash scripts/run_inference.sh 0 configs/models/YOUR_MODEL.py /path/to/YOUR_MODEL-10000 /path/to/frames
```
推理结果会保存在``/path/to/frames_YOUR_MODEL``路径下。

### 模型固化
将checkpoint固化为PB文件：
```shell
bash scripts/run_freeze.sh configs/models/YOUR_MODEL.py /path/to/YOUR_MODEL-10000
```
其中固化的输入placeholder的size可以通过修改``configs/models/YOUR_MODEL.py``来进行配置。

## 测试样片效果
我们提供了若干用于测试增强效果的视频片段，并且给出了昇腾视频增强对于这些片段的处理效果，包括单一的去噪、插帧、人脸增强、HDR色彩增强和超分辨率等算法。

| 片段         | 分辨率 | 帧率 | 链接                                                         | 备注 |
| ------------- | --- | --- | ------------------------------------------------------------ | --- |
| 超分原片      | 1080P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/City-1080p.mp4 |  |
| 2倍超分        | 2160P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/City-1080p-2x_vsr.mp4 |  |
| 4倍超分        | 4320P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/City-1080p-4x_vsr.mp4 |  |
| 去噪原片      | 1080P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/City-Noisy-1080p.mp4 |  |
| 去噪效果      | 1080P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/City-Noisy-1080p_Denoised.mp4 |  |
| 人脸原片      | 1080P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Face.mp4 |  |
| 人脸增强效果      | 1080P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Face-Enhancement.mp4 |  |
| 插帧原片      | 1080P  | 23.976 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Waterdrop-24FPS-1080p.mp4 |  |
| 2倍插帧        | 1080P | 47.952 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Waterdrop-48FPS-1080p.mp4 |  |
| 4倍插帧        | 1080P | 95.904 |https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Waterdrop-96FPS-1080p.mp4 |  |
| SDR原片       | 2160P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Color-SDR-1080p.mp4 |  |
| HDR无色彩增强 |  2160P | 25 | https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Color_HLG-1080p.mp4 | 需要播放器或者屏幕支持HLG |
| HDR昇腾色彩增强   | 2160P | 25 |https://obs-ascend-test.obs.cn-east-2.myhuaweicloud.com/vsr/Color-Enhanced-HLG-1080p.mp4 | 需要播放器或者屏幕支持HLG |

## 离线推理参考

昇腾[Sample仓库](https://gitee.com/ascend/samples)提供了超分模型EDVR的[离线推理案例](https://gitee.com/ascend/samples/tree/master/python/level2_simple_inference/6_other/video_super_resolution)以供参考

