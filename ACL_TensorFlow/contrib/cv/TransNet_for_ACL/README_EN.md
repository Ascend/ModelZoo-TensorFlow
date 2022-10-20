English|[中文](README.md)

# TransNet Inference for Tensorflow 

This repository provides a script and recipe to Inference of the TransNet model for fast detection of common shot transitions.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/TransNet_for_ACL
```

### 2. Setup Environment

```shell
apt install ffmpeg
pip3 install -r requirements.txt
```

### 3. Download demo video

1. Download a demo video of **'BigBuckBunny.mp4'**  [Download Link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/Dataset/BigBuckBunny.mp4)

2. Move video file to **'scripts'** and convert it to bin files:
```
cd scripts
mkdir input_bins
python3 video_pre_postprocess.py --video_path BigBuckBunny.mp4 --output_path input_bins --mode preprocess
```
The video file will be converted to bin fils under **input_bins/**.

### 4. Offline Inference

**Convert pb to om:**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om

  [**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/TransNet_for_ACL.zip)

  ```
  atc --model=transnet_tf.pb --framework=3 --output=transnet_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="TransNet/inputs:1,100,27,48,3" --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference results

Shot transitions frame id of demo video:
```
[1      37]
[41    284]
[285   377]
[378   552]
[553  1144]
[1145 1345]
[1346 1441]
```

## Reference
[1] https://github.com/soCzech/TransNet
