## Attention-Based Pedestrian Attribute Analysis (JLPLS-PAA)

## Environment
- Python 3.7
- Tensorflow-1.15

## Datasets
- PETA

## Installation
- Clone JLPLS-PAA repository. We'll call the directory that you cloned jlpls-paa_tf_hw80211537 as $ROOT_PATH.
    ```Shell
  git clone --recursive https://gitee.com/liuajian/modelzoo/tree/master/contrib/TensorFlow/Research/cv/jlpls-paa/jlpls-paa_tf_hw80211537
    ```

## Usage(Bare Metal)
- Bare Metal:
- ip: 192.168.88.103 username: test_user06
- $root: cd /home/test_user06/LAJ_PED/jlpls-paa/jlpls-paa_tf_hw80211537
- command: source ~/env.sh 
- train: python3.7 train_PedAttr_bare.py
- test: python3.7 test_PedAttr_bare.py
- result: vim /home/test_user06/LAJ_PED/Jobs_Ped/scores/2021-6-1/Test_scores.txt


## Usage(CLOUD)
- Account: hw80211537
- Obs: ajian3
- data_url: obs://ajian3/PETA
- train_url: Jobs
- train: python train_PedAttr_yun.py
- test: python test_PedAttr_yun.py

If you do not train the model by yourself, you can download our [trained model](https://pan.baidu.com/s/1IZAAnIWIeegz8bxYt-25PQ). code: 62ik

## Results
   ```Shell
   ------------------------
   |  Method   |    mA    |
   |   Paper   |   83.64  |
   |   Ours    |   83.63  |
   ------------------------
   Note that the metric of mA is the final indicator, and the higher value means better performance.
   
  ```
## Citation
  ```Shell
Please cite the following papers in your publications if it helps your research:
@article{tan2019attention,
author = {Tan, Zichang and Yang, Yang and Wan, Jun and Wan, Hanyuan and Guo, Guodong and Li, Stan},
year = {2019},
month = {07},
pages = {1-1},
title = {Attention-Based Pedestrian Attribute Analysis},
journal = {IEEE Transactions on Image Processing},
}
  ```
## Questions
 
Please contact 'ajianliu92@gmail.com'








