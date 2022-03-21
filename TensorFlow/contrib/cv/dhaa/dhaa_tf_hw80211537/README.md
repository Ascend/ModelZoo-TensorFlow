## Deeply-learned Hybrid Representations for Facial Age Estimation (DHAA)

## Environment
- Python 3.7
- Tensorflow-1.15

## Datasets
- morph

## Installation
- Clone DHAA repository. We'll call the directory that you cloned dhaa_tf_hw80211537 as $ROOT_PATH.
    ```Shell
  git clone --recursive https://gitee.com/liuajian/modelzoo/tree/master/contrib/TensorFlow/Research/cv/dhaa/dhaa_tf_hw80211537
    ```

## Usage(Bare Metal)
- Bare Metal:
- ip: 192.168.88.103 username: test_user06
- $root: 
- cd /home/test_user06/LAJ_DHAA
- command: 
- source ~/env.sh 
- train: python3.7 train_FaceAge_bare.py
- test: python3.7 test_FaceAge_bare.py
- result: vim /home/test_user06/LAJ_DHAA/Jobs_morph/scores/2021-5-25/Test_scores.txt


## Usage(CLOUD)
- Account: hw80211537
- Obs: ajian3
- data_url: obs://ajian3/morph
- train_url: Jobs
- train: python train_FaceAge_yun.py
- test: python test_FaceAge_yun.py

If you do not train the model by yourself, you can download our [trained model](https://pan.baidu.com/s/1Inrslw1FcGahKwDvWzhEIg). code: ttn5

## Results
   ```Shell
   ------------------------
   |  Method   |   MAE    |
   |   Paper   |   3.0    |
   |   Ours    |   3.0    |
   ------------------------
   Note that the metric of MAE is the final indicator, and the smaller value means better performance.
   
  ```
## Citation
  ```Shell
Please cite the following papers in your publications if it helps your research:
@inproceedings{tan2019deeply,
  title={Deeply-learned Hybrid Representations for Facial Age Estimation.},
  author={Tan, Zichang and Yang, Yang and Wan, Jun and Guo, Guodong and Li, Stan Z},
  booktitle={IJCAI},
  pages={3548--3554},
  year={2019}
}
  ```
## Questions
 
Please contact 'ajianliu92@gmail.com'








