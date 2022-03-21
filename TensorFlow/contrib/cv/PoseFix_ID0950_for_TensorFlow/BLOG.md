### NPU training blog for PoseFix
This blog will introduce the steps for you to train the posefix project on NPU enviroment. For how this model works and all details about its structure, please refer to https://github.com/mks0601/PoseFix_RELEASE
We will start from the original GPU repo all the way to the changes we need for NPU training.

## 1. Clone the original Repo

clone the repo and prepare the dataset
```
git clone https://github.com/mks0601/PoseFix_RELEASE
```



### Training Dataset
Dataset: COCO dataset

Extra files 
person_keypoints_256x192_resnet50_val2017_results.json

https://drive.google.com/drive/folders/10d8oSlCnWD-n3CBARXFj7kDBOKXfSYmf 

result.json 

https://drive.google.com/drive/folders/1blrmQ2CRiCeJDjx7sFzLFOBtlALf2Uxa  

Model files
```
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
```

Follow the original repo guide to put all the data as the following structure. The person_keypoints_256x192_resnet50_val2017_results should replace the name_of_input_pose.json under ${POSE_ROOT}data/COCO/input_pose. (For the folder structure, may refer to https://github.com/mks0601/PoseFix_RELEASE for more details)

```
${POSE_ROOT}
|-- data
|-- |-- MPII
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg
|-- |-- PoseTrack
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- train2018.json
|       |   |-- val2018.json
|       |   `-- test2018.json
|       |-- original_annotations
|       |   |-- train/
|       |   |-- val/
|       |   `-- test/
|       `-- images
|           |-- train/
|           |-- val/
|           `-- test/
|-- |-- COCO
|   `-- |-- input_pose
|       |   |-- name_of_input_pose.json
|       |   |-- test_on_trainset
|       |   |   | -- result.json
|       |-- annotations
|       |   |-- person_keypoints_train2017.json
|       |   |-- person_keypoints_val2017.json
|       |   `-- image_info_test-dev2017.json
|       `-- images
|           |-- train2017/
|           |-- val2017/
|           `-- test2017/
`-- |-- imagenet_weights
|       |-- resnet_v1_50.ckpt
|       |-- resnet_v1_101.ckpt
|       `-- resnet_v1_152.ckpt
```

## 2. Modify the code
Use the npu conversion tool before going into the manual change. This tool will help you import all the necessary headers and fix some basic differences between NPU and GPU training.
Afterwards, modify the code in the model part. Remove anything related with device allocation like gpu:0 and cpu:0. Remove the tower gradients in the posefix.

## 3. Train the model
Create the shell files in the scripts according to you need (if needed for distributed training or just on one NPU)

## 4. Save Model
Save the model at the end of the training and convert it into pb file.

## 5. Convert PB to OM
Convert the pb model to the OM model using atc command or MindStudio. 

## 6. Run inference using OM model
Load the om model on inference enviroment. Refer to Inference.md
