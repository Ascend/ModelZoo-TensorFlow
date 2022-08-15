### Representation Learning with Contrastive Predictive Coding

This dataset consists of sequences of modified MNIST numbers (64x64 RGB). Positive sequence samples contain *sorted* numbers, and negative ones *random* numbers. For example, let's assume that the context sequence length is S=4, and CPC is asked to predict the next P=2 numbers. A positive sample could look like ```[2, 3, 4, 5]->[6, 7]```, whereas a negative one could be ```[1, 2, 3, 4]->[0, 8]```. Of course CPC will only see the patches, not the actual numbers.

### Results

After 10 training epochs, CPC reports a 99% accuracy on the contrastive task. After training, I froze the encoder and trained a MLP on top of it to perform supervised digit classification on the same MNIST data. It achieved 90% accuracy after 10 epochs, demonstrating the effectiveness of CPC for unsupervised feature extraction.


### Dependencies Install
- Execute ```pip install -r requirements.txt``` to install Dependencies.

### Requisites

- Python 3.6.8
- Keras 2.2.5
- Tensorflow 1.15.0 
- numpy 1.19.5 
- scipy 1.5.4
- Pillow 7.2.0
- matplotlib 3.0.3
### 【执行结果打屏信息】
```
390/390 [==============================] - 174s 445ms/step - loss: 0.6817 - binary_accuracy: 0.5275 - val_loss: 0.4455 - val_binary_accuracy: 0.7937
390/390 [==============================] - 100s 256ms/step - loss: 0.2324 - binary_accuracy: 0.9091 - val_loss: 0.0887 - val_binary_accuracy: 0.9716
390/390 [==============================] - 91s 233ms/step - loss: 0.0686 - binary_accuracy: 0.9784 - val_loss: 0.0319 - val_binary_accuracy: 0.9896
390/390 [==============================] - 90s 230ms/step - loss: 0.0352 - binary_accuracy: 0.9893 - val_loss: 0.0212 - val_binary_accuracy: 0.9952
390/390 [==============================] - 90s 232ms/step - loss: 0.0190 - binary_accuracy: 0.9937 - val_loss: 0.0117 - val_binary_accuracy: 0.9968
390/390 [==============================] - 91s 233ms/step - loss: 0.0153 - binary_accuracy: 0.9958 - val_loss: 0.0170 - val_binary_accuracy: 0.9928
390/390 [==============================] - 92s 236ms/step - loss: 0.0142 - binary_accuracy: 0.9960 - val_loss: 0.0189 - val_binary_accuracy: 0.9956
390/390 [==============================] - 91s 234ms/step - loss: 0.0109 - binary_accuracy: 0.9974 - val_loss: 0.0068 - val_binary_accuracy: 0.9992
390/390 [==============================] - 90s 231ms/step - loss: 0.0064 - binary_accuracy: 0.9984 - val_loss: 0.0072 - val_binary_accuracy: 0.9976
390/390 [==============================] - 91s 234ms/step - loss: 0.0043 - binary_accuracy: 0.9990 - val_loss: 0.0058 - val_binary_accuracy: 0.9980
```
### 【执行任务OBS链接】
npu训练在裸机上执行，无obs链接

### 【裸机屏幕输出】
```
cpc_npu.log
```

### 【数据集OBS链接】
```
obs://cpc-npu/data
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=lPiNrJ/5bTALQ0CINnqN9NAkCkwLFa+5jgwpzOgJKmFz8QDMvZpcOZJuhVd784/sePwhglUMpyKK9Hsp7Vk6LiImgocmVtwJ7H10VJJPdEzhQSfxqHOo196P4mhKY+c/h7NhTWWcezWu3DdCoCKOS649enyQKEbwGDzKdJoO+4+rI1t/EKZEQM2ynrDarybWTxKJxIzXqo5hCnsGLDVnHmVaDg8oDH2Gbm0ekdo1J5ob66ZgCKHftFQOrctVON9k21k1FfruXhd7LmKxl4EJtWprvYHsr8ftIg7xFTQIEE4C6cXBR7aZnq4620qYsDMYQNxa/rHIHdYlZRfI7cBN+gUCrA2yXNis4oWC+DqcxC6tIes4xBncf1jeZ7Ry1SEkSGwe+O3NK1g6QAwZm4SQ0YHXYdyqDaAO6xTDLA/Bor963g19km8wnW5d6iAHEVH01NvdOo6qImvlcrr8UIxtmgAsDT/uOf5sD4e9PeLVZAs7afuGRqkfqOVuNKbRfSft

提取码:
123456

*有效期至: 2023/08/10 22:43:09 GMT+08:00
### 【Ascend NPU INFO NOTICE】
```
------------------ INFO NOTICE START------------------
INFO, your task have used Ascend NPU, please check your result.
2022-08-04 21:41:51.738760: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 91
2022-08-04 21:41:51.872238: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 101
2022-08-04 21:41:51.985385: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 111
2022-08-04 21:41:52.135082: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 121
2022-08-04 21:41:52.420963: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 131
2022-08-04 21:41:52.531044: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 141
2022-08-04 21:41:52.677375: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 151
2022-08-04 21:41:52.815935: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 161
2022-08-04 21:41:52.955870: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 171
2022-08-04 21:41:53.203352: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 181
2022-08-04 21:41:53.562792: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 191
2022-08-04 21:41:53.691520: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 201
2022-08-04 21:41:53.825070: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 211
2022-08-04 21:41:53.964409: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 221
2022-08-04 21:41:54.105231: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 231
2022-08-04 21:41:54.233242: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 241
2022-08-04 21:41:54.372433: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 251
2022-08-04 21:41:54.504895: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 261
2022-08-04 21:41:54.640009: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 271
2022-08-04 21:41:54.744031: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 281
2022-08-04 21:41:54.848709: I tf_adapter/kernels/geop_npu.cc:805] The model has been compiled on the Ascend AI processor, current graph id is: 291
------------------ INFO NOTICE END------------------
```
### 【Final】
gpu训练日志文件cpc_gpu.log，精度为98.75%
npu训练日志文件cpc_npu.log，精度为99.90%。
### 【启动命令】
```bash
python3 train_model.py
```
