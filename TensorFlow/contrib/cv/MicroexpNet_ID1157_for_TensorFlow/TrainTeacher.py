# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import  layers
from tensorflow.keras import Model
from  tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from npu_bridge.npu_init import *

def define_model():
    pre_mode = InceptionV3(include_top=False,weights="/cache/dataset/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",input_shape=(256,256,3),pooling='max')
    for layer in pre_mode.layers:
        layer.trainable = True
    last_layer=pre_mode.get_layer('mixed10')
    last_output=last_layer.output
    x=layers.GlobalAvgPool2D()(last_output)

    x=layers.Dense(8,activation='softmax')(x)
    mode = Model(inputs=pre_mode.input,outputs=x)
    mode.compile(optimizer=Adam(lr=1e-04), loss='categorical_crossentropy',metrics=['accuracy'])
    return mode

def plot_training(history,GraphName):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.savefig(GraphName)

def trainTeacher(valpath,trainpath,graphname):
    model= define_model()
    batchSize = 64
    # learningRate = 1e-04
    epochs = 25
    testStep = 20
    displayStep = 20
    valPath = valpath
    trainPath = trainpath
    GraphName =graphname
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        trainPath, target_size=(256, 256), batch_size=batchSize, class_mode='categorical'
    )

    validation_generator = train_datagen.flow_from_directory(
        valPath, target_size=(256, 256), batch_size=batchSize, class_mode='categorical'
    )
    history_t1 = model.fit_generator(
        train_generator,epochs=epochs,steps_per_epoch=displayStep,validation_data=validation_generator,validation_steps=displayStep,class_weight='auto'
    )
    plot_training(history_t1,GraphName)
    model.save_weights('/cache/dataset/database/TeacherExpNet_CK.h5')