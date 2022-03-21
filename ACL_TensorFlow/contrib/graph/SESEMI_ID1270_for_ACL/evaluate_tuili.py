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
"""Utility functions for semi-supervised learning evaluate."""
import os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
'''
with tf.device("/cpu:0"):
    model = tf.keras.models.load_model('./mnist.h5')
    tf.saved_model.simple_save(
      tf.keras.backend.get_session(),
      "./h5_savedmodel/",
      inputs={"image": model.input},
      outputs={"scores": model.output}
    )

reconstructed_model = keras.models.load_model("save_model")
'''

y_pred1 = np.fromfile('input_x/20211222_094059/input_x_1_output_0.bin', dtype='float32')
y_pred2 = np.fromfile('input_x/20211222_094059/input_x_2_output_0.bin', dtype='float32')
y_pred3 = np.fromfile('input_x/20211222_094059/input_x_3_output_0.bin', dtype='float32')
y_pred4 = np.fromfile('input_x/20211222_094059/input_x_4_output_0.bin', dtype='float32')
y_pred5 = np.fromfile('input_x/20211222_094059/input_x_5_output_0.bin', dtype='float32')
y_pred6 = np.fromfile('input_x/20211222_094059/input_x_6_output_0.bin', dtype='float32')
y_pred7 = np.fromfile('input_x/20211222_094059/input_x_7_output_0.bin', dtype='float32')
y_pred8 = np.fromfile('input_x/20211222_094059/input_x_8_output_0.bin', dtype='float32')
y_pred1 = y_pred1.reshape(10000, 10)
y_pred2 = y_pred2.reshape(10000, 10)
y_pred3 = y_pred3.reshape(10000, 10)
y_pred4 = y_pred4.reshape(10000, 10)
y_pred5 = y_pred5.reshape(10000, 10)
y_pred6 = y_pred6.reshape(10000, 10)
y_pred7 = y_pred7.reshape(10000, 10)
y_pred8 = y_pred8.reshape(10000, 10)
y_pred = np.zeros((80000, 10))
y_pred[0: 10000, :] = y_pred1
y_pred[10000: 20000, :] = y_pred2
y_pred[20000: 30000, :] = y_pred3
y_pred[30000: 40000, :] = y_pred4
y_pred[40000: 50000, :] = y_pred5
y_pred[50000: 60000, :] = y_pred6
y_pred[60000: 70000, :] = y_pred7
y_pred[70000: 80000, :] = y_pred8
y_pred = y_pred.reshape((len(y_pred) // 8, 8, -1))
y_pred = y_pred.mean(axis=1)
y_pred = np.argmax(y_pred, axis=1)
y_pred = y_pred.astype(np.int32)
true2 = np.fromfile('bin/y_test.bin', dtype='int32')
y_true = true2

print('*' * 100)
error = 1.0 - accuracy_score(y_true, y_pred)
print('classification error rate: {:.4f}'.format(error), '\n')
print('*' * 100)
