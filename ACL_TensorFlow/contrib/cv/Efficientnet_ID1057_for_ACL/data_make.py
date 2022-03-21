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
import tensorflow as tf
import numpy as np
import sys, os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy
import numpy as np
import os
import glob
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from preprocess import preprocess_for_train, preprocess_for_eval
# import matplotlib.pyplot as plt


def parse_data_train(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["label"]
    images = tf.image.decode_jpeg(parsed_features["image"])
    h = tf.cast(parsed_features['height'], tf.int64)
    w = tf.cast(parsed_features['width'], tf.int64)
    c = tf.cast(parsed_features['channels'], tf.int64)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = images/255.0
    images = preprocess_for_eval(images, 224, 224, 0.875)#0.83
    return images, label



is_training = True
batch_size = 50
image_num = 50000
epochs = 1
tf_data= "/sdb1/xiaoqiqi/tfrecord/valid"


def tf_data_list(tf_data_path):
    filepath = tf_data_path
    tf_data_list = []
    file_list = os.listdir(filepath)
    for i in file_list:
        tf_data_list.append(os.path.join(filepath,i))
    print("-----------------------------------------------------")
    print(tf_data_list)
    return tf_data_list



dataset = tf.data.TFRecordDataset(tf_data_list(tf_data))
dataset = dataset.map(parse_data_train,num_parallel_calls=2)

if is_training:
    #dataset.map(_parse_augmentation)
    dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.repeat(epochs)
else:
   dataset = dataset.repeat(1)

dataset = dataset.batch(batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, labels_batch = iterator.get_next()
print(images_batch, labels_batch)






config = tf.ConfigProto()
sess = tf.Session(config=config)
with gfile.FastGFile('./pb_model_npu/model_001.pb', 'rb') as f:
     graph_def = tf.GraphDef()
     graph_def.ParseFromString(f.read())
     sess.graph.as_default()
     tf.import_graph_def(graph_def, name='')
 
 # 获取输入tensor
x = tf.get_default_graph().get_tensor_by_name("input_1:0")
pred = tf.get_default_graph().get_tensor_by_name("output_1:0")  
label  = []
for step in range(int(image_num/batch_size)):
        x_in,y_in = sess.run([images_batch,labels_batch])
        # y_pred= sess.run(pred, feed_dict={x: x_in })
        y_in = np.squeeze(y_in,1)
        print(y_in.shape)
        x_in.tofile("./data/xdata/"+str(step)+".bin")
        label += y_in.tolist()
label = np.array(label)
np.save( "./data/ydata/imageLabel.npy", label)

