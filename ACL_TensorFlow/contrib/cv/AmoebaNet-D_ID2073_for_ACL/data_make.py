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
import inception_preprocessing
'''
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
'''

def _dataset_parser(serialized_proto):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.parse_single_example(serialized_proto, keys_to_features)

    bbox = None

    image = features['image/encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = inception_preprocessing.preprocess_image(
        image=image,
        output_height=299,
        output_width=299,
        is_training=False,
        # If eval_from_hub, do not scale the images during preprocessing.
        scaled_images=True,
        bbox=bbox)

    label = tf.cast(
        tf.reshape(features['image/class/label'], shape=[]), dtype=tf.int32)

    return image, label




is_training = False
batch_size = 64
image_num = 50000
epochs = 1
tf_data= "/home/test_user03/tf_records/valid"


file_pattern = os.path.join('/home/test_user03/tf_records', 'validation/validation-*')
print('================file_pattern===============')
print(file_pattern)

dataset = tf.data.Dataset.list_files("/home/test_user03/tf_records/validation/validation-*", shuffle=False)
print('================dataset===============')
print(dataset)
def fetch_dataset(filename):
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset

dataset = dataset.apply(tf.data.experimental.parallel_interleave(fetch_dataset, cycle_length=64, sloppy=True))


#dataset = dataset.apply(tf.data.experimental.map_and_batch(_dataset_parser, batch_size=batch_size, num_parallel_batches=8, drop_remainder=True))





dataset = dataset.map(_dataset_parser)
#dataset = dataset.map(parse_data_train,num_parallel_calls=2)

if is_training:
    #dataset.map(_parse_augmentation)
    dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.repeat(epochs)
else:
   dataset = dataset.repeat(1)

dataset = dataset.batch(batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, labels_batch = iterator.get_next()
print("==============images_batch, labels_batch=============")
print(images_batch, labels_batch)






config = tf.ConfigProto()
sess = tf.Session(config=config)
'''
with gfile.FastGFile('./pb_model_npu/model_001.pb', 'rb') as f:
     graph_def = tf.GraphDef()
     graph_def.ParseFromString(f.read())
     sess.graph.as_default()
     tf.import_graph_def(graph_def, name='')
 

x = tf.get_default_graph().get_tensor_by_name("input_1:0")
pred = tf.get_default_graph().get_tensor_by_name("output_1:0")  
'''
label  = []
for step in range(int(image_num/batch_size)):
        x_in,y_in = sess.run([images_batch,labels_batch])
        #print("===============x_in,y_in=================")
        #print(x_in.shape)
        #print(y_in.shape)
        # y_pred= sess.run(pred, feed_dict={x: x_in })
        #y_in = np.squeeze(y_in,1)
        x_in.tofile("/home/test_user03/tpu-3/models/official/amoeba_net/bin/"+str(step)+".bin")
        print("==================================x_in.shape==========================================")
        print(x_in.shape)
        label += y_in.tolist()
        #print(label)
        print(step)
label = np.array(label)
np.save( "/home/test_user03/tpu-3/models/official/amoeba_net/imageLabel.npy", label)