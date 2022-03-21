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
from npu_bridge.npu_init import *

import functools
import os

import tensorflow as tf

from tensorflow.python.platform import gfile
from functions import data_config
from preprocessing import imagenet_preprocessing
from utils.data_util import preprocess_image
import numpy as np
import glob
import random
from tqdm import tqdm
import argparse

# os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1' 
os.environ['ASCEND_VISIBLE_DEVICES'] = '2' 


val_batch_size = 1
val_regex = 'validation-*'
data_dir = '/home/sunshk/assemble/data/TFRecord_food101'
dataset_name = 'food101'
pb_path = '/home/sunshk/assemble-multi/Assemble-CNN/pb_model/npu.pb'


def set_seed(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)





def parse_example_proto(example_serialized):
  # Dense features in Example proto.
  feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
  }

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  return features['image/encoded'], label


def parse_record(raw_record):
  image_buffer, label = parse_example_proto(raw_record)

  image = preprocess_image(image_buffer=image_buffer,
                           is_training=False,
                           num_channels=3,
                           dtype=tf.float32,
                           use_random_crop=False,
                           image_size=256,
                           bbox=None,
                           autoaugment_type='good',
                           with_drawing_bbox=False,
                           dct_method='INTEGER_ACCURATE',
                           preprocessing_type='imagenet_224_256a')

  return image, label



def generate_input_label_predict():
  ds = tf.data.Dataset.list_files(data_dir + '/' + val_regex)
  ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=20)

  ds = ds.map(parse_record)
  ds = ds.batch(val_batch_size, drop_remainder=True)
  iterator = ds.make_one_shot_iterator()
  images, labels = iterator.get_next()
  dconf = data_config.get_config(dataset_name)
  num_val_images = dconf.num_images['validation']

  np_i = 0
  correct_cnt = 0
  with tf.Session(config=npu_config_proto()) as sess:
    with gfile.FastGFile(pb_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')  # 导入计算图
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in tqdm(range(int(num_val_images / val_batch_size) + 1)):
      try:
        np_image, np_label = sess.run([images, labels])
        np_image.tofile('./bin_input/{0:05d}.bin'.format(i))
        np_predict = sess.run('output:0', feed_dict={'input:0': np_image})
        np_label.tofile('./bin_label/{0:05d}.bin'.format(i))
        np_predict.tofile('./bin_predict/{0:05d}.bin'.format(i))
        np_i += np_predict.shape[0]
        correct_cnt += np.sum(np_predict == np_label)
      except tf.errors.OutOfRangeError:
        break
    assert np_i == num_val_images

    metric = correct_cnt / np_i
  return metric



def get_predicts_from_bin(bin_root):
    file_names = glob.glob(os.path.join(bin_root, '*.bin'))
    file_names.sort(key=lambda x : int(x.split('/')[-1][:5]))
    predict = []
    for file in file_names:
        predict.append(np.fromfile(file, dtype=np.int32))
        #print(predict[-1])
    np_predict = np.array(predict)
    
    return np_predict

def get_predicts_from_binv2(bin_root):
    file_names = glob.glob(os.path.join(bin_root, '*.bin'))
    file_names.sort(key=lambda x : int(x.split('/')[-1].split('_')[0]))
    predict = []
    for file in file_names:
        predict.append(np.fromfile(file, dtype=np.int32))
        #print(predict[-1])
    np_predict = np.array(predict)
    
    return np_predict

def compare_gt(args):
  np_predict = get_predicts_from_binv2(args.predict_path)
  np_label = get_predicts_from_bin(args.gt_path)
  dconf = data_config.get_config(dataset_name)
  num_val_images = dconf.num_images['validation']

  np_i = 0
  correct_cnt = 0
  for i in tqdm(range(int(num_val_images / val_batch_size))):
    try:
      correct_cnt += np.sum(np_predict[np_i][0] == np_label[np_i][0])
      np_i += 1
    except tf.errors.OutOfRangeError:
      break
  assert np_i == num_val_images

  metric = correct_cnt / np_i
  print('=====FINISH Comparison===== Top1 Accuracy: ', metric)
  return metric

    

def main():
  set_seed(2)
  parser = argparse.ArgumentParser(description='Offline Inference')
  parser.add_argument('--gen', action='store_true', default=False)
  parser.add_argument('--predict_path', type=str)
  parser.add_argument('--gt_path', type=str)
    
  args = parser.parse_args()

  if args.gen:
    generate_input_label_predict()
  else:
    compare_gt(args)


if __name__ == '__main__':
  main()
