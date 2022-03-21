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

from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
import argparse

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import Input
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from PIL import Image
from keras.models import Model

parser = argparse.ArgumentParser(description='unsupervised evaluating')
parser.add_argument('--dataset', required=True, default='',
                    help='Training dataset.')
args = parser.parse_args()

market = {}
market['TEST'] = os.path.join('./dataset/Market', 'bounding_box_test')
market['TEST_NUM'] = 19732
market['QUERY'] = os.path.join('./dataset/Market', 'query')
market['QUERY_NUM'] = 3368

duke = {}
duke['TEST'] = os.path.join('./dataset/Duke', 'bounding_box_test')
duke['TEST_NUM'] = 17661
duke['QUERY'] = os.path.join('./dataset/Duke', 'query')
duke['QUERY_NUM'] = 2228


def extract_feature(dir_path, net):
  """

  :param dir_path: dir path
  :param net: net
  :return: feature infos
  """

  features = []
  infos = []
  num = 0
  for image_name in os.listdir(dir_path):
    arr = image_name.split('_')
    person = int(arr[0])
    camera = int(arr[1][1])
    image_path = os.path.join(dir_path, image_name) 
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = net.predict(x)
    features.append(np.squeeze(feature))
    infos.append((person, camera))

  return features, infos


# use GPU to calculate the similarity matrix
query_t = tf.placeholder(tf.float32, (None, None))
test_t = tf.placeholder(tf.float32, (None, None))
query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def evaluate(dataset, model):
  """
  :param dataset: dataset
  :param model: model
  :return: print results
  """
  TEST = dataset['TEST']
  TEST_NUM = dataset['TEST_NUM']
  QUERY = dataset['QUERY']
  QUERY_NUM = dataset['QUERY_NUM']
  # load model
  net = load_model(model)
  net = Model(input=net.input, output=net.get_layer('avg_pool').output)

  test_f, test_info = extract_feature(TEST, net)
  query_f, query_info = extract_feature(QUERY, net)

  match = []
  junk = []

  for q_index, (qp, qc) in enumerate(query_info):
    tmp_match = []
    tmp_junk = []
    for t_index, (tp, tc) in enumerate(test_info):
      if tp == qp and qc != tc:
        tmp_match.append(t_index)
      elif tp == qp or tp == -1:
        tmp_junk.append(t_index)
    match.append(tmp_match)
    junk.append(tmp_junk)

  result = sess.run(tensor, {query_t: query_f, test_t: test_f})
  result_argsort = np.argsort(result, axis=1)

  mAP = 0.0
  CMC = np.zeros([len(query_info), len(test_info)])
  for idx in range(len(query_info)):
    recall = 0.0
    precision = 1.0
    hit = 0.0
    cnt = 0
    ap = 0.0
    YES = match[idx]
    IGNORE = junk[idx]
    for i in list(reversed(range(0, TEST_NUM))):
      k = result_argsort[idx][i]
      if k in IGNORE:
        continue
      else:
        cnt += 1
        if k in YES:
          CMC[idx, cnt - 1:] = 1
          hit += 1
        
        tmp_recall = hit / len(YES)
        tmp_precision = hit / cnt
        ap = ap + (tmp_recall - recall) * ((precision + tmp_precision)/2)
        recall = tmp_recall
        precision = tmp_precision
      if hit == len(YES):
        break
    mAP += ap

  rank_1 = np.mean(CMC[:, 0])
  rank_5 = np.mean(CMC[:, 4])
  rank_10 = np.mean(CMC[:, 9])
  rank_20 = np.mean(CMC[:, 19])
  mAP /= QUERY_NUM 

  print ('%s - %s' % (model, TEST))
  print ('1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f' % (rank_1, rank_5, rank_10, rank_20, mAP))
  print ()
  sys.stdout.flush()

name = args.dataset.upper()
if name == 'DUKE':
  dataset = duke
elif name == 'MARKET':
  dataset = market

for i in range(25, 26):
   model = './dataset/Market/checkpoint/%d.ckpt' % i
   evaluate(dataset, model)
