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
# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
#tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100

with open('../dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0
def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr

def _eval(sess, model):
  auc_sum = 0.0
  score_arr = []
  index = 0
  content = ""
  for _, uij in DataInputTest(test_set, test_batch_size):
    index += 1
    for i in range(5):
        np.array(uij[i]).astype("int32").tofile("input_bins/pl{}/{}.bin".format(i+1,str(index).zfill(6)))
    content += "name:{}.bin  shape:Placeholder_1:{};Placeholder_2:{};Placeholder_4:{},{};Placeholder_5:{}\n".format(str(index).zfill(6),np.array(uij[0]).shape[0],np.array(uij[1]).shape[0],np.array(uij[3]).shape[0],np.array(uij[3]).shape[1],np.array(uij[4]).shape[0])
    auc_, score_ = model.eval(sess, uij)
    
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
  with open("dataset_conf.txt","w") as f:
    f.write(content)
  test_gauc = auc_sum / len(test_set)
  Auc = calc_auc(score_arr)

  return test_gauc, Auc

def _test(sess, model):
  auc_sum = 0.0
  score_arr = []
  predicted_users_num = 0
  print("test sub items")
  for _, uij in DataInputTest(test_set, predict_batch_size):
    if predicted_users_num >= predict_users_num:
        break
    score_ = model.test(sess, uij)
    score_arr.append(score_)
    predicted_users_num += predict_batch_size
  return score_[0]

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
  saver = tf.train.Saver()
  ckpt_state = tf.train.get_checkpoint_state("./save_path/")
  model_path = os.path.join("./save_path/",os.path.basename(ckpt_state.model_checkpoint_path))
  print("Restore from {}".format(model_path))
  saver.restore(sess,model_path)
  
  print("test_gauc: %.4f\t test_auc: %.4f" % _eval(sess,model))
  sys.stdout.flush()