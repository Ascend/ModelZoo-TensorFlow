# -*- coding: utf-8 -*-
# From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
from npu_bridge.npu_init import *

MODEL_DIR = '/cache/dataset/imagenet'
#DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  #bs=1
  bs = 100
  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer2"
  custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
  custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
  #config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
  with tf.Session(config=config) as sess:
  #with tf.Session() as sess:
    preds = []
    n_batches = math.ceil(float(len(inps)) // float(bs))
    for i in range(n_batches):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        #print(inp.shape)
        #print(np.array(inp).shape)
        inp = np.concatenate(inp, 0)
        #print(inp.shape)
        #pred = sess.run(softmax, {'ExpandDims:0': inp})
        pred = sess.run(softmax, {'InputTensor:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  # #如果没有这个目录，就创建
  # if not os.path.exists(MODEL_DIR):
  #   os.makedirs(MODEL_DIR)
  # filename = DATA_URL.split('/')[-1]
  # filepath = os.path.join(MODEL_DIR, filename)
  # #如果没有这个文件
  # if not os.path.exists(filepath):
  #     #定义了一个函数用来显示下载进度
  #   def _progress(count, block_size, total_size):
  #       #sys.stdout.write是print的默认格式
  #     sys.stdout.write('\r>> Downloading %s %.1f%%' % (
  #         filename, float(count * block_size) / float(total_size) * 100.0))
  #       #刷新输出
  #     sys.stdout.flush()
  #   #下载指定udl的文件到指定文件路径
  #   filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
  #     #输出指定路径的内容
  #   print()
  #   statinfo = os.stat(filepath)
  #   print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  #   #解压缩文件
  # tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.io.gfile.GFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    #_ = tf.import_graph_def(graph_def, name='')
    # Import model with a modification in the input tensor to accept arbitrary
    # batch size.
    input_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='InputTensor')
    _ = tf.import_graph_def(graph_def, name='',
                            input_map={'ExpandDims:0': input_tensor})


  # Works with an arbitrary minibatch size.


  config = tf.compat.v1.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer1"
  custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
  custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
  #config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
  with tf.compat.v1.Session(config=config) as sess:
  #with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()
