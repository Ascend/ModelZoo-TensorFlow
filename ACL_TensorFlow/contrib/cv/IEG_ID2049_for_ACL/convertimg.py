import numpy as np
import os
from npu_bridge.npu_init import *

import collections
import math
import sys

from absl import flags
from PIL import Image
from six.moves import cPickle
import sklearn.metrics as sklearn_metrics
import tensorflow.compat.v1 as tf

def load_batch(fpath, label_key='labels'):
  """Internal utility for parsing CIFAR data.

  Args:
    fpath: path the file to parse.
    label_key: key for label data in the retrieve dictionary.

  Returns:
    A tuple `(data, labels)`.
  """

  with tf.io.gfile.GFile(fpath, 'rb') as f:
    if sys.version_info < (3,):
      d = cPickle.load(f)
    else:
      d = cPickle.load(f, encoding='bytes')
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
        d_decoded[k.decode('utf8')] = v
      d = d_decoded
  data = d['data']
  labels = d[label_key]

  data = data.reshape(data.shape[0], 3, 32, 32)
  return data, labels

def cifar10_load_data(root):
  """Loads CIFAR10 dataset.

  Args:
    root: path that saves data file.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  if not root:
    return tf.keras.datasets.cifar10.load_data()

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(root, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(root, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  x_train = x_train.transpose(0, 2, 3, 1)
  x_test = x_test.transpose(0, 2, 3, 1)

  x_test = x_test.astype(x_train.dtype)
  y_test = y_test.astype(y_train.dtype)

  return (x_train, y_train), (x_test, y_test)





def convertimg(src,dest):
    (x_train, y_train), (x_test,y_test)=cifar10_load_data(src)
    print("xtest的shape",str(x_test.shape))
    print("xtrain的shape", str(x_train.shape))
    print("ytest的shape",str(y_test.shape))
    print("ytrain的shape", str(y_train.shape))

    for i in range(100):
        im = Image.fromarray(x_test[i])
        im.save(dest+"/cifar10_test_"+str(i)+".png")


if __name__ == '__main__':
    src="./data/cifar-10-python/cifar-10-batches-py"
    dest="./toimg"
    convertimg(src=src,dest=dest)
