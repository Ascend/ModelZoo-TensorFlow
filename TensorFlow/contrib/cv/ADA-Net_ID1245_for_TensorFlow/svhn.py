# ==============================================================================
# MIT License

# Copyright (c) 2019 Qin Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from npu_bridge.npu_init import *

import os
import sys
from scipy.io import loadmat

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf
from dataset_utils import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './svhn/', "")
tf.app.flags.DEFINE_integer('num_labeled_examples', 1000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032

def maybe_download_and_extract():
    # Training set
    print("Loading training data...")
    print("Preprocessing training data...")
    train_data = loadmat(FLAGS.data_dir + '/train_32x32.mat')
    train_x = (-127.5 + train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("Loading test data...")
    test_data = loadmat(FLAGS.data_dir + '/test_32x32.mat')
    test_x = (-127.5 + test_data['X']) / 255.
    test_x = test_x.transpose((3, 0, 1, 2))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    np.save('{}/train_images'.format(FLAGS.data_dir), train_x)
    np.save('{}/train_labels'.format(FLAGS.data_dir), train_y)
    np.save('{}/test_images'.format(FLAGS.data_dir), test_x)
    np.save('{}/test_labels'.format(FLAGS.data_dir), test_y)


def load_svhn():
    maybe_download_and_extract()
    train_images = np.load('{}/train_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_images = np.load('{}/test_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    return (train_images, train_labels), (test_images, test_labels)


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_svhn()
    dirpath = os.path.join(FLAGS.data_dir, 'seed' + str(FLAGS.dataset_seed))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(FLAGS.dataset_seed)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    print(rand_ix)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

    labeled_ind = np.arange(FLAGS.num_labeled_examples)
    labeled_train_images, labeled_train_labels = _train_images[labeled_ind], _train_labels[labeled_ind]
    _train_images = np.delete(_train_images, labeled_ind, 0)
    _train_labels = np.delete(_train_labels, labeled_ind, 0)
    convert_images_and_labels(labeled_train_images,
                              labeled_train_labels,
                              os.path.join(dirpath, 'labeled_train.tfrecords'))
    convert_images_and_labels(train_images, train_labels,
                              os.path.join(dirpath, 'unlabeled_train.tfrecords'))
    convert_images_and_labels(test_images,
                              test_labels,
                              os.path.join(dirpath, 'test.tfrecords'))

    # Construct dataset for validation
    train_images_valid, train_labels_valid = labeled_train_images, labeled_train_labels
    test_images_valid, test_labels_valid = \
        _train_images[:FLAGS.num_valid_examples], _train_labels[:FLAGS.num_valid_examples]
    unlabeled_train_images_valid = np.concatenate(
        (train_images_valid, _train_images[FLAGS.num_valid_examples:]), axis=0)
    unlabeled_train_labels_valid = np.concatenate(
        (train_labels_valid, _train_labels[FLAGS.num_valid_examples:]), axis=0)
    convert_images_and_labels(train_images_valid,
                              train_labels_valid,
                              os.path.join(dirpath, 'labeled_train_val.tfrecords'))
    convert_images_and_labels(unlabeled_train_images_valid,
                              unlabeled_train_labels_valid,
                              os.path.join(dirpath, 'unlabeled_train_val.tfrecords'))
    convert_images_and_labels(test_images_valid,
                              test_labels_valid,
                              os.path.join(dirpath, 'test_val.tfrecords'))

def inputs(datadir,train=True, validation=False,shuffle=True, num_epochs=None,batch_size=100):
    if validation:
        if train:
            filenames = 'labeled_train_val.tfrecords'
            
        else:
            filenames = 'test_val.tfrecords'
            
    else:
        if train:
            filenames = 'labeled_train.tfrecords'
        else:
            filenames = 'test.tfrecords'
            

    filenames = os.path.join(datadir,'seed' + str(FLAGS.dataset_seed), filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)

    if shuffle:
        dataset = dataset.repeat(num_epochs).shuffle(20).batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        image , label = iterator.get_next()
        return image,label
        # return dataset.repeat(num_epochs).shuffle(20).batch(batch_size)
    else:
        dataset = dataset.repeat(num_epochs).batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        image , label = iterator.get_next()
        return image,label
        # return dataset.repeat(num_epochs).batch(batch_size)

            

def unlabeled_inputs(datadir,batch_size=100,
                     validation=False,
                     shuffle=True,num_epochs=None):
    if validation:
        filenames = 'unlabeled_train_val.tfrecords'
        # num_examples = NUM_EXAMPLES_TRAIN - FLAGS.num_valid_examples
    else:
        filenames = 'unlabeled_train.tfrecords'
        # num_examples = NUM_EXAMPLES_TRAIN

    filenames = os.path.join('seed' + str(FLAGS.dataset_seed), filenames)
    filenames=os.path.join(datadir, filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_exmp)
    if shuffle:
        dataset = dataset.repeat(num_epochs).shuffle(20).batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        image , _ = iterator.get_next()
        return image
        # return dataset.repeat(num_epochs).shuffle(20).batch(batch_size,iterator = dataset.make_one_shot_iterator())
    else:
        dataset = dataset.repeat(num_epochs).batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        image , _ = iterator.get_next()
        return image
        # return dataset.repeat(num_epochs).batch(batch_size)



def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()

