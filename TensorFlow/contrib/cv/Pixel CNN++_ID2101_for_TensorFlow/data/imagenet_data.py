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

import os
import sys
import tarfile
from six.moves import urllib

import numpy as np
from imageio import imread

def fetch(url, filepath):
    filename = url.split('/')[-1]
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    print(url)
    filepath, headers = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def maybe_download_and_extract(data_dir):
    # more info on the dataset at http://image-net.org/small/download.php
    # downloads and extracts the two tar files for train/val

    train_dir = os.path.join(data_dir, 'train_32x32')
    if not os.path.exists(train_dir):
        train_url = 'http://image-net.org/small/train_32x32.tar' # 4GB
        filepath = os.path.join(data_dir, 'train_32x32.tar')
        fetch(train_url, filepath)
        print('unpacking the tar file', filepath)
        tarfile.open(filepath, 'r').extractall(data_dir) # creates the train_32x32 folder

    test_dir = os.path.join(data_dir, 'valid_32x32')
    if not os.path.exists(test_dir):
        test_url = 'http://image-net.org/small/valid_32x32.tar' # 154MB
        filepath = os.path.join(data_dir, 'valid_32x32.tar')
        fetch(test_url, filepath)
        print('unpacking the tar file', filepath)
        tarfile.open(filepath, 'r').extractall(data_dir) # creates the valid_32x32 folder

def maybe_preprocess(data_dir):

    npz_file = os.path.join(data_dir, 'imgnet_32x32.npz')
    if os.path.exists(npz_file):
        return # all good

    trainx = []
    train_dir = os.path.join(data_dir, 'train_32x32')
    for f in os.listdir(train_dir):
        if f.endswith('.png'):
            print('reading', f)
            filepath = os.path.join(train_dir, f)
            trainx.append(imread(filepath).reshape((1,32,32,3)))
    trainx = np.concatenate(trainx, axis=0)

    testx = []
    test_dir = os.path.join(data_dir, 'valid_32x32')
    for f in os.listdir(test_dir):
        if f.endswith('.png'):
            print('reading', f)
            filepath = os.path.join(test_dir, f)
            testx.append(imread(filepath).reshape((1,32,32,3)))
    testx = np.concatenate(testx, axis=0)

    np.savez(npz_file, trainx=trainx, testx=testx)


def load(data_dir, subset='train'):
    if not os.path.exists(data_dir):
        print('creating folder', data_dir)
        os.makedirs(data_dir)
    maybe_download_and_extract(data_dir)
    maybe_preprocess(data_dir)
    imagenet_data = np.load(os.path.join(data_dir,'imgnet_32x32.npz'))
    return imagenet_data['trainx'] if subset == 'train' else imagenet_data['testx']



class DataLoader(object):
    """ an object that generates batches of CIFAR-10 data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, **kwargs):
        """ 
        - data_dir is location where the files are stored
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = load(os.path.join(data_dir,'small_imagenet'), subset=subset)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        self.p += self.batch_size

        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


