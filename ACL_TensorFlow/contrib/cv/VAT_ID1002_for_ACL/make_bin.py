# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
from scipy import linalg
import pickle
import sys
import glob

NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 50000
NUM_EXAMPLES_TEST = 10000
DATA_DIR = './dataset'
np.random.seed(1)

def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

# Loading training data
train_images = np.zeros((NUM_EXAMPLES_TRAIN, 3 * 32 * 32), dtype=np.float32)
train_labels = []
for i, data_fn in enumerate(
        sorted(glob.glob(DATA_DIR + '/cifar-10-batches-py/data_batch*'))):
    batch = unpickle(data_fn)
    train_images[i * 10000:(i + 1) * 10000] = batch['data']
    train_labels.extend(batch['labels'])
train_images = (train_images - 127.5) / 255.
train_labels = np.asarray(train_labels, dtype=np.int64)

rand_ix = np.random.permutation(NUM_EXAMPLES_TRAIN)
train_images = train_images[rand_ix]

# Loading test data
test = unpickle(DATA_DIR + '/cifar-10-batches-py/test_batch')
test_images = test['data'].astype(np.float32)
test_images = (test_images - 127.5) / 255.
test_labels = np.asarray(test['labels'], dtype=np.int64)

# Apply ZCA whitening
components, mean, _ = ZCA(train_images)
np.save('{}/components'.format(DATA_DIR), components)
np.save('{}/mean'.format(DATA_DIR), mean)
test_images = np.dot(test_images - mean, components.T)
test_images = test_images.reshape(
    (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TEST, -1)).astype('float32')

test_images.tofile('./data/Xtest.bin')
test_labels.tofile('./data/Ytest.bin')
print('bin file saved!')
