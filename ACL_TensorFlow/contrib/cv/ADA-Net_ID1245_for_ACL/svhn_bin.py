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
import os
import sys
from scipy.io import loadmat
from six.moves import urllib
import numpy as np

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './SVHN/', "")


def maybe_download_and_extract():
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    filepath_train_mat = os.path.join(FLAGS.data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(FLAGS.data_dir, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

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



def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    image = np.load(FLAGS.data_dir+'test_images.npy').astype(np.float32)
    label = np.load(FLAGS.data_dir+'test_labels.npy').astype(np.float32)
    image = np.reshape(image,[image.shape[0],32, 32, 3])
    
    return image,label

if __name__ == '__main__': 
    
    image,label = load_data()
    if  not os.path.exists('image'):
        os.makedirs('image')
    for i, x in enumerate(image):
        name = str(i).zfill(5)
        x = x.reshape(1,32,32,3)
        x.astype(np.float32).tofile(f"image/{name}.bin")