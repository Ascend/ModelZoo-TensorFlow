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
import numpy as np
import os
from absl import flags,app
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path', default='./MNIST_data',
    help=('Directory to store dataset data'))
flags.DEFINE_string(
    'bin_path', default='./bin',
    help=('Directory to store bin data'))
flags.DEFINE_integer(
    'batch_size', default=1,
    help=('Batch size for inference,need to be divided by sum_samples.'))


def main(argv):
    if len(argv) > 3:
        raise app.UsageError('Too many command-line arguments.')
    image_path = os.path.join(FLAGS.bin_path,'image_' + str(FLAGS.batch_size))
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    label_path = os.path.join(FLAGS.bin_path, 'label_' + str(FLAGS.batch_size))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    mnist = input_data.read_data_sets(FLAGS.dataset_path, one_hot=False, reshape=False)

    i = 1
    while i <= 256*20:
        batch_images, batch_labels = mnist.test.next_batch(FLAGS.batch_size)
        name_image = ''.join([image_path, '/', str(i), '_image', '.bin'])
        name_label = ''.join([label_path, '/', str(i), '_label', '.txt'])
        image = batch_images.astype(np.float32)
        label = batch_labels.astype(np.int64)
        image.tofile(name_image)
        np.savetxt(name_label,label)
        i += 1
    print('done!')

if __name__ == '__main__':
    app.run(main)
