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

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path', default='./data',
    help=('Directory to store dataset data'))
flags.DEFINE_string(
    'bin_path', default='./bin',
    help=('Directory to store bin data'))
flags.DEFINE_integer(
    'batch_size', default=1,
    help=('Batch size for inference,need to be divided by sum_samples.'))


def batch_generator(data, batch_size=100):
    def generate():
        for idx in range(0, len(data), batch_size):
            yield data[idx:(idx + batch_size)]
    return generate

def data_array(expected_n, x_data, y_data):
    array = np.zeros(expected_n, dtype=[
        ('x', np.float32, (32, 32, 3)),            
        ('y', np.int32, ())  # We will be using -1 for unlabeled
        ])
    array['x'] = x_data
    array['y'] = y_data
    return array




def main(argv):
    if len(argv) > 3:
        raise app.UsageError('Too many command-line arguments.')
    image_path = os.path.join(FLAGS.bin_path,'image_' + str(FLAGS.batch_size))
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    label_path = os.path.join(FLAGS.bin_path, 'label_' + str(FLAGS.batch_size))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    data_path = os.path.join(FLAGS.dataset_path, 'cifar10_gcn_zca_v2.npz')    
    file_data = np.load(data_path)
    test_data = data_array(10000, file_data['test_x'], file_data['test_y'])
    test_batches = batch_generator(test_data,FLAGS.batch_size)

    i = 0
    for batch in test_batches():
        name_image = ''.join([image_path, '/', str(i), '_image', '.bin'])
        name_label = ''.join([label_path, '/', str(i), '_label', '.txt'])
        image = batch['x'].astype(np.float32)
        label = batch['y'].astype(np.float32)
        image.tofile(name_image)
        np.savetxt(name_label, label)
        i+=1


if __name__ == '__main__':
    app.run(main)