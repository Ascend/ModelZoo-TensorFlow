# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import os
import absl.app as app
import absl.flags as flags

import numpy
import tensorflow as tf
import utils

# in_val_data_dir = r"C:\Users\zhj\Desktop\study_files\CANN\llr_ood_genomics\between_2011-2016_in_val"
# ood_val_data_dir = r"C:\Users\zhj\Desktop\study_files\CANN\llr_ood_genomics\between_2011-2016_ood_val"
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_boolean(
    'in_val_data', True,
    'False is ood_val_data, True is in_val_data')
flags.DEFINE_string('out_dir', './log/bin_data/in_val_data',
                    'Directory where to write log and models.')
in_val_data_dir = r"./log/original_data/between_2011-2016_in_val"
ood_val_data_dir = r"./log/original_data/between_2011-2016_ood_val"
in_val_file_pattern = 'in_val'
ood_val_file_pattern = 'ood_val'

def main():
    num_epoch=1
    batch_size=1
    in_val_data = FLAGS.in_val_data
    out_dir = FLAGS.out_dir
    #in_val_num = 999774
    #ood_val_num = 6000000
    if in_val_data:
        data_dir = in_val_data_dir
        file_pattern = in_val_file_pattern
    else:
        data_dir = ood_val_data_dir
        file_pattern = ood_val_file_pattern

    data_file_list = [
         os.path.join(data_dir, x)
         for x in tf.gfile.ListDirectory(data_dir)
         if file_pattern in x and '.tfrecord' in x
    ]
    tf.logging.info('data_file_list=%s', data_file_list)
    dataset = tf.data.TFRecordDataset(
        data_file_list).map(lambda v: utils.parse_single_tfexample(v, v))

    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    val_x = []
    val_y = []
 

    i=1

    with tf.Session() as sess:
        try:
            while(i<10001):
                x,y = sess.run([features['x'],features['y']])
                val_x.append(x)
                val_y.append(y)
                if i%100==0:
                    val_x = numpy.array(val_x)
                    val_x = val_x.reshape([100, 250])
                    val_y = numpy.array(val_y)
                    val_y = val_y.reshape([100, 1])
                    val_x.tofile(os.path.join(out_dir, "{}.bin".format(i)))
                    val_x = []
                    val_y = []
                    print(i)
                i = i+1
        except tf.errors.OutOfRangeError:
            print(i)

if __name__ == '__main__':
    main()
