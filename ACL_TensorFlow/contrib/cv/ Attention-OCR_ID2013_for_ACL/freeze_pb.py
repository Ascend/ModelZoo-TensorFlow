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

from tensorflow.compat.v1 import flags
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import os, sys
import argparse



import os
import time

import common_flags
import datasets
import data_provider
import numpy as np
import PIL.Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


FLAGS = flags.FLAGS
common_flags.define()


# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')


def get_dataset_image_size(dataset_name):
    # Ideally this info should be exposed through the dataset interface itself.
    # But currently it is not available by other means.
    ds_module = getattr(datasets, dataset_name)
    height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
    return width, height

def load_images(file_pattern, batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
                                    dtype='uint8')
    # for i in range(batch_size):
    #     path = file_pattern % i
    #     print("Reading %s" % path)
    #     pil_image = PIL.Image.open(tf.io.gfile.GFile(path, 'rb'))
    #     images_actual_data[i, ...] = np.asarray(pil_image)
    return images_actual_data


def create_model(batch_size, dataset_name):
    width, height = get_dataset_image_size(dataset_name)
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(
        num_char_classes=dataset.num_char_classes,
        seq_length=dataset.max_sequence_length,
        num_views=dataset.num_of_views,
        null_code=dataset.null_code,
        charset=dataset.charset)
    raw_images = tf.compat.v1.placeholder(
        tf.uint8, shape=[batch_size, height, width, 3], name='input')
    images = tf.map_fn(data_provider.preprocess_image, raw_images,
                       dtype=tf.float32)
    endpoints = model.create_base(images, labels_one_hot=None)
    return raw_images, endpoints


def run(checkpoint, batch_size, dataset_name, image_path_pattern):
    images_placeholder, endpoints = create_model(batch_size,
                                                 dataset_name)
    images_data = load_images(image_path_pattern, batch_size,
                              dataset_name)

    # session_creator = monitored_session.ChiefSessionCreator(
    #     checkpoint_filename_with_path=checkpoint)
    # with monitored_session.MonitoredSession(
    #         session_creator=session_creator) as sess:
    #     predictions = sess.run(endpoints.predicted_text,
    #                            feed_dict={images_placeholder: images_data})
    # return [pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()]

    # output_node_names = 'AttentionOcr_v1/chars_logit, AttentionOcr_v1/chars_log_prob, AttentionOcr_v1/predicted_chars, AttentionOcr_v1/predicted_scores, AttentionOcr_v1/predicted_text, AttentionOcr_v1/predicted_length, AttentionOcr_v1/predicted_conf, AttentionOcr_v1/normalized_seq_conf'
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=checkpoint,
            output_node_names='AttentionOcr_v1/predicted_chars',
            # output_node_names=output_node_names,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/attention_ocr.pb',
            clear_devices=False,
            initializer_nodes=''
        )
        print("done")


if __name__ == '__main__':
    ckpt_number = 1000000
    checkpoint = '/home/zixuan/attention_ocr/my_ckpt_6_9/' + 'model.ckpt-' + str(ckpt_number)
    # predictions = run(checkpoint, FLAGS.batch_size, FLAGS.dataset_name,
    #                   FLAGS.image_path_pattern)
    predictions = run(checkpoint, 1, FLAGS.dataset_name,
                      FLAGS.image_path_pattern)


