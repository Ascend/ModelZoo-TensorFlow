# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import configs.config as config
import tensorflow as tf
import tensorflow_transform as tft
import os


def input_fn_tfrecord(tag, batch_size=4096, lines_per_sample=4096,
                      num_epochs=1, num_parallel=16, perform_shuffle=False):

    tf_transform_output = tft.TFTransformOutput(config.metadata_path)

    path = config.record_path + '/' + tag
    all_files = os.listdir(path)
    files = [os.path.join(path,f) for f in all_files if f.startswith('part')]
    print ("Dataset files: ", files)
    raw_dataset = tf.data.TFRecordDataset(files)
    raw_dataset = raw_dataset.repeat(num_epochs)
    raw_dataset = raw_dataset.batch(int(batch_size // lines_per_sample))


    # this function appears to require each element to be a vector
    # batching should mean that this is always true
    # one possible alternative for any problematic case is tf.io.parse_single_example
    parsed_dataset = raw_dataset.apply(
        tf.data.experimental.parse_example_dataset(
            tf_transform_output.transformed_feature_spec(),
            num_parallel_calls=num_parallel
        )
    )

    # a function mapped over each dataset element
    # will separate label, ensure that elements are two-dimensional (batch size, elements per record)
    # adds print_display_ids injection
    def consolidate_batch(elem):
        label = elem.pop('label')
        reshaped_label = tf.reshape(label, [batch_size, label.shape[-1]])
        reshaped_elem = {
            key: tf.reshape(elem[key], [batch_size, elem[key].shape[-1]])
            for key in elem
        }

        return reshaped_elem, reshaped_label

    parsed_dataset = parsed_dataset.map(
        consolidate_batch,
        num_parallel_calls=num_parallel
    )
    if perform_shuffle:
        parsed_dataset = parsed_dataset.shuffle(int(1))

    parsed_dataset = parsed_dataset.prefetch(1)

    return parsed_dataset
