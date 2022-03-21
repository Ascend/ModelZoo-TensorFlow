# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_FILE_SHUFFLE_BUFFER = 100

_READ_RECORD_BUFFER = 8 * 1000 * 1000

def _filter_max_length(example, max_length=256):
    return tf.logical_and(tf.size(example[0]) <= max_length,
                          tf.size(example[1]) <= max_length)


def _get_example_length(example):
    length = tf.reduce_sum(example["input_mask"], axis=[-1])
    return length

def _create_min_max_boundaries(seq_len_buckets):
    bucket_boundaries = seq_len_buckets
    buckets_min = [0] + bucket_boundaries[:-1]
    buckets_max = bucket_boundaries

    return buckets_min, buckets_max

def _batch_examples(dataset, seq_len_buckets, max_tockens_num, max_length):
    buckets_min, buckets_max = _create_min_max_boundaries(seq_len_buckets)

    bucket_batch_sizes = [max_tockens_num // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    seq_len_buckets = tf.constant(seq_len_buckets, dtype=tf.int64)

    def example_to_bucket_id(example):
        seq_length = _get_example_length(example)
        conditions_c = tf.logical_and(
            tf.less(buckets_min, seq_length),
            tf.less_equal(seq_length, buckets_max)
        )
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)
        grouped_dataset = grouped_dataset.batch(bucket_batch_size)

        def slice_batch(features, bucket_id):

            def sliced_seq_len(k):
                if features[k].shape.as_list()[-1] != max_length:
                    return -1
                return seq_len_buckets[bucket_id]

            sliced_feature = {
                k: tf.slice(features[k], [0, 0],
                            [bucket_batch_size, sliced_seq_len(k)]) for k in features
            }

            return sliced_feature

        grouped_dataset = grouped_dataset.map(lambda example: slice_batch(example, bucket_id))

        return grouped_dataset

    ds = dataset.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn
    )

    return ds