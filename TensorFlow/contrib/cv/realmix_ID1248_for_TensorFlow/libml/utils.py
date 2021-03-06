# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
#
"""Utilities."""
from npu_bridge.npu_init import *

import glob
import os
import re

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

_GPUS = None
FLAGS = flags.FLAGS
flags.DEFINE_bool('log_device_placement', False, 'For debugging purpose.')


def get_config():
    config = tf.ConfigProto()
    if len(get_available_gpus()) > 1:
        config.allow_soft_placement = True
    if FLAGS.log_device_placement:
        config.log_device_placement = True
    config.gpu_options.allow_growth = True
    return config


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)


def smart_shape(x):
    s = x.shape
    st = tf.shape(x)
    return [s[i] if s[i].value is not None else st[i] for i in range(4)]


def ilog2(x):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def find_latest_checkpoint(dir, glob_term='model.ckpt-*.meta'):
    """Replacement for tf.train.latest_checkpoint.

    It does not rely on the "checkpoint" file which sometimes contains
    absolute path and is generally hard to work with when sharing files
    between users / computers.
    """
    r_step = re.compile('.*model\.ckpt-(?P<step>\d+)\.meta')
    matches = glob.glob(os.path.join(dir, glob_term))
    matches = [(int(r_step.match(x).group('step')), x) for x in matches]
    ckpt_file = max(matches)[1][:-5]
    return ckpt_file


def get_latest_global_step(dir):
    """Loads the global step from the latest checkpoint in directory.
  
    Args:
      dir: string, path to the checkpoint directory.
  
    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(find_latest_checkpoint(dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def get_latest_global_step_in_subdir(dir):
    """Loads the global step from the latest checkpoint in sub-directories.

    Args:
      dir: string, parent of the checkpoint directories.

    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    sub_dirs = (x for x in glob.glob(os.path.join(dir, '*')) if os.path.isdir(x))
    step = 0
    for x in sub_dirs:
        step = max(step, get_latest_global_step(x))
    return step


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.

    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def gpu(x):
    return '/gpu:%d' % (x % max(1, len(get_available_gpus())))


def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        local_device_protos = device_lib.list_local_devices(session_config=config)
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS


def average_gradients(tower_grads):
    # Adapted from:
    #  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. For each tower, a list of its gradients.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if len(tower_grads) <= 1:
        return tower_grads[0]

    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grad = tf.reduce_mean([gv[0] for gv in grads_and_vars], 0)
        average_grads.append((grad, grads_and_vars[0][1]))
    return average_grads


def para_list(fn, *args):
    """Run on multiple GPUs in parallel and return list of results."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return zip(*[fn(*args)])
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device('/cpu:0'):
                outputs.append(fn(*x))
    return zip(*outputs)


def para_mean(fn, *args):
    """Run on multiple GPUs in parallel and return means."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device('/cpu:0'):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.reduce_mean(x, 0) for x in zip(*outputs)]
    return tf.reduce_mean(outputs, 0)


def para_cat(fn, *args):
    """Run on multiple GPUs in parallel and return concatenated outputs."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device('/cpu:0'):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.concat(x, axis=0) for x in zip(*outputs)]
    return tf.concat(outputs, axis=0)


## Adapted from brain-research/realistic-ssl-evaluation/master/lib/tf_utils.py
## Used for out-of-distribution example robustness experiments.
def hash_float(x, big_num=1000 * 1000):
    """Hash a tensor 'x' into a floating point number in the range [0, 1)."""
    return tf.cast(
        tf.string_to_hash_bucket_fast(x, big_num), tf.float32
    ) / tf.constant(float(big_num))


def make_set_filter_fn(elements):
    """Constructs a TensorFlow "set" data structure.

    Note that sets returned by this function are uninitialized. Initialize them
    by calling `sess.run(tf.tables_initializer())`

    Args:
        elements: A list of non-Tensor elements.

    Returns:
        A function that when called with a single tensor argument, returns
        a boolean tensor if the argument is in the set.
    """
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            elements, tf.tile([1], [len(elements)])
        ),
        default_value=0,
    )

    return lambda x: tf.equal(table.lookup(tf.dtypes.cast(x, tf.int32)), 1)


def filter_fn_from_comma_delimited(string):
    """Parses a string of comma delimited numbers, returning a filter function.

    This utility function is useful for parsing flags that represent a set of
    options, where the default is "all options".

    Args:
        string: e.g. "1,2,3", or empty string for "set of all elements"

    Returns:
        A function that when called with a single tensor argument, returns
        a boolean tensor that evaluates to True if the argument is in the set.
        If 'string' argument is None, the set is understood to contain all
        elements and the function always returns a True tensor.
    """
    if string:
        return make_set_filter_fn(list(map(int, string.split(","))))
    else:
        return lambda x: tf.constant(True)
