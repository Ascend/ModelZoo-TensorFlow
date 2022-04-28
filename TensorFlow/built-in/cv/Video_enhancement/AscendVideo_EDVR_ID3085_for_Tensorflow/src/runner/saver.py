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

import os
import re

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from src.utils.logger import logger


def get_variables_in_checkpoint_file(file_name):
    """Get all the variables given the checkpoint file

    Args:
        file_name: str, ckpt file.

    Returns:
        Dict of tensor name to tensor.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        logger.error(str(e))
        if "corrupted compressed block contents" in str(e):
            logger.error("It's likely that your checkpoint file has been compressed "
                         "with SNAPPY.")


def loading_variables(sess, variables, checkpoint, strict=False):
    """Loading specific variables given session and checkpoint.
    """
    if not strict:
        var_dic = get_variables_in_checkpoint_file(checkpoint)
        var_missing = []
        var_restore = []

        for v in variables:
            if v.name.split(':')[0] in var_dic:
                var_restore.append(v)
                logger.info('Match: {} {} {}/{}'.format(
                    v.name, 
                    v.dtype, 
                    v.shape, 
                    var_dic[v.name.split(':')[0]]))
            else:
                logger.info('Miss: {} {}'.format(v.name, v.shape))
                var_missing.append(v.name)
        assert len(variables) == len(var_restore) + len(var_missing)

        saver = tf.train.Saver(var_list=var_restore)
        saver.restore(sess, checkpoint)
    else:
        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, checkpoint)
    logger.info("Loading checkpoints...{} Success".format(checkpoint))

    # Get the step information in ckpt file, may be used for continual training.
    recover_step = 0
    regex = re.compile('[A-Za-z.]*-([0-9]*).?[A-Za-z0-9]*$')
    try:
        b, = regex.search(checkpoint).groups()
        if b is not None and b != '':
            recover_step = int(b) + 1
    except:
        pass
    return recover_step


def restore(sess, var_list, directory, checkpoint, strict=False):
    """Restore variables from ckpt.
    """
    if os.path.exists(checkpoint + '.meta'):
        logger.info(f'Found checkpoint {checkpoint}.')
        ckpt_name = checkpoint
    else:
        logger.info(f'Cannot find checkpoint {checkpoint}. Searching in {directory} ...')
        ckpt = tf.train.get_checkpoint_state(directory)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_name = os.path.join(directory, ckpt_name)
            logger.info(f'Found checkpoint {ckpt_name}.')
        else:
            logger.error("Reading checkpoints... ERROR")
            raise ValueError(f'Cannot find checkpoint in {directory}')
    return loading_variables(sess, var_list, ckpt_name, strict=strict)


def strict_loading(sess, scope, directory, checkpoint):
    """Strict loading **every single variable** in the scope.
    """
    if scope == '':
        logger.info(f"Reading checkpoints (no given scope) ...")
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    else:
        logger.info(f"Reading checkpoints for scope '{scope}' ...")
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)
    return restore(sess, variables, directory, checkpoint, strict=True)


def loose_loading(sess, scope, directory, checkpoint):
    """Loading variables in the scope, but allow missing keys or variables.
    """
    if scope == '':
        logger.info(f"Reading checkpoints (no given scope) ...")
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    else:
        logger.info(f"Reading checkpoints for scope '{scope}' ...")
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)
    var_dic = get_variables_in_checkpoint_file(checkpoint)
    var_missing = []
    var_restore = []

    for v in variables:
        loading_cond = v.name.split(':')[0] in var_dic and (scope in v.name.split(':')[0])
        if loading_cond:
            var_restore.append(v)
            logger.info('Match: {} {} Expect {} / Get {}'.format(v.name, v.dtype, v.shape, var_dic[v.name.split(':')[0]]))
        else:
            logger.info('Miss: {} {}'.format(v.name, v.shape))
            var_missing.append(v.name)
    assert len(variables) == len(var_restore) + len(var_missing)
    return restore(sess, var_restore, directory, checkpoint, strict=True)
