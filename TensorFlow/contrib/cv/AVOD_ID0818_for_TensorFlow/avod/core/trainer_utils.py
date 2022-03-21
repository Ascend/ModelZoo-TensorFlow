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


from npu_bridge.npu_init import *
import os
import tensorflow as tf

slim = tf.contrib.slim


def load_checkpoints(checkpoint_dir, saver):

    # Load latest checkpoint if available
    all_checkpoint_states = tf.train.get_checkpoint_state(
        checkpoint_dir)
    if all_checkpoint_states is not None:
        all_checkpoint_paths = \
            all_checkpoint_states.all_model_checkpoint_paths
        # Save the checkpoint list into saver.last_checkpoints
        saver.recover_last_checkpoints(all_checkpoint_paths)
    else:
        print('No checkpoints found')


def get_global_step(sess, global_step_tensor):
    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    return global_step


def create_dir(dir):
    """
    Checks if a directory exists, or else create it

    Args:
        dir: directory to create
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_model_weights(sess, checkpoint_dir):
    """Restores the model weights.

    Loads the weights loaded from checkpoint dir onto the
    model. It ignores the missing weights since this is used
    to load the RPN weights onto AVOD.

    Args:
        sess: A TensorFlow session
        checkpoint_dir: Path to the weights to be loaded
    """

    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
    init_fn(sess)

