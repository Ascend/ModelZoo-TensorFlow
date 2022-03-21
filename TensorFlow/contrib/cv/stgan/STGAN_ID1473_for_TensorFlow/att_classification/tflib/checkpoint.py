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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """Load checkpoint.

    This function add some useless ops to the graph. It is better
    to use tf.train.init_from_checkpoint(...).
    """
    try:
        if os.path.isdir(ckpt_dir_or_file):
            ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_dir_or_file)
        print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
        return True
    except:
        return False


def init_from_checkpoint(ckpt_dir_or_file, assignment_map={'/': '/'}):
    # Use the checkpoint values for the variables' initializers. Note that this
    # function just changes the initializers but does not actually run them, and
    # you should still run the initializers manually.
    tf.train.init_from_checkpoint(ckpt_dir_or_file, assignment_map)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_dir_or_file)
