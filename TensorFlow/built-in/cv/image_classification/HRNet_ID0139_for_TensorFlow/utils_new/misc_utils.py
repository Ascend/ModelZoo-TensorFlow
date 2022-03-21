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

"""Miscellaneous utility functions."""

import tensorflow as tf
from npu_bridge.npu_init import *
from utils_new.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

def auto_barrier(mpi_comm=None):
  """Automatically insert a barrier for multi-GPU training, or pass for single-GPU training.

  Args:
  * mpi_comm: MPI communication object
  """

  if FLAGS.enbl_multi_gpu:
    mpi_comm.Barrier()
  else:
    pass

def is_primary_worker(scope='global'):
  """Check whether is the primary worker of all nodes (global) or the current node (local).

  Args:
  * scope: check scope ('global' OR 'local')

  Returns:
  * flag: whether is the primary worker
  """

  if scope == 'global':
    #return True if not FLAGS.enbl_multi_gpu else mgw.rank() == 0
    return True if not FLAGS.enbl_multi_gpu else get_rank_id() == 0
  elif scope == 'local':
    #return True if not FLAGS.enbl_multi_gpu else mgw.local_rank() == 0
    return True if not FLAGS.enbl_multi_gpu else get_local_rank_id() == 0
  else:
    raise ValueError('unrecognized worker scope: ' + scope)
