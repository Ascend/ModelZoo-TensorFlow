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
import tensorflow as tf
from src.utils.logger import logger


def broadcast_global_variables(sess, device, root_rank=0):
    """A helper function to broadcast the variables across the devices in
    distributed training.

    Args:
        sess: tf.Session instance.
        device: str, possible choices in ('npu').
        root_rank: int, the root node rank of the cluster. Default 0.
    
    Raises:
        ValueError, when device is not in ('npu').
    """
    if device == 'npu':
        npu_broadcast(sess, root_rank)
    else:
        raise ValueError


def npu_broadcast(sess, root_rank=0):
    """Broadcast the variables in NPU environment.
    
    We use hccl interface to do the broadcast.

    Args:
        sess: tf.Session instance.
        root_rank: int, the root node rank of the cluster. Default 0.
    """
    from npu_bridge.hccl import hccl_ops
    logger.info(f'Broadcast variables from root_rank {root_rank} ...')
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            outputs = hccl_ops.broadcast(tensor=[var], root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    bcast = tf.group(op_list)
    sess.run(bcast)


def allreduce_avg(tensor, device, ranksize):
    """A helper function to perform the reduce mean across the devices in
    distributed engine.

    Args:
        tensor: tensor to reduce average.
        device: str, possible choices in ('npu').
        ranksize: int, the number of the nodes in the cluster.
    
    Raises:
        ValueError, when device is not in ('npu').
    """
    if device == 'npu':
        return npu_allreduce_avg(tensor, ranksize)
    else:
        raise NotImplementedError


def npu_allreduce_avg(tensor, ranksize):
    """Reduce mean across the devices in NPU environment.

    Args:
        tensor: tensor to reduce average.
        ranksize: int, the number of the nodes in the cluster.
    
    Returns:
        tensor, reduced average tensor.
    """
    from npu_bridge.hccl import hccl_ops
    # There is no 'mean' reduction in allreduce ops. Use 'sum' instead.
    # See https://support.huaweicloud.com/mprtg-A800_9000_9010/atlasprtg_13_0024.html
    tensor_reduced = hccl_ops.allreduce(tensor / ranksize, "sum")
    return tensor_reduced
