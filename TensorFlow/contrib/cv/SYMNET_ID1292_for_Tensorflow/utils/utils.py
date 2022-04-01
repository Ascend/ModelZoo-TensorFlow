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
import tensorflow as tf
import time, cv2, os, argparse
import os.path as osp
from PIL import Image
import numpy as np

from . import config as cfg
from . import aux_data
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
################################################################################
#                               tools for solvers                              #
################################################################################

def create_session():
    """create tensorflow session"""
    #######################  add  ########################
    configs = tf.ConfigProto()
    custom_op = configs.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")

    # mix add white
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    # custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")

    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        '{"output":"/cache/profiling","task_trace":"on", "aicpu":"on"}')
    # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

    # close
    configs.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    # close
    configs.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    #######################  add  ########################
    return tf.Session(config=configs)



def display_args(args, logger, verbose=False):
    """print some essential arguments"""
    if verbose:
        ignore = []
        for k,v in args.__dict__.items():
            if not callable(v) and not k.startswith('__') and k not in ignore:
                logger.info("{:30s}{}".format(k,v))
    else:
        logger.info('Name:       %s'%args.name)
        logger.info('Network:    %s'%args.network)
        logger.info('Data:       %s'%args.data)
        logger.info('FC layers:  At {fc_att}, Cm {fc_compress}, Cls {fc_cls}'.format(
            **args.__dict__))



def duplication_check(args):
    if args.force:
        return
    elif args.trained_weight is None or args.trained_weight.split('/')[0] != args.name:
        assert not osp.exists(osp.join(cfg.WEIGHT_ROOT_DIR, args.name)), \
            "weight dir with same name exists (%s)"%(args.name)
        assert not osp.exists(osp.join(cfg.LOG_ROOT_DIR, args.name)), \
            "log dir with same name exists (%s)"%(args.name)
        

def formated_czsl_result(report):
    fstr = '[{name}/{epoch}] rA:{real_attr_acc:.4f}|rO:{real_obj_acc:.4f}|Cl/T1:{top1_acc:.4f}|T2:{top2_acc:.4f}|T3:{top3_acc:.4f}'

    return fstr.format(**report)


################################################################################
#                                glove embedder                                #
################################################################################

class Embedder:
    """word embedder (for various vector type)
    __init__(self)
    """

    def __init__(self, vec_type, vocab, data):
        self.vec_type = vec_type

        if vec_type != 'onehot':
            self.embeds = self.load_word_embeddings(vec_type, vocab, data)
            self.emb_dim = self.embeds.shape[1]
        else:
            self.emb_dim = len(vocab)
    
    def get_embedding(self, i):
        """actually implements __getitem__() function"""
        if self.vec_type == 'onehot':
            return tf.one_hot(i, depth=self.emb_dim, axis=1)
        else:
            i_onehot = tf.one_hot(i, depth=self.embeds.shape[0], axis=1)
            return tf.matmul(i_onehot, self.embeds)


    def load_word_embeddings(self, vec_type, vocab, data):
        tmp = aux_data.load_wordvec_dict(data, vec_type)
        if type(tmp) == tuple:
            attr_dict, obj_dict = tmp
            attr_dict.update(obj_dict)
            embeds = attr_dict
        else:
            embeds = tmp

        embeds_list = []
        for k in vocab:
            if k in embeds:
                embeds_list.append(embeds[k])
            else:
                raise NotImplementedError('some vocabs are not in dictionary: %s'%k)

        embeds = np.array(embeds_list, dtype=np.float32)

        print ('Embeddings shape = %s'%str(embeds.shape))
        return embeds





################################################################################
#                                network utils                                 #
################################################################################


def repeat_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,1,1,2,2,2,3,3,3)"""
    
    result_shape = tensor.shape.as_list()
    for i,v in enumerate(result_shape):
        if v is None:
            result_shape[i] = tf.shape(tensor)[i]
    result_shape[axis] *= multiple

    tensor = tf.expand_dims(tensor, axis+1)
    mul = [1]*len(tensor.shape)
    mul[axis+1] = multiple
    tensor = tf.tile(tensor, mul)
    tensor = tf.reshape(tensor, result_shape)

    return tensor


def tile_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,2,3,1,2,3,1,2,3)"""
    mul = [1]*len(tensor.shape)
    mul[axis] = multiple

    return tf.tile(tensor, mul)


def activation_func(name):
    if name == "none":
        return (lambda x:x)
    elif name == "sigmoid":
        return tf.sigmoid
    elif name == "relu":
        return tf.nn.relu
    else:
        raise NotImplementedError("activation function %s not implemented"%name)

