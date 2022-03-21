# convert checkpoint to pb

# -*- coding: utf-8 -*-
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
#
#    http:/www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implies.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
如要checkpoint转pb请先打通训练、推理代码
请将hash表功能移到前后处理中
参考方法：
       请修改model.py中
       tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),tf.int32)
       为 tgt_sos_id = tf.constant(1,tf.int32)    (请根据数据集修改，为起始符）
       tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),tf.int32)
       为 tgt_eos_id = tf.constant(2,tf.int32)    (请根据数据集修改，为结束符）
'''

import tensorflow as tf
from nmt.nmt import create_hparams, argparse, add_arguments, create_or_load_hparams
from nmt.inference import get_model_creator, start_sess_and_load_model, load_data
from nmt import model_helper
import numpy as np
import  os


def freeze_pb():
    '''请指定需要转pb的checkpoint文件'''
    ckpt_path = '../nmt/ckpt_12000/translate.ckpt-12000'
    pb_file = './model/nmt.pb'
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    hparams = create_hparams(FLAGS)
    jobid = FLAGS.jobid
    num_workers = FLAGS.num_workers

    loader_hparams = False
    if FLAGS.ckpt:
        ckpt_dir = os.path.dirname(FLAGS.ckpt)
        ckpt_hparams_file = os.path.join(ckpt_dir, 'hparams')
        if tf.gfile.Exists(ckpt_hparams_file) or FLAGS.hparams_path:
            hparams = create_or_load_hparams(
                ckpt_dir, hparams, FLAGS.hparams_path,
                save_hparams=False
            )
            loaded_hparams = True
    out_dir = '../nmt/ckpt_12000'
    if not loaded_hparams:
        assert out_dir
        hparams = create_or_load_hparams(
            out_dir, hparams, FLAGS.hparams_path,
            save_hparams=(jobid == 0)
        )
    hparams.inference_indices = None
    model_creator = get_model_creator(hparams)
    infer_model = model_helper.create_infer_model(model_creator, hparams, None)
    sess, loaded_infer_model = start_sess_and_load_model(infer_model, ckpt_path)
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    input_graph_def=sess.graph.as_graph_def(),
                                                                    output_node_names=[
                                                                        'dynamic_seq2seq/decoder/decoder/TensorArrayStack_1/TensorArrayGatherV3'
                                                                    ])
    for node in output_graph_def.node:
        if node.device == './device:GPU:0':
            node.device = ''
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


freeze_pb()

