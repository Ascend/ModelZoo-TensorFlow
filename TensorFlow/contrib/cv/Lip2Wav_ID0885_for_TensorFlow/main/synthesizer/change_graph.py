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
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
from synthesizer.models import create_model
from synthesizer.models import Tacotron
#from model import Model
#from config import cfg
#from tfflat.base import Trainer
#from tfflat.utils import mem_info
import argparse
from synthesizer.hparams import hparams

import os
import os.path as osp

def load_model(sess, model_path,weights_dict):
    #TODO(global variables ?? how about _adam weights)
    variables = tf.global_variables()
   
    # vis_var_keep_dic = []
    variables_to_restore = {}
    changed_variables = {}
    for v in variables:
        v_name = v.name.split(':')[0]
        if v_name in weights_dict:
            # print('Varibles restored: %s' % v.name)
            #variables_to_restore.append(v)
            variables_to_restore[v] = weights_dict[v_name]
            # vis_var_keep_dic.append(v.name.split(':')[0])
        else:
            # print('Unrestored Variables: %s' % v.name)
            pass
    # print('Extra Variables in ckpt', set(var_keep_dic) - set(vis_var_keep_dic))
    print("len of variables_to_restore is {}".format(len(variables_to_restore)))
    if len(variables_to_restore) > 0:
        for tensor, value in variables_to_restore.items():
            sess.run(tensor.assign(value))
   
            
    else:
        print('No variables in {} fits the network'.format(model_path))

from tensorflow.python.platform import gfile
GRAPH_PB_PATH = '/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/main/npu_tacotron.pb' #/home/jiayansuo/leo/PoseFix_RELEASE/output/pb_dump/COCO/posefix.pb'
weights_dict={}
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)
    print(names)
    wts = [n for n in graph_nodes if n.op=='Const']
    from tensorflow.python.framework import tensor_util
    for n in wts:
        print("Name of the node - %s" % n.name)
        print("Value - ")
        weights_dict[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
#args = parse_args()
#cfg.set_args(args.gpu_ids, args.continue_train)
tf.reset_default_graph()

#placeholders for initialize
inputs = tf.placeholder(tf.float32, shape=(hparams.tacotron_batch_size, 90, 48, 48, 3), name ="inputs"),

input_lengths = tf.placeholder(tf.int32, shape=(hparams.tacotron_batch_size,), name="input_lengths"),
        
targets = tf.placeholder(tf.float32, shape=(hparams.tacotron_batch_size, hparams.mel_step_size, hparams.num_mels), 
        name="mel_targets"),

split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), 
        name="split_infos"),

# SV2TTS
speaker_embeddings = tf.placeholder(tf.float32, shape=(hparams.tacotron_batch_size, 256), 
        name="speaker_embeddings")

trainer = Tacotron(hparams)

trainer.initialize(inputs, input_lengths, speaker_embeddings, targets, split_infos=split_infos)

#trainer = Trainer(Model(), cfg)
model_path = "/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/main/synthesizer/saved_models/logs-final_train5/taco_pretrained/tacotron_model.ckpt-36000"

load_model(trainer.sess, model_path, weights_dict)

saver = tf.train.Saver()
saver.save(trainer.sess, "/home/jiayansuo/Lip2wav_train/Lip2Wav-master_npu_20210602003231/main/npu_changed_graph_0805.ckpt")
