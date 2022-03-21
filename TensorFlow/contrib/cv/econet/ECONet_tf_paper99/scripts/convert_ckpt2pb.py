import random
import os
import argparse
import logging
import sys

sys.path.append('../')

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import tensorboard
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from model.econet import ECONet


os.environ["GLOBAL_LOG_LEVEL"] = "3"
os.environ["SLOG_PRINT_TO_STDOUT"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description="Model Conversion from `.ckpt` to `.pb`")
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51'], required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)

CLASS_INFO = {
    'ucf101': 101,
    'hmdb51': 51 
}


def main():
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    output_name = args.output_name 

    target_folder = 'pb_model'
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    clip_holder = tf.placeholder(tf.float32, [None, 224, 224, 3], name='clip_holder')
    label_holder = tf.placeholder(tf.int64, [None,], name='label_holder')
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    net_opt = {
       'weight_decay': 5e-4, 
       'net2d_keep_prob': 1.,
       'net3d_keep_prob': 1.,
       'num_segments': 4,
       'num_classes': CLASS_INFO[args.dataset] 
    }

    logits, end_points = ECONet(clip_holder, opt=net_opt, is_training=False)

    predict_class = tf.identity(logits, name='output')

    print("CKPT=>PB | Model conversion started")
    with tf.Session() as sess:
        tf.io.write_graph(sess.graph_def, target_folder, 'tmp_model.pb')   
        freeze_graph.freeze_graph(
		        input_graph=os.path.join(target_folder, 'tmp_model.pb'),   
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=ckpt_path, 
		        output_node_names='output',  
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph=os.path.join(target_folder, output_name),   
		        clear_devices=False,
		        initializer_nodes='')

    print("CKPT=>PB | Model conversion succeeded. The model path is [{}]".format(os.path.join(target_folder, output_name)))

if __name__ == '__main__':
    main()
