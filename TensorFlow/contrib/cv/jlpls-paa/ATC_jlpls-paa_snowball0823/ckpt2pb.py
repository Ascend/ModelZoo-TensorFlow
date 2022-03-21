# -*-coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os, time, argparse, sys
import numpy as np
# import util.utils as utils
import util.resnet_JLPLS as resnet_JLPLS

ckpt_path = "../../Jobs_Ped/models/2021-6-1/model-2021-6-1.ckpt-39"

def main():
    # global seting
    tf.reset_default_graph()
    # model input setting
    data_url = '/root/code/LAJ_PED/PETA'
    list_file = os.path.join(data_url, 'txt/peta_test.txt')
    label_dim = 35
    batch_size_p = tf.compat.v1.placeholder(tf.int32, name='batch_size')
    image_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image_batches')
    label_batch_p = tf.compat.v1.placeholder(tf.int32, shape=(None, 35), name='labels')
    # batch_size_p, label_batch_p, label_dim, train_file are not used to get loss, not in test process
    # so when transform to om model, the image_batch_p is the only needed parameters
    logits, pre_labels, total_loss, accuracy = resnet_JLPLS.construct_resnet(50, image_batch_p, batch_size_p,
        label_batch_p, label_dim, is_training=False, reuse=False, train_file=list_file)
    # get graph
    graph = tf.get_default_graph()
    op = graph.get_operations()
    # for i, m in enumerate(op):
    #     try:
    #         print(m.values()[0])
    #     except Exception as err:
    #         print("Error: "+str(err))
    #         break
    # exit(0)
    with tf.Session() as sess:
        gd = sess.graph.as_graph_def()
                
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        for node in gd.node:
            if 'Sub' in node.op:
                print(node.op)
                input()
        # tf.train.write_graph(sess.graph_def, '../../pbModel', 'JSPJAA_Input.pb')
        tf.train.write_graph(gd, '../../pbModel', 'JSPJAA_Input.pb')

        freeze_graph.freeze_graph(
              input_graph='../../pbModel/JSPJAA_Input.pb',
              input_saver='',
              input_binary=False,
              input_checkpoint=ckpt_path,
              output_node_names='truediv',  ### cross_entropy/cross_entropy & DE_color/fc_logit/dense/MatMul
              restore_op_name='save/restore_all',
              filename_tensor_name='save/Const:0',
              output_graph='../../pbModel/JSPJAA_tf.pb',
              clear_devices=False,
              initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()
