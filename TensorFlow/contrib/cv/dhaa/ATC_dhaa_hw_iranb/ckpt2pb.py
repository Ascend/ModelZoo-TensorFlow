# -*-coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import util.resnet_DHAA as resnet_DHAA
from  util.resnet_DHAA import RseNet_G, define_resnet_dhaa

ckpt_path = "../../Jobs_morph/models/2021-5-25/model-2021-5-25.ckpt-39"

def main():
    tf.reset_default_graph()
    label_dim = 101
    image_batch_p = tf.compat.v1.placeholder(tf.float32,shape=(None, 224, 224, 3), name='image_batches')
    thetas_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 6, 2, 3), name='labels')  ### shape=(None, 6, 2, 3)
    batch_size_p = tf.compat.v1.placeholder(tf.int32, name='batch_size')
    label_batch_p = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='labels')
    isTraining_p = False
    reuse=False
    
    ch = 64
    resnet_G = RseNet_G(18)
    residual_block = resnet_G.residual_block
    residual_list = resnet_G.residual_list

    # logits_list = define_resnet_dhaa(ch, residual_list, residual_block, None, image_batch_p, thetas_batch_p, label_dim, isTraining_p, reuse)
    result_list = resnet_DHAA.construct_resnet(18, image_batch_p, thetas_batch_p, batch_size_p, label_batch_p, label_dim,
        is_training=False, reuse=False)


    graph = tf.get_default_graph()
    op = graph.get_operations()
    for i, m in enumerate(op):
        try:
            print("index:",i,m.values()[0])
        except:
            break
    
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
                
        tf.train.write_graph(gd, './pbModel', 'model.pb')

        freeze_graph.freeze_graph(
              input_graph='./pbModel/model.pb',
              input_saver='',
              input_binary=False,
              input_checkpoint=ckpt_path,
              output_node_names='h3/logits/dense/MatMul',  
              restore_op_name='save/restore_all',
              filename_tensor_name='save/Const:0',
              output_graph='./pbModel/DHAA_tf.pb',
              clear_devices=False,
              initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()