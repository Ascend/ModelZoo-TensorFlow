from __future__ import division
from __future__ import print_function

import dateutil.tz
import datetime
import argparse
import pprint

import sys
sys.path.append('misc')
sys.path.append('stageII')

from datasets import TextDataset
from utils import mkdir_p
from config import cfg, cfg_from_file
from model import CondGAN
from trainer import CondGANTrainer


import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
# from model import network  # network是你们自己定义的模型结构（代码结构）


model_path = "/home/TestUser07/stackgan/ckt_logs/birds/stageII_2021_10_12_15_06_28/model_164000.ckpt"  # 设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前



def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args




def main():
    # args = parse_args()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 4)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    filename_train = '%s/train' % (datadir)
    dataset.train = dataset.get_data(filename_train)
    ckt_logs_dir = "ckt_logs/%s/%s_%s" % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    model = CondGAN(lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio), hr_lr_ratio=dataset.hr_lr_ratio)

    algo = CondGANTrainer(model=model, dataset=dataset, ckt_logs_dir=ckt_logs_dir)


    tf.reset_default_graph()
    # 设置输入网络的数据维度，根据训练时的模型输入数据的维度自行修改
    input_node = tf.placeholder(tf.float32, shape=(64, 1024))
    # output_node = algo.build_test(input_node)# 神经网络的输出
    output1,output2 = algo.build_test(input_node)  # 神经网络的输出
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
    # 保存模型图（结构），为一个json文件
    tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
    # 将模型参数与模型图结合，并保存为pb文件
    # freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'g_net/g_OT/conv2d3/g_net/g_OT/conv2d3, hr_g_net/hr_gen/conv2d5/hr_g_net/hr_gen/conv2d5',
    #                           'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")
    freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path,
                              'g_net/g_OT/conv2d3/g_net/g_OT/conv2d3',
                              'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")
    
    freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path,
                              'hr_g_net/hr_gen/conv2d5/hr_g_net/hr_gen/conv2d5',
                              'save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model2.pb', False, "")
    print("done")


if __name__ == '__main__':
    main()