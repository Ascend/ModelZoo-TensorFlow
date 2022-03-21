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
# -*- coding:utf-8 -*-
"""
An implementation of acGAN using TensorFlow (work in progress).
"""
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import tensorflow as tf
import numpy as np
from model import MemNet
import os
import glob
import cv2
#import moxing as mox

def main(_):
    tf_flags = tf.app.flags.FLAGS

    #data_dir = "/home/ma-user/modelarts/inputs/data_url_0"
    #os.makedirs(data_dir)
    #mox.file.copy_parallel(tf_flags.data_url,data_dir)
    #mox.file.copy(src_url = "obs://bigelow/datasets3/BSD.tfrecords", dst_url = data_dir)
    # gpu config.
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9

    config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    #custom_op.parameter_map["mix_compile_mode"].b = True
    # custom_op.parameter_map["dynamic_input"].b = True
    # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
    # custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("getnext:[-1 ,-1,-1,-1]")
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    # config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF


    if tf_flags.phase == "train":
        with tf.Session(config=npu_config_proto(config_proto=config)) as sess: 
            # sess = tf.Session(config=config) # when use queue to load data, not use with to define sess
            train_model = MemNet.MemNet(sess, tf_flags)
            train_model.train(tf_flags.training_steps, tf_flags.summary_steps, 
                tf_flags.checkpoint_steps, tf_flags.save_steps)
    else:
        with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
            test_model = MemNet.MemNet(sess, tf_flags)
            test_model.load(tf_flags.checkpoint)
            # test Set12
            # get psnr and ssim outside
            save_path = "./datasets/Set12_Recovery"
            for image_file in glob.glob(tf_flags.testing_set):
                print("testing {}...".format(image_file))
                # testing_set is path/*.jpg.
                c_image = np.reshape(cv2.resize(cv2.imread(image_file, 0), (tf_flags.img_size, tf_flags.img_size)), 
                    (1, tf_flags.img_size, tf_flags.img_size, 1)) / 255.
                # In Caffe, Tensorflow, might must divide 255.?
                recovery_image = test_model.test(c_image)
                # save image
                cv2.imwrite(os.path.join(save_path, image_file.split("/")[3]), 
                    np.uint8(recovery_image[0, :].clip(0., 1.) * 255.))
                # recovery_image[0, :], 3D array.
            print("Testing done.")
    #train_url = "obs://bigelow/test/"   #对应设置中的OBS path
    #model_dir = "/cache/result"
    #os.makedirs(model_dir)
    #mox.file.copy_parallel(model_dir,train_url)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string("output_dir", "model_output",
                               "checkpoint and summary directory.") #output_dir
    tf.app.flags.DEFINE_string("phase", "train", 
                               "model phase: train/test.")
    tf.app.flags.DEFINE_string("training_set", "",
                               "dataset path for training.") #traing set
    tf.app.flags.DEFINE_string("testing_set", "", 
                               "dataset path for testing.")
    tf.app.flags.DEFINE_integer("img_size", 256, 
                                "testing image size.")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "batch size for training.")
    tf.app.flags.DEFINE_integer("training_steps", 100000, 
                                "total training steps.")
    tf.app.flags.DEFINE_integer("summary_steps", 100, 
                                "summary period.")
    tf.app.flags.DEFINE_integer("checkpoint_steps", 1000, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_integer("save_steps", 100, 
                                "checkpoint period.")
    tf.app.flags.DEFINE_string("checkpoint", None, 
                                "checkpoint name for restoring.")
    tf.app.run(main=main)
