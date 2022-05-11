"""
SRNet - Editing Text in the Wild
Model training.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

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
import argparse

from npu_bridge.npu_init import *

import tensorflow as tf
from model import SRNet
import numpy as np
import os
import cfg
from utils import *
from datagen import srnet_datagen, get_input_data
import time

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


def getParas(code_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="test", action="store_true")
    #parser.add_argument("--train_url", type=str)
    #parser.add_argument("--data_url", type=str)
    parser.add_argument("--data_dir", dest="dataDir", type=str, default=os.path.join(code_dir, "trainData"))
    parser.add_argument("--output_dir", dest="resultDir", type=str, default=os.path.join(r"/cache/out"))
    return parser.parse_args()


def main():
    code_dir = os.path.dirname(__file__)
    print(code_dir)
    args = getParas(code_dir)
    print("******************")
    print(args.dataDir)
    print("******************")

    example_data_dir = os.path.join(code_dir, "examples", "labels")
    vgg19Path = os.path.join(args.dataDir, "vgg19_weights_tf_dim_ordering_tf_kernels_notop.pb")
    example_result_dir = os.path.join(args.resultDir, "predictResult")
    tensorboardDir = os.path.join(args.resultDir, "trainResult", "train_logs")
    checkpoint_savedir = os.path.join(args.resultDir, "trainResult", "checkpoints")

    #if not args.test:
    #    os.makedirs(args.dataDir)
    #    mox.file.copy_parallel(src_url=args.data_url, dst_url=args.dataDir)

    if args.test:
        max_iter = cfg.test_max_iter
    else:
        max_iter = cfg.max_iter

    # define train_name
    if not cfg.train_name:
        train_name = get_train_name()
    else:
        train_name = cfg.train_name

    # define model
    print_log('model compiling start.', content_color=PrintColor['yellow'])
    model = SRNet(vgg19Path=vgg19Path,
                  tensorboardDir=tensorboardDir,
                  shape=cfg.data_shape,
                  name=train_name)
    print_log('model compiled.', content_color=PrintColor['yellow'])

    # define data generator
    # datagen()之中包含yield，所以不会正真执行，而是返回一个生成器(迭代器)
    gen = srnet_datagen(args.dataDir, cfg.batch_size)

    with model.graph.as_default():
        init = tf.global_variables_initializer()
        trainCfg = tf.ConfigProto()
        custom_op = trainCfg.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        trainCfg.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        with tf.Session(config=npu_config_proto()) as sess:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # load pretrained weights or initialize variables
            if cfg.pretrained_ckpt_path:
                print_log('weight loading start.', content_color=PrintColor['yellow'])
                saver.restore(sess, cfg.pretrained_ckpt_path)
                print_log('weight loaded.', content_color=PrintColor['yellow'])
            else:
                print_log('weight initialize start.', content_color=PrintColor['yellow'])
                sess.run(init)
                print_log('weight initialized.', content_color=PrintColor['yellow'])

            # train
            print_log('training start.', content_color=PrintColor['yellow'])
            for step in range(max_iter):
                global_step = step + 1

                start_time = time.time()
                # train and get loss
                d_loss, g_loss, d_log, g_log = model.train_step(sess, global_step, *next(gen))

                # show loss
                #if global_step % cfg.show_loss_interval == 0 or step == 0:
                #    print_log("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))
                print("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}   time: {:.4f}".format(global_step, d_loss, g_loss, (time.time() - start_time)))

                # write tensorboard
                if global_step % cfg.write_log_interval == 0:
                    write_summary(model.d_writer, model.g_writer, d_log, g_log, global_step)

                # gen example
                if global_step % cfg.gen_example_interval == 0:
                    savedir = os.path.join(example_result_dir, train_name,
                                           'iter-' + str(global_step).zfill(len(str(max_iter))))
                    predict_data_list(model, sess, savedir, get_input_data(example_data_dir))
                    print_log("example generated in dir {}".format(savedir), content_color=PrintColor['green'])

                # save checkpoint
                if global_step % cfg.save_ckpt_interval == 0:
                    savedir = os.path.join(checkpoint_savedir, train_name, 'iter')
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    save_checkpoint(sess, saver, savedir, global_step)
                    print_log("checkpoint saved in dir {}".format(savedir), content_color=PrintColor['green'])

            print_log('training finished.', content_color=PrintColor['yellow'])
            pb_savepath = os.path.join(checkpoint_savedir, train_name, 'final.pb')
            save_pb(sess, pb_savepath, ['o_sk', 'o_t', 'o_b', 'o_f'])
            print_log('pb model saved in dir {}'.format(pb_savepath), content_color=PrintColor['green'])

    #if not args.test:
    #    mox.file.copy_parallel(src_url=args.resultDir, dst_url=r'obs:\\cann-nju-srnet\trainout')


if __name__ == '__main__':
    main()

