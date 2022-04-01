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

import tensorflow as tf
from model import SRNet
import numpy as np
import os
import cfg
from utils import *
from datagen import srnet_datagen, get_input_data
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("version", metavar="V", type=str)
    parser.add_argument("lr", metavar="L", nargs='?', const=0.001, type=float)
    args = parser.parse_args()
    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    if os.path.exists(os.path.join(cfg.trainoutRoot, args.version)):
        os.mkdir(os.path.join(cfg.trainoutRoot, args.version))

    #path
    tensorboard_dir = os.path.join(cfg.trainoutRoot, args.version, 'train_logs')
    lossFilePath = os.path.join(cfg.trainoutRoot, args.version, 'loss.txt')
    fd = os.open(lossFilePath, os.O_WRONLY | os.O_CREAT);
    # define train_name
    if not cfg.train_name:
        train_name = get_train_name()
    else: 
        train_name = cfg.train_name
    
    # define model
    print_log('model compiling start.', content_color = PrintColor['yellow'])
    model = SRNet(learning_rate=args.lr, dir= tensorboard_dir, shape = cfg.data_shape, name = train_name)
    print_log('model compiled.', content_color = PrintColor['yellow'])

    # define data generator
    gen = srnet_datagen()
    
    with model.graph.as_default():
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
            
            # load pretrained weights or initialize variables
            if cfg.pretrained_ckpt_path:
                print_log('weight loading start.', content_color = PrintColor['yellow'])
                saver.restore(sess, cfg.pretrained_ckpt_path)
                print_log('weight loaded.', content_color = PrintColor['yellow'])
            else:
                print_log('weight initialize start.', content_color = PrintColor['yellow'])
                sess.run(init)
                print_log('weight initialized.', content_color = PrintColor['yellow'])

            
            # train
            print_log('training start.', content_color = PrintColor['yellow'])
            for step in range(cfg.max_iter):
                global_step = step + 1
                
                # train and get loss
                d_loss, g_loss, d_log, g_log = model.train_step(sess, global_step, *next(gen))

                # show loss
                if global_step % cfg.show_loss_interval == 0 or step == 0:
                    print_log ("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))
                    lossReportStr = "step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}\n".format(global_step, d_loss, g_loss)
                    os.write(fd, lossReportStr.encode())
                
                # write tensorboard
                if global_step % cfg.write_log_interval == 0:
                    write_summary(model.d_writer, model.g_writer, d_log, g_log, global_step)

                # gen example
                if global_step % cfg.gen_example_interval == 0:
                    savedir = os.path.join(cfg.trainoutRoot, args.version, 'examples', 'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                    predict_data_list(model, sess, savedir, get_input_data())
                    print_log ("example generated in dir {}".format(savedir), content_color = PrintColor['green'])

                # save checkpoint
                if global_step % cfg.save_ckpt_interval == 0:
                    savedir = os.path.join(cfg.trainoutRoot, args.version, 'checkpoints', 'iter')
                    save_checkpoint(sess, saver, savedir, global_step)
                    print_log ("checkpoint saved in dir {}".format(savedir), content_color = PrintColor['green'])

            print_log('training finished.', content_color = PrintColor['yellow'])
            pb_savepath = os.path.join(cfg.trainoutRoot, args.version, 'checkpoints', 'final.pb')
            save_pb(sess, pb_savepath, ['o_sk', 'o_t', 'o_b', 'o_f'])
            print_log('pb model saved in dir {}'.format(pb_savepath), content_color = PrintColor['green'])
    os.close(fd)
if __name__ == '__main__':
    main()
