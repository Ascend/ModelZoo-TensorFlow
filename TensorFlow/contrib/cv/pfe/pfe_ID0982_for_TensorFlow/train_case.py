"""Train case for PFE
"""

# MIT License
#
# Copyright (c) 2019 Yichun Shi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
import os
import sys
import time
# import imp
import importlib
import argparse
import tensorflow as tf
import numpy as np

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network import Network

import shutil


def main(args):

    # I/O
    config_file = args.config_file
    print("current work path ", os.path.abspath('.'))
    # config = imp.load_source('config', config_file)
    config = importlib.import_module(".".join(config_file.split("/"))[:-3])
    if args.name:
        config.name = args.name

    trainset = Dataset("data/list_casia_mtcnncaffe_aligned_nooverlap_train_case.txt")


    network = Network()
    network.initialize(config, trainset.num_classes)

    # Initialization for running

    log_dir = utils.create_log_dir(config, config_file)

    print(log_dir)

    # summary_writer = tf.summary.FileWriter(log_dir, network.graph)
    summary_writer = tf.compat.v1.summary.FileWriter(log_dir, network.graph)
    if config.restore_model:
        network.restore_model(config.restore_model, config.restore_scopes)  #add 


    proc_func = lambda images: preprocess(images, config, True)
    trainset.start_batch_queue(config.batch_format, proc_func=proc_func)

    # Main Loop

    print('\nStart Training\nname: {}\n# epochs: {}\nepoch_size: {}\nbatch_size: {}\n'.format(
            config.name, config.num_epochs, config.epoch_size, config.batch_format['size']))
    global_step = 0
    start_time = time.time()
    for epoch in range(config.num_epochs):

        # Training
        for step in range(config.epoch_size):
            # Prepare input
            learning_rate = utils.get_updated_learning_rate(global_step, config)
            batch = trainset.pop_batch_queue()

            wl, sm, global_step = network.train(batch['image'], batch['label'], learning_rate, config.keep_prob)

            wl['lr'] = learning_rate

            # Display
            if step % config.summary_interval == 0:
                duration = time.time() - start_time
                start_time = time.time()
                utils.display_info(epoch, step, duration/100, wl)
                summary_writer.add_summary(sm, global_step=global_step)

        # Save the model
        network.save_model(log_dir, global_step)



if __name__=="__main__":

    print(tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="The path to the training configuration file",
                        type=str)
    parser.add_argument("--name", help="Rename the log dir",
                        type=str, default=None)
    parser.add_argument("--code_dir", help="Code dir in MA",
                        type=str, default=None)
    args = parser.parse_args()
    print("*"*10, "main.py: ", sys.argv[0], "*"*10)
    main(args)
