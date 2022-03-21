# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from npu_bridge.npu_init import *
import os
import scipy.misc
import numpy as np
import json
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables, expand_path, timestamp

import tensorflow as tf
#import moxing as mox

flags = tf.app.flags
flags.DEFINE_integer("epoch", 2, "Epoch to train [25]")#add (25)
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "imagenet", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.JPEG", "Glob pattern of filename of input images [*]")#add
flags.DEFINE_string("data_dir", "./data", "path to datasets [e.g. $HOME/data]")
flags.DEFINE_string("out_dir", "./out", "Root directory for outputs [e.g. $HOME/out]")
flags.DEFINE_string("out_name", "",
                    "Folder (under out_root_dir) for all outputs. Generated automatically if left blank []")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Folder (under out_root_dir/out_name) to save checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Folder (under out_root_dir/out_name) to save samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")#add
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("export", False, "True for exporting with new batch size")
flags.DEFINE_boolean("freeze", False, "True for exporting with new batch size")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
flags.DEFINE_integer("sample_freq", 20, "sample every this many iterations")#add (200)
flags.DEFINE_integer("ckpt_freq", 10, "save checkpoint every this many iterations")#add (200)
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_string("z_dist", "uniform_signed", "'normal01' or 'uniform_unsigned' or uniform_signed")
flags.DEFINE_boolean("G_img_sum", False, "Save generator image summaries in log")
flags.DEFINE_boolean("train_svm", True, "True for training svm, False for loading svm [False]")#add
# flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
tf.flags.DEFINE_string('data_url', '', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '', 'saved model directory.')
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    #mox.file.copy_parallel(FLAGS.data_url, './data/{}/'.format(FLAGS.dataset))

    # expand user name and environment variables
    FLAGS.data_dir = expand_path(FLAGS.data_dir)
    FLAGS.out_dir = expand_path(FLAGS.out_dir)
    FLAGS.out_name = expand_path(FLAGS.out_name)
    FLAGS.checkpoint_dir = expand_path(FLAGS.checkpoint_dir)
    FLAGS.sample_dir = expand_path(FLAGS.sample_dir)

    if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
    if FLAGS.input_width is None: FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height

    # output folders
    if FLAGS.out_name == "":
        FLAGS.out_name = '{} - {} - {}'.format(timestamp(), FLAGS.data_dir.split('/')[-1],
                                               FLAGS.dataset)  # penultimate folder of path
        if FLAGS.train: #add 20210901.020523 - data - imagenet - x32.z100.uniform_signed.y32.b64/checkpoint/
            FLAGS.out_name += ' - x{}.z{}.{}.y{}.b{}'.format(FLAGS.input_width, FLAGS.z_dim, FLAGS.z_dist,
                                                             FLAGS.output_width, FLAGS.batch_size)

    FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)#add /out/20210901.020523 - data - imagenet - x32.z100.uniform_signed.y32.b64/checkpoint/
    FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)

    if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)

    with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
        flags_dict = {k: FLAGS[k].value for k in FLAGS}
        json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                y_dim=10,
                z_dim=FLAGS.z_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir,
                out_dir=FLAGS.out_dir,
                max_to_keep=FLAGS.max_to_keep)
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                z_dim=FLAGS.z_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                data_dir=FLAGS.data_dir,
                out_dir=FLAGS.out_dir,
                max_to_keep=FLAGS.max_to_keep)

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            load_success, load_counter = dcgan.load(FLAGS.checkpoint_dir)
            if not load_success:
                raise Exception("Checkpoint not found in " + FLAGS.checkpoint_dir)

            # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
            #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
            #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
            #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
            #                 [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            if FLAGS.export:
                export_dir = os.path.join(FLAGS.checkpoint_dir, 'export_b' + str(FLAGS.batch_size))
                dcgan.save(export_dir, load_counter, ckpt=True, frozen=False)

            if FLAGS.freeze:
                export_dir = os.path.join(FLAGS.checkpoint_dir, 'frozen_b' + str(FLAGS.batch_size))
                dcgan.save(export_dir, load_counter, ckpt=False, frozen=True)

            if FLAGS.visualize:
                OPTION = 1
                visualize(sess, dcgan, FLAGS, OPTION, FLAGS.sample_dir)

    #mox.file.copy_parallel(src_url="./out", dst_url=FLAGS.train_url)


if __name__ == '__main__':
    tf.app.run()
