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

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''
from npu_bridge.npu_init import *


from utils.data_process import load_data
import nets.TinyMobileFaceNet as TinyMobileFaceNet
import nets.MobileFaceNet as MobileFaceNet
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=1)
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--pretrained_model', type=str, default='./output/ckpt_best',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--model_type', default=0, help='MobileFaceNet or TinyMobileFaceNet')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        args = get_parser()

        # define placeholder
        inputs = tf.placeholder(name='img_inputs',
                                shape=[None, args.image_size[0], args.image_size[1], 3],
                                dtype=tf.float32)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                              name='phase_train')

        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        w_init_method = slim.initializers.xavier_initializer()
        if args.model_type == 0:
            prelogits, net_points = MobileFaceNet.inference(images=inputs,
                                                            phase_train=phase_train_placeholder,
                                                            weight_decay=args.weight_decay)
        else:
            prelogits, net_points = TinyMobileFaceNet.inference(images=inputs,
                                                                phase_train=phase_train_placeholder,
                                                                weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"       
        custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/home/test_user07/MobileFaceNet_Tensorflow/fusion_switch.cfg")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess = tf.Session(config=config)
        
        # saver to load pretrained model or save model
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if args.pretrained_model:
            print('Restoring pretrained model: %s' % args.pretrained_model)
            ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_accuracy = {}

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('testing...')
        for ver_step in range(len(ver_list)):
            start_time = time.time()
            data_sets, issame_list = ver_list[ver_step]
            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
            nrof_batches = data_sets.shape[0] // args.test_batch_size
            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                start_index = index * args.test_batch_size
                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            duration = time.time() - start_time
            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)

            print("total time %.3f to evaluate %d images of %s" % (duration,
                                                                   data_sets.shape[0],
                                                                   ver_name_list[ver_step]))
            print('Accuracy: %1.3f' % (np.mean(accuracy)))
            print('Testing Done' )

            coord.request_stop()
            coord.join(threads)
