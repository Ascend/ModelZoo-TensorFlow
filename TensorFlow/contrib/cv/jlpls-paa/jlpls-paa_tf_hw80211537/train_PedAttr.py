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
import os, time, argparse, sys
import numpy as np
import util.utils as utils
from util.utils import Dataset
import util.resnet_JLPLS as resnet_JLPLS

def main(args):
    #### Set GPU options ###
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    config.gpu_options.allow_growth = True

    ### Setting Parameters ###
    mean_div = []
    lr_decay_epochs = []
    mean_div += (float(i) for i in args.mean_div_str.split('-'))
    lr_decay_epochs += (int(i) for i in args.lr_decay_epochs_str.split('-'))
    label_dim = 35

    ### Make folders of logs and models ###
    image_size = (args.image_size, args.image_size)
    subdir = args.subdir ### subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(args.train_url, 'models', subdir)
    outputs_dir = os.path.join(args.train_url, 'outputs', subdir)
    scores_dir = os.path.join(args.train_url, 'scores', subdir)
    for phase in args.phases:
        utils.make_if_not_exist(os.path.join(outputs_dir, phase))
    utils.make_if_not_exist(model_dir)
    utils.make_if_not_exist(scores_dir)
    utils.write_arguments_to_file(args, os.path.join(scores_dir, 'result.txt'))
    ### Load data ###
    data_url = args.data_url + 'PETA_dataset/'
    list_file = args.data_url + 'txt/peta_train.txt'
    pretrained_model = args.data_url + 'pretrain/'

    st = time.time()
    data_train = utils.get_ped_dataset(data_url, list_file)
    image_list, label_list = utils.get_image_paths_and_labels(data_train)
    epoch_size = int(len(image_list) / args.batch_size)
    need_hour, need_mins, need_secs = utils.convert_secs2time(time.time() - st)
    print('Load Train Data: num_images={}/num_ID={}, Time_coat={:02d}:{:02d}:{:02d}'.format(
        len(image_list), len(label_list), need_hour, need_mins, need_secs))

    ### Create a new Graph and set it as the default ###
    # with tf.Graph().as_default():
    ### Repeatedly running this block with the different graph will generate same value
    seed = np.random.randint(1, 100)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    ### Construction  placeholder
    batch_size_p = tf.compat.v1.placeholder(tf.int32, name='batch_size')
    isTraining_p = tf.compat.v1.placeholder(tf.bool, name='isTraining')
    image_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name='image_batches')
    label_batch_p = tf.compat.v1.placeholder(tf.int32, shape=(None, 35), name='labels')
    learning_rate_p = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=learning_rate_p, global_step=global_step,
        decay_steps=args.max_nrof_epochs * epoch_size, decay_rate=1, staircase=True)
    dataset_train = Dataset(args, image_list, label_list, mode='train')

    ### Build the inference graph ###
    logits, pre_labels, total_loss, accuracy = resnet_JLPLS.construct_resnet(50, image_batch_p, batch_size_p,
        label_batch_p, label_dim, is_training=isTraining_p, reuse=False, train_file=list_file)
    ### Create a saver and train_op, and save the last three models
    saver, saver_restore, trainable_list, restore_trainable_list = utils.get_saver_resnet_tf()
    train_op = utils.get_train_op(total_loss, trainable_list, global_step, args.optimizer, learning_rate)
    ### Start running operations on the Graph.
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        print('Restoring pretrained model: %s' % pretrained_model)
        source_ckpt = tf.train.get_checkpoint_state(pretrained_model)
        saver_restore.restore(sess, source_ckpt.model_checkpoint_path)
        print('Running training')
        fid_train = open(os.path.join(scores_dir, 'Train_loss.txt'), 'w')
        sess.run(global_step, feed_dict=None)
        for epoch in range(args.check_point, args.max_nrof_epochs + 1):  # 1~40
            current_lr = utils.get_lr(args.lr, lr_decay_epochs, epoch, lr_decay=args.lr_decay_factor)
            ### Train for one epoch
            cont = utils.train_Resnet_Race(
                sess, epoch, dataset_train, learning_rate_p, isTraining_p, batch_size_p,
                image_batch_p, label_batch_p, current_lr, args.batch_size,
                epoch_size, mean_div, logits, accuracy, total_loss, train_op,
                global_step, learning_rate, outputs_dir, args.phases[0], fid_train, args.data_name)
            ### Save model ###
            utils.save_variables_and_metagraph(sess, saver, model_dir, subdir, epoch)
        ## End sess ###
        fid_train.close()
        sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--train_url", type=str, default='/home/ajliu/LAJ/HUAWEI_v2/Jobs_Ped/')
    parser.add_argument("--data_url", type=str, default='/home/ajliu/LAJ/HUAWEI_v2/PETA/')

    parser.add_argument("--subdir", type=str, default='2021-5-21')
    parser.add_argument('--data_name', type=str, default='PETA')
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--mean_div_str", type=str,default='127.5-1.0')
    parser.add_argument("--data_augment_str", type=str, default='1-1-0-1-0-1',
                        help ='[0]:Random Rotate [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std [5]:random erasing')
    parser.add_argument("--color_para_str", type=str, default='8-0.2-0.005', help ='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_str", type=str, default='0-0')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default='ADAM')
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_nrof_epochs", type=int, default=40)
    parser.add_argument("--lr_decay_epochs_str", type=str, default='20-30')
    parser.add_argument("--check_point", type=int, default=0)
    parser.add_argument("--lr_decay_factor", type=float, default=0.1)
    parser.add_argument("--pre_train_file", type=str, default='.')
    parser.add_argument("--phases", type=list, default=['train', 'dev', 'test'])
    parser.add_argument("--train_preprocess_threads", type=int, default=8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
