#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
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

import numpy as np
import tf_models
from sklearn.preprocessing import scale
from npu_bridge.npu_init import *
import tensorflow as tf
#from tensorflow.contrib.keras.python.keras.backend import learning_phase
#from tensorflow.contrib.keras.python.keras.layers import concatenate, Conv3D
from tensorflow.python.keras.backend import learning_phase
from tensorflow.python.keras.layers import concatenate, Conv3D
from nibabel import load as load_nii
import os
import argparse
import tensorflow.python.keras as keras
import time
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator import npu_ops
# for NPU
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
# for NPU
def parse_inputs():
    def _str_to_bool(s):
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description='train the model')
    parser.add_argument('-r', '--root-path', dest='root_path', default='/npu/traindata/densenet3d')
    parser.add_argument('-sp', '--save-path', dest='save_path', default='dense24_correction')
    parser.add_argument('-lp', '--load-path', dest='load_path', default='dense24_correction')
    parser.add_argument('-ow', '--offset-width', dest='offset_w', type=int, default=12)
    parser.add_argument('-oh', '--offset-height', dest='offset_h', type=int, default=12)
    parser.add_argument('-oc', '--offset-channel', dest='offset_c', nargs='+', type=int, default=12)
    parser.add_argument('-ws', '--width-size', dest='wsize', type=int, default=38)
    parser.add_argument('-hs', '--height-size', dest='hsize', type=int, default=38)
    parser.add_argument('-cs', '--channel-size', dest='csize', type=int, default=38)
    parser.add_argument('-ps', '--pred-size', dest='psize', type=int, default=12)
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=2)
    parser.add_argument('-e', '--num-epochs', dest='num_epochs', type=int, default=5)
    parser.add_argument('-c', '--continue-training', dest='continue_training', type=bool, default=False)
    parser.add_argument('-nc', '--n4correction', dest='correction', type=bool, default=False)
    parser.add_argument('-mn', '--model_name', dest='model_name', type=str, default='dense24')
    parser.add_argument('-mul_rank_size', '--mul_rank_size', dest='mul_rank_size', type=int, default=1)
    parser.add_argument('-mul_device_id', '--mul_device_id', dest='mul_device_id', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_id', dest='gpu_id', type=str, default='0')
    parser.add_argument('-per', '--performance', dest='performance', type=bool, default=False)
    parser.add_argument('--autotune', dest='autotune', type=_str_to_bool, default=False)
    return vars(parser.parse_args())


def acc_tf(y_pred, y_true):
    correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, -1), tf.int32), tf.cast(tf.argmax(y_true, -1), tf.int32))
    return 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_patches_3d(data, labels, centers, hsize, wsize, csize, psize, preprocess=True):
    """

    :param data: 4D nparray (h, w, c, ?)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """
    patches_x, patches_y = [], []
    offset_p = (hsize - psize) / 2
    offset_p = int(offset_p)
    for i in range(len(centers[0])):
        h, w, c = centers[0, i], centers[1, i], centers[2, i]
        h_beg = min(max(0, h - hsize / 2), 240 - hsize)
        w_beg = min(max(0, w - wsize / 2), 240 - wsize)
        c_beg = min(max(0, c - csize / 2), 155 - csize)

        h_beg = int(h_beg)
        w_beg = int(w_beg)
        c_beg = int(c_beg)

        ph_beg = h_beg + offset_p
        pw_beg = w_beg + offset_p
        pc_beg = c_beg + offset_p
        vox = data[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize, :]
        vox_labels = labels[ph_beg:ph_beg + psize, pw_beg:pw_beg + psize, pc_beg:pc_beg + psize]
        patches_x.append(vox)
        patches_y.append(vox_labels)
    return np.array(patches_x), np.array(patches_y)


def positive_ratio(x):
    return float(np.sum(np.greater(x, 0))) / np.prod(x.shape)


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def segmentation_loss(y_true, y_pred, n_classes):
    y_true = tf.reshape(y_true, (-1, n_classes))
    y_pred = tf.reshape(y_pred, (-1, n_classes))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))


def vox_preprocess(vox):
    vox_shape = vox.shape
    vox = np.reshape(vox, (-1, vox_shape[-1]))
    vox = scale(vox, axis=0)
    return np.reshape(vox, vox_shape)


def one_hot(y, num_classes):
    y_ = np.zeros([len(y), num_classes])
    y_[np.arange(len(y)), y] = 1
    return y_


def dice_coef_np(y_true, y_pred, num_classes):
    """

    :param y_true: sparse labels
    :param y_pred: sparse labels
    :param num_classes: number of classes
    :return:
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_true = y_true.flatten()
    y_true = one_hot(y_true, num_classes)
    y_pred = y_pred.flatten()
    y_pred = one_hot(y_pred, num_classes)
    intersection = np.sum(y_true * y_pred, axis=0)
    return (2. * intersection) / (np.sum(y_true, axis=0) + np.sum(y_pred, axis=0))


def vox_generator(all_files, n_pos, n_neg,correction= False):
    path = options['root_path']
    while 1:
        for file in all_files:
            print("======", file)
            if correction:
                flair = load_nii(os.path.join(path, file, file + '_flair_corrected.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2_corrected.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1_corrected.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce_corrected.nii.gz')).get_data()
            else:
                flair = load_nii(os.path.join(path, file, file + '_flair.nii.gz')).get_data()
                t2 = load_nii(os.path.join(path, file, file + '_t2.nii.gz')).get_data()
                t1 = load_nii(os.path.join(path, file, file + '_t1.nii.gz')).get_data()
                t1ce = load_nii(os.path.join(path, file, file + '_t1ce.nii.gz')).get_data()

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = load_nii(os.path.join(path, file, file+'_seg.nii.gz')).get_data()

            foreground = np.array(np.where(labels > 0))
            background = np.array(np.where((labels == 0) & (flair > 0)))

            # n_pos = int(foreground.shape[1] * discount)
            foreground = foreground[:, np.random.permutation(foreground.shape[1])[:n_pos]]
            background = background[:, np.random.permutation(background.shape[1])[:n_neg]]

            centers = np.concatenate((foreground, background), axis=1)
            centers = centers[:, np.random.permutation(n_neg+n_pos)]

            yield data_norm, labels, centers


def label_transform(y, nlabels):
    return [
            keras.utils.to_categorical(np.copy(y).astype(dtype=np.bool),
                                       num_classes=2).reshape([y.shape[0], y.shape[1], y.shape[2], y.shape[3], 2]),

            keras.utils.to_categorical(y,
                                       num_classes=nlabels).reshape([y.shape[0], y.shape[1], y.shape[2], y.shape[3], nlabels])
            ]


def data_shard(rank_size, device_id, X_train, batch_size=None):
    if rank_size != 1:
        sample_one_device = int(len(X_train) / rank_size)
        X_train = X_train[device_id * sample_one_device:(device_id + 1) * sample_one_device]
    return X_train
 
 
 
 
 
def broadcast_global_variables(root_rank, index):
    op_list = []
    for var in tf.global_variables():
        if "float" in var.dtype.name:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))
    return tf.group(op_list)
    
def train():

    DATA_PATH = options['root_path']
    NUM_EPOCHS = options['num_epochs']
    LOAD_PATH = options['load_path']
    SAVE_PATH = options['save_path']
    PSIZE = options['psize']
    HSIZE = options['hsize']
    WSIZE = options['wsize']
    CSIZE = options['csize']
    model_name= options['model_name']
    BATCH_SIZE = options['batch_size']
    continue_training = options['continue_training']

    files = []
    num_labels = 5
    with open(DATA_PATH+'/train.txt') as f:
        for line in f:
            files.append(line[:-1])
    files = data_shard(options['mul_rank_size'], options['mul_device_id'],files)
    print( '%d training samples' % len(files))

    flair_t2_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))
    t1_t1ce_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 2))
    flair_t2_gt_node = tf.placeholder(dtype=tf.int32, shape=(None, PSIZE, PSIZE, PSIZE, 2))
    t1_t1ce_gt_node = tf.placeholder(dtype=tf.int32, shape=(None, PSIZE, PSIZE, PSIZE, 5))

    if model_name == 'dense48':
        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat_large(input=t1_t1ce_node, name='t1')
    elif model_name == 'no_dense':

        flair_t2_15, flair_t2_27 = tf_models.PlainCounterpart(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.PlainCounterpart(input=t1_t1ce_node, name='t1')

    elif model_name == 'dense24':

        flair_t2_15, flair_t2_27 = tf_models.BraTS2ScaleDenseNetConcat(input=flair_t2_node, name='flair')
        t1_t1ce_15, t1_t1ce_27 = tf_models.BraTS2ScaleDenseNetConcat(input=t1_t1ce_node, name='t1')
    else:
        print(' No such model name ')

    t1_t1ce_15 = concatenate([t1_t1ce_15, flair_t2_15])
    t1_t1ce_27 = concatenate([t1_t1ce_27, flair_t2_27])

    flair_t2_15 = Conv3D(2, kernel_size=1, strides=1, padding='same', name='flair_t2_15_cls')(flair_t2_15)
    flair_t2_27 = Conv3D(2, kernel_size=1, strides=1, padding='same', name='flair_t2_27_cls')(flair_t2_27)
    t1_t1ce_15 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_15_cls')(t1_t1ce_15)
    t1_t1ce_27 = Conv3D(num_labels, kernel_size=1, strides=1, padding='same', name='t1_t1ce_27_cls')(t1_t1ce_27)

    flair_t2_score = flair_t2_15[:, 13:25, 13:25, 13:25, :] + \
                     flair_t2_27[:, 13:25, 13:25, 13:25, :]

    t1_t1ce_score = t1_t1ce_15[:, 13:25, 13:25, 13:25, :] + \
                    t1_t1ce_27[:, 13:25, 13:25, 13:25, :]

    loss = segmentation_loss(flair_t2_gt_node, flair_t2_score, 2) + \
           segmentation_loss(t1_t1ce_gt_node, t1_t1ce_score, 5)

    acc_flair_t2 = acc_tf(y_pred=flair_t2_score, y_true=flair_t2_gt_node)
    acc_t1_t1ce = acc_tf(y_pred=t1_t1ce_score, y_true=t1_t1ce_gt_node)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        lr = 5e-4
        if int(options['mul_rank_size']) != 1:
         lr = lr * int(options['mul_rank_size'])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # for NPU
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        if int(options['mul_rank_size']) != 1:
            optimizer = npu_distributed_optimizer_wrapper(optimizer)
            optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)
        else:
            optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
        optimizer = optimizer.minimize(loss)
        # for NPU

    saver = tf.train.Saver(max_to_keep=15)
    data_gen_train = vox_generator(all_files=files, n_pos=200, n_neg=200,correction = options['correction'])


    # for npu
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    # modify for npu autotune start
    auto_tune = options['autotune']
    print("Autotune module is " + str(auto_tune))
    if auto_tune:
        print("[INFO] Enable autotone module, true")
        custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

    # modify for npu autotune end

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    # for npu

    with tf.Session(config=config) as sess:
        if continue_training:
            saver.restore(sess, LOAD_PATH)
        else:
            sess.run(tf.global_variables_initializer())
            if int(options['mul_rank_size']) != 1:
                bcast_op = broadcast_global_variables(0, 1)
                sess.run(bcast_op)
        for ei in range(NUM_EPOCHS):
            for pi in range(len(files)):
                acc_pi, loss_pi = [], []
                data, labels, centers = next(data_gen_train)
                n_batches = int(np.ceil(float(centers.shape[1]) / BATCH_SIZE))
                performance =  options['performance']
                if performance is True:
                    n_batches = 3
                duration = 0
                for nb in range(n_batches):
                    start_time = time.time()
                    offset_batch = min(nb * BATCH_SIZE, centers.shape[1] - BATCH_SIZE)
                    data_batch, label_batch = get_patches_3d(data, labels, centers[:, offset_batch:offset_batch + BATCH_SIZE], HSIZE, WSIZE, CSIZE, PSIZE, False)
                    label_batch = label_transform(label_batch, 5)
                    _, l, acc_ft, acc_t1c = sess.run(fetches=[optimizer, loss, acc_flair_t2, acc_t1_t1ce],
                                                   feed_dict={flair_t2_node: data_batch[:, :, :, :, :2],
                                                              t1_t1ce_node: data_batch[:, :, :, :, 2:],
                                                              flair_t2_gt_node: label_batch[0],
                                                              t1_t1ce_gt_node: label_batch[1],
                                                              learning_phase(): 1})
                    acc_pi.append([acc_ft, acc_t1c])
                    loss_pi.append(l)
                    n_pos_sum = np.sum(np.reshape(label_batch[0], (-1, 2)), axis=0)
                    duration += (time.time() - start_time)
                    print( 'epoch-patient: %d, %d, iter: %d-%d, p%%: %.4f, loss: %.4f, acc_flair_t2: %.2f%%, acc_t1_t1ce: %.2f%%, time cust: %.4f' % \
                          (ei + 1, pi + 1, nb + 1, n_batches, n_pos_sum[1]/float(np.sum(n_pos_sum)), l, acc_ft, acc_t1c, duration))
                    duration = 0
                print( 'patient loss: %.4f, patient acc: %.4f' % (np.mean(loss_pi), np.mean(acc_pi)))

            saver.save(sess, SAVE_PATH, global_step=ei)
            print( 'model saved')


if __name__ == '__main__':
    options = parse_inputs()
    # os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_id']
    train()
