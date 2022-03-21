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


import glob,cv2,time,random,sys,re,math,copy
from numpy import zeros,ones
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve
import scipy, os
from six import iteritems

def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)
def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)
def write_arguments_to_file(args, filename):
    """
    :param args:
    :param filename:
    :return: write args parameter to file
    """
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs
def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name + '/mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(name + '/stddev', stddev)
        tf.summary.scalar(name + '/max', tf.reduce_max(var))
        tf.summary.scalar(name + '/min', tf.reduce_min(var))
        tf.summary.histogram(name + '/histogram', var)

def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3):
    '''
    img is a 3-D variable (ex: tf.Variable(image, validate_shape=False) ) and  HWC order
    '''
    # HWC order
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    channel = tf.shape(img)[2]
    area = tf.cast(width*height, tf.float32)

    erase_area_low_bound = tf.cast(tf.round(tf.sqrt(sl * area * r1)), tf.int32)
    erase_area_up_bound = tf.cast(tf.round(tf.sqrt((sh * area) / r1)), tf.int32)
    h_upper_bound = tf.minimum(erase_area_up_bound, height)
    w_upper_bound = tf.minimum(erase_area_up_bound, width)

    h = tf.random.uniform([], erase_area_low_bound, h_upper_bound, tf.int32)
    w = tf.random.uniform([], erase_area_low_bound, w_upper_bound, tf.int32)

    x1 = tf.random.uniform([], 0, height+1 - h, tf.int32)
    y1 = tf.random.uniform([], 0, width+1 - w, tf.int32)

    erase_area = tf.cast(tf.random.uniform([h, w, channel], 0, 255, tf.int32), tf.uint8)

    erasing_img = img[x1:x1+h, y1:y1+w, :].assign(erase_area)

    return tf.cond(tf.random.uniform([], 0, 1) > probability, lambda: img, lambda: erasing_img)


def random_erase_np_v2(img, probability = 0.3, sl = 0.02, sh = 0.4, r1 = 0.3):

    if np.random.uniform() > probability:
        return img

    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    area = width * height

    erase_area_low_bound = np.round( np.sqrt(sl * area * r1) ).astype(np.int)
    erase_area_up_bound = np.round( np.sqrt((sh * area) / r1) ).astype(np.int)
    if erase_area_up_bound < height:
        h_upper_bound = erase_area_up_bound
    else:
        h_upper_bound = height
    if erase_area_up_bound < width:
        w_upper_bound = erase_area_up_bound
    else:
        w_upper_bound = width

    h = np.random.randint(erase_area_low_bound, h_upper_bound)
    w = np.random.randint(erase_area_low_bound, w_upper_bound)

    x1 = np.random.randint(0, height+1 - h)
    y1 = np.random.randint(0, width+1 - w)

    x1 = np.random.randint(0, height - h)
    y1 = np.random.randint(0, width - w)
    # img_ori = img * 1
    img[x1:x1+h, y1:y1+w, :] = np.random.randint(0, 255, size=(h, w, channel)).astype(np.uint8)

    return img

def get_ped_dataset(path, train_file):
    dataset = []
    with open( train_file ) as f:
        for eachline in f:
            contents = eachline.strip().split(' ')
            image_path = os.path.join( path, contents[0] )
            labels = [int(x) for x in contents[1:]]
            # labels = np.array( labels, dtype=np.int )

            dataset.append( (image_path, labels) )
    return dataset
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += [dataset[i][0]]
        labels_flat += [dataset[i][1]]
    return image_paths_flat, labels_flat

class ImageClass():
    """
    Stores the paths of images for a given video
    input: video_name, image_paths
    output: class(include three functions)
    """
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
    def __len__(self):
        return len(self.image_paths)


def get_aug_flag(flag, set):
    return tf.equal(set, flag)
def align_imagee_py(image_decoded, target_image_size):
    image_decoded = image_decoded.decode()
    image_size = cv2.imread(image_decoded).shape
    size_h, size_w = image_size[0], image_size[1]
    resize_flag = False
    if (size_h != target_image_size[0]) or (size_w != target_image_size[1]):
        resize_flag = True
        size_h, size_w = target_image_size[0], target_image_size[1]
        size_h = np.array(size_h).astype('int64')
        size_w = np.array(size_w).astype('int64')
    return size_h, size_w, resize_flag
def resize_py_image(image_decoded, image_size):
    image_size = tuple(image_size)
    image_resized = cv2.resize(image_decoded, image_size)
    return image_resized
def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    # return misc.imrotate(image, angle, 'bicubic')
    return scipy.ndimage.interpolation.rotate(image, angle)
def is_preprocess_imagenet(image, flag):
    image = tf.cast(image, tf.float32)
    if flag:
        #### Preprocess: imagr normalization per channel
        imagenet_mean = \
            np.array([0.4914, 0.4822, 0.4465], dtype=np.float32) * 255.0
        imagenet_std = \
            np.array([0.2023, 0.1994, 0.2010], dtype=np.float32) * 255.0
        image = (image - imagenet_mean) / imagenet_std
    return image
def is_standardization(image, flag=True):
    if flag:return tf.image.per_image_standardization(image)
    else:return image
def random_rotate_np(img, prob = 0.5):
    if np.random.uniform() > prob:
        return img
    angle = np.random.uniform(-10,10,1)[0]
    angle = angle * 3.14 / 180
    img = scipy.ndimage.interpolation.rotate(img, angle, reshape=False)
    return img
def distort_color(image, color_ordering, alpha=8, beta=0.2, gamma=0.05):
    if (color_ordering ==1) or (color_ordering ==0):
        image = image
    elif color_ordering ==2:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
    elif color_ordering ==3:
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==4:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==5:
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==6:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==7:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha/255)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
    elif color_ordering ==8:
        image = tf.image.random_contrast(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
    elif color_ordering ==9:
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_saturation(image, lower=1.0-beta, upper=1.0+beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
        image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
    elif color_ordering ==10:
        image = tf.image.random_saturation(image, lower=1.0 - beta, upper=1.0 + beta)
        image = tf.image.random_hue(image, max_delta=gamma)
        image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
        image = tf.image.random_brightness(image, max_delta=alpha / 255)
    else:
        print('color_ordering is error!', color_ordering)
        exit(0)
    return image


class Dataset():
    def __init__(self, config, images_list, labels_list, mode, test_flip = False):
        self.config = config
        self.image_size = (config.image_size, config.image_size)
        self.buffer_size = int(len(images_list) / self.config.batch_size)
        # self.seed = config.seed

        self.mean_div = []
        disorder_para = []
        data_augment = []
        color_para = []
        self.mean_div += (float(i) for i in config.mean_div_str.split('-'))
        color_para += (float(i) for i in config.color_para_str.split('-'))
        data_augment += (int(i) for i in config.data_augment_str.split('-'))
        self.RANDOM_ROTATE = data_augment[0]
        self.RANDOM_FLIP = data_augment[1]
        self.RANDOM_CROP = data_augment[2]
        self.RANDOM_COLOR = data_augment[3]
        self.is_std = data_augment[4]
        self.RANDOM_ERASING = data_augment[5]
        self.c_alpha = int(color_para[0])
        self.c_beta = color_para[1]
        self.c_gamma = color_para[2]

        self.mode = mode
        self.test_flip = test_flip

        labels_list = np.stack(labels_list, axis=0)
        self.input_tensors = self.inputs_for_training(images_list, labels_list )
        self.nextit = self.input_tensors.make_one_shot_iterator().get_next()

    def inputs_for_training(self, images_list, labels_list):
        dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=-1)
        dataset = dataset.shuffle(self.buffer_size).batch(self.config.batch_size).repeat(self.config.max_nrof_epochs)

        return dataset

    def _parse_function(self, filename, label):
        image = tf.image.decode_image(tf.io.read_file(filename), channels=3)
        ### @0: get info of rgb_size
        rgb_size_h, rgb_size_w, resize_flag = \
            tuple(tf.py_func(align_imagee_py, [filename, self.image_size], [tf.int64, tf.int64, tf.bool]))
        ### @1: resize image_color_modal
        image = tf.cond(resize_flag,
                        lambda: tf.py_func(resize_py_image, [image, (rgb_size_w, rgb_size_h)], tf.uint8),
                        lambda: image)
        ### @3: Distort_color
        image = tf.cond(get_aug_flag(self.RANDOM_COLOR, 1),
                        lambda: distort_color(image, self.c_alpha, self.c_beta, self.c_gamma),
                        lambda: tf.identity(image))
        if self.test_flip:
            image_flip = tf.image.flip_left_right( image )
            image_flip = tf.cast(image_flip, tf.float32)
        ### @4: Random flip
        if self.RANDOM_FLIP:
            do_flip = tf.random.uniform([]) > 0.5
            image = tf.cond(do_flip,
                        lambda: tf.image.flip_left_right( image ),
                        lambda: tf.identity(image) )
        else:
            image = tf.identity(image)

        image.set_shape(self.image_size + (3,))

        ### @2: RANDOM_ROTATE
        # if self.RANDOM_ROTATE == 1: rotate_flag += 1
        image = tf.cond(get_aug_flag(self.RANDOM_ROTATE, 1),
                        lambda: tf.py_func(random_rotate_np, [image], tf.uint8 ),
                        lambda: tf.identity(image) )
        ### @5 Random Erasing
        image = tf.cond(get_aug_flag( self.RANDOM_ERASING, 1),
                        lambda: tf.py_func(random_erase_np_v2, [image], tf.uint8),
                        lambda: tf.identity(image))
        # ### @5: Crop_Resize
        # if (self.RANDOM_CROP == 1): crop_flag = int(not crop_flag)
        # image = tf.cond(get_aug_flag(crop_flag, 1),
        #                 lambda: tf.random_crop(image, self.config.image_size + (3,), seed=seed),
        #                 lambda: tf.py_func(resize_py_image, [image, self.config.image_size], tf.uint8))
        # ### FIXED_STANDARDIZATION
        image = (tf.cast(image, tf.float32) - self.mean_div[0]) / self.mean_div[1]
        image = is_preprocess_imagenet(image, self.is_std)
        image = tf.cast(image, tf.float32)
        image.set_shape(self.image_size + (3,))
        if self.test_flip:
            image_ori = tf.expand_dims(image, axis=0)
            image_flip = tf.expand_dims(image_flip, axis=0)
            image = tf.concat([image_ori, image_flip], axis=0)
        return image, label, filename

def get_saver_resnet_tf( max_to_keep = 1024):
    def show_vars(trainable_list):
        for i in range(len(trainable_list)):
            var_name = trainable_list[i].name
            print('{} {}'.format(i, var_name))
    ### save global_variables ###
    global_list = tf.compat.v1.global_variables()
    print('***** Network parameter Info *******')
    print('*** global={}'.format(len(global_list)))
    ### Only save trainable ###
    trainable_list = tf.compat.v1.trainable_variables()
    print('*** trainable={}'.format(len(trainable_list)))
    #### save trainable + bn ###
    bn_moving_vars = [g for g in global_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]
    trainable_list += bn_moving_vars
    print('*** trainable@bn_moving_vars={}'.format(len(trainable_list)))
    saver = tf.compat.v1.train.Saver(trainable_list, max_to_keep=max_to_keep)
    ### move out some variable ###
    restore_trainable_list = [t for t in trainable_list if 'logits' not in t.name]
    restore_trainable_list = [t for t in restore_trainable_list if '_reduce' not in t.name]
    ###
    # show_vars(trainable_list)
    # show_vars(restore_trainable_list)
    saver_restore = tf.compat.v1.train.Saver(restore_trainable_list)
    return saver, saver_restore, trainable_list, restore_trainable_list


def get_train_op(total_loss, trainable_list, global_step, optimizer, learning_rate):

    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    elif optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Invalid optimization algorithm')
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys)
    with tf.control_dependencies(update_ops):
        grads = opt.compute_gradients(total_loss, var_list=trainable_list)
        train_op = opt.apply_gradients(grads, global_step=global_step)
    # train_op = slim.learning.create_train_op(total_loss, opt, global_step=global_step)
    return train_op

def get_lr(initial_lr, lr_decay_epochs, epoch_idx, lr_decay=0.1):
    lr = initial_lr
    for s in lr_decay_epochs:
        if epoch_idx >= s:
            lr *= lr_decay
    return lr


def write_Resnet_Race(save_path, image_batch, label_batch, filename_batch, mean_div, batch_it,
                         logits, accuracy, data_name, fid, is_save_images=True, is_std=False, flag='@'):
    ### parameters ###

    return True

def write_Resnet_Race_v2(save_path, image_batch, rp_images_batch, label_batch, filename_batch, mean_div, batch_it,
                         logits, accuracy, data_name, fid, is_save_images=False, is_std=False, flag='@'):
    ### parameters ###
    num_colorspace = 1
    logit_acc_mean = 0.0
    accuracy_mean = 0.0
    sum_prob_score = 0.0
    sum_exp_score = 0.0
    def realProb(logits):
        x = np.array(logits)
        if np.isinf(np.sum(np.exp(x))):
            return 0, 0
        y = np.exp(x[0]) / np.sum(np.exp(x))
        return x[0], y

    if logits.shape[0] == (2 * image_batch.shape[0]):
        image_batch = np.concatenate((image_batch, image_batch), axis=0)
        rp_images_batch = np.concatenate((rp_images_batch, rp_images_batch), axis=0)
        label_batch = np.concatenate((label_batch, label_batch), axis=0)
        filename_batch = np.concatenate((filename_batch, filename_batch), axis=0)
    for frame_ind in range(len(image_batch)):
        frame_all = batch_it * len(image_batch) + frame_ind
        sample_label = label_batch[frame_ind]
        P_list = filename_batch[frame_ind].split('/')
        P = [P_list[-3], 'G({})'.format(str(sample_label)), P_list[-1]]
        sample_name = '_'.join(P)
        if frame_ind == 0:first_name = P_list[-3]
        assert first_name == P_list[-3]
        ### testing ###
        if (save_path.find('dev') == -1) and (save_path.find('test') == -1):
            continue
        ### compute logical score ###
        out = np.argmax(np.array(logits[frame_ind]))
        logit_acc = int(out == sample_label)
        logit_acc_mean += float(logit_acc)
        accuracy_mean += accuracy
        prob_score, exp_score = realProb(logits[frame_ind])
        sum_prob_score += prob_score
        sum_exp_score += exp_score
    if (save_path.find('dev') == -1) and (save_path.find('test') == -1):return True
    else:
        ### write score ###
        fid.write(flag + '@GT_' + str(sample_label) + ',' + str(prob_score / (frame_ind + 1)) + ',' +
            str(exp_score / (frame_ind + 1)) + '\n')
        print(
        '* batch_it={}, flag={}, sample_name={}, batch_size={}, all_samples={}, logit_acc/accuracy={}/{}'.format(
            batch_it + 1, flag, sample_name, (frame_ind + 1), frame_all + 1, str(logit_acc_mean / (frame_ind + 1)),
            str(accuracy_mean / (frame_ind + 1))))

def train_Resnet_Race(sess, epoch, dataset_train, learning_rate_p, isTraining_p, batch_size_p,
                      image_batch_p, label_batch_p, lr, batch_size, epoch_size, mean_div,
                      logits, accuracy, total_loss, train_op, global_step,
                      learning_rate, outputs_dir, phase, fid, data_name):
    ### Training loop ###
    batch_number = 0
    train_time = 0
    interval_save = 20
    while batch_number < epoch_size:
        st = time.time()
        image_batch, label_batch, filename_batch = sess.run(list(dataset_train.nextit))
        feed_dict = {learning_rate_p: lr, isTraining_p: True, batch_size_p: batch_size, image_batch_p: image_batch, label_batch_p: label_batch}
        tensor_list = [total_loss, train_op, global_step, learning_rate, accuracy, logits]
        if batch_number % interval_save == 0:
            loss_,_, step_, lr_, accuracy_, logits_ = sess.run(tensor_list, feed_dict=feed_dict)
            ### Generate intermediate results
            write_Resnet_Race(os.path.join(outputs_dir, phase), image_batch, label_batch,
                filename_batch, mean_div, batch_number, logits_, accuracy_, data_name, fid)
        else:
            loss_, _, step_, lr_, accuracy_, logits_ = sess.run(tensor_list, feed_dict = feed_dict)
        duration = time.time() - st
        print('FAR: Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\t''Lr %2.5f'
            %(epoch, batch_number + 1, epoch_size, duration, loss_, accuracy_, lr_))
        if batch_number % 100 == 0:
            fid.write('FAR: Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\t''Lr %2.5f\n'
                %(epoch, batch_number + 1, epoch_size, duration, loss_, accuracy_, lr_))
        batch_number += 1
        train_time += duration
    return True

def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
    ### Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)

def test_Resnet_Race(sess, image_list, dataset_test, label_batch_p, isTraining_p, batch_size_p,
                     image_batch_p, batch_size, mean_div, logits,
                     accuracy, total_loss, outputs_dir, phase, fid, data_name, is_training=False, test_flip=False):

    # assert len(image_list) % batch_size == 0
    print('Running forward pass on evaluate set')
    AllN_s = len(image_list)
    nrof_batches = AllN_s // batch_size
    nrof_images = nrof_batches * batch_size
    # AllN = nrof_images * 2 if test_flip else nrof_images
    test_batches = nrof_batches
    pre_labels_array = np.zeros((nrof_images, 35), np.float32)
    labels_array = np.zeros((nrof_images, 35), np.float32)
    ### Testing loop
    start_idx = 0
    for batch_it in range(test_batches):  # num of video
        image_batch, label_batch, filename_batch = sess.run(list(dataset_test.nextit))
        if test_flip:
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch[:,0], label_batch_p: label_batch}
            tensor_list = [accuracy, logits]
            accuracy_1, logits_1 = sess.run(tensor_list, feed_dict=feed_dict)
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch[:, 1], label_batch_p: label_batch }
            tensor_list = [accuracy, logits]
            accuracy_2, logits_2 = sess.run(tensor_list, feed_dict=feed_dict)
            logits_ = (logits_1 + logits_2)/2
            accuracy_ = (accuracy_1 + accuracy_2)/2
            pre_labels_ = logits_
            pre_labels_ = pre_labels_ > 0
            image_batch = image_batch[:,0]
        else:
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch, label_batch_p: label_batch}
            tensor_list = [accuracy, logits]
            accuracy_, logits_ = sess.run(tensor_list, feed_dict=feed_dict)
            pre_labels_ = logits_
            pre_labels_ = pre_labels_ > 0

        c_n = pre_labels_.shape[0]
        # pre_labels_array[start_idx:start_idx+c_n] = pre_labels
        pre_labels_array[start_idx:start_idx + c_n] = pre_labels_
        labels_array[start_idx:start_idx + c_n] = label_batch
        start_idx += c_n
        write_Resnet_Race(os.path.join(outputs_dir, phase), image_batch, label_batch,
                          filename_batch, mean_div, batch_it, logits_, accuracy_, data_name, fid)
        if batch_it % 10 == 0:
            print('.', end='')
            sys.stdout.flush()
    print('')
    Result = instance_evaluation(pre_labels_array, labels_array)
    return Result

def instance_evaluation(p_labels, g_labels):
# p_labels-->predicted labels,   g_label-->ground-truth labels
    N,L = g_labels.shape
    accuracy_pos = np.zeros((L,1))
    accuracy_neg = np.zeros((L,1))

    pos_index = (g_labels==1)
    neg_index = (g_labels==0)
    pos_cnt = np.sum(pos_index, 0)
    neg_cnt = np.sum(neg_index, 0)

    zero_flag = pos_cnt==0
    pos_cnt[zero_flag] = 1
    if(np.sum(zero_flag) > 0):
        print ("some attributes have no test image")

    zero_flag = neg_cnt == 0
    neg_cnt[zero_flag] = 1
    if (np.sum(zero_flag) > 0):
        print ("some attributes have no test image")

    flag = p_labels==g_labels
    for i in range(0,L):
        accuracy_pos[i] = np.sum( flag[pos_index[:,i],i] )/float(pos_cnt[i])
        accuracy_neg[i] = np.sum( flag[neg_index[:,i],i] )/float(neg_cnt[i])
    accuracy_all = (accuracy_pos + accuracy_neg)/2.0
    accuracy_mean = np.mean(accuracy_all)

    # instance evaluation
    # print accuracy_all
    # print accuracy_mean

    gt_pos_index = np.zeros((N,L),dtype='int')
    pt_pos_index = np.zeros((N,L),dtype='int')
    gt_pos_index[g_labels == 1] = 1
    pt_pos_index[p_labels == 1] = 1
    t = gt_pos_index + pt_pos_index
    tmp = np.sum(t >= 2, 1)
    tmp1 = np.sum(t>=1,1)
    tmp2 = np.sum(gt_pos_index >= 1, 1)
    tmp3 = np.sum(pt_pos_index >= 1, 1)

    f1 = tmp1 != 0
    f2 = tmp2 != 0
    f3 = tmp3 != 0

    tmp = tmp.astype(np.float32)
    tmp1 = tmp1.astype(np.float32)
    tmp2 = tmp2.astype(np.float32)
    tmp3 = tmp3.astype(np.float32)

    #tmp.dtype = 'float'
    #tmp1.dtype = 'float'
    #tmp2.dtype = 'float'
    #tmp3.dtype = 'float'


    instance_accuracy = np.sum(tmp[f1]/ tmp1[f1] )/np.sum(f1)
    instance_recall = np.sum(tmp[f2]/tmp2[f2]  )/np.sum(f2)
    instance_precision = np.sum(tmp[f3]/tmp3[f3] )/np.sum(f3)
    instance_F1 = 2*instance_precision*instance_recall/(instance_recall + instance_precision)

    # print instance_accuracy
    # print instance_recall
    # print instance_precision
    # print instance_F1

    Result = {}
    Result['accuracy_all'] = accuracy_all
    Result['accuracy_mean'] = accuracy_mean
    Result['instance_accuracy'] = instance_accuracy
    Result['instance_recall'] = instance_recall
    Result['instance_precision'] = instance_precision
    Result['instance_F1'] = instance_F1

    return Result

