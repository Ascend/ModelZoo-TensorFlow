import time, random, sys, cv2, os, scipy
import tensorflow as tf
import numpy as np
from six import iteritems
from random_erasing import random_erasing, random_erase_np_v2
from skimage import transform as trans

def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)
def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)
def get_father_path(pwd):
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    return father_path
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

def random_rotate(img, lms):
    lms_new = lms
    # angle = np.random.uniform(-10,10,1)[0]
    angle = tf.random_uniform((1,), minval=-10, maxval=10, dtype=tf.float32)[0]
    angle = angle*3.14/180
    img = tf.contrib.image.rotate( img, angle )
    cos, sin = tf.cos(angle), tf.sin(angle)
    rotate_mat = tf.Variable([[cos, -sin], [sin, cos]], trainable=False, dtype=tf.float32)
    rotate_mat = tf.reshape( rotate_mat, [2,2] )
    lms_new = lms/224 - 0.5
    lms_new = tf.matmul(lms_new, rotate_mat)
    lms_new += 0.5
    lms_new = lms_new * 224
    return img, lms_new

def random_rotate_np(img, lms, prob = 0.5):
    if np.random.uniform() > prob:
        return img, lms
    lms_new = lms
    angle = np.random.uniform(-10,10,1)[0]
    angle = angle * 3.14 / 180
    img = scipy.ndimage.interpolation.rotate(img, angle, reshape=False)
    cos, sin = np.cos(angle), np.sin(angle)
    rotate_mat = np.array( [[cos, -sin], [sin, cos]], dtype=np.float32 )
    lms_new = lms / 224 - 0.5
    lms_new = np.matmul(lms_new, rotate_mat)
    lms_new += 0.5
    lms_new = lms_new * 224
    return img, lms_new

def identity_imglms_rotate(image, lms):
    return image, lms

def identity_imglms(image, lms_two):
    return image, lms_two[0]

def distort_color(image, alpha=8, beta=0.2, gamma=0.05):
    image = tf.image.random_brightness(image, max_delta=alpha / 255)
    image = tf.image.random_contrast(image, lower=1.0 - beta, upper=1.0 + beta)
    image = tf.image.random_hue(image, max_delta=gamma)
    image = tf.image.random_saturation(image, lower=1.0 - beta, upper=1.0 + beta)
    return image

def flip_imglms(image, lms_two):
    image_flip = tf.image.flip_left_right(image)
    lms_flip = lms_two[1]
    return image_flip, lms_flip

def get_aug_flag(flag, set):
    return tf.equal(set, flag)

def align_imagee_py(image_decoded, target_image_size):
    # image_decoded = image_decoded.decode()
    # image_size = cv2.imread(image_decoded).shape
    # image_size = misc.imread(image_decoded).shape
    image_size = [224, 224]
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

def is_preprocess_imagenet(image, flag):
    image = tf.cast(image, tf.float32)
    if flag:
        #### Preprocess: imagr normalization per channel
        imagenet_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32) * 255.0
        imagenet_std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32) * 255.0
        image = (image - imagenet_mean) / imagenet_std
    return image

def is_standardization(image, flag=True):
    if flag:return tf.image.per_image_standardization(image)
    else:return image

def get_trans_theta(landmarks):
    pts_dst = [
        [[20.5, 26.2], [43.5, 26.24], [32, 39.04], [23.04, 52.5], [41.6, 52.5]], #global 1
        [[17.6, 24.8], [46.4, 24.8], [32, 40.8], [20.8, 57.6], [44, 57.6]], #global 2
        [[32, 16], [32, 32], [44, 32], [20, 32]], #eye
        [[8, 16], [32, 16], [56, 16], [32, 48]], #nose
        [[16, 32], [32, 32], [48, 32], [32, 16]], #mouth
    ]

    idx_ori = [
        [0,1,7,10,11], # global 1
        [0,1,7,10,11], # global 2
        [22,0,16,18], # eye
        [0,6,1,7], #nose
        [10,14,11,15], #mouth
    ]

    thetas = np.zeros((6, 2, 3), dtype=np.float32)
    thetas[0] = np.array([[1,0,0],[0,1,0]], dtype=np.float32)

    for i in range( len(idx_ori) ):
        p_dst = pts_dst[i]
        p_dst = np.array( p_dst, dtype=np.float32 )
        p_dst = p_dst/32 - 1

        #------parse landmarks----
        p_ori = np.zeros( (len(idx_ori[i]), 2), dtype=np.float32 )
        for j in range( len(idx_ori[i]) ):
            p_ori[j] = landmarks[idx_ori[i][j]]

        p_ori = p_ori * 224 / (4.0*224)
        p_ori = p_ori/28 - 1

        tform = trans.SimilarityTransform()
        tform.estimate( p_dst, p_ori )
        M = tform.params[0:2, :]
        thetas[i + 1] = M

    return thetas

class Dataset():
    def __init__(self, config, images_list, labels_list, lms_list, mode, test_flip=False):
        self.config = config
        self.image_size = (config.image_size, config.image_size)
        self.buffer_size = int(len(images_list) / self.config.batch_size)
        # self.seed = config.seed

        self.mean_div = []
        data_augment = []
        color_para = []
        self.mean_div += (float(i) for i in config.mean_div_str.split('-'))
        color_para += (float(i) for i in config.color_para_str.split('-'))
        data_augment += (int(i) for i in config.data_augment_str.split('-'))
        ### '1-1-0-1-0-1'
        self.RANDOM_ROTATE = data_augment[0]
        self.RANDOM_FLIP = data_augment[1]
        self.RANDOM_CROP = data_augment[2]
        self.RANDOM_COLOR = data_augment[3]
        self.is_std = data_augment[4]
        self.RANDOM_ERASING = data_augment[5]

        self.c_alpha = int(color_para[0])
        self.c_beta = color_para[1]
        self.c_gamma = color_para[2]

        self.test_flip = test_flip
        lms_list = np.stack(lms_list, axis=0)
        self.input_tensors = self.inputs_for_training(images_list, labels_list, lms_list)
        self.nextit = self.input_tensors.make_one_shot_iterator().get_next()

    def inputs_for_training(self, images_list, labels_list, lms_list):
        dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list, lms_list))
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=-1)
        dataset = dataset.shuffle(self.buffer_size).batch(self.config.batch_size).repeat(self.config.max_nrof_epochs)
        return dataset

    def _parse_function(self, filename, label, lms_two):
        image = tf.image.decode_image(tf.io.read_file(filename), channels=3)
        ### @0: get info of rgb_size
        rgb_size_h, rgb_size_w, resize_flag = \
            tuple(tf.py_func(align_imagee_py, [filename, self.image_size], [tf.int64, tf.int64, tf.bool]))
        ### @1: resize image_color_modal
        image = tf.cond(resize_flag,
                    lambda: tf.py_func(resize_py_image, [image, (rgb_size_w, rgb_size_h)], tf.uint8), lambda: image)
        ### @3: Distort_color
        image = tf.cond(get_aug_flag(self.RANDOM_COLOR, 1),
                    lambda: distort_color(image, self.c_alpha, self.c_beta, self.c_gamma), lambda: tf.identity(image))
        if self.test_flip:
            image_flip, lms_flip = flip_imglms(image, lms_two)
            image_flip = tf.cast(image_flip, tf.float32)
        ### @4: Random flip
        if self.RANDOM_FLIP:
            do_flip = tf.random.uniform([]) > 0.5
            image, lms = tf.cond(do_flip, lambda: flip_imglms(image, lms_two), lambda: identity_imglms(image, lms_two))
        else:
            image, lms = identity_imglms(image, lms_two)
        image.set_shape(self.image_size + (3,))
        lms = tf.cast(lms, dtype=tf.float32)
        ### @2: RANDOM_ROTATE
        image, lms = tf.cond(get_aug_flag(self.RANDOM_ROTATE, 1),
                        lambda: tuple(tf.py_func(random_rotate_np, [image, lms], [tf.uint8, tf.float32])),
                        lambda: identity_imglms_rotate(image, lms))
        # image, lms = tf.cond(get_aug_flag(self.RANDOM_ROTATE, 1),
        #                      lambda: random_rotate(image, lms),
        #                      lambda: identity_imglms_rotate(image, lms))
        ### @5 Random Erasing
        image = tf.cond(get_aug_flag(self.RANDOM_ERASING, 1),
                        lambda: tf.py_func(random_erase_np_v2, [image], tf.uint8),
                        lambda: tf.identity(image))
        # ### @5: Crop_Resize
        # if (self.RANDOM_CROP == 1): crop_flag = int(not crop_flag)
        # image = tf.cond(get_aug_flag(crop_flag, 1),
        #                 lambda: tf.random_crop(image, self.config.image_size + (3,), seed=seed),
        #                 lambda: tf.py_func(resize_py_image, [image, self.config.image_size], tf.uint8))
        #
        ### FIXED_STANDARDIZATION
        image = (tf.cast(image, tf.float32) - self.mean_div[0]) / self.mean_div[1]
        image = is_preprocess_imagenet(image, self.is_std)
        image = tf.cast(image, tf.float32)
        image.set_shape(self.image_size + (3,))
        thetas = tf.py_func(get_trans_theta, [lms], tf.float32)
        thetas.set_shape((6,2,3))

        if self.test_flip:
            image_ori = tf.expand_dims(image, axis=0)
            thetas_ori = tf.expand_dims(thetas, axis=0)
            image_flip = tf.expand_dims(image_flip, axis=0)
            thetas_flip = tf.py_func(get_trans_theta, [lms_flip], tf.float32)
            thetas_flip.set_shape((6, 2, 3))
            thetas_flip = tf.expand_dims(thetas_flip, axis=0)
            image = tf.concat([image_ori, image_flip], axis=0)
            thetas = tf.concat([thetas_ori, thetas_flip], axis=0)

        return image, label, filename, thetas

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

def write_Resnet_Race(save_path, image_batch, label_batch, pre_labels_, filename_batch, mean_div, batch_it,
                         logits, accuracy, data_name, fid, is_save_images=True, is_std=False, flag='@'):
    for c in range(len(label_batch)):
        str_line = str(c) + ' ' + str(pre_labels_[c]) + ' ' + str(label_batch[c]) + '\n'
        fid.write(str_line)
    return True

def train_Resnet_Race(sess, epoch, dataset_train, learning_rate_p, isTraining_p, batch_size_p,
                      image_batch_p, label_batch_p, thetas_batch_p, lr, batch_size, epoch_size, mean_div,
                      logits, accuracy, MAE, total_loss, train_op, global_step,
                      learning_rate, outputs_dir, phase, fid, data_name):
    ### Training loop ###
    batch_number = 0
    train_time = 0
    interval_save = 20
    while batch_number < epoch_size:
        st = time.time()
        image_batch, label_batch, filename_batch, thetas_batch = sess.run(list(dataset_train.nextit))
        feed_dict = {learning_rate_p: lr, isTraining_p: True, batch_size_p: batch_size, image_batch_p: image_batch,
                     label_batch_p: label_batch, thetas_batch_p: thetas_batch}
        tensor_list = [train_op, total_loss, global_step, learning_rate, accuracy, MAE, logits]
        if batch_number % interval_save == 0:
            _, loss_, step_, lr_, accuracy_, MAE_, logits_ = sess.run(tensor_list, feed_dict=feed_dict)
            ### Generate intermediate results
        else:
            _, loss_, step_, lr_, accuracy_, MAE_, logits_ = sess.run(tensor_list, feed_dict=feed_dict)
        duration = time.time() - st
        print('FAR: Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\tMAE %2.3f\t''Lr %2.5f'
            %(epoch, batch_number + 1, epoch_size, duration, loss_, accuracy_, MAE_, lr_))
        if batch_number % interval_save == 0:
            fid.write('FAR: Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\tMAE %2.3f\t''Lr %2.5f\n'
                %(epoch, batch_number + 1, epoch_size, duration, loss_, accuracy_, MAE_, lr_))
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

def test_Resnet_Race(sess, image_list, dataset_test, label_batch_p, isTraining_p, batch_size_p, thetas_batch_p,
                     image_batch_p, batch_size, mean_div, logits, pre_labels, accuracy, MAE, total_loss, outputs_dir,
                     phase, fid, data_name, is_training=False, test_flip=False):

    # assert len(image_list) % batch_size == 0
    print('Running forward pass on evaluate set')
    AllN_s = len(image_list)
    nrof_batches = AllN_s // batch_size
    nrof_images = nrof_batches * batch_size
    # AllN = nrof_images * 2 if test_flip else nrof_images
    test_batches = nrof_batches
    pre_labels_array = np.zeros((nrof_images, 1), np.float32)
    labels_array = np.zeros((nrof_images, 1), np.float32)

    ### Training loop ###
    start_idx = 0
    for batch_it in range(test_batches):
        st = time.time()
        image_batch, label_batch, filename_batch, thetas_batch = sess.run(list(dataset_test.nextit))
        if test_flip:
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch[:, 0],
                         label_batch_p: label_batch, thetas_batch_p: thetas_batch[:,0]}
            tensor_list = [accuracy, logits, MAE, pre_labels]
            accuracy_1, logits_1, MAE_1, pre_labels_1 = sess.run(tensor_list, feed_dict=feed_dict)
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch[:, 1],
                        label_batch_p: label_batch, thetas_batch_p: thetas_batch[:, 1]}
            tensor_list = [accuracy, logits, MAE, pre_labels]
            accuracy_2, logits_2, MAE_2, pre_labels_2 = sess.run(tensor_list, feed_dict=feed_dict)
            pre_labels_ = (pre_labels_1 + pre_labels_2)/2
            logits_ = (logits_1 + logits_2) / 2
            image_batch = image_batch[:, 0]
        else:
            feed_dict = {isTraining_p: False, batch_size_p: batch_size, image_batch_p: image_batch,
                         label_batch_p: label_batch, thetas_batch_p: thetas_batch}
            tensor_list = [accuracy, logits, MAE, pre_labels]
            accuracy_, logits_, MAE_, pre_labels_ = sess.run(tensor_list, feed_dict=feed_dict)
        for c in range(len(label_batch)):
            print(c, pre_labels_[c], label_batch[c])
        c_n = pre_labels_.shape[0]
        # pre_labels_array[start_idx:start_idx+c_n] = pre_labels
        pre_labels_array[start_idx:start_idx + c_n] = pre_labels_
        labels_array[start_idx:start_idx + c_n] = label_batch
        start_idx += c_n
        write_Resnet_Race(os.path.join(outputs_dir, phase), image_batch, label_batch, pre_labels_, filename_batch,
                          mean_div, batch_it, logits_, accuracy, data_name, fid)
        if batch_it % 2 == 0:
            # print(start_idx, end=' ')
            sys.stdout.flush()
    MAE_total = np.mean(np.abs(pre_labels_array - labels_array))
    return MAE_total


