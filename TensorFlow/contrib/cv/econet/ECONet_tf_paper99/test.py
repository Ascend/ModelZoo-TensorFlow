import random
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

import tensorflow as tf
import tensorboard
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

from dataset import Video_3D
from model.econet import ECONet
from opts import parser

from transforms import resize, get_center_crop, get_multi_scale_crop, get_random_horizontal_flip, stack_then_normalize


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CLASS_INFO = {
    'ucf101': 101,
    'hmdb51': 51 
}

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_random_patch(frame_list, patch_size):

    ih, iw = frame_list[0].shape[:2]
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    def _get_random_patch(frame):
        frame = frame[iy:iy + ip, ix:ix + ip, :]
        return frame

    return [_get_random_patch(f) for f in frame_list]


def get_center_patch(frame_list, patch_size):

    ih, iw = frame_list[0].shape[:2]
    ip = patch_size

    ix = int((iw - ip) / 2)
    iy = int((ih - ip) / 2)

    def _get_center_patch(frame):
        frame = frame[iy:iy + ip, ix:ix + ip, :]
        return frame

    return [_get_center_patch(f) for f in frame_list]


def _get_data_label_from_info(train_info_tensor, dataset_path, is_training, num_segments):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    clip_holder, label_holder = tf.py_func(
        process_video, [train_info_tensor, dataset_path, is_training, num_segments], [tf.float32, tf.int64]) 
    return clip_holder, label_holder


def process_video(data_info, dataset_path, is_training, num_segments, data_augment=None):
    """ Get video clip and label from data info list."""
    video = Video_3D(data_info, dataset_path)
    clip_seq, label_seq = video.get_frames(num_segments, is_training=is_training)

    if is_training:
        clip_seq = get_multi_scale_crop(clip_seq, patch_size=224, scales=[1, .875, .75, .66])
        clip_seq = get_random_horizontal_flip(clip_seq)
    else:
        clip_seq = resize(clip_seq, patch_size=256)
        clip_seq = get_center_crop(clip_seq, patch_size=224)

    normalize_list = [104, 117, 128]
    clip_seq = stack_then_normalize(clip_seq, normalize_list)

    return clip_seq, label_seq


def main():
    # Load options
    args = parser.parse_args()

    # Initialize Logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO)
    logger = logging.getLogger('econet')

    logger.info('************* YOUR SETTINGS *************')
    for arg in vars(args):
        logger.info("%20s: %s" %(str(arg), str(getattr(args, arg))))
    logger.info('')

    assert args.modality.lower() in ['rgb'], logger.error('Only RGB data is supported')

    # Preload data filelist 
    test_file = args.val_list

    with open(test_file, 'r') as f:
        test_info = list()
        for line in f.readlines():
            test_info.append(line.strip().split(' '))

    num_test_sample = len(test_info)
    test_epoch_step = int(np.ceil(num_test_sample/args.batch_size))

    test_info_tensor = tf.constant(test_info)

    num_segments = args.num_segments 

    # Build dataset
    test_info_dataset = tf.data.Dataset.from_tensor_slices(
        (test_info_tensor))
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset_path=args.dataset_path, is_training=False, num_segments=num_segments), num_parallel_calls=args.workers)

    test_dataset = test_dataset.batch(args.batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=2*args.batch_size)

    test_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types, test_dataset.output_shapes)

    test_init_op = test_iterator.make_initializer(test_dataset)
    test_clip_holder, test_label_holder = test_iterator.get_next()
    test_clip_holder = tf.reshape(test_clip_holder, [-1, 224, 224, 3])

    clip_holder = tf.placeholder(tf.float32, [None, 224, 224, 3], name='clip_holder')
    label_holder = tf.placeholder(tf.int64, [None,], name='label_holder')
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

    # Network definition
    num_classes = CLASS_INFO[args.dataset]

    net_opt = {
       'weight_decay': args.weight_decay, 
       'net2d_keep_prob': 1.,
       'net3d_keep_prob': 1.,
       'num_segments': num_segments,
       'num_classes': num_classes 
    }

    logits, end_points = ECONet(clip_holder, opt=net_opt, is_training=is_training)

    pred_classes = tf.argmax(logits, axis=1)

    test_acc_op, test_acc_update = tf.metrics.accuracy(labels=label_holder, predictions=pred_classes)

    # For NPU usage
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = False
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=config)

    # For GPU usage
    # sess = tf.Session()

    sess.run(tf.local_variables_initializer()) 

    # Load pretrained model
    saver = tf.train.Saver()
    logger.info('Loading checkpoint from {}...'.format(args.resume_path))
    saver.restore(sess, args.resume_path)

    # Test Phase
    logger.info('Start Test Phase...')
    logger.info('Number of test samples: [{}]'.format(num_test_sample))
    sess.run(test_init_op)

    # start test process        
    for _ in range(test_epoch_step):
        clip, label = sess.run([test_clip_holder, test_label_holder])
        test_iter_acc = sess.run(test_acc_update, feed_dict={clip_holder: clip, label_holder: label, is_training: False})

    test_acc = sess.run(test_acc_op, feed_dict={clip_holder: clip, label_holder: label, is_training: False})

    logger.info('Avg Test Acc.: {:.2f}'.format(test_acc*100))

    sess.close()


if __name__ == '__main__':
    main()
