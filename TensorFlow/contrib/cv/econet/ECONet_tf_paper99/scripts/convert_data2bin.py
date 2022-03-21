import random
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import pickle
import sys

sys.path.append('../')

import tensorflow as tf
import tensorboard

from dataset import Video_3D
from model.econet import ECONet
from opts import parser

from transforms import resize, get_center_crop, get_multi_scale_crop, get_random_horizontal_flip, stack_then_normalize


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description="Binary data conversion")
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51'], required=True)
parser.add_argument('--data_path', type=str, required=True)

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
    clip_holder, label_holder, data_name_holder = tf.py_func(
        process_video, [train_info_tensor, dataset_path, is_training, num_segments], [tf.float32, tf.int64, tf.string]) 
    return clip_holder, label_holder, data_name_holder


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
    data_name = os.path.basename(video.path)

    return clip_seq, label_seq, data_name


def main():
    args = parser.parse_args()

    # Preload data filelist 
    test_file = 'splits_txt/{}/{}_val_split_1_rawframes.txt'.format(args.dataset, args.dataset)
    dataset_path = args.data_path
    batch_size = 1 

    with open(test_file, 'r') as f:
        test_info = list()
        for line in f.readlines():
            test_info.append(line.strip().split(' '))

    num_test_sample = len(test_info)
    test_epoch_step = int(np.ceil(num_test_sample/batch_size))

    test_info_tensor = tf.constant(test_info)

    num_segments = 4

    # Build dataset
    test_info_dataset = tf.data.Dataset.from_tensor_slices(
        (test_info_tensor))
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset_path=dataset_path, is_training=False, num_segments=num_segments), num_parallel_calls=8)

    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=2*batch_size)

    test_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types, test_dataset.output_shapes)

    test_init_op = test_iterator.make_initializer(test_dataset)
    test_clip_holder, test_label_holder, data_name_holder = test_iterator.get_next()
    test_clip_holder = tf.reshape(test_clip_holder, [-1, 224, 224, 3])

    # Specific Hyperparams
    sess = tf.Session()

    # sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) 

    # Test Phase
    print('Start conversion...')
    print('Number of test samples: [{}]'.format(num_test_sample))
    sess.run(test_init_op)

    label_dict = {}

    if not os.path.exists('bin'):
        os.mkdir('bin')

    if not os.path.exists('bin/{}'.format(args.dataset)):
        os.mkdir('bin/{}'.format(args.dataset))

    # start test process        
    with tqdm(range(test_epoch_step)) as t:
        for _ in t:
            clip, label, data_name = sess.run([test_clip_holder, test_label_holder, data_name_holder])
            basename = bytes.decode(data_name[0])
            clip.tofile('bin/{}/'.format(args.dataset)+basename+'.bin')

            label_dict[basename] = int(label)

            t.update()

    sess.close()

    with open('bin/{}_label.pkl'.format(args.dataset), 'wb') as f:
        pickle.dump(label_dict, f)

if __name__ == '__main__':
    main()
