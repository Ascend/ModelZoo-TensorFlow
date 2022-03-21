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
from transforms import resize, get_center_crop, get_multi_scale_crop, get_random_horizontal_flip, stack_then_normalize
from model.econet import ECONet
from opts import parser


os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"
os.environ["ASCEND_SLOG_PRINT_TO_STDOUT`"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
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

    # Create log directory
    log_dir = os.path.join('./experiments', get_timestamp())
    mkdir(log_dir)
    ckp_dir = os.path.join(log_dir, 'ckpt')
    mkdir(ckp_dir)

    # Initialize Logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO, filename=os.path.join(log_dir, 'log.txt'))
    logger = logging.getLogger('econet')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)

    logger.info("Log dir is [{}]".format(log_dir))

    logger.info('************* YOUR SETTINGS *************')
    for arg in vars(args):
        logger.info("%20s: %s" %(str(arg), str(getattr(args, arg))))
    logger.info('')

    assert args.modality.lower() in ['rgb'], logger.error('Only RGB data is supported')

    # Preload data filelist 
    train_file = args.train_list
    test_file = args.val_list

    with open(train_file, 'r') as f:
        train_info = list()
        for line in f.readlines():
            train_info.append(line.strip().split(' '))

    with open(test_file, 'r') as f:
        test_info = list()
        for line in f.readlines():
            test_info.append(line.strip().split(' '))

    num_train_sample = len(train_info)
    num_test_sample = len(test_info)

    train_info_tensor = tf.constant(train_info)
    test_info_tensor = tf.constant(test_info)

    num_segments = args.num_segments

    # Build training dataset
    train_info_dataset = tf.data.Dataset.from_tensor_slices(
        (train_info_tensor))
    train_info_dataset = train_info_dataset.shuffle(buffer_size=num_train_sample)
    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset_path=args.dataset_path, is_training=True, num_segments=num_segments), num_parallel_calls=args.workers)
        
    train_dataset = train_dataset.repeat().batch(args.batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=2*args.batch_size)

    # Build test dataset
    test_info_dataset = tf.data.Dataset.from_tensor_slices(
        (test_info_tensor))
    test_dataset = test_info_dataset.map(lambda x: _get_data_label_from_info(
        x, dataset_path=args.dataset_path, is_training=False, num_segments=num_segments), num_parallel_calls=args.workers)

    test_dataset = test_dataset.batch(args.batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=2*args.batch_size)

    train_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes)

    test_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types, test_dataset.output_shapes)

    train_init_op = train_iterator.make_initializer(train_dataset)
    test_init_op = test_iterator.make_initializer(test_dataset)

    train_clip_holder, train_label_holder = train_iterator.get_next()
    test_clip_holder, test_label_holder = test_iterator.get_next()

    train_clip_holder = tf.reshape(train_clip_holder, [-1, 224, 224, 3])
    test_clip_holder = tf.reshape(test_clip_holder, [-1, 224, 224, 3])

    clip_holder = tf.placeholder(tf.float32, [None, 224, 224, 3], name='clip_holder')
    label_holder = tf.placeholder(tf.int64, [None,], name='label_holder')
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    net2d_dropout_holder = tf.placeholder(tf.float32, name='net2d_dropout_holder')
    net3d_dropout_holder = tf.placeholder(tf.float32, name='net3d_dropout_holder')

    # Network definition
    num_classes = CLASS_INFO[args.dataset]

    net_opt = {
       'weight_decay': args.weight_decay, 
       'net2d_keep_prob': net2d_dropout_holder,
       'net3d_keep_prob': net3d_dropout_holder,
       'num_segments': num_segments,
       'num_classes': num_classes 
    }

    logits, end_points = ECONet(clip_holder, opt=net_opt, is_training=is_training)

    pred_classes = tf.argmax(logits, axis=1)

    train_acc_op, train_acc_update = tf.metrics.accuracy(labels=label_holder, predictions=pred_classes)
    test_acc_op, test_acc_update = tf.metrics.accuracy(labels=label_holder, predictions=pred_classes)

    # Loss calculation, excluding l2-norm
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=logits))

    # Specific Hyperparams
    train_epoch_step = int(np.ceil(num_train_sample/args.batch_size))
    test_epoch_step = int(np.ceil(num_test_sample/args.batch_size))

    logger.info('Num. training samples: [{}] | [{}] iters/epoch'.format(num_train_sample, train_epoch_step))
    logger.info('    Num. test samples: [{}] | [{}] iters/epoch'.format(num_test_sample, test_epoch_step))

    # Set learning rate schedule
    global_index = tf.Variable(0, trainable=False, name='learning_rate')
    init_lr = args.lr
    lr_boundaries = [int(s*train_epoch_step) for s in args.lr_steps]
    lr_values = [init_lr / (args.lr_decay**i) for i in range(len(lr_boundaries)+1)]
    learning_rate = tf.train.piecewise_constant(
        global_index, lr_boundaries, lr_values)

    # Initialize tensorboard summary
    loss_summary = tf.summary.scalar('loss', loss)
    lr_summary = tf.summary.scalar('learning_rate', learning_rate)
    train_acc_summary = tf.summary.scalar('acc/train_acc', train_acc_op)

    test_acc_summary = tf.summary.scalar('acc/test_acc', test_acc_op)

    # Optimizer set-up
    # For batch normalization, we then use this updata_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=args.momentum).minimize(loss, global_step=global_index)
    
    # For NPU usage
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = False
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=config)

    train_summary = tf.summary.merge([loss_summary, lr_summary, train_acc_summary])
    test_summary = tf.summary.merge([test_acc_summary])

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op)

    global_saver = tf.train.Saver()

    if args.resume_path is not None:
        var_list_no_fc = [v for v in tf.global_variables() if 'fc' not in v.name]

        # For finetuning usage
        load_saver = tf.train.Saver(var_list=var_list_no_fc)
        logger.info('Loading checkpoint from {}...'.format(args.resume_path))
        load_saver.restore(sess, args.resume_path)

    step = 0 
    best_acc = 0.
    best_epoch = 0
    
    for epoch in range(1, args.epochs+1):
        logger.info('Epoch: [{}] - Start Training Phase...'.format(epoch))

        # Local variables re-initilization for accuracy computation
        sess.run(tf.local_variables_initializer()) 
        sess.run(train_init_op)
        duration_list = []
        for idx in range(train_epoch_step):
            step += 1
            start_time = time.time()

            clip, label = sess.run([train_clip_holder, train_label_holder])
            _, loss_now, per_iter_acc, summary = sess.run([optimizer, loss, train_acc_update, train_summary], \
                                                 feed_dict={clip_holder: clip, label_holder: label, 
                                                            is_training: True,
                                                            net2d_dropout_holder: args.net2d_dropout,
                                                            net3d_dropout_holder: args.net3d_dropout})

            summary_writer.add_summary(summary, step)
            duration = time.time() - start_time
            duration_list.append(duration)
            cur_lr = sess.run(learning_rate)
            # Responsible for printing relevant results
            if idx % args.print_freq == 0:
                logger.info('Epoch: [%d] Iter: % -4d, loss: %-.4f, acc: %.2f, lr: %f \
                                ( %.2f sec/batch)' %
                            (epoch, idx, loss_now, per_iter_acc*100, cur_lr, float(duration)))

        train_acc = sess.run(train_acc_op)
        
        epoch_duration = sum(duration_list[1:])/(len(duration_list)-1)
        logger.info('Epoch: [{}] Avg. Training Acc.: {:.3f} Avg. Iter Time: {:.4f}'.format(epoch, train_acc*100, epoch_duration))
         
        # Test Phase
        logger.info('Epoch: [{}] - Start Test Phase...'.format(epoch))

        sess.run(test_init_op)

        for _ in range(test_epoch_step):
            clip, label = sess.run([test_clip_holder, test_label_holder])
            test_iter_acc = sess.run(test_acc_update, \
                                        feed_dict={clip_holder: clip, label_holder: label, is_training: False,
                                                net2d_dropout_holder: 1.,
                                                net3d_dropout_holder: 1.})

        test_acc, summary = sess.run([test_acc_op, test_acc_summary], \
                                         feed_dict={clip_holder: clip, label_holder: label, is_training: False,
                                                    net2d_dropout_holder: 1.,
                                                    net3d_dropout_holder: 1.})                                        
        summary_writer.add_summary(summary, epoch)

        logger.info('Epoch: [{}], Avg Test Acc.: {:.2f}'.format(epoch, test_acc*100))

        summary_writer.flush()

        global_saver.save(sess, os.path.join(ckp_dir, 'latest.ckpt'))
        logger.info('Saving latest checkpoint to {}...'.format(os.path.join(ckp_dir, 'latest.ckpt')))

        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            global_saver.save(sess, os.path.join(ckp_dir, 'best.ckpt'))
            logger.info('Saving best checkpoint to {}...'.format(os.path.join(ckp_dir, 'best.ckpt')))

        logger.info('Best Acc. is {:.2f} in Epoch [{}]'.format(best_acc*100, best_epoch))

    summary_writer.close()
    sess.close()


if __name__ == '__main__':
    main()
