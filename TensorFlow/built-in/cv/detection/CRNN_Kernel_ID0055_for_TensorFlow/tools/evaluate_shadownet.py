#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : evaluate_shadownet.py
# @IDE: PyCharm Community Edition
"""
Evaluate the crnn model on the synth90k test dataset
"""
import argparse
import os.path as ops
import os
import math
import time
import sys
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import glog as log
import tqdm
from sklearn.metrics import confusion_matrix
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

cur_path = os.path.abspath(os.path.dirname(__file__))
working_dir = os.path.join(cur_path, '../')
sys.path.append(working_dir)


from crnn_model import crnn_net
from config import global_config
from data_provider import shadownet_data_feed_pipline
from data_provider import tf_io_pipline_fast_tools
from local_utils import evaluation_tools


CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str,default='data/',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-c', '--char_dict_path', type=str,default='data/char_dict_bak/char_dict_en.json',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,default='data/char_dict_bak/ord_map_en.json',
                        help='Directory where ord map dictionaries for the dataset were stored')
    parser.add_argument('-w', '--weights_path', type=str, required=True,
                        help='Path to pre-trained weights')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=False,
                        help='Whether to display images')
    parser.add_argument('-p', '--process_all', type=args_str2bool, nargs='?', const=False,
                        help='Whether to process all test dataset')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')




def evaluate_shadownet(dataset_dir, weights_path, char_dict_path,
                       ord_map_dict_path, is_visualize=False,
                       is_process_all_data=False):
    """

    :param dataset_dir:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_visualize:
    :param is_process_all_data:
    :return:
    """
    # prepare dataset
    test_dataset = shadownet_data_feed_pipline.CrnnDataFeeder(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        flags='test'
    )
    test_images, test_labels, test_images_paths,test_label_length = test_dataset.inputs(
        batch_size=128
    )
    x, y = np.meshgrid(np.arange(CFG.ARCH.MAX_LENGTH), np.arange(128))
    indexes = np.concatenate([y.flatten()[:, None], x.flatten()[:, None]], axis=1)
    indexes = tf.constant(indexes, dtype=tf.int64)
    test_labels = tf.SparseTensor(indexes, tf.reshape(test_labels, [-1]), np.array([128, CFG.ARCH.MAX_LENGTH], dtype=np.int64))

    # set up test sample count
    if is_process_all_data:
        log.info('Start computing test dataset sample counts')
        t_start = time.time()
        test_sample_count = test_dataset.sample_counts()
        log.info('Test dataset sample counts: {:d}'.format(test_sample_count))
        log.info('Computing test dataset sample counts finished, cost time: {:.5f}'.format(time.time() - t_start))
        num_iterations = int(math.ceil(test_sample_count / 128))
    else:
        num_iterations = 1

    # declare crnn net
    shadownet = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )
    # set up decoder
    decoder = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    # compute inference result
    test_inference_ret = shadownet.inference(
        inputdata=test_images,
        name='shadow_net',
        reuse=False
    )
    test_decoded, test_log_prob = tf.nn.ctc_greedy_decoder(
        test_inference_ret,
        CFG.ARCH.SEQ_LENGTH * np.ones(128),
        merge_repeated=True
    )

    # recover image from [-1.0, 1.0] ---> [0.0, 255.0]
    test_images = tf.multiply(tf.add(test_images, 1.0), 127.5, name='recoverd_test_images')

    # Set saver configuration
    saver = tf.train.Saver()
    
    # NPU CONFIG
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    custom_op.parameter_map["mix_compile_mode"].b = False  # 混合计算
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    # Set sess configuration
    #sess_config = tf.ConfigProto(allow_soft_placement=True)
    #sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    #sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH

    #sess = tf.Session(config=sess_config)
    sess = tf.Session(config=config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        log.info('Start predicting...')

        per_char_accuracy = 0.0
        full_sequence_accuracy = 0.0

        total_labels_char_list = []
        total_predictions_char_list = []
        epoch_tqdm = tqdm.tqdm(range(num_iterations))

        while True:
            try:
                for epoch in epoch_tqdm:
                    t_start = time.time()
                    test_predictions_value, test_images_value, test_labels_value, test_images_paths_value = sess.run(
                        [test_decoded, test_images, test_labels, test_images_paths])
                    test_images_paths_value = np.reshape(
                        test_images_paths_value,
                        newshape=test_images_paths_value.shape[0]
                    )
                    test_images_paths_value = [tmp.decode('utf-8') for tmp in test_images_paths_value]
                    test_images_names_value = [ops.split(tmp)[1] for tmp in test_images_paths_value]
                    test_labels_value = decoder.sparse_tensor_to_str(test_labels_value)
                    test_predictions_value = decoder.sparse_tensor_to_str(test_predictions_value[0])

                    per_char_accuracy += evaluation_tools.compute_accuracy(
                        test_labels_value, test_predictions_value, display=False, mode='per_char'
                    )
                    full_sequence_accuracy += evaluation_tools.compute_accuracy(
                        test_labels_value, test_predictions_value, display=False, mode='full_sequence'
                    )

                    for index, test_image in enumerate(test_images_value):
                        log.info('Predict {:s} image with gt label: {:s} **** predicted label: {:s}'.format(
                            test_images_names_value[index],
                            test_labels_value[index],
                            test_predictions_value[index]))

                        if is_visualize:
                            plt.imshow(np.array(test_image, np.uint8)[:, :, (2, 1, 0)])
                            plt.show()

                        test_labels_char_list_value = [s for s in test_labels_value[index]]
                        test_predictions_char_list_value = [s for s in test_predictions_value[index]]

                        if not test_labels_char_list_value or not test_predictions_char_list_value:
                            continue

                        if len(test_labels_char_list_value) != len(test_predictions_char_list_value):
                            min_length = min(len(test_labels_char_list_value),
                                             len(test_predictions_char_list_value))
                            test_labels_char_list_value = test_labels_char_list_value[:min_length - 1]
                            test_predictions_char_list_value = test_predictions_char_list_value[:min_length - 1]

                        assert len(test_labels_char_list_value) == len(test_predictions_char_list_value), \
                            log.error('{}, {}'.format(test_labels_char_list_value, test_predictions_char_list_value))

                        total_labels_char_list.extend(test_labels_char_list_value)
                        total_predictions_char_list.extend(test_predictions_char_list_value)
                        if is_visualize:
                            plt.imshow(np.array(test_image, np.uint8)[:, :, (2, 1, 0)])
                    epoch_tqdm.set_description('Epoch {:d} cost time: {:.5f}s'.format(epoch, time.time() - t_start))
                if num_iterations == 1:
                    raise tf.errors.OutOfRangeError
            except tf.errors.OutOfRangeError:
                log.error('End of tfrecords sequence')
                break
            except Exception as err:
                log.error(err)
                break

        epoch_tqdm.close()
        avg_per_char_accuracy = per_char_accuracy / num_iterations
        avg_full_sequence_accuracy = full_sequence_accuracy / num_iterations
        log.info('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
        log.info('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))
        print('Mean test per char accuracy is {:5f}'.format(avg_per_char_accuracy))
        print('Mean test full sequence accuracy is {:5f}'.format(avg_full_sequence_accuracy))
        # compute confusion matrix
        cnf_matrix = confusion_matrix(total_labels_char_list, total_predictions_char_list)
        np.set_printoptions(precision=2)
        #evaluation_tools.plot_confusion_matrix(cm=cnf_matrix, normalize=True)

        #plt.show()


if __name__ == '__main__':
    """
    test code
    """
    args = init_args()

    print('weight path :{}'.format(args.weights_path))
    evaluate_shadownet(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights_path,
        char_dict_path=args.char_dict_path,
        ord_map_dict_path=args.ord_map_dict_path,
        is_visualize=args.visualize,
        is_process_all_data=args.process_all
    )
