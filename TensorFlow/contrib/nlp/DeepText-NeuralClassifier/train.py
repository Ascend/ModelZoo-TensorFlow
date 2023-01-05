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
# ==============================================================================

from npu_bridge.npu_init import *

import codecs
import os
import sys
import time

import tensorflow as tf
# from get_data import get_format_txt
import util
from config import Config
from data_processor import DataProcessor
from evaluate import Evaluator
from model.attentive_convolution import AttentiveConvNetEstimator
from model.fasttext import FastTextEstimator
from model.model_helper import ModelHelper
from model.region_embedding import RegionEmbeddingEstimator
from model.text_cnn import TextCNNEstimator
from model.text_dpcnn import TextDPCNNEstimator
from model.text_drnn import TextDRNNEstimator
from model.text_rnn import TextRNNEstimator
from model.text_vdcnn import TextVDCNNEstimator

# Just prevent auto-reformat-code of pycharm from deleting import
TextRNNEstimator, FastTextEstimator, TextCNNEstimator, TextVDCNNEstimator, \
RegionEmbeddingEstimator, AttentiveConvNetEstimator, TextDPCNNEstimator, \
TextDRNNEstimator


def get_standard_label(file):
    """Get the standard label of validation and test file
    Text format: Label
    Label could be flattened or hierarchical which is separated by "--".
    Args:
        file: File to read
    Return:
        label_list
    """
    label_list = []
    for line in codecs.open(file, "r", encoding='utf8'):
        label_list.append(line.strip("\n").split("\t")[0])
    return label_list


def eval_predict(estimator, data_processor, train_tfrecord_file, evaluator,
                 labels, logger, stage, config, log=True):
    probs = estimator.predict(
        input_fn=lambda: data_processor.dataset_input_fn(
            tf.estimator.ModeKeys.PREDICT, train_tfrecord_file,
            config.train.predict_batch_size, 1), hooks=None)
    (precision_list, recall_list, fscore_list, standard_list) = \
        evaluator.evaluate(
            probs, labels, data_processor.label_map,
            threshold=config.eval.threshold, multi_label=config.eval.multi_label,
            is_flat=config.eval.is_flat
            )
    log_str = "%s performance, precision: %f, recall: %f, " \
              "f1: %f,standard: %d" % (
                  stage, precision_list[0][evaluator.MICRO_AVERAGE],
                         recall_list[0][evaluator.MICRO_AVERAGE],
                         fscore_list[0][evaluator.MICRO_AVERAGE],
                  standard_list[0][evaluator.MACRO_AVERAGE])
    if log:
        logger.info(log_str)

    return fscore_list[0][evaluator.MICRO_AVERAGE], log_str


def Train(config):
    logger = util.Logger(config)
    data_processor = DataProcessor(config, logger=logger)
    evaluator = Evaluator(config.eval.eval_dir)
    if os.path.exists(
            config.data.tfrecord_dir + "/" + os.path.basename(
                config.data.train_text_file) + ".tfrecord"):
        logger.info(
            "Data has been processed before. Directly read from %s" %
            config.data.tfrecord_dir)
        data_processor.load_all_dict()
    else:
        logger.info("Processed text files")
        data_processor.process_from_text_file(use_exists_dict=False)
    params = dict()
    params["label_size"] = len(data_processor.label_map)
    params["feature_names"] = config.feature_common.feature_names.split(",")
    if config.train.track_timeline:
        timeline_dir = config.model_common.checkpoint_dir + "/timeline/"
        if not os.path.exists(timeline_dir):
            os.makedirs(timeline_dir)
        hook = [tf.estimator.ProfilerHook(save_steps=1000, output_dir=timeline_dir)]
    else:
        hook = None
    if config.model_common.tensorboard_dir:
        tensorboard_dir = config.model_common.tensorboard_dir
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        merged_summary = tf.summary.merge_all()
        hook = [tf.train.SummarySaverHook(save_steps=100, output_dir=tensorboard_dir, summary_op=merged_summary)]
    else:
        hook = None

    assert config.model_common.model_type in ModelHelper.VALID_MODEL_TYPE
    model_name = config.model_common.model_type + "Estimator"
    train_tfrecord_file = data_processor.train_file
    validate_tfrecord_file = data_processor.validate_file
    test_tfrecord_file = data_processor.test_file
    train_labels = get_standard_label(data_processor.train_feature_file)
    validate_labels = get_standard_label(data_processor.validate_feature_file)
    test_labels = get_standard_label(data_processor.test_feature_file)
    tag_fscore = 0
    best_validate_fscore = 0
    test_log_str = ""
    start_time = time.time()
    for i in range(1, config.train.num_epochs + 1):
        if i <= config.train.num_epochs_static_embedding:
            params["static_embedding"] = True
        else:
            params["static_embedding"] = False
        params["epoch"] = i
        estimator = globals()[model_name](data_processor, params)
        logger.info("Start training epoch %d" % i)
        start_train_time = time.time()
        estimator.train(input_fn=lambda: data_processor.dataset_input_fn(
            tf.estimator.ModeKeys.TRAIN, train_tfrecord_file,
            config.train.batch_size,
            1), hooks=hook)
        train_time = time.time() - start_train_time
        logger.info("Epoch %d 训练时间: %d second" % (i, train_time))
        logger.info("Start evaluate at epoch %d" % i)
        if config.train.eval_train_data:
            eval_predict(estimator, data_processor, train_tfrecord_file,
                         evaluator, train_labels, logger, "Epoch %d train" % i,
                         config)
        fscore, _ = eval_predict(
            estimator, data_processor, validate_tfrecord_file, evaluator,
            validate_labels, logger, "Epoch %d validate" % i, config)
        logger.info("Epoch {} validate score: {} ****** last best validate score: {}".format(i, fscore, best_validate_fscore))
        if fscore > best_validate_fscore:
            best_validate_fscore = fscore
            _, test_log_str = eval_predict(
                estimator, data_processor, test_tfrecord_file, evaluator,
                test_labels, logger, "Best test", config, False)
            f = open('Best.txt', 'w+')
            f.write(str(fscore)+'\t'+str(i)+'\n')
            f.close()
            evaluator.save()
            logger.info("Write best validate score {} to file".format(best_validate_fscore))
        estimator.export_savedmodel(config.model_common.export_model_dir,
                                    data_processor.serving_input_receiver_fn)
        time_used = time.time() - start_time
        start_time = time.time()
        logger.info("Epoch %d cost time: %d second" % (i, time_used))
        # if fscore < best_validate_fscore:
        #     tag_fscore += 1
        # else:
        #     tag_fscore = 0

        # if tag_fscore > 5:
        #     logger.warn("Early stop after {} epoch".format(tag_fscore))
        #     break
    logger.info(test_log_str)


def main(_):
    # get_format_txt()
    Train(config=sys.argv[1])
    # Train(config=Config(config_file=sys.argv[1]))
    # Train(config=Config(config_file='conf/textcnn_char.config'))

if __name__ == '__main__':
    tf.app.run()
