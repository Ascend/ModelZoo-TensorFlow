# -*- coding: utf-8 -*-
#
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
#
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
# import npu_device
# npu_device.open().as_default()
from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.train import train
from engines.predict import Predictor
from engines.utils.word2vec import Word2VecUtils
from config import mode, classifier_config, word2vec_config
import json
import os


if __name__ == '__main__':
    logger = get_logger('./logs')
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
    # 训练分类器
    if mode == 'train_classifier':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        logger.info('model: {}'.format(classifier_config['classifier']))
        train(data_manage, logger)
    # 测试分类
    elif mode == 'interactive_predict':
        logger.info(json.dumps(classifier_config, indent=2))
        data_manage = DataManager(logger)
        logger.info('mode: predict_one')
        logger.info('model: {}'.format(classifier_config['classifier']))
        predictor = Predictor(data_manage, logger)
        predictor.predict_one('warm start')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)
            print(results)
    # 训练词向量
    elif mode == 'train_word2vec':
        logger.info(json.dumps(word2vec_config, indent=2))
        logger.info('mode: train_word2vec')
        w2v = Word2VecUtils(logger)
        w2v.train_word2vec()
    # 训练词向量
    elif mode == 'test':
        logger.info('mode: test')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.predict_test()
    # 保存pb格式的模型用于tf-severing接口
    elif mode == 'save_model':
        logger.info('mode: save_pb_model')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.save_model()

