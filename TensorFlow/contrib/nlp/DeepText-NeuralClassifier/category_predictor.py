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

import codecs
import sys

import numpy as np
import tensorflow as tf

import util
from config import Config
from predict import Predictor


class CategoryPredictor(Predictor):
    def __init__(self, config_file, name_to_real_id_file):
        config = Config(config_file=config_file)
        self.name_to_real_id_map = dict()
        for line in codecs.open(name_to_real_id_file, "r",
                                encoding=util.CHARSET):
            content = line.strip("\n").split("\t")
            self.name_to_real_id_map[content[1]] = content[0]
        super(CategoryPredictor, self).__init__(config)

    def predict_text(self, text, top_n=1, threshold=0.0):
        probs = super(CategoryPredictor, self).predict([text])[0]
        ids = []
        names = []
        ori_weights = []
        weights = []
        indexes = np.argsort(probs[0])

        i = 0
        length = len(probs[0]) - 1
        total_weight = 0
        while i < top_n and length - i >= 0:
            prob_index = length - i
            if probs[0][prob_index] < threshold:
                break
            name = self.data_processor.id_to_label_map[indexes[prob_index]]
            names.append(name)
            ids.append(self.name_to_real_id_map[name])
            total_weight += probs[0][indexes[prob_index]] * \
                probs[0][indexes[prob_index]]
            ori_weights.append(probs[0][indexes[prob_index]])
            i += 1
        total_weight = np.sqrt(total_weight)
        for ori_weight in ori_weights:
            weights.append(ori_weight / total_weight)
        return ids, names, ori_weights, weights


def main(_):
    config = Config(config_file=sys.argv[1])
    predictor = CategoryPredictor(config, sys.argv[2])
    print(predictor.predict_text("0\t2-年级 2-英语\t\t", 2, 0.2))


if __name__ == '__main__':
    tf.app.run()
