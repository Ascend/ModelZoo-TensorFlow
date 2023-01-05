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

import unittest

import numpy as np

from evaluate import Evaluator


def softmax(np_array):
    """Softmax a np.array
    Return:
         softmax value
    """
    max_np = np.max(np_array, axis=1, keepdims=True)
    softmax_np = np.exp(np_array - max_np)
    z = np.sum(softmax_np, axis=1, keepdims=True)
    softmax_np = softmax_np / z
    return softmax_np


class TestEval(unittest.TestCase):
    def test_calculate_level_performance(self):
        evaluator = Evaluator(".")
        id_to_label_map = dict({0: "其他", 1: "教育--中小学",
                                2: "教育--学历教育--高等教育",
                                3: "教育--学历教育--其他", 4: "教育--其他",
                                5: "体育--健身", 6: "体育--其他"})
        right_count_category = dict()
        predict_count_category = dict()
        standard_count_category = dict()
        for label_id, label_name in id_to_label_map.items():
            right_count_category[label_name] = label_id + 1
            predict_count_category[label_name] = label_id + 3
            standard_count_category[label_name] = label_id + 2
        predict_count_category["教育--学历教育--高等教育"] = 0
        standard_count_category["教育--学历教育--其他"] = 0
        precision_dict, recall_dict, fscore_dict = \
            evaluator.calculate_level_performance(
                id_to_label_map, right_count_category, predict_count_category,
                standard_count_category, exclude_method="start")

        self.assertAlmostEqual(precision_dict[evaluator.MACRO_AVERAGE],
                               0.56812169)
        self.assertAlmostEqual(recall_dict[evaluator.MACRO_AVERAGE],
                               0.663690476)
        self.assertAlmostEqual(fscore_dict[evaluator.MACRO_AVERAGE],
                               0.612198785)
        self.assertAlmostEqual(precision_dict[evaluator.MICRO_AVERAGE],
                               0.794117647)
        self.assertAlmostEqual(recall_dict[evaluator.MICRO_AVERAGE],
                               0.964285714)
        self.assertAlmostEqual(fscore_dict[evaluator.MICRO_AVERAGE],
                               0.870967741)

        precision_dict, recall_dict, fscore_dict = \
            evaluator.calculate_level_performance(
                id_to_label_map, right_count_category, predict_count_category,
                standard_count_category, exclude_method="contain")

        self.assertAlmostEqual(precision_dict[evaluator.MACRO_AVERAGE],
                               0.41666666)
        self.assertAlmostEqual(recall_dict[evaluator.MACRO_AVERAGE],
                               0.757936507)
        self.assertAlmostEqual(fscore_dict[evaluator.MACRO_AVERAGE],
                               0.537725225)
        self.assertAlmostEqual(precision_dict[evaluator.MICRO_AVERAGE],
                               0.916666666)
        self.assertAlmostEqual(recall_dict[evaluator.MICRO_AVERAGE],
                               0.785714285)
        self.assertAlmostEqual(fscore_dict[evaluator.MICRO_AVERAGE],
                               0.846153846)

    def test_evaluate(self):
        evaluator = Evaluator(".")
        # def evaluate(probs, labels, label_map, threshold, is_flat=False,
        #             other_text='其他'):
        labels = ["体育", "教育--中小学", "其他", "教育--中小学", "体育--健身",
                  "教育--中小学", "教育--其他", "体育--健身", "体育", "其他",
                  "其他", "教育--中小学", "教育--其他", "体育--健身", "体育"]
        self.assertEqual(len(labels), 15)
        label_map = dict({"其他": 0, "教育--中小学": 1,
                          "教育--其他": 2, "体育--健身": 3, "体育": 4})
        probs = np.asarray(
            [[0.16665406, 0.17128651, 0.22790066, 0.15240852, 0.28175025],
             [0.16373185, 0.23643395, 0.212988, 0.19349667, 0.19334954],
             [0.27977045, 0.14908171, 0.19945701, 0.18531397, 0.18637687],
             [0.15806427, 0.31092174, 0.17966731, 0.17499493, 0.17635175],
             [0.20422085, 0.12716979, 0.18246204, 0.2451357, 0.24101162],

             [0.26758713, 0.12934757, 0.22408755, 0.26109785, 0.11787991],
             [0.25730898, 0.16906949, 0.15162034, 0.23466986, 0.18733132],
             [0.12713102, 0.29404712, 0.16372075, 0.2841079, 0.1309932],
             [0.2241528, 0.24569175, 0.21768124, 0.14138361, 0.1710906],
             [0.14378468, 0.27834602, 0.17227523, 0.13321389, 0.27238019],

             [0.2291905, 0.16647999, 0.20320596, 0.17203644, 0.22908711],
             [0.29787804, 0.2404769, 0.13746795, 0.16435064, 0.15982647],
             [0.14112081, 0.23134903, 0.24136793, 0.14952184, 0.23664038],
             [0.21353831, 0.19339162, 0.20902374, 0.1875536, 0.19649273],
             [0.22287845, 0.20906594, 0.17496322, 0.16571431, 0.22737809]])
        self.assertEqual(probs.shape[0], 15)
        self.assertEqual(probs.shape[1], 5)

        (confusion_matrix_list, precision_list, recall_list, fscore_list,
         right_count_list, predict_count_list, standard_count_list) = \
            evaluator.evaluate(probs, labels, label_map, threshold=0.22,
                               is_prob=True, is_flat=False)

        self.assertEqual(len(precision_list), 3)
        self.assertEqual(len(recall_list), 3)
        self.assertEqual(len(fscore_list), 3)
        self.assertAlmostEqual(precision_list[0][evaluator.MACRO_AVERAGE], 0.85)
        self.assertAlmostEqual(recall_list[0][evaluator.MACRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[0][evaluator.MACRO_AVERAGE],
                               0.629629629)
        self.assertAlmostEqual(precision_list[0][evaluator.MICRO_AVERAGE],
                               0.666666666)
        self.assertAlmostEqual(recall_list[0][evaluator.MICRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[0][evaluator.MICRO_AVERAGE],
                               0.571428571)

        self.assertAlmostEqual(precision_list[1][evaluator.MACRO_AVERAGE], 0.75)
        self.assertAlmostEqual(recall_list[1][evaluator.MACRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[1][evaluator.MACRO_AVERAGE], 0.6)
        self.assertAlmostEqual(precision_list[1][evaluator.MICRO_AVERAGE],
                               0.666666666)
        self.assertAlmostEqual(recall_list[1][evaluator.MICRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[1][evaluator.MICRO_AVERAGE],
                               0.571428571)

        self.assertAlmostEqual(precision_list[2][evaluator.MACRO_AVERAGE], 0.7)
        self.assertAlmostEqual(recall_list[2][evaluator.MACRO_AVERAGE],
                               0.416666666)
        self.assertAlmostEqual(fscore_list[2][evaluator.MACRO_AVERAGE],
                               0.522388059)
        self.assertAlmostEqual(precision_list[2][evaluator.MICRO_AVERAGE], 0.5)
        self.assertAlmostEqual(recall_list[2][evaluator.MICRO_AVERAGE],
                               0.428571428)
        self.assertAlmostEqual(fscore_list[2][evaluator.MICRO_AVERAGE],
                               0.461538461)

        predicts = ["体育", "教育--中小学", "其他", "教育--中小学", "体育--健身",
                    "其他", "其他", "教育--中小学", "教育--中小学", "教育--中小学",
                    "其他", "教育--中小学", "教育--其他", "其他", "其他"]
        (confusion_matrix_list, precision_list, recall_list, fscore_list,
         right_count_list, predict_count_list, standard_count_list) = \
            evaluator.evaluate(predicts, labels, label_map, threshold=0.22,
                               is_prob=False, is_flat=False)

        self.assertEqual(len(precision_list), 3)
        self.assertEqual(len(recall_list), 3)
        self.assertEqual(len(fscore_list), 3)
        self.assertAlmostEqual(precision_list[0][evaluator.MACRO_AVERAGE],
                               0.875)
        self.assertAlmostEqual(recall_list[0][evaluator.MACRO_AVERAGE],
                               0.479166666)
        self.assertAlmostEqual(fscore_list[0][evaluator.MACRO_AVERAGE],
                               0.619230769)
        self.assertAlmostEqual(precision_list[0][evaluator.MICRO_AVERAGE],
                               0.666666666)
        self.assertAlmostEqual(recall_list[0][evaluator.MICRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[0][evaluator.MICRO_AVERAGE],
                               0.571428571)

        self.assertAlmostEqual(precision_list[1][evaluator.MACRO_AVERAGE],
                               0.78571428)
        self.assertAlmostEqual(recall_list[1][evaluator.MACRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[1][evaluator.MACRO_AVERAGE],
                               0.611111111)
        self.assertAlmostEqual(precision_list[1][evaluator.MICRO_AVERAGE],
                               0.666666666)
        self.assertAlmostEqual(recall_list[1][evaluator.MICRO_AVERAGE], 0.5)
        self.assertAlmostEqual(fscore_list[1][evaluator.MICRO_AVERAGE],
                               0.571428571)

        self.assertAlmostEqual(precision_list[2][evaluator.MACRO_AVERAGE], 0.75)
        self.assertAlmostEqual(recall_list[2][evaluator.MACRO_AVERAGE],
                               0.541666666)
        self.assertAlmostEqual(fscore_list[2][evaluator.MACRO_AVERAGE],
                               0.629032258)
        self.assertAlmostEqual(precision_list[2][evaluator.MICRO_AVERAGE],
                               0.571428571)
        self.assertAlmostEqual(recall_list[2][evaluator.MICRO_AVERAGE],
                               0.571428571)
        self.assertAlmostEqual(fscore_list[2][evaluator.MICRO_AVERAGE],
                               0.571428571)


if __name__ == '__main__':
    unittest.main()
