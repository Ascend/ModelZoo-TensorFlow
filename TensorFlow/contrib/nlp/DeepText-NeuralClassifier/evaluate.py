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
import json
import math
import os
import sys
import time
import numpy as np
import tensorflow as tf
import pickle
import util
from config import Config
from predict import Predictor
import sklearn.metrics
import pandas as pd

# import logging
# logging.basicConfig(level=logging.INFO, filename='logging.log', format="%(levelname)s:%(asctime)s:%(message)s")
class Evaluator(object):
    MACRO_AVERAGE = "macro avg"
    MICRO_AVERAGE = 'micro avg'
    """Not thread safe, will keep the latest eval result
    """

    def __init__(self, eval_dir):
        self.confusion_matrix_list = None
        self.precision_list = None
        self.recall_list = None
        self.fscore_list = None
        self.right_list = None
        self.predict_list = None
        self.standard_list = None

        self.eval_dir = eval_dir
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

    @staticmethod
    def _calculate_prf(right_count, predict_count, standard_count):
        """Calculate precision, recall, fscore
        Args:
            standard_count: Standard count
            predict_count: Predict count
            right_count: Right count
        Returns:
            precision, recall, f_score
        """
        precision, recall, f_score = 0, 0, 0
        if predict_count > 0:
            precision = right_count / predict_count
        if standard_count > 0:
            recall = right_count / standard_count
        if precision + recall > 0:
            f_score = precision * recall * 2 / (precision + recall)

        return precision, recall, f_score

    def calculate_level_performance(
            self, id_to_label_map, right_count_category, predict_count_category,
            standard_count_category, other_text='其他',
            exclude_method="contain"):
        """Calculate the level performance.
        Args:
            id_to_label_map: Label id to label name.
            other_text: Text to judge the other label.
            right_count_category: Right count.
            predict_count_category: Predict count.
            standard_count_category: Standard count.
            exclude_method: The method to judge the other label. Can be
                            contain(label_name contains other_text) or
                            start(label_name start with other_text).
        Returns:
            precision_dict, recall_dict, fscore_dict.
        """
        other_label = dict()
        for label_id, label_name in id_to_label_map.items():
            if exclude_method == "contain":
                if other_text in label_name:
                    other_label[label_name] = 1
            elif exclude_method == "start":
                if label_name.startswith(other_text):
                    other_label[label_name] = 1
            else:
                raise TypeError("Cannot find exclude_method: " + exclude_method)

        precision_dict = dict()
        recall_dict = dict()
        fscore_dict = dict()
        precision_dict[self.MACRO_AVERAGE] = 0
        recall_dict[self.MACRO_AVERAGE] = 0
        fscore_dict[self.MACRO_AVERAGE] = 0
        right_total = 0
        predict_total = 0
        standard_total = 0

        for label_id, label_name in id_to_label_map.items():
            if label_name in other_label:
                continue
            (precision_dict[label_name], recall_dict[label_name],
             fscore_dict[label_name]) = self._calculate_prf(
                right_count_category[label_name],
                predict_count_category[label_name],
                standard_count_category[label_name])
            right_total += right_count_category[label_name]
            predict_total += predict_count_category[label_name]
            standard_total += standard_count_category[label_name]
            precision_dict[self.MACRO_AVERAGE] += precision_dict[label_name]
            recall_dict[self.MACRO_AVERAGE] += recall_dict[label_name]
            fscore_dict[self.MACRO_AVERAGE] += fscore_dict[label_name]
        num_label_eval = len(id_to_label_map) - len(other_label)

        precision_dict[self.MACRO_AVERAGE] = precision_dict[self.MACRO_AVERAGE] / num_label_eval
        recall_dict[self.MACRO_AVERAGE] = recall_dict[self.MACRO_AVERAGE] / num_label_eval
        fscore_dict[self.MACRO_AVERAGE] = 0 \
            if (recall_dict[self.MACRO_AVERAGE] +
                precision_dict[self.MACRO_AVERAGE]) == 0 else \
            2 * precision_dict[self.MACRO_AVERAGE] * \
            recall_dict[self.MACRO_AVERAGE] / \
            (recall_dict[self.MACRO_AVERAGE]
             + precision_dict[self.MACRO_AVERAGE])

        right_count_category[self.MICRO_AVERAGE] = right_total
        predict_count_category[self.MICRO_AVERAGE] = predict_total
        standard_count_category[self.MICRO_AVERAGE] = standard_total

        (precision_dict[self.MICRO_AVERAGE], recall_dict[self.MICRO_AVERAGE],
         fscore_dict[self.MICRO_AVERAGE]) = \
            self._calculate_prf(right_total, predict_total, standard_total)
        return precision_dict, recall_dict, fscore_dict

    def get_threshold(self, dict_map, probs, labels):
        threshold = []
        for idx_label in range(len(probs[0, :])):
            rightlabel = 0
            for i in range(len(labels)):
                for item in labels[i].split(','):
                    if item == dict_map[idx_label]:
                        rightlabel += 1

            idx_list = probs[:, idx_label].argsort(axis=0)
            idx_list = idx_list[::-1]
            fscore_list = []
            TP = 0
            for i in range(len(labels)):

                true_label = labels[idx_list[i]].split(',')
                for m in range(len(true_label)):
                    if true_label[m] == dict_map[idx_label]:
                        TP += 1
                        break
                if rightlabel == 0:
                    fscore_list.append(0)
                    continue
                recall = TP / rightlabel
                prcs = TP / (i + 1)
                if (recall + prcs) == 0:
                    fscore_list.append(0)
                else:
                    fscore_list.append(2 * recall * prcs / (recall + prcs))

            threshold.append(probs[idx_list[fscore_list.index(max(fscore_list))], idx_label])
        print(str(threshold))
        return threshold

    def evaluate(self, predicts, labels, label_map=True, threshold=0, multi_label=False,
                 is_prob=True, is_flat=False, other_text='其他'):
        """Eval the predict result.
        Args:
            predicts: Predict probability or
                      predict text label(is_prob is false).
            labels: Standard label.
            label_map: Label dict. If is_prob is false and label_map is None,
                       label_map will be generated using labels.
            threshold: Threshold to filter probs.
            is_prob: The predict is prob list or label id.
            is_flat: If true, only calculate flat result.
                     Else, calculate hierarchical result.
            other_text: Label name contains other_text will not be calculate.
        Returns:
            confusion_matrix_list contain all result,
            filtered_confusion_matrix_list contains result that max predict prob
                is greater than threshold and will be used to calculate prf,
            precision_list, recall_list, fscore_list,
            right_count_list, predict_count_list, standard_count_list
        """
        if type(predicts) == type([0]):
            probs = np.array(predicts)
        else:
            prob_np = []

            for predict in predicts:
                if is_prob:
                    prob_np.append(np.array(predict, dtype=np.float32)[0])

            probs = np.array(prob_np)

#        multi_label =config.eval.multi_label
        labels = labels[0:len(probs)]

        result_dir = 'result'  # config.eval.acc_report_file
        test_classname = labels

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # 标签词典
        dict_class_int2string = {value: key for key, value in label_map.items()}

        # 获得最优的阈值
        threshold = self.get_threshold(dict_class_int2string, probs, labels)
        file = open('pickle_example.pickle', 'wb')
        pickle.dump(dict_class_int2string, file)
        file.close()
        np.save('labels', np.array(labels))
        np.save('probs', probs)
        if multi_label:
            unknown_labels = set(
                [y for x in set(test_classname) for y in x.split(',') if y not in dict_class_int2string.values()])
        else:
            unknown_labels = set([y for y in set(test_classname) if y not in dict_class_int2string.values()])
        known_label_count = len(dict_class_int2string)
        for i, v in enumerate(unknown_labels):
            dict_class_int2string[known_label_count + i] = v
        dict_class_string2int = dict(zip(dict_class_int2string.values(), dict_class_int2string.keys()))
        data = {}
        if not multi_label:
            # print(len(probs))
            # print(probs.shape)
            y_pred = probs.argmax(axis=1)
            y_true = [dict_class_string2int[x] for x in test_classname]
            miss = (y_pred != y_true).astype(int).sum()
            data["predict_label"] = [dict_class_int2string[x] for x in y_pred]
            data["predict_confidence"] = probs.max(1)
        else:
            # 多标签矩阵
            I = np.eye(len(dict_class_int2string))
            y_true_labels = [[dict_class_string2int[y] for y in x.split(',')] for x in test_classname]
            y_true = np.array([sum(I[x].astype(int)).tolist() for x in y_true_labels])

            # 多标签预测结果.  y_pred与y_true的shape不一致，y_true有y_pred未覆盖的label
            y_pred = (probs >= threshold).astype(int)
            y_pred = np.append(y_pred, np.zeros((y_true.shape[0], y_true.shape[1] - y_pred.shape[1])), axis=1)
            miss = (y_true == y_pred).all(axis=1).astype(int).sum()
  #          top_k = 3
  #          top_k_index = probs.argsort(axis=1)[:, ::-1][:, :top_k]
  #          r, c = top_k_index.shape
  #          xv, yv = np.meshgrid(range(c), range(r))
  #          prob_top_k = probs[yv, top_k_index]
  #          data["predict_label"] = [
  #              ','.join([dict_class_int2string[i] for i, p in zip(indexs, probs) if p >= threshold]) for indexs, probs
  #              in zip(top_k_index, prob_top_k)]
  #          data["predict_confidence"] = [
  #              ','.join(['{:.4f}'.format(p) for i, p in zip(indexs, probs) if p >= threshold]) for indexs, probs in
  #              zip(top_k_index, prob_top_k)]
#
        # 输出测试结果
        # count = len(probs)
        # accuracy = float(count - miss) / count
        # acc_summary = "{}{}, dataset \n\tevaluate result :  count|miss|accuracy  =  {}|{}|{:.2%}\n".format( \
        #     time.asctime(), \
        #     ', multilabel', \
        #     count, miss, accuracy)
        #
        # print(acc_summary)
        # with open(acc_report_file, 'a') as f:
        #     f.write(acc_summary)
        #
        # # 输出原表格带预测列
        # res = pd.DataFrame(data)
        # columns = ["predict_label", "predict_confidence"]
        # columns = columns + [x for x in list(res.columns) if not x in columns]
        # res[columns].to_csv(test_result_file, sep='\t', index=False, float_format='%.4f')

        target_names = [dict_class_int2string[x] for x in range(len(dict_class_int2string))]

        # label_list_all = list(set(dict_class_int2string.values()))

        # 评估报告
        report = sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred,
                                                       labels=range(len(target_names)), target_names=target_names,
                                                       output_dict=True)
        
        # logging.info(f'评估报告: \n{report}')
        # logging.info(f'参数: \n y_true大小:{len(y_true)}\n y_pred大小:{len(y_pred)}\n labels:{len(labels)}\n target_names:{len(target_names)}')
        
        report_df = pd.DataFrame(report).T
        # support改为int型
        report_df['support'] = report_df['support'].astype(int)
        # 按support排序，不排序average部分
        report_df = report_df.iloc[:len(target_names)].sort_values(by=['support'], ascending=False).append(
            report_df.iloc[len(target_names):])
        #     report_df.to_csv(classification_report_file, sep='\t', index=True, float_format='%.4f')
        #
        # # 混淆矩阵
        # if not multi_label and not confusion_matrix_file is None:
        #     confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred,
        #                                                         labels=[x for x in range(len(dict_class_int2string))])
        #     # print("confusion_matrix={}".format(confusion_matrix))
        #     cm = pd.DataFrame(confusion_matrix, index=target_names, columns=target_names)
        #     cm.to_csv(confusion_matrix_file, sep='\t', index=True)

        precision_list = [dict(zip(report_df.index, report_df.precision))]
        recall_list = [dict(zip(report_df.index, report_df.recall))]
        fscore_list = [dict(zip(report_df.index, report_df['f1-score']))]
        standard_category_count_list = [dict(zip(report_df.index, report_df.support))]

        (self.precision_list, self.recall_list,
         self.fscore_list, self.standard_list) = (
             precision_list, recall_list, fscore_list,
            standard_category_count_list)
        return precision_list, recall_list, fscore_list, standard_category_count_list

    @staticmethod
    def save_confusion_matrix(file_name, confusion_matrix):
        """Save confusion matrix
        Args:
            file_name: File to save to.
            confusion_matrix: Confusion Matrix.
        Returns:
        """
        with codecs.open(file_name, "w", encoding=util.CHARSET) as cm_file:
            cm_file.write("\t")
            for category_fist in sorted(confusion_matrix.keys()):
                cm_file.write(category_fist + "\t")
            cm_file.write("\n")
            for category_fist in sorted(confusion_matrix.keys()):
                cm_file.write(category_fist + "\t")
                for category_second in sorted(confusion_matrix.keys()):
                    cm_file.write(
                        str(confusion_matrix[category_fist][
                                category_second]) + "\t")
                cm_file.write("\n")

    def save_prf(self, file_name, precision_category, recall_category,
                 fscore_category,
                 standard_category):
        """Save precision, recall, fscore
        Args:
            file_name: File to save to.
            precision_category: Precision dict.
            recall_category: Recall dict.
            fscore_category: Fscore dict.
            right_category: Right dict.
            predict_category: Predict dict.
            standard_category: Standard dict.
        Returns:
        """

        def _format(category):
            """Format evaluation string.
            Args:
                category: Category evaluation to format.
            Returns:
            """
            if category == self.MACRO_AVERAGE:
                return "%s, precision: %f, recall: %f, fscore: %f, " % (
                    category, precision_category[category],
                    recall_category[category], fscore_category[category])
            else:
                return "%s, precision: %f, recall: %f, fscore: %f, " \
                       "standard_count: %d" % (
                           category, precision_category[category],
                           recall_category[category], fscore_category[category],
                           standard_category[category])

        with codecs.open(file_name, "w", encoding=util.CHARSET) as prf_file:
            prf_file.write(_format(self.MACRO_AVERAGE) + "\n")
            prf_file.write(_format(self.MICRO_AVERAGE) + "\n")
            prf_file.write("\n")
            for category in precision_category:
                if category != self.MICRO_AVERAGE and \
                        category != self.MACRO_AVERAGE:
                    prf_file.write(_format(category) + "\n")

    def save(self):
        """Save the latest evaluation.
        """
        if self.confusion_matrix_list is not None:
            for i, confusion_matrix in enumerate(self.confusion_matrix_list):
                if i == 0:
                    eval_name = "all"
                else:
                    eval_name = "level_%s" % i
                self.save_confusion_matrix(
                    self.eval_dir + "/" + eval_name + "_confusion_matrix",
                    confusion_matrix)
                self.save_prf(
                    self.eval_dir + "/" + eval_name + "_prf",
                    self.precision_list[i], self.recall_list[i],
                    self.fscore_list[i], self.standard_list[i])
        self.save_prf(
            self.eval_dir + "/" + "all_prf",
            self.precision_list[0], self.recall_list[0],
            self.fscore_list[0], self.standard_list[0])


def main(_):
    config = Config(config_file='conf/fasttext_token_char.config')
    predictor = Predictor(config)
    predict_probs = []
    standard_labels = []
    logger = util.Logger(config)
    if not os.path.exists(config.eval.eval_dir):
        os.makedirs(config.eval.eval_dir)
    with codecs.open(config.eval.eval_dir + "/predict.txt", "w",
                     encoding=util.CHARSET) as predict_file:
        texts = []
        for line in codecs.open(config.eval.text_file, "r",
                                encoding='gb18030'):
            line = line.strip("\n")
            texts.append(line)
        batch_size = config.eval.batch_size
        epochs = math.ceil(len(texts) / batch_size)

        for i in range(epochs):
            predicts = predictor.predict(
                texts[i * batch_size: (i + 1) * batch_size])
            for k in range(len(predicts)):
                predict_result = "Nil\t0"
                predict = predicts[k]
                line = texts[i * batch_size + k]
                if predict is not None:
                    predict_np = np.array(predict[0], dtype=np.float32)
                    predict_label = predictor.data_processor.id_to_label_map[
                        np.argmax(predict_np)]
                    predict_result = "%s\t%f" % (
                        predict_label, np.max(predict_np))
                    predict_probs.append(predict[0])
                    standard_labels.append(line.split("\t")[0])
                predict_file.write(predict_result + "\t" + line + "\n")
    evaluator = Evaluator(config.eval.eval_dir)
    multi_label = config.eval.multi_label
    (precision_list, recall_list, fscore_list, standard_list) = evaluator.evaluate(
        predict_probs, standard_labels, predictor.data_processor.label_map,
        config.eval.threshold, multi_label)
    logger.info(
        "Test performance, precision: %f, recall: %f, f1: %f,  standard: %d" % (
            precision_list[0][evaluator.MICRO_AVERAGE],
            recall_list[0][evaluator.MICRO_AVERAGE],
            fscore_list[0][evaluator.MICRO_AVERAGE],
            standard_list[0][evaluator.MICRO_AVERAGE],

        ))
    evaluator.save()


if __name__ == '__main__':
    tf.app.run()

