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

"""
Function: Feature(bin) -> Result(ACER)
Date: 2021.7.2
Author: AJ
"""
import os, sys, copy, argparse
import numpy as np
from sklearn.metrics import roc_curve

def video_2_label(video_name):
    label = int(video_name.split('_')[-1])
    label = 0 if label == 1 else 1
    data_name = video_name.split('_')[0]
    return label, data_name

def metric_zcx(scores_dev, labels_dev, scores_test, labels_test):
    def get_eer_threhold(fpr, tpr, threshold):
        differ_tpr_fpr_1 = tpr + fpr - 1.0
        right_index = np.argmin(np.abs(differ_tpr_fpr_1))
        best_th = threshold[right_index]
        eer = fpr[right_index]
        return eer, best_th
    scores_dev = np.array(scores_dev)
    labels_dev = np.array(labels_dev)
    scores_test = np.array(scores_test)
    labels_test = np.array(labels_test)

    dev_fpr, dev_tpr, dev_thre = roc_curve(labels_dev, scores_dev, pos_label=1)
    dev_eer, dev_best_thre = get_eer_threhold(dev_fpr, dev_tpr, dev_thre)

    real_scores = scores_test[labels_test == 1]
    attack_scores = scores_test[labels_test == 0]
    APCER = np.mean(np.array(attack_scores >= dev_best_thre, np.float32))
    BPCER = np.mean(np.array(real_scores < dev_best_thre, np.float32))

    test_fpr, test_tpr, test_threshold = roc_curve(labels_test, scores_test, pos_label=1)
    test_EER, test_best_thre = get_eer_threhold(test_fpr, test_tpr, test_threshold)
    results = [test_EER, APCER, BPCER, (APCER + BPCER) / 2.0]
    return dev_best_thre, results

### <real:label==1, fake:label==0> (metric)
def performance(phases, scores_dir, score_ind=1):
    '''
    :param phases: ['train', 'dev', 'test']
    :param score_ind: 1:prob_score 2:exp_score
    '''
    predicts_test = []
    predicts_dev = []
    scores_test = []
    labels_test = []
    scores_dev = []
    labels_dev = []
    for phase in phases[1:]:
        locals()['predicts_' + phase] = []
        score_fid = open(os.path.join(scores_dir, phase.capitalize() + '_scores.txt'), 'r')
        lines = score_fid.readlines()
        score_fid.close()
        for line in lines:
            label = 1 - int(line.split(',')[0].split('_')[-1])
            locals()['predicts_' + phase].append([line.split(',')[0], float(line.split(',')[score_ind]), label])
            locals()['scores_' + phase].append(float(line.split(',')[score_ind]))  ### '1_275_1_1_1_G(0)_0001.jpg@GT_0'
            locals()['labels_' + phase].append(int(label))

    Dev_best_thre, results = metric_zcx(scores_dev, labels_dev, scores_test, labels_test)
    print('@_ZCX: Dev_best_thre={} EER={} APCER={} BPCER={} ACER={}'.
          format(Dev_best_thre, results[0], results[1], results[2], results[3]))

    return Dev_best_thre, results[1], results[2], results[3]

def evaluate_metric(scores_dir, phases=['train', 'dev', 'test']):
    for score_ind in range(1, 3):
        if score_ind == 1: score_type = 'logit_score'
        else: score_type = 'depth_score'
        ### <score_ind==1:prob_score score_ind==2:exp_score>
        Dev_best_thre, APCER, NPCER, ACER = performance(phases, scores_dir, score_ind)
        result_txt = os.path.join(scores_dir, 'result.txt')
        ### Write results ###
        if not os.path.exists(result_txt):
            lines = []
        else:
            fid = open(result_txt, 'r')
            lines = fid.readlines()
            fid.close()
        str_line = '%s thre %.4f\tAPCER %.4f\tNPCER %.4f\tACER %.4f\t'%(
            score_type, Dev_best_thre, APCER*100, NPCER*100, ACER*100)
        fid = open(result_txt, 'w')
        line_new = str_line + '\tmodal_iter %d\n'%(42)
        lines.append(line_new)
        fid.writelines(lines)
        fid.close()

def write_ScoreImages(feature_folder, fid):
    ### parameters ###
    logit_acc_mean = 0.0
    depth_acc_mean = 0.0
    accuracy_mean = 0.0
    accuracy = 1
    map_score = 1
    bat_it = 0
    def realProb(logits):
        x = np.array(logits)
        if np.isinf(np.sum(np.exp(x))):
            return 0
        y = np.exp(x[0]) / np.sum(np.exp(x))
        return y

    feats = os.listdir(feature_folder)
    for feat in feats:
        bat_it += 1
        depth_map = np.fromfile(os.path.join(feature_folder, feat), dtype='float32').reshape([1, 32, 32, 1])
        filename = feat.split('_output')[0]
        label, _ = video_2_label(filename)

        assert depth_map.shape[-1] == 1
        depth_map_image = np.squeeze(depth_map, axis=-1)
        depth_map_image = (depth_map_image * 255.0)
        ### compute depth score ###
        depth_map_image_norm = depth_map_image / 255.0  ### 0~1
        depth_binary = np.array(depth_map > depth_map.min(), np.float32)
        depth_mean_score = np.sum(depth_map_image_norm) / np.sum(depth_binary[..., 0])
        # assert np.min(depth_batch[frame_ind, :, :, :]) == 0
        # print(np.sum(depth_map_image_norm), np.sum(depth_binary[..., 0]), depth_mean_score)
        # assert depth_mean_score <= 20.0
        depth_score = [depth_mean_score, 1.0 - depth_mean_score]
        out = np.argmax(np.array(depth_score))
        depth_acc = int(out == label)
        depth_acc_mean += float(depth_acc)
        map_score = realProb(depth_score)

        ### write score ###
        fid.write(filename + '@GT_' + str(label) + ',' + str(0) + ',' + str(map_score) + '\n')
        print('*{}/{}: depth_acc={} logit_acc/accuracy={}/{}'.format(
            'phase', str(bat_it), str(depth_acc_mean / (0 + 1)), str(logit_acc_mean / bat_it), str(accuracy_mean / bat_it)))


if __name__ == "__main__":
    dev_path = sys.argv[1]
    test_path = sys.argv[2]
    fid = open('./offline_inference/Dev_scores.txt', 'w')
    write_ScoreImages(dev_path, fid)
    fid = open('./offline_inference/Test_scores.txt', 'w')
    write_ScoreImages(test_path, fid)
    scores_dir = './offline_inference/'
    evaluate_metric(scores_dir)
