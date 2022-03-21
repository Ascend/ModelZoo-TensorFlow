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

import argparse
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np
import os
import json

def write_accuracy(result_file, result_content):
    encode_json = json.dumps(result_content, sort_keys=False, indent=4)
    with open(result_file,'w') as json_file:
        json_file.write(encode_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_result', type=str, default='./output/bench_out/SEGDEC-NET/')
    parser.add_argument('--result_file', type=str, default='./output/ACC.json')
    parser.add_argument('--data_type', type=str, default=np.float32)
    parser.add_argument('--label', type=str, default='./output/labels/')

    args = parser.parse_args()
    samples_outcome = []
    for infer_bin_file in os.listdir(args.infer_result):
        if infer_bin_file.endswith('.bin'):
            label_file = infer_bin_file.split('_out')[0].split('davinci_')[1] + '.bin'
            decision = np.fromfile(os.path.join(args.infer_result, infer_bin_file), dtype=args.data_type)
            decision = 1.0/(1+np.exp(-np.squeeze(decision)))
            label = np.fromfile(os.path.join(args.label, label_file), dtype=args.data_type)
            samples_outcome.append((decision, np.max(label)))

    if len(samples_outcome) > 0:
        samples_outcome = np.matrix(np.array(samples_outcome))

        idx = np.argsort(samples_outcome[:,0], axis=0)
        idx = idx[::-1]
        samples_outcome = np.squeeze(samples_outcome[idx, :])

        P = np.sum(samples_outcome[:, 1])
        TP = np.cumsum(samples_outcome[:, 1] == 1).astype(np.float32).T
        FP = np.cumsum(samples_outcome[:, 1] == 0).astype(np.float32).T

        recall = TP / P
        precision = TP /(TP + FP)

        f_measure = 2 * np.multiply(recall, precision) / (recall + precision)
        idx =np.argmax(f_measure)

        best_f_measure = f_measure[idx]
        best_thr = samples_outcome[idx,0]
        best_FP = FP[idx]
        best_FN = P-TP[idx]

        precision_, recall_, thresholds = precision_recall_curve(samples_outcome[:,-1], samples_outcome[:,0])
        FPR, TPR, _ = roc_curve(samples_outcome[:,1], samples_outcome[:,0])
        AUC = auc(FPR,TPR)
        AP = auc(recall_,precision_)

        accuracy_results = {'map': round(AP, 4)}
        print('accuracy_results',accuracy_results)
        print('AUC=%f, and AP=%f,with best thr=%f,at f-measure=%.3f and FP=%d, FN=%d' % (AUC, AP, best_thr, best_f_measure, best_FP, best_FN))

    else:
        print('ERROR')