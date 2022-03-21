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

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--query_path', type=str, default='')
parser.add_argument('--test_path', type=str, default='')
parser.add_argument('--test_num', type=int, default=19732)
parser.add_argument('--query_num', type=int, default=3368)
args = parser.parse_args()


def extract_feature(dir_path):
    features = []
    infos = []
    num = 0
    for name in os.listdir(dir_path):
        arr = name.split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        feature_path = os.path.join(dir_path, name)
        feature = np.genfromtxt(feature_path, delimiter=' ')
        features.append(np.squeeze(feature))
        infos.append((person, camera))

    return features, infos


if __name__ == '__main__':

    query_path = args.query_path
    test_path = args.test_path

    queries = os.listdir(query_path)
    tests = os.listdir(test_path)

    test_features, test_info = extract_feature(test_path)
    query_features, query_info = extract_feature(query_path)

    match = []
    junk = []

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index, (tp, tc) in enumerate(test_info):
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    print(np.array(query_features).shape)
    print(np.array(test_features).shape)
    query_norm = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    test_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
    query_norm = query_norm.astype(np.float32)
    test_norm = test_norm.astype(np.float32)
    print(query_norm.dtype)
    print(test_norm.dtype)
    result = np.matmul(query_norm, np.transpose(test_norm))
    result_argsort = np.argsort(result, axis=1)

    mAP = 0.0
    CMC = np.zeros([len(query_info), len(test_info)])
    for idx in range(len(query_info)):
        recall = 0.0
        precision = 1.0
        hit = 0.0
        cnt = 0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        for i in list(reversed(range(0, args.test_num))):
            k = result_argsort[idx][i]
            if k in IGNORE:
                continue
            else:
                cnt += 1
                if k in YES:
                    CMC[idx, cnt-1:] = 1
                    hit += 1
            
                tmp_recall = hit/len(YES)
                tmp_precision = hit/cnt
                ap = ap + (tmp_recall - recall)*((precision + tmp_precision)/2)
                recall = tmp_recall
                precision = tmp_precision
            if hit == len(YES):
                break
        mAP += ap

    rank_1 = np.mean(CMC[:,0])
    rank_5 = np.mean(CMC[:,4])
    rank_10 = np.mean(CMC[:,9])
    rank_20 = np.mean(CMC[:,19])
    mAP /= args.query_num 

    print ('1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f'%(rank_1, rank_5, rank_10, rank_20, mAP))
    print ()
