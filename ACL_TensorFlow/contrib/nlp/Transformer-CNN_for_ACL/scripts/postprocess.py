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
import os
import sys

import numpy as np
from sklearn.metrics import r2_score
import csv
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--infer_result", type=str, default="./results")
parser.add_argument("--ground_truth", type=str, default="./test.csv")
args = parser.parse_args()
PATH = args.infer_result
TRUE_LABEL = args.ground_truth
def test():
    y_predict, y_true = [], []
    first_row = True
    for line in csv.reader(open(TRUE_LABEL, "r")):
        if first_row:
            first_row = False
            continue
        y_true.append(float(line[1]))
    props = pickle.load(open("model.pkl", "rb"))
    file_list = sorted(os.listdir(PATH))
    for item in file_list:
        test_out = np.fromfile(PATH+os.sep+item, dtype=np.float32)
        res = np.zeros(len(props))
        for prop in props:
            if len(props) == 1:
                res[prop] = np.mean(test_out);
            else:
                res[prop] = np.mean(y[prop]);
            if props[prop][2] == "regression":
                res[prop] = (res[prop] - 0.9) / 0.8 * (props[prop][4] - props[prop][3]) + props[prop][4]
        y_predict.append(res[0])

    score = r2_score(y_true, y_predict)
    return score

if __name__ == "__main__":
    score = test()
    print("COEFFICIENT OF DETERMINATION:{}".format(score))