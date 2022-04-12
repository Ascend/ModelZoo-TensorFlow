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
import os.path
import argparse
import numpy as np
max_length = 64

def main(args):
    acc = []
    for i in range(10):
        file_path = args.bin_path+"/"+str(i)+"/batch_y.bin"
        predict_filepath = args.predict_path+"/"+str(i)+"/res_output_0.txt"
        batch_y = np.fromfile(file_path,dtype=np.float32)
        y_pred = read_predict(predict_filepath)
        batch_y = batch_y.reshape((-1, 2))
        wrong_idx = []
        for mm in range(len(y_pred)):
            if (y_pred[mm, 0] > y_pred[mm, 1] and batch_y[mm, 0] == 0) or (
                    y_pred[mm, 0] <= y_pred[mm, 1] and batch_y[mm, 0] == 1):
                wrong_idx.append(mm)

        train_accuracy = (len(y_pred) - len(wrong_idx)) / len(y_pred)
        acc.append(train_accuracy)
        print(train_accuracy)
    acc = np.array(acc)
    print('accuracy : {}'.format(np.mean(acc)))


def read_predict(predict_filename):
    predicts = []
    with open(predict_filename, 'r') as f:
        for line in f.readlines():
            predict = line.strip().split()
            predicts.append(predict)
    return np.array(predicts)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # 输入ckpt模型路径
    parser.add_argument('--predict_path', type=str,
                        help='the path of inference result', default="./predict")
    # 输出pb模型的路径
    parser.add_argument('--bin_path', type=str,
                        help='the path of batch_y.bin.', default="./bin")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))