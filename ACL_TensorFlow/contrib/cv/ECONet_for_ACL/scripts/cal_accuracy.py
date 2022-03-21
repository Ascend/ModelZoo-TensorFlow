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
import argparse
import pickle
import os

parser = argparse.ArgumentParser(description="Offline Inference Accuracy Computation")
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)

def main():
    args = parser.parse_args()

    label_path = args.label_path
    output_path = args.output_path

    with open(label_path, 'rb') as f:
        label_dict = pickle.load(f)

    output_num = 0.
    top1_num = 0.
    top5_num = 0.
    print("Start accuracy computation")
    for par, dir_list, file_list in os.walk(output_path):
        for file_name in file_list:
            if file_name.endswith('_output_0.bin'):
                output_num += 1
                output_logit = np.fromfile(os.path.join(par, file_name), dtype='float32')
                inf_label = int(np.argmax(output_logit))
                inf_top5 = np.argsort(-output_logit)[0:5]
                img_key = file_name.replace('_output_0.bin', '')
                if inf_label == label_dict[img_key]:
                    top1_num += 1
                if label_dict[img_key] in inf_top5:
                    top5_num += 1
                print("{: <30} gt label:{: >3}, predict results:{}".format(img_key, label_dict[img_key], str(inf_top5)))

    top1_acc = top1_num / output_num
    print("Totol image num: %d, Top1 accuarcy: %.4f, Top5 accuarcy: %.4f"%(output_num, top1_acc, top5_num/output_num))

if __name__ == '__main__':
    main()