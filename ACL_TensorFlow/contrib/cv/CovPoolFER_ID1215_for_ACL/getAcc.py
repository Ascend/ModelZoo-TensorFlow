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
import argparse
import os
import sys

import numpy as np

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_om_out', type=str,
                        help='dir where to load om.',default= "/root/Neesky/output/20220117_165253/")
    parser.add_argument('--input_reference_out', type=str,
                        help='dir where to load reference file.',default= "/root/Neesky/SFEW_100/Reference/")
    return parser.parse_args(argv)
def main(args):
    lists = os.listdir(args.input_om_out)
    tot = 0
    alltot = 0
    for i in range(len(lists)):
        file1 = open(args.input_om_out + lists[i], "r")
        file2 = open(args.input_reference_out + str(lists[i].split("_")[0]) + ".txt", "r")
        lines1 = file1.readlines()
        l1 = []
        for line in lines1:
            l1 = [int(l) for l in line.split()]
        lines2 = file2.readlines()
        l2 = [int(l) for l in lines2[0].split()]

        print("预测:", l1)
        print("实际:", l2)
        for i in range(len(l1)):
            if l1[i] == l2[i]:
                tot = tot + 1
            alltot = alltot + 1
        file1.close()
        file2.close()
    print("总准确率", tot / alltot)
if __name__ == "__main__" :
    main(parse_arguments(sys.argv[1:]))