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
import os
import numpy as np
from absl import flags, app

FLAGS = flags.FLAGS


def main(argv):
    del argv
    LABEL_FLODER = FLAGS.pre
    PREDICT_FLODER = FLAGS.label

    files_label = os.listdir(LABEL_FLODER)
    files_predict = os.listdir(PREDICT_FLODER)
    num, check_num = 0, 0
    result_map = {}
    label_map = {}

    for file in files_label:
        if file.endswith(".txt"):
            tmp = np.loadtxt(LABEL_FLODER + '/' + file, dtype='float32')
            result_map[file.split("_")[0]] = tmp if tmp.ndim is not 0 else [tmp]

    for file in files_predict:
        if file.endswith(".bin"):
            num += 1
            tmp = np.fromfile(PREDICT_FLODER + '/' + file, dtype='float32')
            label_map[file[:-4]] = tmp if tmp.ndim is not 0 else [tmp]
        #     print('label_map:',label_map)
    for i in range(len(label_map)):
        label = label_map['{}'.format(i)][0]
        res = result_map['{}'.format(i)][0]
        if res == label:
            check_num += 1

    accuarcy = check_num / num
    print("Totol num: %d, accuarcy: %.4f" % (num, accuarcy))


if __name__ == '__main__':
    flags.DEFINE_string('label', "./mnist2000bin_label", 'The floder of TRUE result')
    flags.DEFINE_string('pre', "./result/", 'The floder of inference result')
    app.run(main)

