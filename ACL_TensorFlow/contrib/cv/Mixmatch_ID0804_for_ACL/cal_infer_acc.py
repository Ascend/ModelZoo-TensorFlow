# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
    PREDICT_LABEL_FLODER = FLAGS.PREDICT_LABEL_FLODER
    OUTPUT_LABEL_FLODER = FLAGS.OUTPUT_LABEL_FLODER
    files_output = os.listdir(OUTPUT_LABEL_FLODER)
    files_predict = os.listdir(PREDICT_LABEL_FLODER)
    num, check_num = 0, 0
    label_map = {}
    for file in files_output:
        if file.endswith(".txt"):
            tmp = np.loadtxt(OUTPUT_LABEL_FLODER+'/'+file, dtype='float32')
            label_map[file.split(".txt")[0]] = tmp if tmp.ndim is not 0 else [tmp]
    for file in files_predict:
        if file.endswith(".txt"):
            num += 1
            tmp = np.loadtxt(PREDICT_LABEL_FLODER+'/'+file, dtype='float32')
            tmp = tmp if tmp.ndim is not 1 else [tmp]
            for i in range(len(tmp)):
                inf_label_i = int(np.argmax(tmp[i]))
                print(label_map[file.split("_")[0]])
                if(inf_label_i == label_map[file.split("_")[0]][i]):
                    check_num += 1

    top1_accuarcy = check_num/num
    print("Totol num: %d, accuarcy: %.4f"%(num, top1_accuarcy))
if __name__ == '__main__':
    flags.DEFINE_string('PREDICT_LABEL_FLODER', '/home/TestUser03/code/mixmatch/mix_model/out/20221016_14_42_58_783190',\
     'The floder of inference result')
    flags.DEFINE_string('OUTPUT_LABEL_FLODER', '/home/TestUser03/code/mixmatch/mix_model/output_label_01/',\
     'The floder of TRUE result')
    app.run(main)
    # python cal_inference_pref.py --PREDICT_LABEL_FLODER=/home/hang/文档/om/predict_label_gpu_free --OUTPUT_LABEL_FLODER=/home/hang/文档/om/output_label