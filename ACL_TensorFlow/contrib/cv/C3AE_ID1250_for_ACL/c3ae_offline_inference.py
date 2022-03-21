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

import sys
import os
import numpy as np
import argparse
def parse_args():
    '''

    :return:
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='./om_C3AE.om',
                        help="""om path""")
    parser.add_argument('--output_path', default='./output1/,./output2/,./output3',
                        help="""bin file path""")
    parser.add_argument('--data_path', default="./dataset/wiki_crop/",
                        help="""the label data path""")
    parser.add_argument('--inference_path', default="./output0/",
                        help="""the inference result path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def msamePath(output_path, inference_path, model_path):

    if os.path.isdir(inference_path):
        os.system("rm -rf "+inference_path)
    os.system("mkdir "+inference_path)
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path
    print("./msame --model "+model_path + " --input "+output_path +
          " --output "+inference_path + " --outfmt BIN")
    os.system("./msame --model "+model_path + " --input " +
              output_path + " --output "+inference_path + " --outfmt BIN")
    print(inference_path)
    print("[INFO]    Inference finished")


def accuracy(inference_path):
    prediction = []
    target = []
    for root,dirs,files in os.walk(inference_path):
        for file in files:
            if file.endswith("_1.bin"):
                file_url = os.path.join(root,file)
            
                pre = np.fromfile(file_url, dtype='float32').tolist()[0]
            
                ptn = file.split(".")[0].split("_")
            
                if len(ptn) == 3:
                    prediction.append(pre)

                    birth = ptn[1].split("-")
                    taken = ptn[2]

                    if int(birth[1]) < 7:
                        target.append(int(taken) - int(birth[0]))
                    else:
                        target.append(int(taken) - int(birth[0]) - 1)

    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    mae = sum(absError) / len(absError)
    print("AGE MAE: ", '%.4f' % mae)

if __name__ == "__main__":
    args = parse_args()
    msamePath(args.output_path, args.inference_path, args.model_path)
    accuracy(args.inference_path)
