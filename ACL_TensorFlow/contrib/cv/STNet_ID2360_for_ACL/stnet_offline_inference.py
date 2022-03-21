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
    parser.add_argument('--model_path', default='stnet_om.om',
                        help="""om path""")
    parser.add_argument('--output_path', default='./out/',
                        help="""bin file path""")
    parser.add_argument('--data_path', default="./dataset/mnist_sequence1_sample_5distortions5x5.npz",
                        help="""the label data path""")
    parser.add_argument('--inference_path', default="./inference_out/",
                        help="""the inference result path""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def getLabel(label_path):
    mnist_cluttered = np.load(label_path)
    label = mnist_cluttered['y_test']
    return label


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


def segmentation_cls_inference_files(inference_path, sup_labels):
    output_num= 0
    acc=0
    label = sup_labels
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    timefolder = os.listdir(inference_path)
    print(timefolder)
    if len(timefolder) == 1:
        inference_path = inference_path + timefolder[0] + "/"
    else:
        print("there may be some error in inference path: ",inference_path)
    print(inference_path)
    files = len(os.listdir(inference_path))
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    for f in files:
        if f.endswith(".bin"):
            y_in = label[output_num]
            tmp = np.fromfile(f, dtype='float32')
            pred=np.argmax(tmp)
            if pred==y_in:
                acc+=1
            output_num += 1
    print('======acc : {}----total : {}'.format(acc, output_num))
    print('Final Offline Inference Accuracy : ', round(acc / output_num, 4))

if __name__ == "__main__":
    args = parse_args()
    imageLabel = getLabel(args.data_path)
    msamePath(args.output_path, args.inference_path, args.model_path)
    segmentation_cls_inference_files(args.inference_path, imageLabel)

#'./data/mnist_sequence1_sample_5distortions5x5.npz'
#python3 ./msame_inference.py /root/dataset/bin/dpn /root/dataset/bin/dpn/inference /root/310/pb/1202pb/dpn2/dpn.om
