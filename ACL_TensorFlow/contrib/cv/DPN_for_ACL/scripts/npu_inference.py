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
import cv2
import json
import time

batch = 8
clear = False
allNum = 1449
_HEIGHT = 512
_WIDTH = 512

def get_result(confusion_matrix):
    # pixel accuracy
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # mean iou
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def getLabel(output_path):
    label = np.load(output_path + "/label" + "/label.npy")
    print(label.shape)
    return label


def clear_files(output_path):
    os.system("rm -rf %sdata" % output_path)
    os.makedirs(output_path+"data")


def evaluating_cm(log, label, num_classes=21):
    predict = np.argmax(log, axis=-1)
    mask = (label >= 0) & (label < num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    cm = count.reshape(num_classes, num_classes)
    return cm


def evaluating_miou(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU


def BenchmarkPath(output_path, inference_path, model_path):
    """
    使用文件夹推理
    """
    if os.path.isdir(inference_path):
        os.system("rm -rf "+inference_path)
    output_path = "{},{}".format(os.path.join(output_path,"data"),os.path.join(output_path,"distance"))
    print("../Benchmark/out/benchmark --model "+model_path + " --input "+output_path +
          " --output "+inference_path + " --outfmt BIN")
    os.system("../Benchmark/out/benchmark --model "+model_path + " --input " +
              output_path + " --output "+inference_path + " --outfmt BIN")
    print(inference_path)
    print("[INFO] Npu Inference Done!")


def segmentation_cls_inference_files(inference_path, sup_labels):
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    oh, ow, c = 64, 64, 21
    label = sup_labels
    files = os.listdir(inference_path)
    files.sort()
    c_matrix = np.zeros((21, 21))
    for f in files:
        if f.endswith(".bin"):
            print("Start to process {}".format(f))
            y_in = label[output_num]
            tmp = np.fromfile(os.path.join(inference_path,f), dtype='float32')
            tmp = tmp.reshape(batch, oh, ow, c)
            pred = tmp
            c_matrix += evaluating_cm(pred, y_in, num_classes=21)
            output_num += 1
    res = evaluating_miou(c_matrix)
    print(">>>>> ", "Total samples %d\t" % (output_num*batch),
          "MIoU: %.6f" % (res))


if __name__ == "__main__":
    output_path = sys.argv[1]      #预处理生成的bin文件的路径
    inference_path = sys.argv[2]   #npu推理输出的bin文件路径
    model_path = sys.argv[3]       #om model的路径
    imageLabel = getLabel(output_path)
    BenchmarkPath(output_path, inference_path, model_path)
    segmentation_cls_inference_files(inference_path, imageLabel)

