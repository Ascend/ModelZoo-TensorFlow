# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
from tqdm import tqdm
import sys
import os

def msamePath(output_path, inference_path, model_path):
    """
    使用文件夹推理
    """
    if os.path.isdir(inference_path):
        os.system("rm -rf " + inference_path)
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + "data"
    front = "./msame --model " + model_path + " --input " + output_path
    end   = " --output " + inference_path + " --outfmt BIN"
    os.system(front + end)
    print(inference_path)
    print("[INFO]    推理结果生成结束")


def get_matrix(predict, label, num_classes):
    
    mask = (label >= 0) & (label <= num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def get_result(confusion_matrix):
    
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    print(MIoU)
    MIoU = np.nanmean(MIoU)
    
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU


def getlabel(labelpath, n):
    return np.load(labelpath + str(n) + ".npy")


def segmentation_cls_inference_files(inference_path, labelpath):
    
    output_num = 0
    batch = 1
    oh, ow, c = 1024, 2048, 19
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    timefolder = os.listdir(inference_path)
    if len(timefolder) == 1:
        inference_path = inference_path + timefolder[0] + "/"
    else:
        print("there may be some error in reference path: ", inference_path)
    print("inference_path   ", inference_path)
    files = len(os.listdir(inference_path))
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    c_matrix = np.zeros((19, 19))
    for f in tqdm(files):
        if f.endswith(".bin"):
            y_in = getlabel(labelpath, output_num)
            
            tmp = np.fromfile(f, dtype='float32')
            
            tmp = tmp.reshape(batch, oh, ow, c)
            pred = np.argmax(tmp, axis=-1)
            c_matrix += get_matrix(pred, y_in, num_classes=19)
            output_num += 1
    p_acc, m_acc, miou, fmiou = get_result(c_matrix)
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num * batch),
          "MIoU: %.6f" % (miou), p_acc, m_acc, miou, fmiou)


if __name__ == "__main__":
    output_path = sys.argv[1]
    inference_path = sys.argv[2]
    model_path = sys.argv[3]  
    imageLabelpath = sys.argv[4]
    msamePath(output_path, inference_path, model_path)
    segmentation_cls_inference_files(inference_path, imageLabelpath)
