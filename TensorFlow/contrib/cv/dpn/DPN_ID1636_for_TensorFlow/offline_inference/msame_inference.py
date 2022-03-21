import sys
import os
import numpy as np
import cv2
import json
import time

batch = 9
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



def msamePath(output_path, inference_path, model_path):
    """
    使用文件夹推理
    """
    if os.path.isdir(inference_path):
        os.system("rm -rf "+inference_path)
    output_path = output_path if output_path[-1] == "/" else output_path + "/"
    output_path = output_path + "data"
    print("./msame --model "+model_path + " --input "+output_path +
          " --output "+inference_path + " --outfmt BIN")
    os.system("./msame --model "+model_path + " --input " +
              output_path + " --output "+inference_path + " --outfmt BIN")
    print(inference_path)
    print("[INFO]    推理结果生成结束")


def segmentation_cls_inference_files(inference_path, sup_labels):
    # 获得这个文件夹下面所有的bin 然后排序每个读进去 就行
    output_num = 0
    oh, ow, c = 64, 64, 21
    label = sup_labels
    inference_path = inference_path if inference_path[-1] == "/" else inference_path + "/"
    timefolder = os.listdir(inference_path)
    print(timefolder)
    if len(timefolder) == 1:
        inference_path = inference_path + timefolder[0] + "/"
    else:
        print("there may be some error in reference path: ",inference_path)
    print(inference_path)
    files = len(os.listdir(inference_path))
    files = [inference_path + str(i)+"_output_0.bin" for i in range(files)]
    c_matrix = np.zeros((21, 21))
    for f in files:
        if f.endswith(".bin"):
            y_in = label[output_num]
            tmp = np.fromfile(f, dtype='float32')
            tmp = tmp.reshape(batch, oh, ow, c)
            pred = tmp
            c_matrix += evaluating_cm(pred, y_in, num_classes=21)
            output_num += 1
    res = evaluating_miou(c_matrix)
    print(">>>>> ", "共 %d 测试样本 \t" % (output_num*batch),
          "MIoU: %.6f" % (res))


if __name__ == "__main__":
    output_path = sys.argv[1]
    inference_path = sys.argv[2]
    model_path = sys.argv[3]  # model的地址
    imageLabel = getLabel(output_path)
    msamePath(output_path, inference_path, model_path)
    segmentation_cls_inference_files(inference_path, imageLabel)
