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
from npu_bridge.npu_init import *
import tensorflow as tf
import glob
import json
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import moxing as mox
import argparse
from help_modelarts import modelarts_result2obs

MINOVERLAP = 0.5

print("++++++++++++++begin get_map file++++++++++++++++++")
parser = argparse.ArgumentParser()
# bos first
parser.add_argument("--dataset", type=str, default="./output")  ## config.modelarts_data_dir  namely  /cache/DataSet
parser.add_argument("--result", type=str, default="./dataset")  ## config.modelarts_result_dir namely  /cache/result

# for last address
parser.add_argument("--obs_dir", type=str)  ## last output fileaddress s3://yolov2/yolov2forfen/output/V0027/

config1 = parser.parse_args()
print("config1.dataset", config1.dataset)  ## config.modelarts_data_dir  namely  /cache/DataSet
print("config1.result", config1.result)  ## config.modelarts_result_dir namely  /cache/result

# for last address
print("config1.obs_dir", config1.obs_dir)  ## last output fileaddress s3://yolov2/yolov2forfen/output/V0027/



## modelarts(config.result  /cache/result) create file  info_result_dir
accuracy_dir = os.path.join(config1.result, 'accuracy')
mapresults_dir = os.path.join(config1.result, 'mapresults')

if not mox.file.exists(accuracy_dir):
    mox.file.make_dirs(accuracy_dir)

if not mox.file.exists(accuracy_dir + "/detections"):
    mox.file.make_dirs(accuracy_dir + "/detections")

if not mox.file.exists(accuracy_dir + "/groundtruths"):
    mox.file.make_dirs(accuracy_dir + "/groundtruths")

GT_PATH = accuracy_dir + '/groundtruths'
DR_PATH = accuracy_dir + '/detections'

# Create the need file
TEMP_FILES_PATH = os.path.join(config1.result, "temp_files")

if not mox.file.exists(TEMP_FILES_PATH):
    mox.file.make_dirs(TEMP_FILES_PATH)
# results_files_path = "results"
if mox.file.exists(mapresults_dir):
    shutil.rmtree(mapresults_dir)

mox.file.make_dirs(os.path.join(mapresults_dir, "AP"))
mox.file.make_dirs(os.path.join(mapresults_dir, "F1"))
mox.file.make_dirs(os.path.join(mapresults_dir, "Recall"))
mox.file.make_dirs(os.path.join(mapresults_dir, "Precision"))


# read the txt
def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


# analysis the txt
ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
ground_truth_files_list.sort()
gt_counter_per_class = {}
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))  # Get the current file name
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))  # Get the current file name
    lines_list = file_lines_to_list(txt_file)  # Read each line of data in the file

    bounding_boxes = []
    for line in lines_list:  # Traverse all lines in the current file
        class_name, left, top, right, bottom = line.split()  # Analyze specific content
        bbox = left + " " + top + " " + right + " " + bottom
        bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
        # Record the number of each category
        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            gt_counter_per_class[class_name] = 1
            # Temporarily save in a json file for easy subsequent reading
    with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

gt_classes = sorted(list(gt_counter_per_class.keys()))  # All categories
n_classes = len(gt_classes)  # Number of categories

# 解析预测出来的结果
dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()
for class_index, class_name in enumerate(gt_classes):
    bounding_boxes = []
    for txt_file in dr_files_list:  # 遍历当前文件中的所有行
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))  # 得到当前文件名
        temp_path = os.path.join(GT_PATH, (file_id + ".txt"))  # 文件路径
        lines = file_lines_to_list(txt_file)  # 读取文件中的每一行数据
        for line in lines:
            tmp_class_name, confidence, left, top, right, bottom = line.split()  # 解析具体的内容
            if tmp_class_name == class_name:  # 记录检测到的结果
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)  # 将预测到的框 按照置信度从大到小排序
    # 临时保存在json文件中，方便后续读取
    with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

sum_AP = 0.0  # 记录map
count_true_positives = {}


# 计算ap
def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # 在第0个元素插入0
    rec.append(1.0)  # 在最后一个元素插入1
    mrec = rec[:]
    prec.insert(0, 0.0)  # 在第0个元素插入0
    prec.append(0.0)  # 在最后一个元素插入1
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


# 遍历所有的类别，对每个类别进行计算 pr，precision，recall f1
for class_index, class_name in enumerate(gt_classes):
    count_true_positives[class_name] = 0
    dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
    dr_data = json.load(open(dr_file))  # 读取出预测结果
    nd = len(dr_data)  # 预测到的目标个数

    tp = [0] * nd  # true positive
    fp = [0] * nd  # false positive
    score = [0] * nd  # 得分
    score05_idx = 0  # 记录得分大于0.5 索引

    # 遍历所有的预测结果，分别与真实值进行iou比较
    for idx, detection in enumerate(dr_data):
        file_id = detection["file_id"]  # 当前文件的文件id
        score[idx] = float(detection["confidence"])  # 当前文件的预测得分
        if score[idx] > 0.5:  # 得分是否大于0.5
            score05_idx = idx

        gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
        ground_truth_data = json.load(open(gt_file))  # 读取真实标签
        ovmax = -1
        gt_match = -1  # 记录匹配的gt

        bb = [float(x) for x in detection["bbox"].split()]  # 检测得到的bbox
        for obj in ground_truth_data:  # 遍历所有的真实标签
            if obj["class_name"] == class_name:  # 预测值与真实值的标签是否一样
                bbgt = [float(x) for x in obj["bbox"].split()]  # 真实框的bbox
                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
                      min(bb[3], bbgt[3])]  # 真实框与预测框的 交叉区域
                iw = bi[2] - bi[0] + 1  # 交叉区域宽、高
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                      + 1) * (
                                 bbgt[3] - bbgt[1] + 1) - iw * ih  # 预测值与真实值的并集
                    ov = iw * ih / ua  # iou
                    if ov > ovmax:  # 记录iou最大的那个
                        ovmax = ov
                        gt_match = obj  # 记录匹配的那个目标

        min_overlap = MINOVERLAP - 0.2
        if ovmax >= min_overlap:  # 如果iou大于指定阈值 MINOVERLAP
            if not bool(gt_match["used"]):  # 检查下当前与预测结果匹配的那个gt是否使用过
                tp[idx] = 1  # true positive ,本来是正样本，但也被预测为正样本
                gt_match["used"] = True
                count_true_positives[class_name] += 1  # 记录改类别中，预测正确的个数
            else:
                fp[idx] = 1
        else:  # 小于阈值，则认为是 false positive 本来是负样本，但是被模型预测为正样本
            fp[idx] = 1

    cumsum = 0
    for idx, val in enumerate(fp):  # 累计 fp
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):  # 累计tp
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):  # 计算召回率
        rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

    prec = tp[:]
    for idx, val in enumerate(tp):  # 计算准确率
        prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

    # 计算ap
    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    # 计算F1
    F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                       (np.array(prec) + np.array(rec)))

    sum_AP += ap  # 累计所有的ap
    text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "

    # 准备标题，绘制在F1，Recall，Precision
    if len(prec) > 0:
        F1_text = "{0:.2f}".format(F1[score05_idx]) + " = " + class_name + " F1 "
        Recall_text = "{0:.2f}%".format(rec[score05_idx] * 100) + " = " + class_name + " Recall "
        Precision_text = "{0:.2f}%".format(prec[score05_idx] * 100) + " = " + class_name + " Precision "
    else:
        F1_text = "0.00" + " = " + class_name + " F1 "
        Recall_text = "0.00%" + " = " + class_name + " Recall "
        Precision_text = "0.00%" + " = " + class_name + " Precision "

    # 只保留两位小数
    rounded_prec = ['%.2f' % elem for elem in prec]
    rounded_rec = ['%.2f' % elem for elem in rec]

    # 绘制 pr曲线
    plt.plot(rec, prec, '-o')
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

    fig = plt.gcf()
    fig.canvas.set_window_title('AP ' + class_name)

    plt.title('class: ' + text)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    fig.savefig(mapresults_dir + "/AP/" + class_name + ".png")
    plt.cla()

    # 绘制 F1 曲线
    plt.plot(score, F1, "-", color='orangered')
    plt.title('class: ' + F1_text + "\nscore_threhold=0.5")
    plt.xlabel('Score_Threhold')
    plt.ylabel('F1')
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    fig.savefig(mapresults_dir + "/F1/" + class_name + ".png")
    plt.cla()

    # 绘制 召回率曲线
    plt.plot(score, rec, "-H", color='gold')
    plt.title('class: ' + Recall_text + "\nscore_threhold=0.5")
    plt.xlabel('Score_Threhold')
    plt.ylabel('Recall')
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    fig.savefig(mapresults_dir + "/Recall/" + class_name + ".png")
    plt.cla()

    # 绘制准确率曲线
    plt.plot(score, prec, "-s", color='palevioletred')
    plt.title('class: ' + Precision_text + "\nscore_threhold=0.5")
    plt.xlabel('Score_Threhold')
    plt.ylabel('Precision')
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    fig.savefig(mapresults_dir + "/Precision/" + class_name + ".png")
    plt.cla()
plt.close()
# Get map
mAP = sum_AP / n_classes
text = "mAP = {0:.2f}%".format(mAP * 100)
print(text)
print("pr,precision,recall,F1 ")
print("Run testcase success!")

print("------Get result-------")
from help_modelarts import modelarts_result2obs

config1.profiling = False
modelarts_result2obs(config1)

# sess.close()
