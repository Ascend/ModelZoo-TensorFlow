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
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from detect import yolo_eval
from nets.yololoss import yolo_head
from data_process.config import anchors, classes, input_shape, batch_size
from nets.v2net import model_body
import cv2
from PIL import Image
import moxing as mox
import argparse

print("++++++++++++++begin get_info file++++++++++++++++++")
parser = argparse.ArgumentParser()
# bos first
parser.add_argument("--dataset", type=str, default="./output")  ## config.modelarts_data_dir    /cache/DataSet
parser.add_argument("--result", type=str, default="./dataset")  ## config.modelarts_result_dir   /cache/result

# for last address
parser.add_argument("--obs_dir", type=str)  ## last address  s3://yolov2/yolov2forfen/output/V0027/



config1 = parser.parse_args()
print("config1.dataset", config1.dataset)  ## config.modelarts_data_dir    /cache/DataSet
print("config1.result", config1.result)  ## config.modelarts_result_dir   /cache/result

# for last address
print("config1.obs_dir", config1.obs_dir)  ## last address s3://yolov2/yolov2forfen/output/V0027/



# HUA WEI
print("------Hua Wei-----")
init = tf.global_variables_initializer()
# Create session
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
# Get profiling
work_dir = os.getcwd()
modelarts_profiling_dir = os.path.join(work_dir, "npu_profiling")
if not os.path.exists(modelarts_profiling_dir):
    os.makedirs(modelarts_profiling_dir)

print("modelarts_profiling_dir", modelarts_profiling_dir)

options = '{"output": "%s", \
            "task_trace": "on", \
            "aicpu": "on"}' % (modelarts_profiling_dir)

custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)

sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
# K.set_session(sess)
sess.run(init)

## modelarts(config.result /cache/result) create file  info_result_dir
accuracy_dir = os.path.join(config1.result, 'accuracy')
mapresults_dir = os.path.join(config1.result, 'mapresults')

if not mox.file.exists(accuracy_dir):
    mox.file.make_dirs(accuracy_dir)

if not mox.file.exists(accuracy_dir + "/detections"):
    mox.file.make_dirs(accuracy_dir + "/detections")

if not mox.file.exists(accuracy_dir + "/groundtruths"):
    mox.file.make_dirs(accuracy_dir + "/groundtruths")

# if not os.path.exists("accuracy"):
#     os.mkdir("accuracy")
# if not os.path.exists("accuracy/detections"):
#     os.mkdir("accuracy/detections")
# if not os.path.exists("accuracy/groundtruths."):
#     os.mkdir("accuracy/groundtruths")

# model_path = config1.dataset+"/"+config1.model_fortest

model_path = config1.dataset + "/ep100-loss1.568-val_loss18.650.h5"

print("model_path we load", model_path)
model_body.load_weights(model_path)  # 加载权重
yolo_outputs = yolo_head(model_body.output, anchors, len(classes))  # 对网络输出进行编码
input_image_shape = K.placeholder(shape=(2,))
boxes1, scores1, pred_classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.05,
                                          iou_threshold=0.6)

### 得到预测值
total = 0
current = 0
valtxt = config1.dataset + "/2007_test_part.txt"
# with open("../data_process/2007_val.txt", "r") as f:
with open(valtxt, "r") as f:
    for tmp in f.readlines():
        tmp = config1.dataset + tmp
        total += 1
        test_path = tmp.split(" ")[0]
        img_name = test_path.split("/")[-1].split(".jpg")[0]
        image = cv2.imread(test_path)
        origin_img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，转换为RGB
        image = cv2.resize(image, input_shape)
        image = image / 255
        image = np.expand_dims(image, 0)
        sess = K.get_session()
        starttime = time.time()
        boxes, scores, cls = sess.run(
            [boxes1, scores1, pred_classes],
            feed_dict={
                model_body.input: image,
                input_image_shape: [input_shape[1], input_shape[0]],
                K.learning_phase(): 0
            })
        endtime = time.time()
        current += float(endtime - starttime)
        print('Found {} boxes for {}'.format(len(boxes), test_path))

        with open(accuracy_dir + "/detections/" + img_name + ".txt", "w") as f:
            for data in zip(cls, scores, boxes):
                f.write(" ".join([str(classes[data[0]]), str(data[1]),
                                  str(data[2][0]), str(data[2][1]),
                                  str(data[2][2]), str(data[2][3])]))
                f.write("\n")
print("fps:", 1 / (current / total))

# 得到真实值
all_files = []
# with open("../data_process/2007_val.txt","r") as f:
with open(valtxt, "r") as f:
    all_files = f.readlines()
    for file in all_files:
        file = config1.dataset + file
        image = cv2.imread(file.split(" ")[0])
        origin_shape = image.shape[0:2]
        x_scale, y_scale = float(input_shape[1]) / origin_shape[1], float(input_shape[0]) / origin_shape[0]
        filename = file.split(" ")[0].split("/")[-1].split(".jpg")[0]
        with open(accuracy_dir + "/groundtruths/" + filename + ".txt", "w", encoding='utf_8') as ff:
            infos = file.split(" ")[1:]
            print(file)
            for info in infos:
                classname = classes[int(info.split(",")[-1].strip())]
                tmp = list(map(int, info.split(",")[:-1]))
                tmps = [str(tmp[0] * x_scale), str(tmp[1] * y_scale), str(tmp[2] * x_scale), str(tmp[3] * y_scale)]
                result = " ".join(tmps)
                ff.write(classname + " " + result + "\n")

print("------Get result-------")
from help_modelarts import modelarts_result2obs

config1.profiling = False
modelarts_result2obs(config1)
#
# sess.close()
