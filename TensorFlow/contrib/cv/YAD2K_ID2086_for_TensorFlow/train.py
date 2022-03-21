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

from npu_bridge.npu_init import *

import tensorflow as tf
from tensorflow.python.keras import backend as k
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from nets.yololoss import yolo_loss
from data_process.config import anchors, classes, input_shape, batch_size
from nets.v2net import model_body
from data_process.data_loader import SequenceData
from keras.layers import Input, Lambda
import argparse
#import moxing as mox

print("++++++++++++++begin ++++++++++++++++++")
parser = argparse.ArgumentParser()
# bos first
parser.add_argument("--dataset", type=str, default="./output")
parser.add_argument("--result", type=str, default="./dataset")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=10)

# last
#parser.add_argument("--obs_dir", type=str)

config1 = parser.parse_args()
print("config1.dataset", config1.dataset)
print("config1.result", config1.result)  ## config.modelarts_result_dir   /cache/result

# last
#print("config1.obs_dir", config1.obs_dir)

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

options = '{"output": "%s", \
            "task_trace": "on", \
            "aicpu": "on"}' % (modelarts_profiling_dir)

custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)

sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
k.set_session(sess)

print(" +++++++++++++++++++++++ start pram ++++++++++")
input_image = Input(shape=(416, 416, 3))  # 输入图片为416*416，三通道（RGB）
boxes_input = Input(shape=(None, 5))  # 表示一张图中所有目标信息
detectors_mask_input = Input(shape=(13, 13, 5, 1))  # 目标掩码，确定目标位于哪一个单元格中的哪一个anchor
matching_boxes_input = Input(shape=(13, 13, 5, 5))  # 目标在单元格中的anchor的编码位置和类别信息
model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                    arguments={'anchors': anchors, 'num_classes': len(classes)})(
    [model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])
print(" +++++++++++++++++++++++ end pram ++++++++++++")

print(" +++++++++++++++++++++++ start create model")
model = Model([model_body.input, boxes_input, detectors_mask_input, matching_boxes_input],
              model_loss)  # 将loss layer加入模型中
# 配置模型
premodel = config1.dataset + '/ckpt.h5'
print(premodel)


# model.compile(
#     optimizer=Adam(learning_rate=0.0001), loss={
#         'yolo_loss': lambda y_true, y_pred: y_pred})

def yolo_choose(y_true, y_pre):
    return y_pre


# 配置模型
model.compile(
    optimizer=Adam(learning_rate=0.0001), loss={
        'yolo_loss': yolo_choose})

model_body.load_weights(premodel, by_name=True, skip_mismatch=True)

# 训练模型
print("++++++++++++++++++++++++++++++++++")
logfile_path = config1.result
print("logfile_path", logfile_path)
print("++++++++++++++++++++++++++++++++++")

logging = TensorBoard(log_dir=logfile_path)  # 指定训练log目录


#  modelArts
## modelarts(config.result /cache/result) info_result_dir
model_storage = os.path.join(config1.result, 'model_storage')

#if not mox.file.exists(model_storage):
#    mox.file.make_dirs(model_storage)
checkpoint = ModelCheckpoint(model_storage + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='loss',
                             save_weights_only=True, save_best_only=False, period=3)  # Save model

# config1.profiling = False


print(config1.result)

# 创建训练和验证数据集
print("++++++++++++++++++++++++++++++++++")
train2007 = config1.dataset + '/2007_train.txt'
print(train2007)
print("++++++++++++++++++++++++++++++++++")
train_sequence = SequenceData(train2007, input_shape, config1.batch_size, anchors, len(classes), config1)
print("++++++++++++++++++++++++++++++++++")
val2007 = config1.dataset + '/2007_val.txt'
print(val2007)
print("++++++++++++++++++++++++++++++++++")
val_sequence = SequenceData(val2007, input_shape, config1.batch_size, anchors, len(classes), config1)

# 训练
model.fit_generator(train_sequence,
                    steps_per_epoch=train_sequence.get_epochs(),
                    validation_data=val_sequence,
                    validation_steps=val_sequence.get_epochs(),
                    validation_freq=1,
                    initial_epoch=0,
                    epochs=config1.epochs,
                    workers=0,
                    callbacks=[checkpoint, logging])

#from help_modelarts import modelarts_result2obs

#print("------Get result-------")
#config1.profiling = True
#modelarts_result2obs(config1)

sess.close()
