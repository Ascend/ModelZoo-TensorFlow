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

from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import argparse
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Multiply, add
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
#import moxing as mox
import os

os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = '0'

parser = argparse.ArgumentParser(description='')
parser.add_argument("--train_url", type=str, default="./output")
parser.add_argument("--data_url", type=str, default="./dataset")
parser.add_argument("--modelarts_data_dir", type=str, default="/cache/fdg_data")
parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
parser.add_argument('--save_dir', dest='save_dir', default='.', help='directory for testing outputs')
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--train_nb', type=int, default=5994)#add

args = parser.parse_args()
config = parser.parse_args()
#mox.file.copy_parallel(src_url=config.data_url, dst_url=config.modelarts_data_dir)

obs_path = config.train_url
obs_result_dir = obs_path + 'result'

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(21 * 1024 * 1024 * 1024))
custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(10 * 1024 * 1024 * 1024))
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
# sess_config.graph_options.rewrite_options.optimizers.extend(["GradFusionOptimizer"]) #分布式场景需要添加
sess = tf.Session(config=sess_config)

K.clear_session()
K.set_session(sess)

BATCH_SIZE = 16
test_nb = 5794
#train_nb = 5994
num_classes = 200
img_size = 448
classes = []

train_path = args.modelarts_data_dir + "/train"
test_path = args.modelarts_data_dir + "/test"
# %%


with open(args.modelarts_data_dir + "/classes.txt") as f:
    for l in f.readlines():
        data = l.split()
        classes.append(data[1])

# %% create data generator

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   zoom_range=[1.0, 2.0],
                                   rotation_range=90,
                                   horizontal_flip=True,
                                   vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    seed=13)

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    seed=13)
# %% finetuning resnet50

input_tensor = Input(shape=(img_size, img_size, 3))
base_model = ResNet50(weights=args.modelarts_data_dir + "/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                      include_top=False, input_tensor=input_tensor)

# %% Implementation of OSME module


def osme_block(in_block, ch, ratio=16):
    z = GlobalAveragePooling2D()(in_block)  # 1
    x = Dense(ch // ratio, activation='relu')(z)  # 2
    x = Dense(ch, activation='sigmoid')(x)  # 3
    return Multiply()([in_block, x])  # 4


s_1 = osme_block(base_model.output, base_model.output_shape[3])
s_2 = osme_block(base_model.output, base_model.output_shape[3])

fc1 = Flatten()(s_1)
fc2 = Flatten()(s_2)

fc1 = Dense(1024, name='fc1')(fc1)
fc2 = Dense(1024, name='fc2')(fc2)

fc = add([fc1, fc2])  # fc1 + fc2

prediction = Dense(num_classes, activation='softmax', name='prediction')(fc)

model = Model(inputs=base_model.input, outputs=prediction)

opt = SGD(lr=0.001, momentum=0.9, decay=0.0001)

# model.load_weights("/home/n-kamiya/models/model_without_MAMC/model_osme_vgg_imagenet.best_loss.hdf5")

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# plot_model(model, to_file="model.png", show_shapes=True)

# %% implement checkpointer and reduce_lr (to prevent overfitting)
checkpointer = ModelCheckpoint(filepath=args.modelarts_result_dir + '/model_osme_resnet50.best_loss.hdf5', verbose=1,
                               save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.0000001)


# %% fit_generator

isExists = os.path.exists(args.modelarts_result_dir)  # 创建 args.modelarts_result_dir 文件夹

# 判断结果
if not isExists:
    # 如果不存在则创建目录
    # 创建目录操作函数
    os.makedirs(args.modelarts_result_dir)

file = open(args.modelarts_result_dir + '/test.txt', 'w')
file.write('helo world!')
file.close()

print("\n\n##################\n\n")
print(args.modelarts_result_dir)
print(args.modelarts_result_dir + '/model_osme_resnet50.best_loss.hdf5')

history = model.fit_generator(train_generator,
                              steps_per_epoch=args.train_nb / BATCH_SIZE,#add   160/16=10
                              epochs=args.epoch,
                              validation_data=validation_generator,
                              validation_steps=64,
                              verbose=1,
                              callbacks=[reduce_lr, checkpointer])

# %% plot results
import datetime

now = datetime.datetime.now()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_without_MAMC accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(args.modelarts_result_dir + "/history_osme_resnet50{0:%d%m}-{0:%H%M%S}.png".format(now))
#plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_without_MAMC loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(args.modelarts_result_dir + "/loss_osme_resnet50{0:%d%m}-{0:%H%M%S}.png".format(now))

#plt.show()
#mox.file.copy_parallel(src_url=args.modelarts_result_dir, dst_url=obs_result_dir)

sess.close()
