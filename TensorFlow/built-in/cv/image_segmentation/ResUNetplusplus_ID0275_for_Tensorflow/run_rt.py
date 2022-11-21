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

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from npu_bridge.npu_init import *
from tensorflow.python.keras import backend as K
from time import time
import random
import argparse
import precision_tool.tf_config as npu_tf_config
from tensorflow.keras.losses import binary_crossentropy
from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss

random.seed(0)
tf.random.set_random_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', dest='save_dir', default='./test/out/checkpoints')
parser.add_argument('--data_path', dest='data_path', default='./new_data/Kvasir-SEG', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')
parser.add_argument('--npu_loss_scale', dest='npu_loss_scale', type=int, default=1)
parser.add_argument('--mode', dest='mode', default='train', choices=('train', 'test', 'train_and_eval'))
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=100)
args = parser.parse_args()

# os.environ["DUMP_GE_GRAPH"] = "2"
# os.environ["DUMP_GRAPH_LEVEL"] = "2"
# os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
# os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"

sess_config = tf.ConfigProto()
sess_config = npu_tf_config.session_dump_config(sess_config, action='fusion_switch')
sess_config.allow_soft_placement = True
sess_config.log_device_placement = False
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess_config.graph_options.optimizer_options.global_jit_level = config_pb2.OptimizerOptions.OFF

custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
#custom_op.parameter_map["dynamic_input"].b = True
#custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

# custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(8*1024 * 1024 * 1024))
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/ma-user/work/rupp/data_dump")
# custom_op.parameter_map["enable_dump_debug"].b = True
# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
# custom_op.parameter_map["enable_dump"].b = True
# custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0")
# custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")

sess = tf.Session(config=sess_config)
K.set_session(sess)

if __name__ == "__main__":
    ## Path
    file_path = "files/"
    model_path = file_path + "resunetplusplus.h5"

    ## Create files folder
    try:
        os.makedirs(file_path)
    except:
        pass

    train_path = os.path.join(args.data_path, "train")
    valid_path = os.path.join(args.data_path, "valid")

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = args.batch_size
    lr = 1e-5
    epochs = args.num_epochs

    train_steps = len(train_image_paths) // batch_size
    valid_steps = len(valid_image_paths) // batch_size

    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    ## Unet
    # arch = Unet(input_size=image_size)
    # model = arch.build_model()

    ## ResUnet
    # arch = ResUnet(input_size=image_size)
    # model = arch.build_model()

    ## ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)
    model = arch.build_model()
    opt = Nadam(lr)
    '''
    # 因为在2021-12版本Keras还不支持Loss Scale，所以不这个方法不行。
    # opt = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    # opt= NPULossScaleOptimizer(opt, loss_scale_manager)
    '''
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    # binary_crossentropy dice_loss
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}unet_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor="val_precision", mode="max")
    reduce_lr = ReduceLROnPlateau(monitor='val_precision', factor=0.1, patience=10, min_lr=1e-6, verbose=1,mode="max")
    early_stopping = EarlyStopping(monitor='val_precision', patience=30, restore_best_weights=False,mode="max")
    tb = TensorBoard(log_dir=file_path, write_grads=True, histogram_freq=0, update_freq=100)
    callbacks = [checkpoint, early_stopping, reduce_lr, tb, csv_logger]
    StartTime1 = time()  # time add
    model.fit_generator(train_gen,
                        validation_data=valid_gen,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        epochs=epochs,
                        callbacks=callbacks)
    EndTime1 = time()  # time add
    print('-------All epoch time : ' + str(EndTime1 - StartTime1))  # time add,Performance calculation requires division
    print('-------Use all time average steps : ' + str(
        (train_steps + valid_steps)* args.batch_size /(EndTime1 - StartTime1)))  # time add,Performance calculation requires division
