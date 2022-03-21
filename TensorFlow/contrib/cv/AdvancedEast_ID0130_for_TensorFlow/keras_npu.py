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

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K

import cfg
from network_tensorflow_changeVGG_npu import East
from losses import quad_loss
from data_generator import gen
import argparse
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', help='data path')
parser.add_argument('--epochs', type=int, default=24, help='epochs')
parser.add_argument('--steps_per_epoch', type=int, default=9000, help='steps_per_epoch')
parser.add_argument('--validation_steps', type=int, default=1000, help='validation_steps')

args = parser.parse_args()
cfg.data_dir = args.data_path

from npu_bridge.npu_init import *

# session config
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

sess = tf.Session(config=sess_config)
K.set_session(sess)

east = East()
east_network = east.east_network()
east_network.summary()

opt_tmp = tf.compat.v1.train.AdamOptimizer(learning_rate=cfg.lr)
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=65536, incr_every_n_steps=1000,
                                                       decr_every_n_nan_or_inf=2, decr_ratio=0.5)
opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
east_network.compile(loss=quad_loss, optimizer=opt)

# if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
#     east_network.load_weights(cfg.saved_model_weights_file_path)
#     print('load model')
east_network.fit_generator(generator=gen(),
                           steps_per_epoch=int(args.steps_per_epoch),
                           epochs=args.epochs,
                           validation_data=gen(is_val=True),
                           validation_steps=int(args.validation_steps),
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])
print('Train Success')
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)


sess.close()
