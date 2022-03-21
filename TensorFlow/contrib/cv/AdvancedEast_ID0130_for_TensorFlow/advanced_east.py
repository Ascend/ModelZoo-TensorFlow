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


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adam
import cfg
from data_generator import gen
from losses import quad_loss
# from network_tf import East
from network_tensorflow_changeVGG_npu import East

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUDA_VISIBLE_DEVICES"] = "1"  # 这里是gpu的序号，指定使用的gpu对象config = tf.ConfigProto()config.gpu_options.allow_growth = True

east = East()
east_network = east.east_network()
east_network.summary()
east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                    # clipvalue=cfg.clipvalue,
                                                    decay=cfg.decay))
if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
    east_network.load_weights(cfg.saved_model_weights_file_path)
east_network.fit_generator(generator=gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           epochs=cfg.epoch_num,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           verbose=1,
                           initial_epoch=cfg.initial_epoch,
                           callbacks=[
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])
east_network.save(cfg.saved_model_file_path)
east_network.save_weights(cfg.saved_model_weights_file_path)
