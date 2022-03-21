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

import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E

def set_up_train(config):
    # Setup tensorflow
    #tf.config.threading.set_inter_op_parallelism_threads(8)
    #tf.config.threading.set_intra_op_parallelism_threads(8)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data, _ = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)

    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model_generator = BIGBIGAN_G(config, weight_init)
    model_discriminator_f = BIGBIGAN_D_F(config, weight_init)
    model_discriminator_h = BIGBIGAN_D_H(config, weight_init)
    model_discriminator_j = BIGBIGAN_D_J(config, weight_init)
    model_encoder = BIGBIGAN_E(config, weight_init)

    # train
    logging.info('Start training...')

    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data)
    # Finished
    logging.info('Training finished ;)')
