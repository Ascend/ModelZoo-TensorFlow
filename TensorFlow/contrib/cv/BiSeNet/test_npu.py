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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import npu_bridge
from npu_bridge.npu_init import *

import tensorflow as tf
from models.bisenet import BiseNet
import configuration
import logging

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    model_config = configuration.MODEL_CONFIG
    train_config = configuration.TRAIN_CONFIG

    g = tf.Graph()
    with g.as_default():
        # Build the test model
        model = BiseNet(model_config, train_config, 32, 'test')
        model.build()

        saver = tf.compat.v1.train.Saver()

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True

        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

        sess = tf.compat.v1.Session(config=config)
        model_path = tf.train.latest_checkpoint(train_config['train_dir'])

        config = train_config['test_data_config']
        total_steps = config['num_examples_per_epoch']//config['batch_size']
        logging.info('Train for {} steps'.format(total_steps))

        local_variables_init_op = tf.local_variables_initializer()

        sess.run(local_variables_init_op)
        saver.restore(sess, model_path)

        for step in range(total_steps):
            predict_loss, loss, accuracy, mean_IOU = sess.run([model.loss, model.total_loss, model.accuracy, model.mean_IOU])
            format_str = 'step %d, loss = %.2f, accuracy = %.2f, mean_IOU = %.2f'
            logging.info(format_str % (step, loss, accuracy[0], mean_IOU[0]))
