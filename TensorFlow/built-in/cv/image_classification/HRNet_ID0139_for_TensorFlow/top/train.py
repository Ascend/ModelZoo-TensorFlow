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
from trainer.trainer import Trainer
from utils_new.multi_gpu_wrapper import MultiGpuWrapper as mgw
import time
from npu_bridge.npu_init import *
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool('enbl_multi_gpu', False, 'Enable training with multiple gpus')
tf.app.flags.DEFINE_string('data_path', '/autotest/CI_daily/CarPeting_HRNet/data/tfrecord', 'path to data tfrecords')
tf.app.flags.DEFINE_string('net_cfg', '/autotest/CI_daily/CarPeting_HRNet/00-access/cfgs/w30_s4.cfg', 'config file of network')
tf.app.flags.DEFINE_bool('eval_only', False, 'Eval mode')
tf.app.flags.DEFINE_bool('resume_training', False, 'resume training')
####################NPU modify start#####################
tf.app.flags.DEFINE_string('precision_mode', default='allow_fp32_to_fp16', help='enable precision mode.')
tf.app.flags.DEFINE_bool('loss_scale_flag', default=False, help='Whether enable loss scale.')
tf.app.flags.DEFINE_integer('loss_scale_value', default=1, help='0:dynamic >=1:static')
tf.app.flags.DEFINE_bool('over_dump', default=False, help='overflow dump flag.')
tf.app.flags.DEFINE_bool('data_dump', default=False, help='data dump flag.')
tf.app.flags.DEFINE_integer('data_dump_step', default=0, help='dump when steps is equal to 0.')
tf.app.flags.DEFINE_bool('profiling', default=False, help='whether profiling.')
tf.app.flags.DEFINE_bool('random_remove', default=False, help='remove random operations in preprocess.')
tf.app.flags.DEFINE_string('model_path', './test/output', 'path to save ckpt and logs')
tf.app.flags.DEFINE_string('data_dump_path', './test/output/data_dump', 'path to save data dump')
tf.app.flags.DEFINE_string('over_dump_path', './test/output/over_dump', 'path to save over dump file')
####################NPU modify end#######################


def main(unused_argv):
    """Main entry.

    Args:
    * unused_argv: unused arguments (after FLAGS is parsed)
    """
    start_time = time.time()
    tf.logging.set_verbosity(tf.logging.INFO)

    ####################NPU modify start#####################
    #if FLAGS.enbl_multi_gpu:
    #    mgw.init()
    config = npu_config_proto(config_proto=tf.ConfigProto())
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    npu_init = npu_ops.initialize_system()
    sess = tf.Session(config=config)
    sess.run(npu_init)
    ####################NPU modify end#######################

    trainer = Trainer(data_path=FLAGS.data_path, netcfg=FLAGS.net_cfg)

    trainer.build_graph(is_train=True)
    trainer.build_graph(is_train=False)

    if FLAGS.eval_only:
        trainer.eval()
    else:
        trainer.train()

    ####################NPU modify start#####################
    npu_shutdown = npu_ops.shutdown_system()
    sess.run(npu_shutdown)
    sess.close()
    ####################NPU modify end#######################

    end_time = time.time()
    e2e_time = end_time - start_time
    print("train+eval total time(s): ", e2e_time)



if __name__ == '__main__':
    tf.app.run()
