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
#
import npu_device as npu
npu.open().as_default()
import os, inspect, warnings, argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import ast
import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = PACK_PATH+'/Checkpoint'


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./datasets', help="path to folder containing images")
    parser.add_argument("--model_dir", default='./model', help="save path of training model")
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size')
    parser.add_argument("--log_steps", type=int, default=300, help="log steps for training")

    # ===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str, help='precision mode')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                        help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                        help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,
                        help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str, help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str, help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str, help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                        help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                        help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,
                        help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,
                        help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')

    ############多p参数##############
    parser.add_argument("--rank_size", default=1, type=int, help="rank size")
    parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS,unparsed

def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu.global_options().dump_config.enable_dump = True
        npu.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu.global_options().dump_config.enable_dump_debug = True
        npu.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu.global_options().profiling_config.profiling_options = profiling_options
    npu.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode == 'allow_mix_precision':
        npu.global_options().modify_mixlist = FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu.global_options().fusion_switch_file = FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu.global_options().auto_tune_mode = "RL,GA"
    npu.open().as_default()



def main():

    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except: pass

    FLAGS, unparsed = args_parse()
    npu_config(FLAGS)
    batch_size = FLAGS.batch_size // FLAGS.rank_size
    dataset = dman.Dataset(data_path=FLAGS.data_path,normalize=FLAGS.datnorm,rank_size=FLAGS.rank_size,device_id=FLAGS.device_id)
    batch_nums = int(dataset.example_num/FLAGS.batch_size)   
    neuralnet = nn.CNN(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        num_class=dataset.num_class, leaning_rate=FLAGS.lr, ckpt_dir=FLAGS.model_dir, rank_size=FLAGS.rank_size)

    tfp.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epochs, batch_size=batch_size,
                 log_steps=FLAGS.log_steps,batch_nums=batch_nums,normalize=True,rank_size=FLAGS.rank_size)
    tfp.test(neuralnet=neuralnet, dataset=dataset, batch_size=batch_size)

if __name__ == '__main__':
    main()

