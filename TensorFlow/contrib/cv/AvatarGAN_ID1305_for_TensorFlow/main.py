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
import argparse
import os
import tensorflow as tf
from model import Cyclegan
from utils import boolean_string
#import moxing as mox
tf.compat.v1.set_random_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument("--gpu_idx", type=str, default="4", help="set gpu for training and testing")
parser.add_argument('--dataset_dir', dest='dataset_dir', default='avatar_data', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=2, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000,
                    help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100,
                    help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0:0. false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                    default='./checkpoint',
                    help='models are saved here')
#./checkpoint
parser.add_argument('--sample_dir', dest='sample_dir',
                    default='./sample',
                    help='sample are saved here')
#./sample
parser.add_argument('--test_dir', dest='test_dir', default='./test_',
                    help='test sample are saved here')
#./test
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True,
                    help='generation network using residule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50,
                    help='max size of image pool, 0 means do not use image pool')
parser.add_argument("--alpha", type=float, default=0.33, help="loss weight for two datasets")
parser.add_argument("--penalize_disc", type=boolean_string, default=False, help="")


# ===================================== TODO: positon in OBS
# data_url:"Data_path_in_OBS"
parser.add_argument("--data_url", type=str, default="./datasets")
# train_url:"OBS_path"
parser.add_argument("--train_url", type=str, default="./output")
# ==========================================================

# ============================== TODO: position in ModelArts
# 在ModelArts容器创建数据存放目录
# data_dir = "/cache/dataset"
# os.makedirs(data_dir)
parser.add_argument("--data_dir", type=str, default="./dataset")

# OBS数据拷贝到ModelArts容器内
# mox.file.copy_parallel(args.data_url, data_dir)
# 在ModelArts容器创建训练输出目录
# model_dir = "/cache/output"
# os.makedirs(model_dir)
parser.add_argument("--model_dir", type=str, default="/home/ma-user/modelarts/outputs/train_url_0/")

args = parser.parse_args()
# ==========================================================

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

## 1
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)

def main(_):
    ## 3
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    # ========================== 手动迁移 ================================= #
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    #profiling 开关
    custom_op.parameter_map["profiling_mode"].b = False
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
    #     '{"output":"/mypath/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":""}')
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        '{"output":"/home/ma-user/modelarts/outputs/train_url_0/","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":""}')
    # 关闭融合规则
    # custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./src/fusion_switch.cfg")

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        model = Cyclegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)


    # tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    # with tf.Session(config=npu_config_proto(config_proto=tfconfig)) as sess:
    #     model = Cyclegan(sess, args)
    #     model.train(args) if args.phase == 'train' \
    #         else model.test(args)
    #     #mox.file.copy_parallel(model_dir, args.train_url)


if __name__ == '__main__':
    ## 2
    tf.app.run()
    # after training, copy: ModelArts->OBS
    #mox.file.copy_parallel(model_dir, args.train_url)
    # ModelArts:"/cache/output" -> OBS_path:"/avatar-gan-npu/GAN_npu_for_TensorFlow/output/"

