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
from MUNIT import MUNIT
import argparse
from utils import *
# from help_modelarts import modelarts_result2obs
# import precision_tool.tf_config as npu_tf_config

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of MUNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--code_dir', type=str, default='code', help='code_dir')
    parser.add_argument('--phase', type=str, default='train', help='train or test or guide')
    parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')
    parser.add_argument('--data_path', type=str, default='summer2winter', help='dataset_name')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')
    parser.add_argument('--obs_dir', type=str, default='./output/', help='obs_dir')

    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')
    parser.add_argument('--direction', type=str, default='a2b', help='direction of style guided image translation')
    parser.add_argument('--guide_img', type=str, default='guide.jpg', help='Style guided image translation')

    parser.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', type=float, default=10.0, help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w', type=float, default=1.0, help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w', type=float, default=1.0, help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w', type=float, default=0.0, help='weight of explicit style augmented cycle consistency loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--n_sample', type=int, default=2, help='number of sampling layers in content encoder')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=3, help='number of scales')

    parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result', type=str, default='results',
                        help='Directory name to save the results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    # parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--bert_loss_scale', type=int, default=0)


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    args.checkpoint_dir = os.path.join(args.result, args.checkpoint_dir)
    check_folder(args.checkpoint_dir)

    # --result_dir
    args.result_dir = os.path.join(args.result, args.result_dir)
    check_folder(args.result_dir)

    # --result_dir
    args.log_dir = os.path.join(args.result, args.log_dir)
    check_folder(args.log_dir)

    # --dump_dir
    args.dump_dir = os.path.join(args.result, "dump")
    check_folder(args.dump_dir)

    # --sample_dir
    args.sample_dir = os.path.join(args.result, args.sample_dir)
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    ############################## npu modify #########################
    config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    # # 混合精度
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
    #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
    # 算子黑名单
    #custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")
    # custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(os.path.join(args.code_dir, "ops_info.json"))
    # print(os.path.isfile(os.path.join(args.code_dir, "ops_info.json")))
    # print(os.path.join(args.code_dir, "ops_info.json"))

    # 判断是否溢出
    # # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.dump_dir)
    # # enable_dump_debug：是否开启溢出检测功能
    # custom_op.parameter_map["enable_dump_debug"].b = True
    # # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    # custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    # custom_op = npu_tf_config.update_custom_op(custom_op, action='overflow')

    # # 关闭全部融合规则
    # config = npu_tf_config.session_dump_config(config, action='fusion_off')

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #off remap
    config = npu_config_proto(config_proto=config)



    # if args.use_fp16 and (args.bert_loss_scale not in [None, -1]):
    #     opt_tmp = custom_op
    #     if args.bert_loss_scale == 0:
    #         loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
    #                                                                decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    #     elif args.bert_loss_scale >= 1:
    #         loss_scale_manager = FixedLossScaleManager(loss_scale=args.bert_loss_scale)
    #     else:
    #         raise ValueError("Invalid loss scale: %d" % args.bert_loss_scale)
    #     # device数是否大于1，如果大于1，进行分布式训练
    #     # if ops_adapter.size() > 1:
    #     #     opt_tmp = NPUDistributedOptimizer(opt_tmp)
    #     #     custom_op = NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=True)
    #     # else:
    #     custom_op = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)

    # open session
    with tf.Session(config=config) as sess:
        gan = MUNIT(sess, args)
    ############################## npu modify #########################

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'guide' :
            gan.style_guide_test()
            print(" [*] Guide finished!")

    #modelarts_result2obs(args)

if __name__ == '__main__':
    main()

