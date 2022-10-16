import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops
from tensorflow.python.framework import graph_util
from ops import *
from UGATIT import UGATIT
import argparse
from utils import *
from logger import Logger
import npu_bridge 
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu import util
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig


import sys
"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / '
                                                                  'U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training '
                                                                  'iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of '
                                                                    'image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    # Quantization argument
    parser.add_argument('--quant', type=str2bool, default=True,
                        help='quantization or not?')
    parser.add_argument('--commitment_cost', type=float, default=2.0, help='commitment cost')
    parser.add_argument('--quantization_layer', type=str, default='123', help='which layer?')
    parser.add_argument('--decay', type=float, default=0.85, help='dictionary learning decay')
    parser.add_argument('--test_train', type=str2bool, default=False, help='if test while training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir

    if args.quant:
        args.checkpoint_dir += '_quant'
        args.result_dir += '_quant'
        args.log_dir += '_quant'
        args.sample_dir += '_quant'

    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
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




config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  
batch_size = 1


ckpt_path = "/home/test_user07/FQ-GAN/FQ-U-GAT-IT/checkpoint_quant/UGATIT_q_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing_123_2.0_0.85/UGATIT.model-2"

def main():
    tf.compat.v1.disable_eager_execution()
    tf.reset_default_graph()
    inputs1 = tf.placeholder(tf.float32, shape=[1, 256, 256, 3], name="input1")

    args = parse_args()
    if args is None:
      exit()
    
    with tf.Session(config=config) as sess:
        gan = UGATIT(sess, args)
        out, cam= gan.generate_b2a(inputs1)

        c_out = tf.identity(out, name='c_out')
        p_cam = tf.identity(cam, name='p_cam')
    
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='p_cam,c_out',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/UGATIT_new.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")

if __name__ == '__main__':
    main()

