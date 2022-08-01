import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops

from npu_bridge.npu_init import *
from MUNIT import MUNIT
import argparse
from utils import *
from tensorflow.python.framework import graph_util

# import precision_tool.tf_config as npu_tf_config
import os
import moxing as mox

# 指定checkpoint路径
ckpt_path = "obs://cann-id0953/final_files/ckpt"

def parse_args():
    desc = "Tensorflow implementation of MUNIT"

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result', type=str, default='results',
                        help='Directory name to save the results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--pb_dir', type=str, default="pb")

    #return parser.parse_args()
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --pb_dir
    args.pb_dir = os.path.join(args.result, args.pb_dir)
    check_folder(args.pb_dir)
    
    # --checkpoint_dir
    args.checkpoint_dir = os.path.join(args.result, args.checkpoint_dir)
    check_folder(args.checkpoint_dir)
   
    return args

def main():
    # parse arguments
    args = parse_args()

    print("===>>>Copy files from {} to modelarts dir:{}".format(ckpt_path, args.checkpoint_dir))
    mox.file.copy_parallel(src_url=ckpt_path, dst_url=args.checkpoint_dir)
    print("Copy finished")

    if args is None:
        exit()

    tf.reset_default_graph()

    # 定义网络的输入节点
    test_image_A = tf.placeholder(tf.float32, [args.batch_size,
                                               args.img_h, args.img_w, args.img_ch], name='test_imageA')
    test_image_B = tf.placeholder(tf.float32, [args.batch_size,
                                               args.img_h, args.img_w, args.img_ch], name='test_imageB')
    test_style = tf.placeholder(tf.float32, [args.batch_size, 1, 1, args.style_dim], name='test_style')

    with tf.Session() as sess:
        gan = MUNIT(sess, args)

        test_content_a, _ = gan.Encoder_A(test_image_A)
        test_content_b, _ = gan.Encoder_B(test_image_B)

        test_fake_A = gan.Decoder_A(content_B=test_content_b, style_A=test_style)
        test_fake_B = gan.Decoder_B(content_A=test_content_a, style_B=test_style)

        test_fake_A = tf.identity(test_fake_A, name = 'outputA')
        test_fake_B = tf.identity(test_fake_B, name = 'outputB')

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.checkpoint_dir, "MUNIT.model-100001"))  # 恢复图并得到数据

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['outputA', 'outputB'])  # 如果有多个输出节点，以逗号隔开
        # 保存模型
        with tf.gfile.GFile(os.path.join(args.pb_dir, "munit.pb"), "wb") as f: 
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    print("done")

if __name__ == '__main__':
    main()
