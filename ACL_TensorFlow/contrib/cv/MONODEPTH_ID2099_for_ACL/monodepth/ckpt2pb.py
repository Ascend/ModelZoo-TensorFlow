import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from monodepth_model import *
from monodepth_main import *
from utils import *
# 添加导入NPU库的头文件
from npu_bridge import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='~/')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default='~/')
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
args = parser.parse_args()


def main():

    config = tf.ConfigProto(allow_soft_placement=True)
    # config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True  # 必须显式开启，在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    ckpt_path = "/home/test_user03/t3/tmp/my_model/model-181250"

    params = monodepth_parameters(
                    encoder=args.encoder,
                    height=args.input_height,
                    width=args.input_width,
                    batch_size=args.batch_size,
                    num_threads=args.num_threads,
                    num_epochs=args.num_epochs,
                    do_stereo=args.do_stereo,
                    wrap_mode=args.wrap_mode,
                    use_deconv=args.use_deconv,
                    alpha_image_loss=args.alpha_image_loss,
                    disp_gradient_loss_weight=args.disp_gradient_loss_weight,
                    lr_loss_weight=args.lr_loss_weight,
                    full_summary=args.full_summary)

    tf.reset_default_graph()


    input1 = tf.placeholder(tf.float32, shape=[2, 256, 512, 3], name="input1")
    net = MonodepthModel(params, "test", input1, None)
    out_left = tf.identity(net.disp_left_est[0], name='out_left')

    with tf.Session(config=config) as sess:
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]
        for node in node_list:
            print("node_name", node)
        tf.train.write_graph(sess.graph_def, '/home/test_user03/t3/xjk/pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='/home/test_user03/t3/xjk/pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='out_left',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='/home/test_user03/t3/xjk/pb_model/monodepth.pb',
            clear_devices=False,
            initializer_nodes='')
    print("done")


if __name__ == '__main__':
    main()