"""
Function: Huawei project for DHAA project for HUAWEI CLOUD
Author: ZiChang Tan
Date: 2021.5.25
BiB:
@inproceedings{tan2019deeply,
  title={Deeply-learned Hybrid Representations for Facial Age Estimation.},
  author={Tan, Zichang and Yang, Yang and Wan, Jun and Guo, Guodong and Li, Stan Z},
  booktitle={IJCAI},
  pages={3548--3554},
  year={2019}
}
"""

import tensorflow as tf
import os, time, argparse, sys
import util.utils as utils
import util.dataset_class as data
from util.utils import Dataset
import util.resnet_DHAA as resnet_DHAA

os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main(args):
    #### Set GPU options ###
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    # config.gpu_options.allow_growth = True

    ### Setting Parameters ###
    mean_div = []
    mean_div += (float(i) for i in args.mean_div_str.split('-'))
    label_dim = 101

    ### Make folders of logs and models ###
    image_size = (args.image_size, args.image_size)
    subdir = args.subdir
    model_dir = os.path.join(args.train_url, 'models', subdir)
    outputs_dir = os.path.join(args.train_url, 'outputs', subdir)
    scores_dir = os.path.join(args.train_url, 'scores', subdir)

    ### Load data ###
    data_url = args.data_url + 'Cropped_91_145/'
    list_file = args.data_url + 'txt/pts_txt/morph_pts_80-20_test.txt'
    data_test = data.get_ped_dataset(data_url, list_file)
    test_image_list, test_label_list, test_lms_list = data.get_image_paths_and_labels(data_test)
    print('Test: num_images={}/num_ID={}'.format(len(test_image_list), len(test_label_list)))

    ### Create a new Graph and set it as the default ###
    # with tf.Graph().as_default():
    ### Construction  placeholder
    batch_size_p = tf.compat.v1.placeholder(tf.int32, name='batch_size')
    isTraining_p = tf.compat.v1.placeholder(tf.bool, name='isTraining')
    image_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3), name='image_batches')
    label_batch_p = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='labels')
    thetas_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 6, 2, 3), name='labels')  ### shape=(None, 6, 2, 3)

    ### Build the inference graph ###
    logits, pre_labels, total_loss, accuracy, MAE = \
        resnet_DHAA.construct_resnet(18, image_batch_p, thetas_batch_p, batch_size_p, label_batch_p, label_dim,
        is_training=isTraining_p, reuse=False)
    ### Create a saver and train_op, and save the last three models
    saver, _, _, _ = utils.get_saver_resnet_tf()
    ### Start running operations on the Graph.
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)
    def evaluate_score(trainedModal, fid):
        with sess.as_default():
            ### Load model ###
            print('load model:', trainedModal)
            saver.restore(sess, trainedModal)
            print('Running Testing')
            ### Get output tensor
            batch_size = args.batch_size
            image_list = test_image_list
            label_list = test_label_list
            lms_list = test_lms_list
            dataset_test = Dataset(args, image_list, label_list, lms_list, mode='test', test_flip=args.test_flip)
            MAE_test = utils.test_Resnet_Race(sess, image_list, dataset_test, label_batch_p, isTraining_p, batch_size_p,
                thetas_batch_p, image_batch_p, batch_size, mean_div, logits, pre_labels, accuracy, MAE, total_loss,
                outputs_dir, args.phase, fid, args.data_name, is_training=False, test_flip=args.test_flip)
            fid.write('----------results--------\n')
            fid.write('MAE: %.2f\t'%(MAE_test))
            print('MAE: %.2f\t'%(MAE_test))

    def offline_eval():
        all_path_ckpt = os.path.join(model_dir, 'checkpoint')
        min_value = 100
        while True:
            if not os.path.exists(all_path_ckpt):continue
            else:break
        iter_before = start_iteration
        fid = open(all_path_ckpt, 'r')
        lines = fid.readlines()
        fid.close()
        fid = open(os.path.join(scores_dir, 'Test_scores.txt'), 'w')
        if len(lines) < 2:
            print('Hasn\'t enough model!')
            return 0
        for i in range(1, len(lines)):
            iter_now = int(lines[i].split('-')[-1][:-2])
            if iter_now - iter_before >= interval_iteration:
                trainedModal = os.path.join(model_dir, 'model-{}.ckpt-{}'.format(subdir, iter_now))
                evaluate_score(trainedModal, fid)
                # min_value = evaluate_metric(iter_now, min_value)
                iter_before = iter_now
                print('end one iteration!')
        fid.close()

    ### Set function ###
    isOnline = args.isOnline
    if isOnline:
        print('come in online')
    else:
        print('come in offline')
        start_iteration = 38
        interval_iteration = 1
        offline_eval()
        sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--isOnline", type=int, default=0)
    parser.add_argument("--train_url", type=str, default='/home/work/modelarts/outputs/Jobs/')
    parser.add_argument("--data_url", type=str, default='/home/work/modelarts/inputs/morph/')

    parser.add_argument("--subdir", type=str, default='2021-5-25')
    parser.add_argument('--data_name', type=str, default='MORPH')
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--mean_div_str", type=str, default='127.5-1.0')
    parser.add_argument("--data_augment_str", type=str, default='0-0-0-0-0-0', help='[0]:max_angle [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std [5]:RANDOM ERASING')
    parser.add_argument("--color_para_str", type=str, default='8-0.2-0.005', help='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_str", type=str, default='0-0')
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--test_preprocess_threads", type=int, default=1)
    parser.add_argument("--test_flip", type=bool, default=False)
    parser.add_argument("--max_nrof_epochs", type=int, default=1)
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
