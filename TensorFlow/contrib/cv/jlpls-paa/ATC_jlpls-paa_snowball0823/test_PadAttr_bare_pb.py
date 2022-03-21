"""
Function: Huawei project for JLPLS-PAA project for Bare Metal
Author: ZiChang Tan
Date: 2021.6.1
BiB:
@article{tan2019attention,
author = {Tan, Zichang and Yang, Yang and Wan, Jun and Wan, Hanyuan and Guo, Guodong and Li, Stan},
year = {2019},
month = {07},
pages = {1-1},
title = {Attention-Based Pedestrian Attribute Analysis},
journal = {IEEE Transactions on Image Processing},
}
"""
import tensorflow as tf
import numpy as np
from numpy import zeros,ones
import os, time, argparse, sys
import util.utils as utils
from util.utils import Dataset, instance_evaluation
import util.resnet_JLPLS as resnet_JLPLS

os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

def main(args):
 

    ### Setting Parameters ###
    mean_div = []
    mean_div += (float(i) for i in args.mean_div_str.split('-'))
    # label dim has been tranformed in the PB model
    label_dim = 35

    ### Make folders of logs and models ###
    subdir = args.subdir ## datetime.strftime(datetime.now(), args.subdir)
    scores_dir = os.path.join(args.train_url, 'scores', subdir)
    ### Load data ###
    data_url = args.data_url + 'PETA_dataset/'
    list_file = args.data_url + 'txt/peta_test.txt'
    data_test = utils.get_ped_dataset(data_url, list_file)
    test_image_list, test_label_list = utils.get_image_paths_and_labels(data_test)
    print('Test: num_images={}/num_ID={}'.format(len(test_image_list), len(test_label_list)))
    
    # use pb model
    saved_model = '/root/code/LAJ_PED/pbModel/JSPJAA_tf.pb'
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(saved_model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            fid = open(os.path.join(scores_dir, 'Test_scores.txt'), 'w')
            # input tensor
            image_input_tensor = sess.graph.get_tensor_by_name("image_batches:0")
            # output tensor
            output_tensor = sess.graph.get_tensor_by_name("truediv:0")
            batch_size = args.batch_size
            image_list = test_image_list
            label_list = test_label_list
            dataset_test = Dataset(args, image_list, label_list, mode='test', test_flip=args.test_flip)
            print('Running forward pass on evaluate set')
            AllN_s = len(image_list)
            nrof_batches = AllN_s // batch_size
            nrof_images = nrof_batches * batch_size
            # AllN = nrof_images * 2 if test_flip else nrof_images
            test_batches = nrof_batches
            pre_labels_array = np.zeros((nrof_images, 35), np.float32)
            labels_array = np.zeros((nrof_images, 35), np.float32)
            ### Testing loop
            start_idx = 0
            for batch_it in range(test_batches):  # num of video
                image_batch, label_batch, filename_batch = sess.run(list(dataset_test.nextit))
                '''
                # =============================
                inner_path = filename_batch[0].decode(encoding='UTF-8').split('/')[-3:]
                inner_path[-1] = inner_path[-1].split('.')[0]
                out_path = os.path.join('./data_bin/{}/'.format(args.phase), '_'.join(inner_path) + ".bin")
                bin_path = os.path.split(out_path)[0]
                if not os.path.exists(bin_path):
                    os.makedirs(bin_path)
                try:
                    image_batch.tofile(out_path)
                except Exception as err:
                    print(out_path)
                    print(type(image_batch))
                    print('Error: '+str(err))
                    input()
                # print(inner_path)
                # input()
                # =============================
                '''
                feed_dict = {image_input_tensor: image_batch}
                logits_ = sess.run(output_tensor, feed_dict=feed_dict)
                pre_labels_ = logits_
                pre_labels_ = pre_labels_ > 0
                c_n = pre_labels_.shape[0]
                # pre_labels_array[start_idx:start_idx+c_n] = pre_labels
                pre_labels_array[start_idx:start_idx + c_n] = pre_labels_
                labels_array[start_idx:start_idx + c_n] = label_batch
                start_idx += c_n
                if batch_it % 10 == 0:
                    print('.', end='')
                    sys.stdout.flush()
            print('')
            Result = instance_evaluation(pre_labels_array, labels_array)
            MeanAcc = 100*Result['accuracy_mean']
            InsACC = 100*Result['instance_accuracy']
            InsRecall = 100*Result['instance_recall']
            InsPrec = 100*Result['instance_precision']
            InsF1 = 100*Result['instance_F1']
            fid.write('----------results--------\n')
            fid.write('accuracy_mean: %.2f\tinstance_accuracy: %.2f\tinstance_recall: %.2f\tinstance_precision: %.2f\tinstance_F1: %.2f\t'
                       %(MeanAcc, InsACC, InsRecall, InsPrec, InsF1))
            print('accuracy_mean: %.2f\tinstance_accuracy: %.2f\tinstance_recall: %.2f\tinstance_precision: %.2f\tinstance_F1: %.2f\t'
                       %(MeanAcc, InsACC, InsRecall, InsPrec, InsF1))
            print('Test Ending!')
            fid.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='1')
    parser.add_argument("--isOnline", type=int, default=0)
    parser.add_argument("--train_url", type=str, default='/home/test_user06/LAJ_PED/Jobs_Ped/')
    parser.add_argument("--data_url", type=str, default='/home/test_user06/LAJ_PED/PETA/')

    parser.add_argument("--subdir", type=str, default='2021-6-1')
    parser.add_argument('--data_name', type=str, default='PETA')
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--mean_div_str", type=str,default='127.5-1.0')
    parser.add_argument("--data_augment_str", type=str, default='0-0-0-0-0-0',
                        help ='[0]:max_angle [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std [5]:RANDOM ERASING')
    parser.add_argument("--color_para_str", type=str, default='8-0.2-0.005', help ='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_str", type=str, default='0-0')
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--test_preprocess_threads", type=int, default=1)
    parser.add_argument("--test_flip", type=bool, default=False)
    parser.add_argument("--max_nrof_epochs", type=int, default=1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
