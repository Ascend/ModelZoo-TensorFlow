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
"""
Function: Input(data_bin) -> Inference(.pb) -> Result(ACER)
Date: 2021.7.2
Author: AJ
"""
import os, sys, copy, argparse
import numpy as np
from sklearn.metrics import roc_curve
import tensorflow as tf

def video_2_label(video_name):
    label = int(video_name.split('_')[-1])
    label = 0 if label == 1 else 1
    data_name = video_name.split('_')[0]
    return label, data_name

def metric_zcx(scores_dev, labels_dev, scores_test, labels_test):
    def get_eer_threhold(fpr, tpr, threshold):
        differ_tpr_fpr_1 = tpr + fpr - 1.0
        right_index = np.argmin(np.abs(differ_tpr_fpr_1))
        best_th = threshold[right_index]
        eer = fpr[right_index]
        return eer, best_th
    scores_dev = np.array(scores_dev)
    labels_dev = np.array(labels_dev)
    scores_test = np.array(scores_test)
    labels_test = np.array(labels_test)

    dev_fpr, dev_tpr, dev_thre = roc_curve(labels_dev, scores_dev, pos_label=1)
    dev_eer, dev_best_thre = get_eer_threhold(dev_fpr, dev_tpr, dev_thre)

    real_scores = scores_test[labels_test == 1]
    attack_scores = scores_test[labels_test == 0]
    APCER = np.mean(np.array(attack_scores >= dev_best_thre, np.float32))
    BPCER = np.mean(np.array(real_scores < dev_best_thre, np.float32))

    test_fpr, test_tpr, test_threshold = roc_curve(labels_test, scores_test, pos_label=1)
    test_EER, test_best_thre = get_eer_threhold(test_fpr, test_tpr, test_threshold)
    results = [test_EER, APCER, BPCER, (APCER + BPCER) / 2.0]
    return dev_best_thre, results

### <real:label==1, fake:label==0> (metric)
def performance(phases, scores_dir, score_ind=1):
    '''
    :param phases: ['train', 'dev', 'test']
    :param score_ind: 1:prob_score 2:exp_score
    '''
    predicts_test = []
    predicts_dev = []
    scores_test = []
    labels_test = []
    scores_dev = []
    labels_dev = []
    for phase in phases[1:]:
        locals()['predicts_' + phase] = []
        score_fid = open(os.path.join(scores_dir, phase.capitalize() + '_scores.txt'), 'r')
        lines = score_fid.readlines()
        score_fid.close()
        for line in lines:
            label = 1 - int(line.split(',')[0].split('_')[-1])
            locals()['predicts_' + phase].append([line.split(',')[0], float(line.split(',')[score_ind]), label])
            locals()['scores_' + phase].append(float(line.split(',')[score_ind]))  ### '1_275_1_1_1_G(0)_0001.jpg@GT_0'
            locals()['labels_' + phase].append(int(label))

    Dev_best_thre, results = metric_zcx(scores_dev, labels_dev, scores_test, labels_test)
    print('@_ZCX: Dev_best_thre={} EER={} APCER={} BPCER={} ACER={}'.
          format(Dev_best_thre, results[0], results[1], results[2], results[3]))

    return Dev_best_thre, results[1], results[2], results[3]

def evaluate_metric(scores_dir, phases=['train', 'dev', 'test']):
    for score_ind in range(2, 3):
        if score_ind == 1: score_type = 'logit_score'
        else: score_type = 'depth_score'
        ### <score_ind==1:prob_score score_ind==2:exp_score>
        Dev_best_thre, APCER, NPCER, ACER = performance(phases, scores_dir, score_ind)
        result_txt = os.path.join(scores_dir, 'result.txt')
        ### Write results ###
        if not os.path.exists(result_txt):
            lines = []
        else:
            fid = open(result_txt, 'r')
            lines = fid.readlines()
            fid.close()
        str_line = '%s thre %.4f\tAPCER %.4f\tNPCER %.4f\tACER %.4f\t'%(
            score_type, Dev_best_thre, APCER*100, NPCER*100, ACER*100)
        fid = open(result_txt, 'w')
        line_new = str_line + '\tmodal_iter %d\n'%(42)
        lines.append(line_new)
        fid.writelines(lines)
        fid.close()

def write_ScoreImages(phase, bat_it, filename, label, depth_mean_div, logits, depth_map, accuracy, fid):

    ### parameters ###
    logit_acc_mean = 0.0
    depth_acc_mean = 0.0
    accuracy_mean = 0.0
    def realProb(logits):
        x = np.array(logits)
        if np.isinf(np.sum(np.exp(x))):
            return 0
        y = np.exp(x[0]) / np.sum(np.exp(x))
        return y

    assert depth_map.shape[-1] == 1
    depth_map_image = np.squeeze(depth_map, axis=-1)
    depth_map_image = (depth_map_image * depth_mean_div[1]) + depth_mean_div[0]

    ### compute depth score ###
    depth_map_image_norm = depth_map_image / 255.0  ### 0~1
    depth_binary = np.array(depth_map > depth_map.min(), np.float32)
    depth_mean_score = np.sum(depth_map_image_norm) / np.sum(depth_binary[..., 0])
    # assert np.min(depth_batch[frame_ind, :, :, :]) == 0
    # print(np.sum(depth_map_image_norm), np.sum(depth_binary[..., 0]), depth_mean_score)
    # assert depth_mean_score <= 20.0
    depth_score = [depth_mean_score, 1.0 - depth_mean_score]
    out = np.argmax(np.array(depth_score))
    depth_acc = int(out == label)
    depth_acc_mean += float(depth_acc)
    map_score = realProb(depth_score)
    ### compute logical score ###
    out = np.argmax(np.array(logits[0]))
    logit_acc = int(out == label)
    logit_acc_mean += float(logit_acc)
    accuracy_mean += accuracy
    logit_score = realProb(logits[0])
    ### write score ###
    fid.write(filename + '@GT_' + str(label) + ',' + str(logit_score) + ',' + str(map_score) + '\n')
    print('*{}/{}: depth_acc={} logit_acc/accuracy={}/{}'.format(
        phase, str(bat_it), str(depth_acc_mean / (0 + 1)), str(logit_acc_mean / (0 + 1)), str(accuracy_mean / (0 + 1))))


def main(args):
    #### Set GPU options ###
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    config.gpu_options.allow_growth = True

    ### Setting Parameters ###
    color_mean_div = []
    depth_mean_div = []
    alpha_beta_gamma = []
    color_mean_div += (float(i) for i in args.color_mean)
    depth_mean_div += (float(i) for i in args.depth_mean)
    alpha_beta_gamma += (float(i) for i in args.alpha_beta_gamma)
    label_dim = 2
    color_size = (args.color_image_size, args.color_image_size)
    depth_size = (args.depth_image_size, args.depth_image_size)
    ### Make folders of logs and models ###
    model_dir = os.path.join(args.train_url, 'models', args.protocol, args.subdir)
    outputs_dir = os.path.join(args.train_url, 'outputs', args.subdir)
    scores_dir = os.path.join(args.train_url, 'scores', args.protocol, args.subdir)
    ### Load data from different domain ###


    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(args.pb_model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ### Input ###
            input_tensor = sess.graph.get_tensor_by_name("color:0")
            # input_label_tensor = sess.graph.get_tensor_by_name("label:0")
            output_depth = sess.graph.get_tensor_by_name("DE_color/conv4_3/Conv2D:0")
            # output_logit = sess.graph.get_tensor_by_name("DE_color/fc_logit/dense/MatMul:0")
            # correct_prediction = \
            #     tf.cast(tf.equal(tf.argmax(output_logit, 1), tf.cast(input_label_tensor, tf.int64)), tf.float32)
            # output_accur = tf.reduce_mean(correct_prediction)

            for phase in args.phases[1:]:
                data_bin_folder = os.path.join(args.bin_root, '{}'.format(phase))
                dataset = os.listdir(data_bin_folder)
                if phase == 'dev':
                    fid = open(os.path.join(scores_dir, 'Dev_scores.txt'), 'w')
                elif phase == 'test':
                    fid = open(os.path.join(scores_dir, 'Test_scores.txt'), 'w')
                print('Running forward pass on evaluate set')
                bat_it = 0
                for data in dataset:
                    bat_it += 1
                    data_bin_path = os.path.join(data_bin_folder, data)
                    data_bin = np.fromfile(data_bin_path, dtype='float32').reshape([1, 256, 256, 3])
                    label, _ = video_2_label(data.split('.bin')[0])
                    feed_dict = {input_tensor: data_bin}
                    depth_map_ = sess.run(output_depth, feed_dict=feed_dict)
                    accuracy_ = 1
                    logits_ = [[0.5, 0.5]]
                    ### Generate intermediate results
                    write_ScoreImages(phase, bat_it, data, label, depth_mean_div, logits_, depth_map_, accuracy_, fid)
                fid.close()

    evaluate_metric(scores_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_model", type=str, default='./offline_inference/pbModel/depthNet_tf.pb')
    parser.add_argument("--bin_root", type=str, default='./offline_inference/data_bin')

    ### fixed paras
    parser.add_argument("--gpu_id", type=str, default='1')
    parser.add_argument("--train_url", type=str, default='/home/ajliu/LAJ/HUAWEI_v2/Jobs_Final/')
    parser.add_argument("--data_url", type=str, default='/home/ajliu/LAJ/pad_datasets/Oulu-300/')
    parser.add_argument("--subdir", type=str, default='001')
    parser.add_argument('--data_name', type=str, default='oulu')
    parser.add_argument("--protocol", type=str, default='oulu_protocal_2')
    parser.add_argument("--data_augment", type=list, default=[0, 0, 0, 0, 0],
                        help='[0]:max_angle [1]:RANDOM_FLIP [2]:RANDOM_CROP [3]:RANDOM_COLOR [4]:is_std')
    parser.add_argument('--net_name', type=str, default='facemap_tf', help='resnet_tf, facemap_tf')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--color_image_size", type=int, default=256)
    parser.add_argument("--depth_image_size", type=int, default=32)
    parser.add_argument("--max_nrof_epochs", type=int, default=1)
    parser.add_argument("--color_mean", type=list, default=[0.0, 127.5])
    parser.add_argument("--depth_mean", type=list, default=[0.0, 255.0])
    parser.add_argument("--disorder_para", type=list, default=[8, 0.2, 0.02], help='[0]:alpha [1]:beta [2]:gamma')
    parser.add_argument("--alpha_beta_gamma", type=list, default=[0.1, 1, 0.1])
    parser.add_argument("--phases", type=list, default=['train', 'dev', 'test'])
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
