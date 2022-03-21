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


import os
import argparse
import numpy as np
import pickle
from eval.utils import calculate_roc, calculate_tar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="/home/HwHiAiUser/arcface/evaluate_output", help="""set obs path""")
    parser.add_argument('--data_dir', default="/home/HwHiAiUser/arcface/dataset", help="""set obs path""")
    return parser.parse_args()


def read_embeding(data_dir, output_dir):
    files = os.listdir(output_dir)
    leng = int(len(files) / 2)
    embeding_num = 0
    embeding_f_num = 0
    embeding = np.zeros(shape=[leng, 512], dtype=np.float32)
    embeding_f = np.zeros(shape=[leng, 512], dtype=np.float32)
    for file in files:
        if file.endswith(".bin"):
            if file.startswith("data_f"):
                tmp = np.fromfile(output_dir + '/' + file, dtype=np.float32)
                tmp_list = file.split('_')
                tmp_num = int(tmp_list[2])
                embeding_f[tmp_num] = tmp
                embeding_f_num += 1
            else:
                tmp = np.fromfile(output_dir + '/' + file, dtype=np.float32)
                tmp_list = file.split('_')
                tmp_num = int(tmp_list[1])
                embeding[tmp_num] = tmp
                embeding_num += 1
    bins, issame_list = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    return np.array(embeding), np.array(embeding_f), issame_list


def evaluate(embeddings, actual_issame, far_target=1e-3, distance_metric=0, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    if distance_metric == 1:
        thresholdes = np.arange(0, 1, 0.0025)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                       distance_metric=distance_metric, nrof_folds=nrof_folds)
    tar, tar_std, far = calculate_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
                                      far_target=far_target, distance_metric=distance_metric, nrof_folds=nrof_folds)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    return tpr, fpr, acc_mean, acc_std, tar, tar_std, far


if __name__ == '__main__':
    args = get_args()
    # tmp = np.fromfile('/home/HwHiAiUser/arcface/data_bin/lfw2/lfw_0.bin', dtype=np.float32)
    # tmp2 = np.fromfile('/home/HwHiAiUser/arcface/evaluate_output/20210812_151335/lfw_0_output_0.bin', dtype=np.float32)
    val_data = {'agedb_30': 'agedb_30.bin',
                'lfw': 'lfw.bin',
                'cfp_ff': 'cfp_ff.bin',
                'cfp_fp': 'cfp_fp.bin',
                'calfw': 'calfw.bin',
                'cplfw': 'cplfw.bin',
                'vgg2_fp': 'vgg2_fp.bin'}
    print('evaluating...')
    for k, v in val_data.items():
        data_dir = os.path.join(args.data_dir, v)
        output_dir = os.path.join(args.output_dir, k)
        filelist = os.listdir(output_dir)
        output_dir = os.path.join(output_dir, filelist[0])
        embds_arr, embds_f_arr, issame = read_embeding(data_dir, output_dir)
        embds_arr = embds_arr / np.linalg.norm(embds_arr, axis=1, keepdims=True) + embds_f_arr / np.linalg.norm(
            embds_f_arr, axis=1, keepdims=True)
        tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds_arr, issame, far_target=1e-3, distance_metric=0)
        print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (k, acc_mean, acc_std, tar, tar_std, far))
    print('done!')
