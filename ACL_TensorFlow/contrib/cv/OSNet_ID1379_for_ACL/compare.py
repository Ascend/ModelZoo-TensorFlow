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
import numpy as np
import os
from sklearn.preprocessing import normalize
from utils.ranking import cmc, mean_ap
from utils.rerank import re_ranking

query_path = './out/query'
gallery_path = './out/gallery'
query_files = os.listdir(query_path)
gallery_files = os.listdir(query_path)


def pairwise_distance(mat1, mat2):
    m = mat1.shape[0]  # query number
    n = mat2.shape[0]  # gallery number
    x = np.repeat(np.sum(np.square(mat1), axis=1, keepdims=True), n, axis=1)  # mxn
    y = np.repeat(np.sum(np.square(mat2), axis=1, keepdims=True), m, axis=1)  # nxm
    y = np.transpose(y)  # mxn
    return x + y - 2 * np.dot(mat1, mat2.T)


def evaluate(query_features, query_labels, query_cams, gallery_features, gallery_labels,
             gallery_cams):
    # query_feature: array, NxD
    # query_cam: array, 1xN
    # query_label: array, 1xN
    # gallery_feature: array, MxD
    # gallery_cam：array, 1xM
    # gallery_label array, 1xM
    distmat = pairwise_distance(query_features, gallery_features)

    print('Applying person re-ranking ...')
    distmat_qq = pairwise_distance(query_features, query_features)
    distmat_gg = pairwise_distance(gallery_features, gallery_features)
    distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    # Compute mean AP
    mAP = mean_ap(distmat, query_labels, gallery_labels, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_labels, gallery_labels, query_cams,
                            gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    cmc_topk = (1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))


def get_data_information(data_root):
    img_path_list = []
    img_name_list = []
    img_cams_list = []
    image_names = os.listdir(data_root)  # the best way is to use sorted list,i.e., sorted()
    image_names = sorted(image_names)[:-1]
    for item in image_names:
        if item[-4:] == '.jpg':
            img_path_list.append(os.path.join(data_root, item))
            img_name_list.append(item.split('_')[0])
            img_cams_list.append(item.split('c')[1][0])
    return img_path_list, np.array(img_name_list), np.array(img_cams_list)


query_img_list, query_name_list, query_cams_list = get_data_information('Market-1501-v15.09.15/query')
gallery_img_list, gallery_name_list, gallery_cams_list = get_data_information('Market-1501-v15.09.15/bounding_box_test')

query_files = os.listdir(query_path)
# query_files.sort(key=lambda x: int(x.split('_')[1]))
gallery_files = os.listdir(gallery_path)
# gallery_files.sort(key=lambda x: int(x.split('_')[1]))
query_results = np.zeros((3368, 512))
gallery_results = np.zeros((19732, 512))

for i, filename in enumerate(query_files):
    path = os.path.join(query_path, filename)
    f = np.loadtxt(path)
    query_results[i, :] = f

query_results = normalize(query_results, norm='l2')

for i, filename in enumerate(gallery_files):
    path = os.path.join(gallery_path, filename)
    f = np.loadtxt(path)
    gallery_results[i, :] = f

gallery_results = normalize(gallery_results, norm='l2')

evaluate(query_results, query_name_list, query_cams_list,
         gallery_results, gallery_name_list, gallery_cams_list)






