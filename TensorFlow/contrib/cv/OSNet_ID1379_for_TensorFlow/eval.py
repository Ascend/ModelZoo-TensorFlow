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

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import SGD
from utils.triplet import generator_batch_test
from sklearn.preprocessing import normalize
import os
import argparse
import numpy as np
from osnet import OSNet
from utils.ranking import cmc, mean_ap
from utils.rerank import re_ranking


def pairwise_distance(mat1, mat2):
    m = mat1.shape[0]  # query number
    n = mat2.shape[0]  # gallery number
    x = np.repeat(np.sum(np.square(mat1), axis=1, keepdims=True), n, axis=1)  # mxn
    y = np.repeat(np.sum(np.square(mat2), axis=1, keepdims=True), m, axis=1)  # nxm
    y = np.transpose(y)  # mxn
    return x + y - 2 * np.dot(mat1, mat2.T)


def evaluate(query_features, query_labels, query_cams,  gallery_features, gallery_labels,
             gallery_cams):
    #query_feature: array, NxD
    #query_cam: array, 1xN
    #query_label: array, 1xN
    #gallery_feature: array, MxD
    #gallery_cam：array, 1xM
    #gallery_label array, 1xM
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
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_labels, gallery_labels, query_cams,
                            gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    cmc_topk = (1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))


def pairwise_distance(mat1, mat2):
    m = mat1.shape[0]  # query number
    n = mat2.shape[0]  # gallery number
    x = np.repeat(np.sum(np.square(mat1), axis=1, keepdims=True), n, axis=1)  # mxn
    y = np.repeat(np.sum(np.square(mat2), axis=1, keepdims=True), m, axis=1)  # nxm
    y = np.transpose(y)  # mxn
    return x + y - 2 * np.dot(mat1, mat2.T)


def evaluate(query_features, query_labels, query_cams,  gallery_features, gallery_labels,
             gallery_cams):
    #query_feature: array, NxD
    #query_cam: array, 1xN
    #query_label: array, 1xN
    #gallery_feature: array, MxD
    #gallery_cam：array, 1xM
    #gallery_label array, 1xM
    distmat = pairwise_distance(query_features, gallery_features)

    print('Applying person re-ranking ...')
    distmat_qq = pairwise_distance(query_features, query_features)
    distmat_gg = pairwise_distance(gallery_features, gallery_features)
    distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    # Compute mean AP
    mAP = mean_ap(distmat, query_labels, gallery_labels, query_cams, gallery_cams)
    print('Final Accuracy accuracy: {:4.1%}'.format(mAP))

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_labels, gallery_labels, query_cams,
                            gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    cmc_topk = (1, 5, 10)
    print('CMC Scores:')
    for k in cmc_topk:
        print('top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))


def get_data_information(data_root):
    img_path_list = []
    img_name_list = []
    img_cams_list = []
    image_names = os.listdir(data_root) #the best way is to use sorted list,i.e., sorted()
    image_names = sorted(image_names)[:-1]
    for item in image_names:
        if item[-4:] == '.jpg':
            img_path_list.append(os.path.join(data_root, item))
            img_name_list.append(item.split('_')[0])
            img_cams_list.append(item.split('c')[1][0])
    return img_path_list, np.array(img_name_list), np.array(img_cams_list)


def main():
    args = parser.parse_args()
    # build model to extract features
    model = OSNet(751).model
    print("done")
    print(model.summary())
    model.load_weights("osnet.h5")
    model.summary()
    dense_feature = model.get_layer('features').output
    model_extract_features = Model(inputs=model.input, outputs=dense_feature)
    model_extract_features.compile(loss=['categorical_crossentropy'], optimizer=SGD(lr=0.1), metrics=['accuracy'])

    # image_path, image_names, image_cams
    query_dir = os.path.join(args.data_path, 'query')
    gallery_dir = os.path.join(args.data_path, 'bounding_box_test')
    query_img_list, query_name_list, query_cams_list = \
        get_data_information(query_dir)
    gallery_img_list, gallery_name_list, gallery_cams_list = \
        get_data_information(gallery_dir)

    # obtain features
    query_generator = generator_batch_test(query_img_list, args.img_width,
            args.img_height, args.batch_size, shuffle=False)
    query_features = model_extract_features.predict(query_generator, verbose=1,
            steps=len(query_img_list)//args.batch_size if len(query_img_list)%args.batch_size==0 \
            else len(query_img_list)//args.batch_size+1)
    query_features = normalize(query_features, norm='l2')
    assert len(query_img_list) == query_features.shape[0], "something wrong with query samples"

    gallery_generator = generator_batch_test(gallery_img_list, args.img_width, args.img_height,
                                             args.batch_size, shuffle=False)
    gallery_features = model_extract_features.predict(gallery_generator,verbose=1,
            steps=len(gallery_img_list)//args.batch_size if len(gallery_img_list)%args.batch_size==0 \
            else len(gallery_img_list)//args.batch_size+1)
    gallery_features = normalize(gallery_features, norm='l2')
    assert len(gallery_img_list) == gallery_features.shape[0], "something wrong with gallery samples"
    #evaluate
    evaluate(query_features, query_name_list, query_cams_list,
             gallery_features, gallery_name_list, gallery_cams_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--data_path', help='path to train_image', type=str, default='/home/dingwei/osnet/dataset/Market-1501-v15.09.15')
    parser.add_argument('--img_width', type=int, default='64')
    parser.add_argument('--img_height', type=int, default='128')
    parser.add_argument('--learning_rate', type=float, default='0.01')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--USE_Label_Smoothing', type=bool, default=True)
    main()