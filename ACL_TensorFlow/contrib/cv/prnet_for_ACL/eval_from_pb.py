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
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
# from api import PRN
import cv2
import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import npu_bridge


def NME(S_bbox, infer, label):
    if label.shape[1] == 2:
        error = np.mean(np.sqrt((label[:, 0] - infer[:, 0]) ** 2 + (label[:, 1] - infer[:, 1]) ** 2))
    elif label.shape[1] == 3:
        error = np.mean(np.sqrt((label[:, 0] - infer[:, 0]) ** 2 + (label[:, 1] - infer[:, 1]) ** 2 + (label[:, 2] - infer[:, 2]) ** 2))
    nme = error / S_bbox
    # print(nme*1000)
    return nme


def to255(arr):
    tmp = np.zeros(arr.shape, dtype=np.uint8)
    index_less_equal_one = arr <= 1
    index_greater_one = arr > 1
    tmp[index_less_equal_one] = arr[index_less_equal_one] * 255
    tmp[index_greater_one] = arr[index_greater_one]

    return tmp


class Infer:

    def __init__(self, args):
        # --------------------------------------------------------------------------------
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        if "autotune" in os.environ:
            print ("autotune value is " + os.environ["autotune"])
            if os.environ["autotune"] == "True":
                print ("autotune is set !")
                custom_op.parameter_map["autotune_tune_mode"].s = tf.compat.as_bytes("RL,GA")
                print ("autotune set success")
            else:
                print ("autotune is not set!")
                
        custom_op.name = "NpuOptimizer"
        # 1???run on Ascend NPU
        custom_op.parameter_map["use_off_line"].b = True

        # 2???recommended use fp16 datatype to obtain better performance
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

        # 3???disable remapping
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        # 4???set graph_run_mode=0???obtain better performance
        custom_op.parameter_map["graph_run_mode"].i = 0
        # --------------------------------------------------------------------------------

        # load model??? set graph input nodes and output nodes
        self.uv_kpt_ind = np.loadtxt('Data/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # create session
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load fronzen graph
        :param model_file:
        :return:
        """
        with tf.gfile.GFile(model_file, "rb") as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        return graph

    def do_infer(self, data):
        out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: data[np.newaxis, :, :, :]})
        out = np.squeeze(out)
        return out * 256 * 1.1

    def get_landmarks(self, pos):
        kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        return kpt



def main(args):
    tf.reset_default_graph()
    infer = Infer(args)

    # ---- init PRN
    # prn = PRN(is_dlib=args.isDlib)

    # ------------- load data
    image_folder = args.inputDir

    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    nme = 0.0  # normalized mean error

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split('/')[-1][:-4]
        # read image
        image = cv2.imread(image_path)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:, :, :3]
        
        pos = infer.do_infer(image / 255.)  # input image has been cropped to 256x256
        S = image.shape[0] * image.shape[1]
        image = image / 255.

        if pos is None:
            continue

        if args.isKpt:
            # get predict landmarks
            predict_kpt = infer.get_landmarks(pos)
            label_kpt = np.loadtxt(os.path.join(image_folder, name + '.txt'))
            error = NME(S, predict_kpt, label_kpt)
            nme += error
            print(name + '\t:\t' + str(error*1000))
    
    nme /= total_num
    print("========================NME(%)========================")
    print(nme*1000)
    print("========================NME(%)========================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='Dataset/TestData/LFPA/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('--isDlib', default=False, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image '
                             'should be cropped in advance')
    parser.add_argument('--isKpt', default=True, type=ast.literal_eval,help='whether to output key points(.txt)')

    parser.add_argument('--batchsize', default=1,help="""batchsize""")
    parser.add_argument('--model_path', default='model/prnet.pb',help="""pb path""")
    parser.add_argument('--input_tensor_name', default = 'input:0',help = """input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='resfcn256/Conv2d_transpose_16/Sigmoid: 0',help="""output_tensor_name""")

    main(parser.parse_args())
