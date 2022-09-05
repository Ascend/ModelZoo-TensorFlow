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
import tensorflow as tf
from tensorflow.python.platform import gfile
import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_path = './eval_data/test_img.jpg'

test_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_vis = test_img
test_img_trans = cv2.resize(test_img, (512, 256), interpolation=cv2.INTER_LINEAR)
test_img_trans = test_img_trans / 127.5 - 1.0

gt_path = './eval_data/gt.png'
gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
gt_trans = cv2.resize(gt, (512, 256), interpolation=cv2.INTER_LINEAR)

CFG = parse_config_utils.lanenet_cfg

pb_path = './pretrained_model/eval.pb'

# ----------- 查看 pbnode -----------
# read graph definition
f = gfile.FastGFile(pb_path, "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='')
for i, n in enumerate(graph_def.node):
    print("=====node====")
    print("Name of the node - %s" % n.name)
# --------------------------------


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def freeze_graph_test(pb_path, img_path):
    # :param pb_path:pb文件的路径
    # :param test_img:测试图片的路径
    # :return:

    f = gfile.FastGFile(pb_path, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # 定义输入输出节点名称
            input_node = sess.graph.get_tensor_by_name("input_tensor:0")
            binary_output_node = sess.graph.get_tensor_by_name("lanenet/binary_seg_out:0")
            pixel_embedding_output_node = sess.graph.get_tensor_by_name("lanenet/instance_seg_out:0")
            img_feed = test_img_trans.astype(np.float32)
            img_feed = np.expand_dims(img_feed, axis=0)

            # img_bin = img_feed
            # img_bin.tofile("./eval_data/test_img.bin")

            binary_output, pixel_embedding_output = sess.run([binary_output_node, pixel_embedding_output_node],
                                                             feed_dict={input_node: img_feed})

            postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_output[0],
                instance_seg_result=pixel_embedding_output[0],
                source_image=image_vis,
                with_lane_fit=True,
                data_source='tusimple'
            )
            mask_image = postprocess_result['mask_image']
            src_image = postprocess_result['source_image']
            # if with_lane_fit:
            #     lane_params = postprocess_result['fit_params']
            #     LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
            #     for i in range(len(lane_params)):
            #         LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

            for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                pixel_embedding_output[:, :, i] = minmax_scale(pixel_embedding_output[:, :, i])
            embedding_image = np.array(pixel_embedding_output, np.uint8)

            # plt.figure('mask_image')
            # plt.imshow(mask_image[:, :, (2, 1, 0)])
            # plt.show()
            # plt.figure('src_image')
            # plt.imshow(image_vis[:, :, (2, 1, 0)])
            # plt.figure('instance_image')
            # plt.imshow(embedding_image[:, :, (2, 1, 0)])
            # plt.figure('binary_image')
            # plt.imshow(binary_output * 255, cmap='gray')
            # plt.show()

            # -------------- 计算准确率 ------------------ #
            gt_gray = cv2.cvtColor(gt_trans, cv2.COLOR_BGR2GRAY)
            mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            WIDTH = mask_image_gray.shape[0]
            HIGTH = mask_image_gray.shape[1]
            tp_count = 0
            tn_count = 0
            for i in range(WIDTH):
                for j in range(HIGTH):
                    if mask_image_gray[i, j] != 0 and gt_gray[i, j] != 0:
                        tp_count = tp_count + 1
                    if mask_image_gray[i, j] == 0 and gt_gray[i, j] == 0:
                        tn_count = tn_count + 1
            Accuracy = (int(tp_count) + int(tn_count)) / (int(WIDTH) * int(HIGTH))

            print("\n# Metric_pb "
                  "\n     Accuracy：{:.3f}".format(Accuracy))

            cv2.imwrite('./eval_output/mask_pb.jpg', mask_image)
            cv2.imwrite('./eval_output/src_pb.jpg', src_image)


if __name__ == '__main__':
    # 测试pb模型
    img_path = './eval_data/test_img.npy'
    freeze_graph_test(pb_path=pb_path, img_path=img_path)
