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

# 通过加载已经训练好的pb模型，执行推理
import tensorflow as tf
import os
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
import time
import numpy as np
import cv2
import sys
sys.path.append("./")
import numpy as np
import mxnet as mx
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from preproccessing.dataset_proc import gen_face, gen_boundbox

MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)
 
def parse_args():
    '''
    用户自定义模型路径、输入、输出
    :return:
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', default='./c3ae_npu_train_v2.pb',
                        help="""pb path""")
    parser.add_argument('--image_path', default='./dataset/wiki_crop/00/81800_1986-06-13_2011.jpg',
                        help="""the data path""")
    parser.add_argument('--input_tensor_name', default='input_2:0',
                        help="""input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='output_1:0',
                        help="""output_tensor_name""")
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

def image_process(img, save_image=False):
    try:
        bounds, lmarks = gen_face(MTCNN_DETECT, img)
        ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)
    except Exception as ee:
        ret = None
        print(img.shape, ee)
    if not ret:
        print("no face")
        return img, None
    padding = 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    bounds, lmarks = bounds, lmarks

    colors = [(0, 0, 255), (0, 0, 0), (255, 0, 0)]
    for pidx, (box, landmarks) in enumerate(zip(bounds, lmarks)):
        trible_box = gen_boundbox(box, landmarks)
        tri_imgs = []
        for bbox in trible_box:
            bbox = bbox + padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            #cv2.imwrite("test.jpg", new_bd_img[w_min:w_max, h_min:h_max, :])
            tri_imgs.append([cv2.resize(new_bd_img[w_min:w_max, h_min:h_max, :], (64, 64))])

        for idx, pbox in enumerate(trible_box):
            pbox = pbox + padding
            h_min, w_min = pbox[0]
            h_max, w_max = pbox[1]
            new_bd_img = cv2.rectangle(new_bd_img, (h_min, w_min), (h_max, w_max), colors[idx], 2)

    return tri_imgs

class Classifier(object):
    #set batchsize:
    args = parse_args()

    def __init__(self):

        # 昇腾AI处理器模型编译和优化配置
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 配置1： 选择在昇腾AI处理器上执行推理run on Ascend NPU
        custom_op.parameter_map["use_off_line"].b = True
        # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
        custom_op.parameter_map["graph_run_mode"].i = 0
        # 配置4：关闭remapping和MemoryOptimizer
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # 加载模型，并指定该模型的输入和输出节点
        args = parse_args()
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
        self.sess = tf.Session(config=config, graph=self.graph)

    def __load_model(self, model_file):
        """
        load frozen graph
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
        """
        do infer
        :param image_data:
        :return:
        """
        
        out = self.sess.run(self.output_tensor, feed_dict={'input_2:0':data[0],'input_3:0':data[1],'input_4:0':data[2]})
           
        return out

    

def main():
    args = parse_args()
    top1_count = 0
    top5_count = 0
    ###data preprocess
    tf.reset_default_graph()
    print("########NOW Start Preprocess!!!#########")
    ###batch process
    print("########NOW Start Batch!!!#########")
    classifier = Classifier()
    ###start inference
    print("########NOW Start inference!!!#########")
    img = cv2.imread(args.image_path)
    data = image_process(img)
    out = classifier.do_infer(data)
    print(out)
    
if __name__ == '__main__':
    main()
