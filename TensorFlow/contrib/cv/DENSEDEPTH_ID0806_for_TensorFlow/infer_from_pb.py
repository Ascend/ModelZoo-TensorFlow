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

# 通过加载已经训练好的pb模型，执行推理
import tensorflow as tf
import os
import argparse
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import time
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from skimage.transform import resize


def parse_args():
    '''
    用户自定义模型路径、输入、输出
    :return:
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', default=1,
                        help="""batchsize""")
    parser.add_argument('--model_path', default=r'./result/pb/test.pb',
                        help="""pb path""")
    parser.add_argument('--image_path', default='./dataset/nyu_test.zip',
                        help="""the data path""")
    parser.add_argument('--input_tensor_name', default='input_1:0',
                        help="""input_tensor_name""")
    parser.add_argument('--output_tensor_name', default='conv3/BiasAdd:0',
                        help="""output_tensor_name""")
    parser.add_argument('--minDepth', type=float, default=10.0, help='Minimum of input depths')
    parser.add_argument('--maxDepth', type=float, default=1000.0, help='Maximum of input depths')

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def read_file(path):
    """
    :param path:
    :return:
    """

    data = extract_zip(path)
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))

    return rgb, depth, crop


def normalize(inputs):
    """
    图像归一化
    :param inputs:
    :return:
    """
    pass


def DepthNorm(x, maxDepth):
    return maxDepth / x


def scale_up(scale, images):
    scaled = []
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def image_process(image_path):
    """
    对输入图像进行一定的预处理
    :param image_path:
    :return:
    """
    rgb, depth, crop = read_file(image_path)
    rgb = rgb / 255

    rgb = rgb.astype(np.float32)
    depth = depth.astype(np.float32)

    images_count = len(rgb)

    return rgb, depth, crop, images_count


class Classifier(object):
    # set batchsize:
    args = parse_args()
    batch_size = int(args.bs)

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
        # 配置4：关闭remapping和MemoryOptimzer
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # 加载模型，并指定该模型的输入和输出节点
        args = parse_args()
        self.maxDepth = args.maxDepth
        self.minDepth = args.minDepth
        self.graph = self.__load_model(args.model_path)
        self.input_tensor = self.graph.get_tensor_by_name(args.input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(args.output_tensor_name)

        # 由于首次执行session run会触发模型编译，耗时较长，可以将session的生命周期和实例绑定
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

    def do_infer(self, batch_images):
        """
        do infer
        :param batch_images:
        :return:
        """
        out_list = []
        total_time = 0
        i = 0
        for images in batch_images:

            # Support multiple RGBs, one RGB image, even grayscale
            if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
            if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))

            t = time.time()
            out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: images})

            if i > 0:
                total_time = total_time + time.time() - t
            i = i + 1

            out = np.clip(DepthNorm(out, maxDepth=self.maxDepth), self.minDepth, self.maxDepth) / self.maxDepth

            out_list.append(out)

        # predictions = np.stack(np.array(out_list))
        # predictions = np.clip(DepthNorm(predictions, maxDepth=self.maxDepth), self.minDepth, self.maxDepth) / self.maxDepth

        return out_list, total_time

    def batch_process(self, rgb, depth):
        """
        batch
        :param rgb: rgb images
        :param depth:depth images
        :return:
        """

        # Get the batch information of the current input data, and automatically adjust the data to the fixed batch
        n_dim = rgb.shape[0]
        batch_size = self.batch_size

        # if data is not enough for the whole batch, you need to complete the data
        m = n_dim % batch_size
        if batch_size > m > 0:
            # The insufficient part shall be filled with images from rgb and depth to n dimension
            print("Advice:\nThe number of test pictures is {}\nbatch size is {}\nThe insufficient part({} images) "
                  "will be filled from existing images\n"
                  "You can change batch size to avoid the problem".format(n_dim, batch_size, m))
            np.random.seed(0)
            fill_idx = np.random.randint(n_dim - m)
            fill_rgb = rgb[fill_idx:fill_idx + m, ...]
            fill_depth = depth[fill_idx:fill_idx + m, ...]
            rgb = np.concatenate((rgb, fill_rgb), axis=0)
            depth = np.concatenate((depth, fill_depth), axis=0)

        # Define the Minis that can be divided into several batches
        mini_rgb = []
        mini_depth = []
        i = 0
        while i < n_dim:
            # Define the Minis that can be divided into several batches
            mini_rgb.append(rgb[i: i + batch_size, :, :, :])
            mini_depth.append(depth[i: i + batch_size, :, :])
            i += batch_size

        mini_rgb = np.asarray(mini_rgb)
        mini_depth = np.asarray(mini_depth)

        return mini_rgb, mini_depth

    def compute_process(self, pre, pred_flip, crop, depth):
        predictions = []
        testSetDepths = []
        for i in range(len(pre)):

            # Compute results
            true_y = depth[i]
            pred_y = scale_up(2, pre[i][:, :, :, 0]) * 10.0

            # Test time augmentation: mirror image estimate
            pred_y_flip = scale_up(2, pred_flip[i][:, :, :, 0]) * 10.0

            # Crop based on Eigen et al. crop
            true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

            # Compute errors per image in batch
            for j in range(len(true_y)):
                predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
                testSetDepths.append(true_y[j])

        predictions = np.stack(predictions, axis=0)
        testSetDepths = np.stack(testSetDepths, axis=0)

        return predictions, testSetDepths

    def compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        return a1, a2, a3, abs_rel, rmse, log_10


def main():
    args = parse_args()
    ###data preprocess
    tf.reset_default_graph()
    print("########NOW Start Preprocess!!!#########")
    rgb, depth, crop, images_count = image_process(args.image_path)
    ###batch process
    print("########NOW Start Batch!!!#########")
    classifier = Classifier()
    batch_rgb, batch_depth = classifier.batch_process(rgb, depth)
    ###start inference
    print("########NOW Start inference!!!#########")
    batch_pred, total_time = classifier.do_infer(batch_rgb)
    batch_pred_flip, total_time_flip = classifier.do_infer(batch_rgb[..., ::-1, :])

    ###compute accuracy

    print("########NOW Start Data Pretreatment!!!#########")
    predictions, testSetDepths = classifier.compute_process(batch_pred, batch_pred_flip, crop, batch_depth)

    batchsize = int(args.bs)
    total_step = int(images_count / batchsize)
    print("########NOW Start Compute Accuary!!!#########")
    e = classifier.compute_errors(predictions, testSetDepths)

    print('+----------------------------------------+')
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))
    print('images number = ', total_step * batchsize)
    print('images/sec = ', (total_step * batchsize) / (total_time + total_time_flip) / 2)
    print('+----------------------------------------+')


if __name__ == '__main__':
    main()
