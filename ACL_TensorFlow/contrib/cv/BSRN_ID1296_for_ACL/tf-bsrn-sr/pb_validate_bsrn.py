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
import argparse
import importlib
import os

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

DEFAULT_DATALOADER = 'basic_loader'
DEFAULT_MODEL = 'bsrn'

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
    tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
    tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
    tf.flags.DEFINE_string('scales', '2,3,4',
                           'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
    tf.flags.DEFINE_string('cuda_device', '-1',
                           'CUDA device index to be used in the validation. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify this to employ GPUs.')

    tf.flags.DEFINE_string('restore_path', None,
                           'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
    tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
    tf.flags.DEFINE_integer('restore_global_step', 0,
                            'Global step of the restored model. Some models may require to specify this.')
    tf.flags.DEFINE_string('obs_dir', "obs://bsrn-test/", "obs result path, not need on gpu and apulis platform")
    tf.flags.DEFINE_string('save_path', None,
                           'Base path of the upscaled images. Specify this to save the upscaled images.')
    tf.flags.DEFINE_string('train_path', './train/', 'Base path of the trained model to be saved.')

    tf.flags.DEFINE_integer('shave_size', 4,
                            'Amount of pixels to crop the borders of the images before calculating quality metrics.')
    tf.flags.DEFINE_boolean('ensemble_only', False, 'Calculate (and save) ensembled image only.')

    tf.flags.DEFINE_string("chip", "gpu", "Run on which chip, (npu or gpu or cpu)")
    tf.flags.DEFINE_string("platform", "linux", 'the platform this code is running on')

    # parse data loader and model first and import them
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
    pre_parser.add_argument('--model', default=DEFAULT_MODEL)
    pre_parsed = pre_parser.parse_known_args()[0]
    if (pre_parsed.dataloader is not None):
        DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
    if (pre_parsed.model is not None):
        MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)

if FLAGS.chip == 'npu':
    from npu_bridge.npu_init import *


def _clip_image(image):
    return np.clip(np.round(image), a_min=0, a_max=255)


def _shave_image(image, shave_size=4):
    return image[shave_size:-shave_size, shave_size:-shave_size]


def _fit_truth_image_size(output_image, truth_image):
    return truth_image[0:output_image.shape[0], 0:output_image.shape[1]]


def _image_psnr(output_image, truth_image):
    diff = truth_image - output_image
    mse = np.mean(np.power(diff, 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr


def _image_rmse(output_image, truth_image):
    diff = truth_image - output_image
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse


def _image_psnr2(output_image, truth_image):
    yr_out = 0.257 * output_image[:, :, 0] + 0.504 * output_image[:, :, 1] + 0.098 * output_image[:, :, 2] + 16.5
    yr_tr = 0.257 * truth_image[:, :, 0] + 0.504 * truth_image[:, :, 1] + 0.098 * truth_image[:, :, 2] + 16.5
    diff = yr_tr - yr_out
    mse = np.mean(np.power(diff, 2))
    psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr


def _image_psnr_tf(output_image, truth_image):
    return tf.image.psnr(output_image, truth_image, max_val=255)


def _image_ssim_tf(output_image, truth_image):
    return tf.image.ssim(output_image, truth_image, max_val=255)


def _image_rmse2(output_image, truth_image):
    yr_out = 0.257 * output_image[:, :, 0] + 0.504 * output_image[:, :, 1] + 0.098 * output_image[:, :, 2] + 16.5
    yr_tr = 0.257 * truth_image[:, :, 0] + 0.504 * truth_image[:, :, 1] + 0.098 * truth_image[:, :, 2] + 16.5
    diff = yr_tr - yr_out
    rmse = np.sqrt(np.mean(diff ** 2))
    return rmse

def _image_ssim(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim

def go():
    # initialize
    FLAGS.bsrn_intermediate_outputs = True
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
    scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
    tf.logging.set_verbosity(tf.logging.INFO)

    # data loader
    dataloader = DATALOADER_MODULE.create_loader()
    dataloader.prepare()

    # image saving session
    if (FLAGS.save_path is not None):
        tf_image_save_graph = tf.Graph()
        with tf_image_save_graph.as_default():
            tf_image_save_path = tf.placeholder(tf.string, [])
            tf_image_save_image = tf.placeholder(tf.float32, [None, None, 3])

            tf_image = tf_image_save_image
            tf_image = tf.round(tf_image)
            tf_image = tf.clip_by_value(tf_image, 0, 255)
            tf_image = tf.cast(tf_image, tf.uint8)

            tf_image_png = tf.image.encode_png(tf_image)
            tf_image_save_op = tf.write_file(tf_image_save_path, tf_image_png)

            tf_image_init = tf.global_variables_initializer()

            if FLAGS.chip == 'gpu':
                os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device  # set GPU:0
                # 设置set_session,与GPU有关
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                # config.gpu_options.per_process_gpu_memory_fraction = 0.3
                # 设置GPU显存按需增长
                # config.gpu_options.allow_growth = True
                tf_image_session = tf.compat.v1.Session(config=config)

            elif FLAGS.chip == 'npu':

                # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
                sess_config = tf.compat.v1.ConfigProto()
                custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
                custom_op.name = "NpuOptimizer"
                # 设置自动调优
                # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
                # if FLAGS.profiling:
                #   custom_op.parameter_map["profiling_mode"].b = True
                #   custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
                #     '{"output":"/home/HwHiAiUser/output","task_trace":"on"}')
                sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
                sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
                custom_op.parameter_map["dynamic_input"].b = True
                custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
                tf_image_session = tf.compat.v1.Session(config=sess_config)
            else:
                config = tf.compat.v1.ConfigProto()
                tf_image_session = tf.Session(config=config)

            tf_image_session.run(tf_image_init)

    # model
    # model = MODEL_MODULE.create_model()
    # model.prepare(is_training=False, global_step=FLAGS.restore_global_step)

    # model > restore
    # model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open("/mnt/data1/2021/hht/huawei/bsrn/frozen_model.pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("sr_input:0")
            input_image_scale = sess.graph.get_tensor_by_name("sr_input_scale:0")

            output = sess.graph.get_tensor_by_name("output:0")

            # 读取测试图片,此处使用假数据
            # im = np.random.randn(2, 60, 60, 3)

            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字

            # out = sess.run(output_list, feed_dict={input_image_tensor: im,
            #                                        input_image_scale: 4})
            # validate
            num_total_outputs = FLAGS.bsrn_recursions // FLAGS.bsrn_recursion_frequency
            modules_average_psnr_dict = {}
            modules_average_ssim_dict = {}
            # modules_average_ssim_dict = {}

            for scale in scale_list:
                modules_average_psnr_dict[scale] = []
                modules_average_ssim_dict[scale] = []
                # modules_average_ssim_dict[scale] = []

            num_images = dataloader.get_num_images()

            for scale in scale_list:
                psnr_list = []
                ssim_list = []
                # ssim_list = []
                for i in range(num_total_outputs + 1):
                    psnr_list.append([])
                    ssim_list.append([])
                    # ssim_list.append([])

                print(num_images)
                for image_index in range(num_images):
                    input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index,
                                                                                     scale=scale)

                    out = sess.run(output, feed_dict={input_image_tensor: [input_image],
                                                      input_image_scale: 4})
                    output_images = np.array_split(out, 16, axis=0)

                    # print(output_images)
                    # print("----------------------------------------------------------")
                    # print("input type:",type(input_image),"shape: ",input_image.shape)
                    # print(type(output_images))
                    # print(len(output_images))
                    # for output_img in output_images:
                    #     print(type(output_img), output_img.shape)
                    output_image_ensemble = np.zeros_like(output_images[0][0])
                    ensemble_factor_total = 0.0

                    for i in range(num_total_outputs):
                        num_recursions = (i + 1) * FLAGS.bsrn_recursion_frequency
                        output_image = output_images[i][0]

                        ensemble_factor = 1.0 / (2.0 ** (num_total_outputs - num_recursions))
                        output_image_ensemble = output_image_ensemble + (output_image * ensemble_factor)
                        ensemble_factor_total += ensemble_factor

                        if not FLAGS.ensemble_only:
                            if FLAGS.save_path is not None:
                                output_image_path = os.path.join(FLAGS.save_path, 't%d' % (num_recursions),
                                                                 'x%d' % (scale),
                                                                 os.path.splitext(image_name)[0] + '.png')
                                tf_image_session.run(tf_image_save_op, feed_dict={tf_image_save_path: output_image_path,
                                                                                  tf_image_save_image: output_image})

                            truth_image = _clip_image(truth_image)
                            output_image = _clip_image(output_image)

                            truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

                            truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
                            output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

                            psnr = _image_psnr2(output_image=output_image_shaved, truth_image=truth_image_shaved)
                            ssim = _image_ssim(output_image_shaved,truth_image_shaved)
                            # ssim = _image_ssim_tf(output_image=output_image_shaved, truth_image=truth_image_shaved)
                            # print("----------------------------------------------")
                            # print("_____________________________________",ssim)
                            tf.logging.info('t%d, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (
                                num_recursions, scale, image_index + 1, num_images, psnr, ssim))

                            psnr_list[i].append(psnr)
                            ssim_list[i].append(ssim)
                            # ssim_list[i].append(ssim)

                    output_image = output_image_ensemble / ensemble_factor_total

                    if FLAGS.save_path is not None:
                        output_image_path = os.path.join(FLAGS.save_path, 'ensemble', 'x%d' % (scale),
                                                         os.path.splitext(image_name)[0] + '.png')
                        tf_image_session.run(tf_image_save_op, feed_dict={tf_image_save_path: output_image_path,
                                                                          tf_image_save_image: output_image})

                    truth_image = _clip_image(truth_image)
                    output_image = _clip_image(output_image)

                    truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

                    truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
                    output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

                    psnr = _image_psnr2(output_image=output_image_shaved, truth_image=truth_image_shaved)
                    ssim = _image_ssim(output_image_shaved, truth_image_shaved)
                    # ssim = _image_ssim_tf(output_image=output_image_shaved, truth_image=truth_image_shaved)

                    tf.logging.info(
                        'ensemble, x%d, %d/%d, psnr=%.2f, ssim=%.2f' % (scale, image_index + 1, num_images, psnr, ssim))

                    psnr_list[num_total_outputs].append(psnr)
                    ssim_list[num_total_outputs].append(ssim)
                    # ssim_list[num_total_outputs].append(ssim)

                for i in range(num_total_outputs + 1):
                    average_psnr = np.mean(psnr_list[i])
                    modules_average_psnr_dict[scale].append(average_psnr)
                    average_ssim = np.mean(ssim_list[i])
                    modules_average_ssim_dict[scale].append(average_ssim)
                    # average_ssim = np.mean(ssim_list[i])
                    # modules_average_ssim_dict[scale].append(average_ssim)

            # finalize
            tf.logging.info('finished')
            for scale in scale_list:
                print('- x%d, PSNR and SSIM:' % (scale))
                print(','.join([('%.3f' % x) for x in modules_average_psnr_dict[scale]]))
                print('')
                print(','.join([('%.3f' % x) for x in modules_average_ssim_dict[scale]]))
                # print('')
                # print(','.join([('%.3f' % x) for x in modules_average_ssim_dict[scale]]))

            if FLAGS.platform.lower() == 'modelarts':
                from help_modelarts import modelarts_result2obs
                modelarts_result2obs(FLAGS)


if __name__ == '__main__':
    tf.app.run(go())
