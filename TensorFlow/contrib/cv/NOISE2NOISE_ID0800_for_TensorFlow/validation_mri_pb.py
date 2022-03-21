# ============================================================================
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

import dnnlib
import os
import numpy as np
import tensorflow as tf
import dnnlib.tflib.tfutil as tfutil
import util
import config_mri_pb
from train_mri import iterate_minibatches, fftshift2d
import dnnlib.submission.submit as submit


def load_dataset(fn, num_images=None, shuffle=False):
    datadir = submit.get_path_from_template(config_mri_pb.validate_config.data_dir)
    if fn.lower().endswith('.pkl'):
        abspath = os.path.join(datadir, fn)
        print('Loading dataset from', abspath)
        img, spec = util.load_pkl(abspath)
    else:
        assert False

    if shuffle:
        perm = np.arange(img.shape[0])
        np.random.shuffle(perm)
        if num_images is not None:
            perm = perm[:num_images]
        img = img[perm]
        spec = spec[perm]

    if num_images is not None:
        img = img[:num_images]
        spec = spec[:num_images]

    # Remove last row/column of the images, we're officially 255x255 now.
    img = img[:, :-1, :-1]

    # Convert to float32.
    assert img.dtype == np.uint8
    img = img.astype(np.float32) / 255.0 - 0.5

    return img, spec


def fftshift3d(x, ifft):
    assert len(x.shape) == 3
    s0 = (x.shape[1] // 2) + (0 if ifft else 1)
    s1 = (x.shape[2] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[:, s0:, :], x[:, :s0, :]], axis=1)
    x = np.concatenate([x[:, :, s1:], x[:, :, :s1]], axis=2)
    return x


def validate(submit_config: dnnlib.SubmitConfig, dataset_test: dict, data_dir: str, minibatch_size: int,
             corrupt_params: dict, pbdir: str, input_tensor_name: str, output_tensor_name: str, post_op: str, tf_config: dict):
    if corrupt_params is None:
        corrupt_params = dict()
    ctx = dnnlib.RunContext(submit_config, config_mri_pb)
    graph = tfutil.load_pb(pbdir)
    print('Loading test set.')
    test_img, test_spec = load_dataset(**dataset_test)
    result_subdir = submit_config.run_dir
    inputs_var = graph.get_tensor_by_name(input_tensor_name)
    denoised = graph.get_tensor_by_name(output_tensor_name)
    with graph.as_default():
        input_shape = [None] + list(test_img.shape)[1:]
        targets_var = tf.placeholder(tf.float32, shape=input_shape, name='targets')
        with tf.name_scope('loss'):
            targets_clamped = tf.clip_by_value(targets_var, -0.5, 0.5)
            denoised_clamped = tf.clip_by_value(denoised, -0.5, 0.5)
            # Keep MSE for each item in the minibatch for PSNR computation:
            loss_clamped = tf.reduce_mean((targets_clamped - denoised_clamped) ** 2, axis=[1, 2])

    session = tfutil.init_tf(tf_config, graph=graph)

    test_db_clamped = 0.0
    test_n = 0.0
    idx = 0
    for (indices, inputs, targets, input_spec_val, input_spec_mask) in iterate_minibatches(test_img, test_spec,
                                                                                           batch_size=minibatch_size,
                                                                                           shuffle=False,
                                                                                           corrupt_targets=False,
                                                                                           corrupt_params=corrupt_params):
        # Construct feed dictionary.
        feed_dict = {inputs_var: inputs, targets_var: targets}

        # Run.

        # Export example result.
        loss_clamped_vals, orig = session.run([loss_clamped, denoised],
                                              feed_dict=feed_dict)

        outputs = orig

        if post_op == 'fspec':
            # 注意：这是相关TF代码的NumPy实现。这是由于NPU不支持tf.complex64而采取的规避操作。仅在验证时进行此项操作。
            denoised_spec = np.fft.fft2(outputs).astype(np.complex64)  # Take FFT of denoiser output.
            denoised_spec = fftshift3d(denoised_spec, False)  # Shift before applying mask.
            spec_mask_c64 = np.asarray(input_spec_mask).astype(np.complex64)
            denoised_spec = input_spec_val * spec_mask_c64 + denoised_spec * (
                    1. - spec_mask_c64)  # Force known frequencies.
            outputs = np.fft.ifft2(fftshift3d(denoised_spec, True))
            outputs = outputs.astype(np.float32)  # Shift back and IFFT.

            targets_clamped_np = np.clip(np.asarray(targets), -0.5, 0.5)
            denoised_clamped_np = np.clip(outputs, -0.5, 0.5)

            loss_clamped_vals = np.mean((targets_clamped_np - denoised_clamped_np) ** 2, axis=(1, 2))

        if test_n == 0:
            prim = [inputs[0], orig[0], outputs[0], targets[0]]
            spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
            pimg = np.concatenate(prim, axis=1) + 0.5
            simg = np.concatenate(spec, axis=1) * 0.03
            img = np.concatenate([pimg, simg], axis=0)
            # scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(os.path.join(result_subdir, 'img%05d.png' % idx))

        # Stats.
        indiv_db = 10 * np.log10(1.0 / loss_clamped_vals)
        test_db_clamped += np.sum(indiv_db)
        test_n += len(indices)

    test_db_clamped /= test_n

    print('test_db_clamped=%.5f' % test_db_clamped)
    session.close()
    ctx.close()
