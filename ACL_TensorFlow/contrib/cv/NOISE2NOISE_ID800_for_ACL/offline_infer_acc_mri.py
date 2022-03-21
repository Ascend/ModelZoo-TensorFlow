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


import os
import argparse
import glob
import cv2 as cv
import scipy
import numpy as np
from acl_utils import load_pkl


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', default=1, type=int,
                        help='batchsize')
    parser.add_argument('--bin_dir', default=r'result/bin/ixi_valid', type=str,
                        help='the bin data path')
    parser.add_argument('--dataset', default=r'datasets/ixi_valid.pkl', type=str,
                        help='the nyu data path')
    parser.add_argument('--width', type=int, default=255, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=255, help='resized image height before inference.')
    parser.add_argument('--post_op', type=str, default=None)
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def read_bin_file(file_path):
    bin_files = os.listdir(file_path)
    loaded_dic = {}
    for bin_file in bin_files:
        file_name = bin_file.split("_")[0]
        loaded_dic.update({file_name: bin_file})

    return loaded_dic


def read_dataset_file(file_path):
    dataset_files = os.listdir(file_path)
    loaded_dic = {}
    for dataset_file in dataset_files:
        file_name = dataset_file.split(".")[0]
        loaded_dic.update({file_name: dataset_file})

    return loaded_dic


def load_dataset(datadir, num_images=None, shuffle=False):
    if os.path.splitext(datadir)[-1].lower().endswith('.pkl'):
        print('[info] Loading dataset from', datadir)
        img, spec = load_pkl(datadir)
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


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


bernoulli_mask_cache = dict()


def corrupt_data(img, spec, params):
    ctype = params['type']
    assert ctype == 'bspec'
    p_at_edge = params['p_at_edge']
    global bernoulli_mask_cache
    if bernoulli_mask_cache.get(p_at_edge) is None:
        h = [s // 2 for s in spec.shape]
        r = [np.arange(s, dtype=np.float32) - h for s, h in zip(spec.shape, h)]
        r = [x ** 2 for x in r]
        r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
        m = (p_at_edge ** (1. / h[1])) ** r
        bernoulli_mask_cache[p_at_edge] = m
        print('[info] Bernoulli probability at edge = %.5f' % m[h[0], 0])
        print('[info] Average Bernoulli probability = %.5f' % np.mean(m))
    mask = bernoulli_mask_cache[p_at_edge]
    keep = (np.random.uniform(0.0, 1.0, size=spec.shape) ** 2 < mask)
    keep = keep & keep[::-1, ::-1]
    sval = spec * keep
    smsk = keep.astype(np.float32)
    spec = fftshift2d(sval / (mask + ~keep), ifft=True)  # Add 1.0 to not-kept values to prevent div-by-zero.
    img = np.real(np.fft.ifft2(spec)).astype(np.float32)
    return img, sval, smsk


augment_translate_cache = dict()


def augment_data(img, spec, params):
    t = params.get('translate', 0)
    if t > 0:
        global augment_translate_cache
        trans = np.random.randint(-t, t + 1, size=(2,))
        key = (trans[0], trans[1])
        if key not in augment_translate_cache:
            x = np.zeros_like(img)
            x[trans[0], trans[1]] = 1.0
            augment_translate_cache[key] = fftshift2d(np.fft.fft2(x).astype(np.complex64))
        img = np.roll(img, trans, axis=(0, 1))
        spec = spec * augment_translate_cache[key]
    return img, spec


def iterate_minibatches(input_img, input_spec, batch_size, shuffle, corrupt_targets, corrupt_params,
                        augment_params=None, max_images=None):
    if augment_params is None:
        augment_params = dict()

    assert input_img.shape[0] == input_spec.shape[0]

    num = input_img.shape[0]
    all_indices = np.arange(num)
    if shuffle:
        np.random.shuffle(all_indices)

    if max_images:
        all_indices = all_indices[:max_images]
        num = len(all_indices)

    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            indices = all_indices[start_idx: start_idx + batch_size]
            inputs, targets = [], []
            spec_val, spec_mask = [], []

            for i in indices:
                img, spec = augment_data(input_img[i], input_spec[i], augment_params)
                inp, sv, sm = corrupt_data(img, spec, corrupt_params)
                inputs.append(inp)
                spec_val.append(sv)
                spec_mask.append(sm)
                if corrupt_targets:
                    t, _, _ = corrupt_data(img, spec, corrupt_params)
                    targets.append(t)
                else:
                    targets.append(img)

            yield indices, inputs, targets, spec_val, spec_mask


def fftshift3d(x, ifft):
    assert len(x.shape) == 3
    s0 = (x.shape[1] // 2) + (0 if ifft else 1)
    s1 = (x.shape[2] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[:, s0:, :], x[:, :s0, :]], axis=1)
    x = np.concatenate([x[:, :, s1:], x[:, :, :s1]], axis=2)
    return x


def main():
    args = parse_args()
    # data load and preprocess
    # load bin data
    print('[info] Start changing pkl to bin...')

    bin_name = sorted(os.listdir(args.bin_dir))

    test_img, test_spec = load_dataset(args.dataset)

    corrupt_params = dict(type='bspec', p_at_edge=0.025)

    test_db_clamped = 0.0
    test_n = 0.0
    idx = 0
    for (indices, _, targets, input_spec_val, input_spec_mask) in iterate_minibatches(test_img, test_spec,
                                                                                      batch_size=args.bs,
                                                                                      shuffle=False,
                                                                                      corrupt_targets=False,
                                                                                      corrupt_params=corrupt_params):

        if idx == len(bin_name):
            break

        denoised_path = os.path.join(args.bin_dir, bin_name[idx])

        denoised_ori = np.fromfile(denoised_path, dtype=np.float32)
        denoised_ori = np.reshape(denoised_ori, (args.bs, args.height, args.width))

        denoised = denoised_ori
        if args.post_op == 'fspec':
            denoised_spec = np.fft.fft2(denoised).astype(np.complex64)  # Take FFT of denoiser output.
            denoised_spec = fftshift3d(denoised_spec, False)  # Shift before applying mask.
            spec_mask_c64 = np.asarray(input_spec_mask).astype(np.complex64)
            denoised_spec = input_spec_val * spec_mask_c64 + denoised_spec * (
                    1. - spec_mask_c64)  # Force known frequencies.
            denoised = np.fft.ifft2(fftshift3d(denoised_spec, True))
            denoised = denoised.astype(np.float32)  # Shift back and IFFT.

        targets_clamped = np.clip(np.asarray(targets), -0.5, 0.5)
        denoised_clamped = np.clip(denoised, -0.5, 0.5)
        # Keep MSE for each item in the minibatch for PSNR computation:
        loss_clamped = np.mean((targets_clamped - denoised_clamped) ** 2, axis=(1, 2))

        # Stats.
        indiv_db = 10 * np.log10(1.0 / loss_clamped)
        test_db_clamped += np.sum(indiv_db)
        test_n += len(indices)
        idx += 1

    test_db_clamped /= test_n
    print("[info] test image:{}".format(int(test_n)))
    print("[info] test_db_clamped: %4.2f" % test_db_clamped)


if __name__ == '__main__':
    main()
