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

import numpy as np
from acl_utils import load_pkl
import argparse
import os


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default=r'datasets/ixi_valid.pkl', type=str,
                        help='input file path.')
    parser.add_argument('--output', default=r'/bin/ixi_valid', type=str,
                        help='Output file.')
    parser.add_argument('--bs', default=1, type=int, help='Batch size.')
    args = parser.parse_args()
    return args


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


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


def main():
    args = parse_args()

    corrupt_params = dict(type='bspec', p_at_edge=0.025)

    print('[info] Start changing pkl to bin...')

    test_img, test_spec = load_dataset(args.input)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    idx = 0
    for (_, inputs, _, _, _) in iterate_minibatches(test_img, test_spec,
                                                    batch_size=args.bs,
                                                    shuffle=False,
                                                    corrupt_targets=False,
                                                    corrupt_params=corrupt_params):
        # indices = indices
        # inputs = inputs
        inputs = np.asarray(inputs)
        output_file_path = os.path.join(args.output, 'img_mir_bs%03d_%05d.bin' % (args.bs, idx))
        inputs.tofile(output_file_path)
        idx += 1
        # if idx == 20:
        #     return
    print("[info] output path:{}".format(args.output))


if __name__ == '__main__':
    main()
