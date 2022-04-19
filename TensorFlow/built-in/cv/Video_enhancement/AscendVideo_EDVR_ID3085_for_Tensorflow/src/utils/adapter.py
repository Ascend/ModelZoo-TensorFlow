# Copyright 2022 Huawei Technologies Co., Ltd
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
from itertools import product

import numpy as np

from .exceptions import *
from .logger import logger


def factor_ceiling(x, factor):
    """Get the ceiling that is divisible by the factor.
    """
    remain = x % factor
    pad = 0
    if remain:
        pad = factor - remain
        x = x + pad
    return x, pad


class NetworkIOAdapter:
    """A network io adapter to process the input images when inference.

    Because of the memory limitation, we may not be able to process the whole 
    frame into a model in a single session.run. In this scenario, we use a 
    dispatch-process-stitching strategy to process the input frames. The 
    NetworkIOAdapter class is used to automatically make patches from the source 
    input frames, and stitching them together to get the whole result, during 
    which each patches can be overlapped.

    There are two basic modes in this adapter when using ckpt to inference:
    1.  Inferece as a whole, where the model is able to process the whole image. 
        In this scenario, the only thing the adapter will do is to pad the input
        frames to satisfy the network smallest feature map. If the smallest 
        feature map of the model is 1/N proportion to the original input, then 
        the size of the input frames should be divisible by N. Therefore, if 
        we find the original frame size does not satisfy this condition, the
        adpater will pad the frames. After inference, the result will be trimmed
        to the expected size.

        Example:
            >>> adapter = NetworkIOAdapter(cfg)
            >>> input_frames = get_data()  # [N, T, H, W, C]
            >>> adapter.register_raw_size(input_frames.shape[2:4])
            >>> print(adapter.patch_mode)
                False
            >>> padded_input = adapter.adapt_input(input_frames)
            >>> result = sess.run(output_tensor, feed_dict={input_node: padded_input})
            >>> final_result = adapter.reverse_adapt(result)

    2.  Inference using patches. In this mode, we have to inference the original
        input frames using split-and-conquer and then stitch them to the expected
        result. The patch size can be configured by the cfg.data.inference.best_patch_size
        for the efficient inference. We first pad the input frames to the factor
        ceiling of the best_patch_size, thus the padded original image can be split
        into several pieces. Then each patch is additionally padded with the 
        overlap size to avoid the discontinuity between two patch results. The 
        pad size of each patch should cover the size of the receptive field of the
        network. The session will inference each padded patch, followed by stitching
        method to aggregate the patch results to a complete one. The padded size 
        of the patch will be first trimmed off, and the patches will be concatenated
        together. Finally, the corrsponding padded size of the whole image will be 
        trimmed to get the final result. 

        Example:
            >>> adapter = NetworkIOAdapter(cfg)
            >>> input_frames = get_data()  # [N, T, H, W, C]
            >>> adapter.register_raw_size(input_frames.shape[2:4])
            >>> print(adapter.patch_mode)
                True
            >>> patches = adapter.extract_image_patches(input_frames)
            >>> num_patches = len(patches)
            >>> patch_per_step = cfg.data.inference.batch_size
            >>> result_patches = []
            >>> for i in range(num_patches//patch_per_step):
            ...     _patch = sess.run(output_tensor, feed_dict={input_node: patches[i:i+patch_per_step]})
            ...     result_patches.extend(_patch)
            >>> final_result = adapter.stitching_patches_to_image(result_patches)

    **A special scenario is to infer with pb file**, where the graph is already 
    freezed. In this scenario, the input size is also fixed, and we use the 
    adapter to automatically determine how to inference. **One must aware that 
    the actual size of the input patch is**:
        cfg.data.inference.input_size + cfg.data.inference.patch_pad_size * 2
    Therefore, one must ensure that the value above can be divisble by the network
    feature map scale factor.

    Args:
        cfg: yacs node, global configuration.
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # network input settings
        self.limited_in_size = cfg.data.inference.max_input_size
        self.best_in_size = cfg.data.inference.best_patch_size
        self.scale = cfg.model.scale
        self.factor_for_adapt_input = cfg.model.factor_for_adapt_input
        self.auto_mode = cfg.data.inference.auto_adapt_input

        if self.auto_mode and not self.limited_in_size:
            raise ValueError('Max input size is required when in auto mode.')
        if self.auto_mode == True:
            self.mode = 'auto'
        else:
            self.mode = None

        if not self.best_in_size:
            self.best_in_size = self.limited_in_size

        self.num_output_frames = cfg.model.num_net_output_frames

        # patch evaluation settings
        self.eval_in_patch = cfg.data.inference.eval_using_patch
        self.eval_pad_size = cfg.data.inference.patch_pad_size
        # size of the input image, before adapted
        self.eval_raw_size = [100000, 100000]
        # size of the input patch when in patch mode, or the image size when in 
        # whole mode, before adapted
        self.eval_in_size = cfg.data.inference.input_size

        self.fixed_in_size_flag = False

        # saved data for output, w.r.t. eval in patches strategy
        self._network_direct_outsize = []
        self._stitching_mode_padsize = []
        self._patch_batch_pad = 0
        self._vtip_stitching_method = False
        self._num_split = (0, 0)

        # saved data for reverse adapt, w.r.t. network input constrains
        self._input_adapt_padsize = [0, 0]

    @property
    def input_size(self):
        # In patch mode, it should be padded size
        if self.eval_in_patch:
            pads_h, pads_w = self.eval_pad_size[0]*2, self.eval_pad_size[1]*2
        else:
            # to keep inline with the original code, we should set 
            # eval_in_patch = eval_raw_size = raw image size
            # when inference the whole image
            pads_h, pads_w = self._input_adapt_padsize

        h = self.eval_in_size[0] + pads_h
        w = self.eval_in_size[1] + pads_w

        return (h, w)

    @property
    def patch_mode(self):
        return self.eval_in_patch

    def adapt_input(self, lr):
        #Used in whole image mode. 
        pads_h, pads_w = self._input_adapt_padsize

        if len(lr.shape) == 4:
            pads = [[0, 0], 
                    [pads_h//2, pads_h-pads_h//2], 
                    [pads_w//2, pads_w-pads_w//2], 
                    [0,0]]
        else:
            pads = [[0, 0], 
                    [0, 0], 
                    [pads_h//2, pads_h-pads_h//2], 
                    [pads_w//2, pads_w-pads_w//2], 
                    [0,0]]

        lr_pads = np.pad(lr, pads, mode='symmetric')
        return lr_pads

    def reverse_adapt(self, data):
        # Used in whole image mode
        pads_h, pads_w = self._input_adapt_padsize
        if data.ndim == 3:
            h, w, c = data.shape
            pads_t, pads_b = pads_h//2, pads_h-pads_h//2
            pads_l, pads_r = pads_w//2, pads_w-pads_w//2
            return data[pads_t*self.scale:h-pads_b*self.scale, 
                        pads_l*self.scale:w-pads_r*self.scale]
        elif data.ndim == 4:
            _, h, w, c = data.shape
            pads_t, pads_b = pads_h//2, pads_h-pads_h//2
            pads_l, pads_r = pads_w//2, pads_w-pads_w//2
            return data[:, 
                        pads_t*self.scale:h-pads_b*self.scale, 
                        pads_l*self.scale:w-pads_r*self.scale]
        else:
            raise ArrayDimensionError(f'Expect input data to have 3 or 4 '
                                      f'dimensions, but got {data.ndim}.')

    def fix_eval_in_size(self):
        # Used for inference with PB file and the input size is fixed.
        fixed_input_size = [self.best_in_size[0] + self.eval_pad_size[0] * 2,
                            self.best_in_size[1] + self.eval_pad_size[1] * 2]
        pad_h, pad_w = self.cal_adapted_size(fixed_input_size)
        assert pad_h == 0 and pad_w == 0, \
            f"Expect to have an input size that is divisible " \
            f"by {self.factor_for_adapt_input} when using a fixed input size, " \
            f"but got {fixed_input_size}. Must ensure that " \
            f"`model.best_in_size + data.eval_padsize*2` divisible by the factor."
        self.eval_in_size = self.best_in_size
        self.limited_in_size = fixed_input_size  # Real input size
        self.fixed_in_size_flag = True

    def register_raw_size(self, raw_size):
        # Override the configured raw_size
        self.eval_raw_size = raw_size

        logger.info(f'Automatically determine inference mode (whether patch or not).')
        if self.mode == 'auto':
            self.eval_in_size = raw_size
            logger.info(f'auto inference mode.')
            # In auto mode, the adapter will automatically define the input size
            h, w = raw_size
            limited_h, limited_w = self.limited_in_size

            if self.fixed_in_size_flag:
                # Remember that in this case:
                #    self.limited_in_size = self.best_in_size + self.eval_padsize * 2
                # **We have also make sure that self.limited_in_size is divisible 
                # by the factor**. See self.fix_eval_in_size()
                # Shall use a different logic to determine whether to eval in 
                # patch or not.
                if h <= limited_h and w <= limited_w:
                    # If the raw input size equals to the fixed size 
                    # (self.limited_in_size), eval in whole.
                    # self.limited_in_size = self.best_in_size + self.eval_padsize * 2 
                    # automatically ensures that self._input_adapt_padsize will be zero.
                    self.eval_in_patch = False
                else:
                    # Else, use a patch mode no matter if the raw size is larger 
                    # or smaller. To use the self.input_size interface consistent, 
                    # set self.eval_in_size to best_in_size hence ensuring:
                    #     self.limited_in_size = self.best_in_size + self.eval_padsize * 2
                    #                          = self.eval_in_size + self.eval_padsize * 2
                    self.eval_in_patch = True
                    self.eval_in_size = self.best_in_size
            else:
                if h * w > limited_w * limited_h:
                    self.eval_in_patch = True
                    self.eval_in_size = (
                        factor_ceiling(min(h, self.best_in_size[0]), self.factor_for_adapt_input)[0],
                        factor_ceiling(min(w, self.best_in_size[1]), self.factor_for_adapt_input)[0],
                    )
                else:
                    self.eval_in_patch = False

        if self.eval_in_patch:
            # Follow the config or the automatic setting
            pass
        else:
            # Adapt the image input to fit the network requirements
            if self.fixed_in_size_flag:
                self._input_adapt_padsize = (
                    (self.limited_in_size[0] - raw_size[0]),
                    (self.limited_in_size[1] - raw_size[1]),
                )
            else:
                # For whole image inference
                self._input_adapt_padsize = self.cal_adapted_size(raw_size)

            if self._input_adapt_padsize[0]:
                logger.info(f'Input height {raw_size[0]} is not a divisible by {self.factor_for_adapt_input}'
                            f', will be padded to {raw_size[0]+self._input_adapt_padsize[0]}')

            if self._input_adapt_padsize[1]:
                logger.info(f'Input width {raw_size[1]} is not a divisible by {self.factor_for_adapt_input}'
                            f', will be padded to {raw_size[1]+self._input_adapt_padsize[1]}')

        logger.info(f'Inference adapter: ')
        logger.info(f'\t          Use patch: {f"{self.eval_in_patch}":>5}')
        logger.info(f'\t     Image Raw size: {self.eval_raw_size}')
        logger.info(f'\tOriginal patch size: {self.eval_in_size}')
        logger.info(f'\t Adapted input size: {self.input_size}')

    def cal_adapted_size(self, raw_size):
        h, w = raw_size
        # In case the input is not divisible by the factor
        _, pad_h = factor_ceiling(h, self.factor_for_adapt_input)
        _, pad_w = factor_ceiling(w, self.factor_for_adapt_input)

        return pad_h, pad_w

    def extract_image_patches(self, data, num_patches_per_step=1):
        # This function is used in patch mode
        return self._extract_image_patches_canonical(data, num_patches_per_step)

    def stitching_patches_to_image(self, data):
        # This function is used in patch mode
        return self._merge_patches_to_images_canonical(data)

    def _extract_image_patches_canonical(self, data, num_patches_per_step=1):
        if data.ndim != 4:
            raise ArrayDimensionError(f'Expect input data to have 4 dimensions, but got {data.ndim}.')

        _, h, w, _ = data.shape
        ph, pw = self.eval_in_size
        # image padding size
        image_pad_right = int(float(w)/pw + 1) * pw - w
        image_pad_bottom = int(float(h)/ph + 1) * ph - h
        image_pad_right = 0 if image_pad_right == pw else image_pad_right
        image_pad_bottom = 0 if image_pad_bottom == ph else image_pad_bottom
        # patch padding size
        patch_pad_top = patch_pad_bottom = self.eval_pad_size[0]
        patch_pad_left = patch_pad_right = self.eval_pad_size[1]
        
        # pad image
        pad_t = patch_pad_top
        pad_b = patch_pad_bottom + image_pad_bottom
        pad_l = patch_pad_left
        pad_r = patch_pad_right + image_pad_right
        img_paded = np.pad(data, ((0, 0),
                                  (pad_t, pad_b),
                                  (pad_l, pad_r),
                                  (0, 0)), mode='symmetric')

        new_h, new_w = img_paded.shape[1:3]
        self._network_direct_outsize = (self.num_output_frames, new_h*self.scale, new_w*self.scale, 3)

        # number of patches
        num_split_y = (h + image_pad_bottom) // ph
        num_split_x = (w + image_pad_right) // pw
        self._num_split = (num_split_y, num_split_x)

        img_patches = []
        for split_j, split_i in product(range(num_split_y), range(num_split_x)):
            # extract patches with extra pad size
            patch_start_y = split_j * ph
            patch_end_y = patch_start_y + ph + patch_pad_top + patch_pad_bottom
            patch_start_x = split_i * pw
            patch_end_x = patch_start_x + pw + patch_pad_left + patch_pad_right
            img_patches.append(img_paded[:, patch_start_y:patch_end_y, patch_start_x:patch_end_x, :])

        img_patches = np.array(img_patches)
        num_patches = img_patches.shape[0]
        batch_pad = (num_patches // num_patches_per_step + 1) * num_patches_per_step - num_patches
        batch_pad = 0 if batch_pad == num_patches_per_step else batch_pad
        self._patch_batch_pad = batch_pad

        # Concatenate all the patches in order.
        if batch_pad > 0:
            img_patches_padded = np.concatenate([
                img_patches,
                np.zeros([batch_pad, *img_patches.shape[1:]], dtype=np.float32),
            ], axis=0)
        else:
            img_patches_padded = img_patches
        return img_patches_padded

    def _merge_patches_to_images_canonical(self, data):
        # This is the reverse processing of the dispatching.
        ph, pw = self.eval_in_size
        num_split_y, num_split_x = self._num_split
        sr_all = np.zeros(self._network_direct_outsize, dtype=np.float32)  # [num_output_frames, h, w, c]
        
        h, w = self.eval_raw_size
        patch_pad_top, patch_pad_left = self.eval_pad_size
        patch_sr = np.array(data)          # should be [num_patches, num_output_frames, h, w, c]
        if patch_sr.ndim == 4 and self.num_output_frames == 1 and patch_sr.shape[1] != 1:
            patch_sr = np.expand_dims(patch_sr, axis=1)

        patch_s_y = patch_pad_top * self.scale
        patch_e_y = (patch_pad_top + ph) * self.scale
        patch_s_x = patch_pad_left * self.scale
        patch_e_x = (patch_pad_left + pw) * self.scale
        patch_id = 0
        for split_j, split_i in product(range(num_split_y), range(num_split_x)):
            im_s_y = split_j * ph * self.scale
            im_e_y = im_s_y + ph * self.scale
            im_s_x = split_i * pw * self.scale
            im_e_x = im_s_x + pw * self.scale
            sr_all[:, im_s_y:im_e_y, im_s_x:im_e_x] = patch_sr[patch_id, :, patch_s_y:patch_e_y, patch_s_x:patch_e_x]
            patch_id += 1

        # Trim the output to expected size.
        sr_all = sr_all[:, :h*self.scale, :w*self.scale]
        return sr_all.squeeze()
