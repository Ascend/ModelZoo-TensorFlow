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

import cv2
import numpy as np
import random
import collections


__all__ = ['_Transform', 'RandomTemporalReverse', 'RandomFlipUpDown',
    'RandomFlipLeftRight', 'Scale', 'Resize', 'RandomCrop',
    'RandomDropChrominanceChannel', 'TempDistCrop', 'RandomSized',
    'RandomReverseColorChannel'
]
        

class _Transform(object):
    """Base transform class.
    """
    def __init__(self, input_dim=4):
        self.input_dim = input_dim


class RandomTemporalReverse(_Transform):
    """Random temporal reverse transform.

    This transform will reverse the multi-frame order.
    """
    def __call__(self, *imgs):
        if random.random() < 0.5:
            imgs = [item[::-1] for item in imgs]
        return imgs


class RandomFlipUpDown(_Transform):
    """Random up-down flip transform.

    This transform will randomly flip the frames upside-down.
    """
    def __init__(self, input_dim=3):
        super().__init__(input_dim)
        if input_dim == 3:
            # HWC
            self.fn = lambda x: x[::-1]
        elif input_dim == 4:
            # BHWC or DHWC
            self.fn = lambda x: x[:, ::-1]
        elif input_dim == 5:
            # BDHWC
            self.fn = lambda x: x[:, :, ::-1]
        else:
            raise NotImplementedError

    def __call__(self, *imgs):
        if random.random() < 0.5:
            imgs = [self.fn(item) for item in imgs]
        return imgs


class RandomFlipLeftRight(_Transform):
    """Random up-down flip transform.

    This transform will randomly flip the frames left-right.
    """
    def __init__(self, input_dim=3):
        super().__init__(input_dim)
        if input_dim == 3:
            # HWC
            self.fn = lambda x: x[:, ::-1]
        elif input_dim == 4:
            # BHWC or DHWC
            self.fn = lambda x: x[:, :, ::-1]
        elif input_dim == 5:
            # BDHWC
            self.fn = lambda x: x[:, :, :, ::-1]
        else:
            raise NotImplementedError

    def __call__(self, *imgs):
        if random.random() < 0.5:
            imgs = [self.fn(item) for item in imgs]
        return imgs


def _resize(img, new_size, input_dim, interpolation=cv2.INTER_LINEAR):
    """Basic resize function.
    """
    if img.shape[-1] == 1:
        expand = True
    else:
        expand = False
    if input_dim == 3:
        img =  cv2.resize(img, new_size, interpolation=interpolation)
    elif input_dim == 4:
        img = [cv2.resize(item, new_size, interpolation=interpolation) for item in img]
        img = np.stack(img, axis=0)
    else:
        raise ValueError('Resize: image dimension must be in [3, 4]')

    if expand:
        img = np.expand_dims(img, -1)

    return img


class Scale(_Transform):
    """Resize the inputs given the scale.
    """
    def __init__(self, input_dim, scales, interpolations=None):
        super().__init__(input_dim)
        self.scale = scales
        self.interpolations = interpolations

    def __call__(self, *imgs):
        h, w = imgs[0].shape[-3:-1]
        ow = int(self.scale * w)
        oh = int(self.scale * h)
        if self.interpolations is None:
            imgs = [_resize(item, (ow, oh), self.input_dim)
                    for item in imgs]
        else:
            imgs = [_resize(item, (ow, oh), self.input_dim, interpolation=interpolation)
                    for item, interpolation in zip(imgs, self.interpolations)]
        return imgs


class Resize(_Transform):
    """Resize the inputs given the target size.
    """
    def __init__(self, input_dim, size, interpolations=None):
        super().__init__(input_dim)
        self.size = size
        self.interpolations = interpolations

    def __call__(self, *imgs):
        new_size = tuple([self.size, self.size])
        if self.interpolations is None:
            imgs = [_resize(item, new_size, self.input_dim)
                    for item in imgs]
        else:
            imgs = [_resize(item, new_size, self.input_dim, interpolation=interpolation)
                    for item, interpolation in zip(imgs, self.interpolations)]
        return imgs


class RandomCrop(_Transform):
    """Random crop the images into patches.

    This function is for multiple input array, i.e. [lr_array, lr2_array, ..., hd_array].
    We want the corresponding crops of these inputs. Therefore, the function accepts
    a base crop_size and the scales of the base crop_size that correspond to the
    expected output patch size of each input array.

    For example, in 4x super resolution we want the training input has the size 
    [64, 64], and the output thus [256, 256] size. Therefore when cropping the paired
    lr and gt data, the corresponding regions of lr and gt are to be cropped. We
    can use a RandomCrop transform with:
    
    Example:
        >>> lr, gt = get_data()  # suppose 4D tensors of data format DHWC
        >>> tr = RandomCrop(input_dim=4, crop_size=(64, 64), scales=(1, 4))
        >>> lr_crop, gt_crop = tr([lr, gt])

    Args:
        input_dim: int, dimension of each input.
        crop_size: list[int], the base size [H, W] of the patch.
        scales: list[int], the scales of the crop for each input.
        bbox: list[int], the bounding box of the crop, [H_ul, W_ul, H_br, W_br],
            where H_ul and W_ul are the height and width of upper-left pixel,
            H_br and W_br are the height and width of the bottom-right pixel.
    """
    def __init__(self, input_dim, crop_size, scales, bbox=None):
        super().__init__(input_dim)
        # Notice, this transformation always based on the first element of the images
        self._h = crop_size[0]
        self._w = crop_size[1]
        self.scales = scales
        self.bbox = bbox

    def crop(self, im, ymin, xmin, ymax, xmax):
        if self.input_dim == 3:
            # HWC
            return im[ymin:ymax, xmin:xmax]
        elif self.input_dim == 4:
            # DHWC or BHWC
            return im[:, ymin:ymax, xmin:xmax]
        elif self.input_dim == 5:
            # BDHWC
            return im[:, :, ymin:ymax, xmin:xmax]

    def __call__(self, *imgs):
        assert len(self.scales) == len(imgs)
        h, w = imgs[0].shape[-3:-1]
        if self.bbox is None:
            h_st, h_ed, w_st, w_ed = 0, h, 0, w
        else:
            h_st, h_ed, w_st, w_ed = self.bbox
        xmin = random.randint(w_st, w_ed - self._w * self.scales[0])
        ymin = random.randint(h_st, h_ed - self._h * self.scales[0])

        augs = []
        for scale, im in zip(self.scales, imgs):
            y_st = ymin * scale // self.scales[0]
            y_ed = y_st + self._h * scale
            x_st = xmin * scale // self.scales[0]
            x_ed = x_st + self._w * scale

            patch = self.crop(im, y_st, x_st, y_ed, x_ed)
            assert patch.shape[-3:-1] ==  (self._h * scale, self._w * scale), \
                f'Expect cropped patch to have size {(self._h * scale, self._w * scale)},' \
                f' but got {patch.shape[-3:-1]} (might be out of range). ' \
                f'For information, im has shape {im.shape}, crop range y: {y_st}-{y_ed}, x: {x_st}-{x_ed}.'
            augs.append(patch)
        return augs


class RandomDropChrominanceChannel(_Transform):
    """Randomly drop chrominance channels. The luminance channel will be replicated.
    """
    def _to_grayscale_3channel(self, x):
        single_x = x[..., 0:1]
        return np.concatenate([single_x, single_x, single_x], axis=-1)

    def __call__(self, *imgs):
        if random.random() < 0.5:
            imgs = [self._to_grayscale_3channel(item) for item in imgs]
        return imgs


class TempDistCrop(_Transform):
    '''Crop video frames with disturbed bboxes along temporal dimension
    '''
    def __init__(self, input_dim, crop_size, scales, dist=0.01, no_padding=True, crop_range=None):
        super(TempDistCrop, self).__init__(input_dim)
        self._h = crop_size[0]
        self._w = crop_size[1]
        self.dist = dist
        self.scales = scales
        self.no_padding = no_padding
        self.crop_range = crop_range

    def crop(self, imgs, T, ymins, xmins, ymaxs, xmaxs):
        assert self.input_dim == 4
        if imgs.shape[0] == 1:
            c_idx = T//2
            return imgs[:, ymins[c_idx]:ymaxs[c_idx], xmins[c_idx]:xmaxs[c_idx]]
        else:
            res = []
            for i in range(T):
                res.append(imgs[i, ymins[i]:ymaxs[i], xmins[i]:xmaxs[i]])

            return np.stack(res, axis=0)

    def pos_disturbe(self, T, ymin, xmin, H, W):
        xmins, ymins = [], []
        x_bias, y_bias = np.random.random(size=T), np.random.random(size=T)
        x_bias = ((x_bias*2-1)*self.dist * self._w).astype(np.int)
        y_bias = ((y_bias*2-1)*self.dist * self._h).astype(np.int)
        for i in range(T):
            xmins.append(np.clip(xmin+x_bias[i], 0, W - self._w))
            ymins.append(np.clip(ymin+y_bias[i], 0, H - self._h))

        return np.array(xmins), np.array(ymins)

    def _pad(self, imgs):
        h, w = imgs.shape[-3:-1]
        if self.no_padding:
            if w < self._w or h < self._h:
                if w < h:
                    ow = self._w
                    oh = int(self._w * h / w)
                else:
                    oh = self._h
                    ow = int(self._h * w / h)
                if self.input_dim == 3:
                    imgs = cv2.resize(imgs, (ow, oh))
                elif  self.input_dim == 4:
                    imgs = [cv2.resize(item, (ow, oh)) for item in imgs]
                else:
                    raise ValueError('TempDistCrop: image dimension must be in [3, 4]')
        else:
            pad_h = max(self._h - h, 0)
            pad_w = max(self._w - w, 0)
            imgs = [np.pad(item, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                    for item in imgs]
            if self.input_dim == 3:
                imgs = np.pad(imgs, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
            elif self.input_dim == 4:
                imgs = [np.pad(item, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                        for item in imgs]
            else:
                raise ValueError('TempDistCrop: image dimension must be in [3, 4]')
        return imgs

    def __call__(self, *imgs):
        assert len(self.scales) == len(imgs)
        # imgs = [self._pad(item) for item in imgs]
        h, w = imgs[0].shape[-3:-1]
        bedge = 140
        if self.crop_range is None:
            # h_st, h_ed, w_st, w_ed = 0, h, 0, w
            h_st, h_ed, w_st, w_ed = bedge, h-bedge, 0, w
        else:
            h_st, h_ed, w_st, w_ed = self.crop_range
        xmin = random.randint(w_st, w_ed - self._w * self.scales[0])
        ymin = random.randint(h_st, h_ed - self._h * self.scales[0])
        T = imgs[0].shape[0]
        xmins, ymins = self.pos_disturbe(T, ymin, xmin, h, w)

        augs = []
        for scale, im in zip(self.scales, imgs):
            y_st = ymins * scale
            y_ed = (ymins + self._h) * scale
            x_st = xmins * scale
            x_ed = (xmins + self._w) * scale

            augs.append(self.crop(im, T, y_st, x_st, y_ed, x_ed))
        return augs


class RandomSized(_Transform):
    """Random resize the input with a ranged scale.
    """
    def __init__(self, input_dim=3, scale=(0.8, 1.2), interpolations=None):
        super().__init__(input_dim)
        self.scale = scale
        self.interpolations = interpolations

    def __call__(self, *imgs):
        org_h, org_w = imgs[0].shape[-3:-1]
        r = random.uniform(self.scale[0], self.scale[1])
        w = int(r * org_w)
        h = int(r * org_h)

        new_size = tuple([w, h])
        if self.interpolations is None:
            imgs = [_resize(item, new_size, self.input_dim, interpolation=cv2.INTER_LINEAR)
                    for item in imgs]
        else:
            imgs = [_resize(item, new_size, self.input_dim, interpolation=interpolation)
                    for item, interpolation in zip(imgs, self.interpolations)]
        return imgs


class RandomReverseColorChannel(_Transform):
    """Randomly shift the color channel.
    """
    def __call__(self, *imgs):
        if random.random() < 0.5:
            imgs = [item[..., ::-1] for item in imgs]
        return imgs
