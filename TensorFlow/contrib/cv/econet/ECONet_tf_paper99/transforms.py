import random
from PIL import Image, ImageOps
import numpy as np


###############################
# Transformation utilities
###############################

def sample_crop_size(im_size, patch_size, scales, max_distort, fix_crop, more_fix_crop):
    image_w, image_h = im_size[0], im_size[1]

    # find a crop size
    base_size = min(image_w, image_h)
    crop_sizes = [int(base_size * x) for x in scales]
    # not enthusiastic about resizing to 224
    crop_h = [patch_size if abs(x - patch_size) < 3 else x for x in crop_sizes]
    crop_w = [patch_size if abs(x - patch_size) < 3 else x for x in crop_sizes]

    # pairs for crop size (for minimum)
    pairs = []
    for i, h in enumerate(crop_h):
        for j, w in enumerate(crop_w):
            if abs(i - j) <= max_distort:
                pairs.append((w, h))

    crop_pair = random.choice(pairs)
    if not fix_crop:
        w_offset = random.randint(0, image_w - crop_pair[0])
        h_offset = random.randint(0, image_h - crop_pair[1])
    else:
        w_offset, h_offset = sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1], more_fix_crop)

    return crop_pair[0], crop_pair[1], w_offset, h_offset


def sample_fix_offset(image_w, image_h, crop_w, crop_h, more_fix_crop):
    offsets = fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h)
    return random.choice(offsets)


def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
    w_step = (image_w - crop_w) // 4
    h_step = (image_h - crop_h) // 4

    ret = list()
    ret.append((0, 0))  # upper left
    ret.append((4 * w_step, 0))  # upper right
    ret.append((0, 4 * h_step))  # lower left
    ret.append((4 * w_step, 4 * h_step))  # lower right
    ret.append((2 * w_step, 2 * h_step))  # center

    if more_fix_crop:
        ret.append((0, 2 * h_step))  # center left
        ret.append((4 * w_step, 2 * h_step))  # center right
        ret.append((2 * w_step, 4 * h_step))  # lower center
        ret.append((2 * w_step, 0 * h_step))  # upper center

        ret.append((1 * w_step, 1 * h_step))  # upper left quarter
        ret.append((3 * w_step, 1 * h_step))  # upper right quarter
        ret.append((1 * w_step, 3 * h_step))  # lower left quarter
        ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

    return ret



###############################
# Main transformation function
###############################


def get_random_crop(frame_list, patch_size):

    ih, iw = frame_list[0].shape[:2]
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    def _get_random_patch(frame):
        frame = frame[iy:iy + ip, ix:ix + ip, :]
        return frame

    return [_get_random_patch(f) for f in frame_list]


def get_center_crop(frame_list, patch_size):

    ih, iw = frame_list[0].size[:2]
    ip = patch_size

    ix = (iw - ip) // 2
    iy = (ih - ip) // 2

    # def _get_center_patch(frame):
    #     frame = frame[iy:iy + ip, ix:ix + ip, :]
    #     return frame

    return [img.crop((iy, ix, iy + ip, ix + ip)) for img in frame_list]


def get_multi_scale_crop(frame_list, patch_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):

    im_size = frame_list[0].size

    crop_w, crop_h, offset_w, offset_h = sample_crop_size(im_size, patch_size, scales, max_distort, fix_crop, more_fix_crop)
    crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in frame_list]
    ret_img_group = [img.resize((patch_size, patch_size), Image.BILINEAR)
                        for img in crop_img_group]
    return ret_img_group


def get_random_horizontal_flip(frame_list):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frame_list]
        return ret
    else:
        return frame_list    


def stack_then_normalize(frame_list, mean_list):
    # Image => numpy.ndarray  RGB => BGR
    frame_np_list = [np.array(img)[:, :, ::-1].astype('float32') for img in frame_list]
    frame_seq = np.stack(frame_np_list)
    # normalization
    frame_seq -= mean_list
    return frame_seq


def resize(frame_list, patch_size):
    return [img.resize((patch_size, patch_size), Image.BILINEAR) for img in frame_list]