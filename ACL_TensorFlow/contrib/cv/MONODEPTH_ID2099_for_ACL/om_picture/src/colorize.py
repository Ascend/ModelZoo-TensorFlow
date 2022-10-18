from __future__ import absolute_import, division, print_function

import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import sys
import time

import argparse
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

# only keep warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, "."))
sys.path.append(os.path.join(path, "./acllite/"))

from constants import IMG_EXT
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_resource import AclLiteResource

INPUT_DIR = '../data/'
OUTPUT_DIR = '../out/'
model_path = '../model/tf_monodepth.om'


def preprocess(picPath):
    input_image = scipy.misc.imread(picPath, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [256, 512], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    return input_images, original_height, original_width


def postprocess(result_list, picPath, original_height, original_width):
    disp_pp = post_process_disparity(result_list[0].squeeze()).astype(np.float32)
    output_directory = os.path.dirname(picPath)
    output_name = os.path.splitext(os.path.basename(picPath))[0]

    np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

def post_process_disparity(disp):
    print(disp)
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def main():
    """
    acl resource initialization
    """
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    # ACL resource initialization
    acl_resource = AclLiteResource()
    acl_resource.init()

    model = AclLiteModel(model_path)
    images_list = [os.path.join(INPUT_DIR, img)
                   for img in os.listdir(INPUT_DIR)
                   if os.path.splitext(img)[1] in IMG_EXT]

    for pic in images_list:
        l_data, original_height, original_width = preprocess(pic)
        result_list = model.execute([l_data, ])
        postprocess(result_list, pic, original_height, original_width)
        break
    print("Execute end")


if __name__ == '__main__':
    main()
