#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.08.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
#
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
#
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

from npu_bridge.npu_init import *
import argparse
import pickle
import sys
import cv2
import os

import numpy as np

from transforms import *
from ssdutils import get_preset_by_name
from utils import load_data_source, str2bool, draw_box
from tqdm import tqdm

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

#-------------------------------------------------------------------------------
def annotate(data_dir, samples, colors, sample_name):
    """
    Draw the bounding boxes on the sample images
    :param data_dir: the directory where the dataset's files are stored
    :param samples:  samples to be processed
    :param colors:   a dictionary mapping class name to a BGR color tuple
    :param colors:   name of the sample
    """
    result_dir = data_dir+'/annotated/'+sample_name.strip()+'/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for sample in tqdm(samples, desc=sample_name, unit='samples'):
        img    = cv2.imread(sample.filename)
        basefn = os.path.basename(sample.filename)
        for box in sample.boxes:
            draw_box(img, box, colors[box.label])
        cv2.imwrite(result_dir+basefn, img)

#-------------------------------------------------------------------------------
def build_sampler(overlap, trials):
    return SamplerTransform(sample=True, min_scale=0.3, max_scale=1.0,
                            min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                            min_jaccard_overlap=overlap, max_trials=trials)

#-------------------------------------------------------------------------------
def build_train_transforms(preset, num_classes, sampler_trials, expand_prob):
    #---------------------------------------------------------------------------
    # Resizing
    #---------------------------------------------------------------------------
    tf_resize = ResizeTransform(width=preset.image_size.w,
                                height=preset.image_size.h,
                                algorithms=[cv2.INTER_LINEAR,
                                            cv2.INTER_AREA,
                                            cv2.INTER_NEAREST,
                                            cv2.INTER_CUBIC,
                                            cv2.INTER_LANCZOS4])

    #---------------------------------------------------------------------------
    # Image distortions
    #---------------------------------------------------------------------------
    tf_brightness = BrightnessTransform(delta=32)
    tf_rnd_brightness = RandomTransform(prob=0.5, transform=tf_brightness)

    tf_contrast = ContrastTransform(lower=0.5, upper=1.5)
    tf_rnd_contrast = RandomTransform(prob=0.5, transform=tf_contrast)

    tf_hue = HueTransform(delta=18)
    tf_rnd_hue = RandomTransform(prob=0.5, transform=tf_hue)

    tf_saturation = SaturationTransform(lower=0.5, upper=1.5)
    tf_rnd_saturation = RandomTransform(prob=0.5, transform=tf_saturation)

    tf_reorder_channels = ReorderChannelsTransform()
    tf_rnd_reorder_channels = RandomTransform(prob=0.5,
                                              transform=tf_reorder_channels)

    #---------------------------------------------------------------------------
    # Compositions of image distortions
    #---------------------------------------------------------------------------
    tf_distort_lst = [
        tf_rnd_contrast,
        tf_rnd_saturation,
        tf_rnd_hue,
        tf_rnd_contrast
    ]
    tf_distort_1 = ComposeTransform(transforms=tf_distort_lst[:-1])
    tf_distort_2 = ComposeTransform(transforms=tf_distort_lst[1:])
    tf_distort_comp = [tf_distort_1, tf_distort_2]
    tf_distort = TransformPickerTransform(transforms=tf_distort_comp)

    #---------------------------------------------------------------------------
    # Expand sample
    #---------------------------------------------------------------------------
    tf_expand = ExpandTransform(max_ratio=4.0, mean_value=[104, 117, 123])
    tf_rnd_expand = RandomTransform(prob=expand_prob, transform=tf_expand)

    #---------------------------------------------------------------------------
    # Samplers
    #---------------------------------------------------------------------------
    samplers = [
        SamplerTransform(sample=False),
        build_sampler(0.1, sampler_trials),
        build_sampler(0.3, sampler_trials),
        build_sampler(0.5, sampler_trials),
        build_sampler(0.7, sampler_trials),
        build_sampler(0.9, sampler_trials),
        build_sampler(1.0, sampler_trials)
    ]
    tf_sample_picker = SamplePickerTransform(samplers=samplers)

    #---------------------------------------------------------------------------
    # Horizontal flip
    #---------------------------------------------------------------------------
    tf_flip = HorizontalFlipTransform()
    tf_rnd_flip = RandomTransform(prob=0.5, transform=tf_flip)

    #---------------------------------------------------------------------------
    # Transform list
    #---------------------------------------------------------------------------
    transforms = [
        ImageLoaderTransform(),
        tf_rnd_brightness,
        tf_distort,
        tf_rnd_reorder_channels,
        tf_rnd_expand,
        tf_sample_picker,
        tf_rnd_flip,
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms

#-------------------------------------------------------------------------------
def build_valid_transforms(preset, num_classes):
    tf_resize = ResizeTransform(width=preset.image_size.w,
                                height=preset.image_size.h,
                                algorithms=[cv2.INTER_LINEAR])
    transforms = [
        ImageLoaderTransform(),
        LabelCreatorTransform(preset=preset, num_classes=num_classes),
        tf_resize
    ]
    return transforms

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Process a dataset for SSD')
    parser.add_argument('--data-source', default='pascal_voc',
                        help='data source')
    parser.add_argument('--data-dir', default='pascal-voc',
                        help='data directory')
    parser.add_argument('--validation-fraction', type=float, default=0.025,
                        help='fraction of the data to be used for validation')
    parser.add_argument('--expand-probability', type=float, default=0.5,
                        help='probability of running sample expander')
    parser.add_argument('--sampler-trials', type=int, default=50,
                        help='number of time a sampler tries to find a sample')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the data samples")
    parser.add_argument('--compute-td', type=str2bool, default='True',
                        help="Compute training data")
    parser.add_argument('--preset', default='vgg300',
                        choices=['vgg300', 'vgg512'],
                        help="The neural network preset")
    parser.add_argument('--process-test', type=str2bool, default='False',
                        help="process the test dataset")
    args = parser.parse_args()

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Validation fraction:  ', args.validation_fraction)
    print('[i] Expand probability:   ', args.expand_probability)
    print('[i] Sampler trials:       ', args.sampler_trials)
    print('[i] Annotate:             ', args.annotate)
    print('[i] Compute training data:', args.compute_td)
    print('[i] Preset:               ', args.preset)
    print('[i] Process test dataset: ', args.process_test)

    #---------------------------------------------------------------------------
    # Load the data source
    #---------------------------------------------------------------------------
    print('[i] Configuring the data source...')
    try:
        source = load_data_source(args.data_source)
        source.load_trainval_data(args.data_dir, args.validation_fraction)
        if args.process_test:
            source.load_test_data(args.data_dir)
        print('[i] # training samples:   ', source.num_train)
        print('[i] # validation samples: ', source.num_valid)
        print('[i] # testing samples:    ', source.num_test)
        print('[i] # classes:            ', source.num_classes)
    except (ImportError, AttributeError, RuntimeError) as e:
        print('[!] Unable to load data source:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Annotate samples
    #---------------------------------------------------------------------------
    if args.annotate:
        print('[i] Annotating samples...')
        annotate(args.data_dir, source.train_samples, source.colors, 'train')
        annotate(args.data_dir, source.valid_samples, source.colors, 'valid')
        if args.process_test:
            annotate(args.data_dir, source.test_samples,  source.colors, 'test ')

    #---------------------------------------------------------------------------
    # Compute the training data
    #---------------------------------------------------------------------------
    if args.compute_td:
        preset = get_preset_by_name(args.preset)
        with open(args.data_dir+'/train-samples.pkl', 'wb') as f:
            pickle.dump(source.train_samples, f)
        with open(args.data_dir+'/valid-samples.pkl', 'wb') as f:
            pickle.dump(source.valid_samples, f)

        with open(args.data_dir+'/training-data.pkl', 'wb') as f:
            data = {
                'preset': preset,
                'num-classes': source.num_classes,
                'colors': source.colors,
                'lid2name': source.lid2name,
                'lname2id': source.lname2id,
                'train-transforms': build_train_transforms(preset,
                                       source.num_classes, args.sampler_trials,
                                       args.expand_probability ),
                'valid-transforms': build_valid_transforms(preset,
                                                           source.num_classes)
            }
            pickle.dump(data, f)

    return 0

if __name__ == '__main__':
    sys.exit(main())
