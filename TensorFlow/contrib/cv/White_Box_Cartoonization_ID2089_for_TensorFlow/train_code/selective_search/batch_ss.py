#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================

from npu_bridge.npu_init import *


import numpy as np
from adaptive_color import label2rgb
from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
from util import switch_color_space
from structure import HierarchicalGrouping


def color_ss_map(image, color_space='Lab', k=10, 
                 sim_strategy='CTSF', seg_num=200, power=1):
    
    img_seg = felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping
    
    while S.num_regions() > seg_num:
        
        i,j = S.get_highest_similarity()
        S.merge_region(i,j)
        S.remove_similarities(i,j)
        S.calculate_similarity_for_new_region()
    
    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image+1)/2
    image = image**power
    image = image/np.max(image)
    image = image*2 - 1
    
    return image


def selective_adacolor(batch_image, seg_num=200, power=1):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)\
                         (image, seg_num, power) for image in batch_image)
    return np.array(batch_out)


if __name__ == '__main__':
    pass
