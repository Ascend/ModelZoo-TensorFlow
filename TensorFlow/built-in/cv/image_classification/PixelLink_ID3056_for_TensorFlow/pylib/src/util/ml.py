#
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
#
from npu_bridge.npu_init import *
import logging
import numpy as np
import util.dec
import util.np

@util.dec.print_calling
def kmeans(samples, k, criteria = None, attempts = 3, flags = None):
    import cv2
    
    if flags == None:
        flags = cv2.KMEANS_RANDOM_CENTERS
    if criteria == None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    samples = np.asarray(samples, dtype = np.float32)
    _,labels,centers = cv2.kmeans(samples, k, criteria, attempts, flags)
    labels = util.np.flatten(labels)
    clusters = [None]*k
    for idx, label in enumerate(labels):
        if clusters[label] is None:
            clusters[label] = []
        clusters[label].append(idx)
        
    for  idx, cluster in enumerate(clusters):
        if cluster == None:
            logging.warn('Empty cluster appeared.')
            clusters[idx] = []
            
    return labels, clusters, centers

