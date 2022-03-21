#                                Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/

#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
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

from __future__ import division
import os, sys, glob
import numpy as np
import nibabel as nib
from skimage.transform import resize
from multiprocessing import Pool, cpu_count

def n4_correction(im_input):
    command = 'N4BiasFieldCorrection -d 3 -i ' + im_input + ' ' + ' -s 3 -c [50x50x30x20] -b [300] -o ' +  im_input.replace('.nii.gz', '_corrected.nii.gz')
    os.system(command)

def batch_works(k):
    if k == n_processes - 1:
        paths = all_paths[k * int(len(all_paths) / n_processes) : ]
    else:
        paths = all_paths[k * int(len(all_paths) / n_processes) : (k + 1) * int(len(all_paths) / n_processes)]
    for path in paths:
        n4_correction(glob.glob(os.path.join(path, '*_t1.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_t1.nii.gz'))[1])
        n4_correction(glob.glob(os.path.join(path, '*_t1ce.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_t2.nii.gz'))[0])
        n4_correction(glob.glob(os.path.join(path, '*_flair.nii.gz'))[0])
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Need at least the input data directory")
    input_path = sys.argv[1]
        
    all_paths = []
    for dirpath, dirnames, files in os.walk(input_path):
        if os.path.basename(dirpath)[0:7] == 'Brats17':
            all_paths.append(dirpath)
            
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))
