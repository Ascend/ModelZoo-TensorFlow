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
#coding=utf-8
import numpy as np
import math
import cv2
#import skimage.measure as measure
import skimage.metrics as measure

def psnr(target, ref, scale):
	#assume RGB image
	target_data = np.array(target)
	target_data = target_data[scale:-scale, scale:-scale]

	ref_data = np.array(ref)
	ref_data = ref_data[scale:-scale, scale:-scale]
	
	diff = ref_data - target_data
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20*math.log10(1.0/rmse)

#这部分测试源代码没有
def ssim(target, ref, scale):
	#assume RGB image
	target_data = np.array(target)
	target_data = target_data[scale:-scale, scale:-scale]

	ref_data = np.array(ref)
	ref_data = ref_data[scale:-scale, scale:-scale]
	
	#_ssim = measure.compare_ssim(target_data, ref_data)
	_ssim = measure.structural_similarity(target_data, ref_data)

	return _ssim












