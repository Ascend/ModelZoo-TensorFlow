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
import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
import cv2

CFG = parse_config_utils.lanenet_cfg

image_path = './eval_data/test_img.jpg'
image_vis = cv2.imread(image_path, cv2.IMREAD_COLOR)

b_out_path = './eval_data/frozen_output_0.bin'
i_out_path = './eval_data/frozen_output_1.bin'
b_out = np.fromfile(b_out_path, dtype=np.int64)
i_out = np.fromfile(i_out_path, dtype=np.float32)
b_out = np.reshape(b_out, (1, 256, 512))
i_out = np.reshape(i_out, (1, 256, 512, 4))

postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
postprocess_result = postprocessor.postprocess(
                binary_seg_result=b_out[0],
                instance_seg_result=i_out[0],
                source_image=image_vis,
                with_lane_fit=True,
                data_source='tusimple'
            )
mask_image = postprocess_result['mask_image']
src_image = postprocess_result['source_image']
gt_path = './eval_data/gt.png'
gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
gt_trans = cv2.resize(gt, (512, 256), interpolation=cv2.INTER_LINEAR)

# -------------- 计算准确率 ------------------ #
gt_gray = cv2.cvtColor(gt_trans, cv2.COLOR_BGR2GRAY)
mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
WIDTH = mask_image_gray.shape[0]
HIGTH = mask_image_gray.shape[1]
tp_count = 0
tn_count = 0
for i in range(WIDTH):
    for j in range(HIGTH):
        if mask_image_gray[i, j] != 0 and gt_gray[i, j] != 0:
            tp_count = tp_count + 1
        if mask_image_gray[i, j] == 0 and gt_gray[i, j] == 0:
            tn_count = tn_count + 1
Accuracy = (int(tp_count) + int(tn_count)) / (int(WIDTH) * int(HIGTH))

print("\n# Metric_om "
      "\n     Accuracy：{:.3f}".format(Accuracy))

cv2.imwrite('./eval_output/mask_om.jpg', mask_image)
cv2.imwrite('./eval_output/src_om.jpg', src_image)
