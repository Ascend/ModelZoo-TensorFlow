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

from generators.pascal import PascalVocGenerator
import cv2
import tqdm
import os
import sys
import numpy as np
from generators.utils import affine_transform, get_affine_transform
import os.path as osp


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
generator = PascalVocGenerator(
    'VOC2007',
    'test',
    shuffle_groups=False,
    skip_truncated=False,
    skip_difficult=True,
)

num_classes = generator.num_classes()
classes = list(generator.classes.keys())
flip_test = True
nms = True
keep_resolution = False
score_threshold = 0.1
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]


for i in tqdm.tqdm(range(4952)):
    image = generator.load_image(i)
    src_image = image.copy()
    txt_content = ''
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0

    tgt_w = generator.input_size
    tgt_h = generator.input_size
        
    npu_PATH = sys.argv[1]
    detections_temp = np.fromfile(os.path.join(npu_PATH,"davinci_{}_output0.bin".format(generator.image_names[i])),\
            dtype="float32")
    detections = detections_temp[:600].reshape(100,6)
    scores = detections[:, 4]
    indices = np.where(scores > score_threshold)[0]

    # select those detections
    detections = detections[indices]
    detections_copy = detections.copy()
    detections = detections.astype(np.float64)
    trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

    for j in range(detections.shape[0]):
        detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
        detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, src_image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, src_image.shape[0])
    for detection in detections:
        xmin = int(round(detection[0]))
        ymin = int(round(detection[1]))
        xmax = int(round(detection[2]))
        ymax = int(round(detection[3]))
        score = '{:.4f}'.format(detection[4])
        class_id = int(detection[5])
        color = colors[class_id]
        class_name = classes[class_id]
        label = '-'.join([class_name, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        txt_content += "{} {} {} {} {} {}\n".format(class_name,score,xmin,ymin,xmax,ymax)
        image_fname = generator.image_names[i]
        npu_predict = sys.argv[2]
        with open(os.path.join(npu_predict,'{}.txt'.format(image_fname)),'w') as f:
            f.write(txt_content)
