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


import os
from tqdm import tqdm
from network_add_bn import East
from postprocess import predict_txt
import cfg

if __name__=='__main__':
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights("./model.h5")
    image_test_dir = os.path.join(cfg.data_dir, 'rename_images/')
    txt_test_dir = os.path.join(cfg.data_dir, 'txt_test')
    test_imgname_list = os.listdir(image_test_dir)
    test_imgname_list = sorted(test_imgname_list)
    print('found %d test images.' % len(test_imgname_list))
    for test_img_name, _ in zip(test_imgname_list,
                                tqdm(range(len(test_imgname_list)))):
        img_path = os.path.join(image_test_dir, test_img_name)
        txt_path = os.path.join(txt_test_dir, 'res_'+test_img_name[:-4] + '.txt')
        predict_txt(east_detect, img_path, txt_path, cfg.pixel_threshold, True)

