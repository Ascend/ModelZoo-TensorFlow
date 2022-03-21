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
import data_processing

TEST_LIST_PATH = '/home/test_user05/test.list'
OUTPUT_DIR = '/home/test_user05/out'
NUM_CLASSES = 101
BATCH_SIZE = 60

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # test data
    test_num = data_processing.get_test_num(TEST_LIST_PATH)
    test_video_indices = range(test_num)

    # Get test data
    batch_index = 0
    batch_data, batch_index = data_processing.get_batches(TEST_LIST_PATH, NUM_CLASSES, batch_index,test_video_indices, BATCH_SIZE)

    batch_clips = batch_data['clips']
    batch_labels = batch_data['labels']
    # save data to bin
    batch_clips.tofile(OUTPUT_DIR + "/batch_clips.bin")
    batch_labels.tofile(OUTPUT_DIR + "/batch_labels.bin")
